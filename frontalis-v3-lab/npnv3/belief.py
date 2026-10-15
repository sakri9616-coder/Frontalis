"""信念ネット: スコープ帰属つき階層ベイズ（Beta–Bernoulli、経験ベイズ近似）＋ニューラル事前ヘッド。

予測対象（SPEC 4.1）
  - 参加者 e が型 k の結果を起こす確率  P(y_{e,k}=1 | e, class(e), context)
  - 文脈 c で自分・世界から型 k の結果が起きる確率  P(y_{c,k}=1 | c)

階層（SPEC 4.2）
  人の層: 個体 ind ← 類 cls ← 一般 gen        文脈の層: 文脈 ctx ← 世界 world
  個体  p_ind = (κ_e m_cls + n1) / (κ_e + n)
  類    m_cls = (κ_c prior_cls + Σ w_i r_i) / (κ_c + Σ w_i),  w_i = n_i/(n_i+κ_e),  r_i = n1_i/n_i
  一般  m_gen = (κ_g E[prior] + Σ_cls w_cls r_cls) / (κ_g + Σ_cls w_cls),  w_cls = W_cls/(W_cls+κ_c),  r_cls = Σ w_i r_i / W_cls
  類の有効標本数は個体の数で上限が決まる（W_cls = Σ w_i < 個体数）。

スコープ帰属（SPEC 4.3）
  φ_person ∝ π_{k,person} Bern(y; p_ind),  φ_ctx ∝ π_{k,ctx} Bern(y; p_ctx,k^person)
  予測 P(y_{e,k}=1) = π̄_person p_ind + π̄_ctx p_ctx^person
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .innate import N_TYPES, OUTCOME_TYPES
from .lexicon import EMB_DIM, Lexicon
from .mlp import MLP, sigmoid

_P_MIN = 1e-4
_P_MAX = 1.0 - 1e-4


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


class Prediction:
    """ある（文脈, 参加者）に対する信念の予測。状態は変えない。"""
    __slots__ = ("p_person", "p_ind", "p_ctxp", "pi_p", "pi_c", "p_sw", "var_person", "var_sw",
                 "known", "head_used", "n_ind", "m_cls")

    def __init__(self, **kw: Any):
        for k, v in kw.items():
            setattr(self, k, v)


class BeliefNet:
    def __init__(self, cfg: Config, lex: Lexicon, rng: Optional[np.random.Generator]):
        self.cfg = cfg
        self.lex = lex
        K = N_TYPES
        self.ind_ids: List[str] = []
        self.ind_cls: List[str] = []
        self.ind_idx: Dict[str, int] = {}
        self.N1 = np.zeros((0, K))
        self.N0 = np.zeros((0, K))
        self.ctx_names: List[str] = []
        self.ctx_idx: Dict[str, int] = {}
        self.CP1 = np.zeros((0, K))   # 文脈で「人が k を起こした」計数（責任 φ_ctx で重み付け）
        self.CP0 = np.zeros((0, K))
        self.CS1 = np.zeros((0, K))   # 文脈で「自分・世界が k を起こした」計数
        self.CS0 = np.zeros((0, K))
        self.pi = np.ones((K, 2))     # π_k の計数: 列 0 = 人のせい、列 1 = 場面のせい（Dirichlet(1,1)）
        self.mag = np.full(K, cfg.mag_init)
        self.mag_n = np.zeros(K)
        self.n_obs = 0                # 学習に使った参加者観測の総数（ヘッドの使用判定）
        self.prior_mean = cfg.prior_a / (cfg.prior_a + cfg.prior_b)
        self.head: Optional[MLP] = None
        self.rb_x = np.zeros((cfg.replay_size, 2 * EMB_DIM))
        self.rb_y = np.zeros((cfg.replay_size, K))
        self.rb_len = 0
        self.rb_ptr = 0
        if not cfg.no_head:
            if rng is None:
                raise ValueError("ヘッドの初期化には乱数が必要です")
            bias = np.full(K, _logit(self.prior_mean))
            self.head = MLP(2 * EMB_DIM, cfg.head_hidden, K, rng, out_bias=bias)

    # ------------------------------------------------------------------ 登録
    def _add_individual(self, pid: str, cls: str) -> int:
        i = len(self.ind_ids)
        self.ind_ids.append(pid)
        self.ind_cls.append(cls)
        self.ind_idx[pid] = i
        z = np.zeros((1, N_TYPES))
        self.N1 = np.vstack([self.N1, z])
        self.N0 = np.vstack([self.N0, z])
        return i

    def _add_context(self, name: str) -> int:
        i = len(self.ctx_names)
        self.ctx_names.append(name)
        self.ctx_idx[name] = i
        z = np.zeros((1, N_TYPES))
        self.CP1 = np.vstack([self.CP1, z]); self.CP0 = np.vstack([self.CP0, z])
        self.CS1 = np.vstack([self.CS1, z]); self.CS0 = np.vstack([self.CS0, z])
        return i

    def check_class(self, pid: str, cls: str) -> None:
        i = self.ind_idx.get(pid)
        if i is not None and self.ind_cls[i] != cls:
            raise ValueError(f"個体 {pid!r} の類が以前（{self.ind_cls[i]!r}）と異なります: {cls!r}")

    # ------------------------------------------------------------------ 集計
    def head_trained(self) -> bool:
        return self.head is not None and self.n_obs >= self.cfg.head_min_samples

    def _class_agg(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """類ごとの (W = Σ w_i, WR = Σ w_i r_i)。"""
        ke = self.cfg.kappa_e
        n = self.N1 + self.N0
        w = n / (n + ke)
        r = np.divide(self.N1, n, out=np.zeros_like(n), where=n > 0)
        wr = w * r
        agg: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for c in sorted(set(self.ind_cls)):
            rows = [i for i, cc in enumerate(self.ind_cls) if cc == c]
            agg[c] = (w[rows].sum(axis=0), wr[rows].sum(axis=0))
        return agg

    def _gen_mean(self, agg: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        num = np.full(N_TYPES, cfg.kappa_g * self.prior_mean)
        den = np.full(N_TYPES, cfg.kappa_g)
        for c in sorted(agg):
            W, WR = agg[c]
            rc = np.divide(WR, W, out=np.zeros_like(W), where=W > 1e-12)
            wc = W / (W + cfg.kappa_c)
            num = num + wc * rc
            den = den + wc
        return num / den, den - cfg.kappa_g      # 平均、有効標本数 Σ w_cls

    def _world_mean(self, C1: np.ndarray, C0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        n = C1 + C0
        w = n / (n + cfg.kappa_e)
        r = np.divide(C1, n, out=np.zeros_like(n), where=n > 0)
        num = cfg.kappa_g * self.prior_mean + (w * r).sum(axis=0)
        den = cfg.kappa_g + w.sum(axis=0)
        return num / den, w.sum(axis=0)

    def _head_prior(self, classes: List[str], ctx: Optional[str]) -> np.ndarray:
        ce = self.lex.ctx_emb(ctx)
        X = np.array([np.concatenate([self.lex.cls_emb(c), ce]) for c in classes])
        return np.clip(self.head.predict_proba(X), _P_MIN, _P_MAX)

    # ------------------------------------------------------------------ 予測
    def predict(self, ctx: Optional[str], participants: List[Dict[str, str]]) -> Prediction:
        cfg = self.cfg
        K = N_TYPES
        ke, kc, kg = cfg.kappa_e, cfg.kappa_c, cfg.kappa_g
        agg = self._class_agg()
        m_gen, W_gen = self._gen_mean(agg)
        v_gen = m_gen * (1 - m_gen) / (kg + W_gen + 1.0)
        # --- 文脈の層（人が起こす k / 自分・世界が起こす k）---
        mw_p, Ww_p = self._world_mean(self.CP1, self.CP0)
        mw_s, Ww_s = self._world_mean(self.CS1, self.CS0)
        vw_p = mw_p * (1 - mw_p) / (kg + Ww_p + 1.0)
        vw_s = mw_s * (1 - mw_s) / (kg + Ww_s + 1.0)
        ci = self.ctx_idx.get(ctx) if ctx is not None else None
        if ci is None:
            p_ctxp, n_c = mw_p.copy(), np.zeros(K)
            p_sw, n_s = mw_s.copy(), np.zeros(K)
        else:
            n_c = self.CP1[ci] + self.CP0[ci]
            p_ctxp = (ke * mw_p + self.CP1[ci]) / (ke + n_c)
            n_s = self.CS1[ci] + self.CS0[ci]
            p_sw = (ke * mw_s + self.CS1[ci]) / (ke + n_s)
        v_ctxp = p_ctxp * (1 - p_ctxp) / (ke + n_c + 1.0) + vw_p
        v_sw = p_sw * (1 - p_sw) / (ke + n_s + 1.0) + vw_s
        # --- スコープの事前 ---
        if cfg.no_scope:
            pi_p = np.ones(K); pi_c = np.zeros(K)
        else:
            tot = self.pi.sum(axis=1)
            pi_p = self.pi[:, 0] / tot
            pi_c = self.pi[:, 1] / tot
        # --- 人の層 ---
        nP = len(participants)
        p_person = np.zeros((nP, K)); p_ind = np.zeros((nP, K)); p_ctxp_all = np.zeros((nP, K))
        var_person = np.zeros((nP, K)); n_ind_all = np.zeros((nP, K)); m_cls_all = np.zeros((nP, K))
        known = []
        use_head = self.head_trained()
        if nP:
            priors = self._head_prior([p["class"] for p in participants], ctx) if use_head else None
        for j, p in enumerate(participants):
            c = p["class"]
            W, WR = agg.get(c, (np.zeros(K), np.zeros(K)))
            prior = priors[j] if use_head else m_gen
            m_cls = (kc * prior + WR) / (kc + W)
            v_cls = m_cls * (1 - m_cls) / (kc + W + 1.0)
            i = self.ind_idx.get(p["id"])
            if i is None:
                n_i = np.zeros(K)
                pi_ind = m_cls
                known.append(False)
            else:
                n_i = self.N1[i] + self.N0[i]
                pi_ind = (ke * m_cls + self.N1[i]) / (ke + n_i)
                known.append(True)
            v_ind = pi_ind * (1 - pi_ind) / (ke + n_i + 1.0)
            p_ind[j] = pi_ind
            p_ctxp_all[j] = p_ctxp
            p_person[j] = pi_p * pi_ind + pi_c * p_ctxp
            var_person[j] = v_ind + v_cls + v_gen + v_ctxp
            n_ind_all[j] = n_i
            m_cls_all[j] = m_cls
        return Prediction(p_person=np.clip(p_person, _P_MIN, _P_MAX), p_ind=p_ind, p_ctxp=p_ctxp_all,
                          pi_p=pi_p, pi_c=pi_c, p_sw=np.clip(p_sw, _P_MIN, _P_MAX),
                          var_person=var_person, var_sw=v_sw, known=known, head_used=use_head,
                          n_ind=n_ind_all, m_cls=m_cls_all)

    # ------------------------------------------------------------------ 責任 φ
    def responsibility(self, pred: Prediction, y: np.ndarray) -> np.ndarray:
        """観測 y_{e,k} の責任 φ_person（行 = 参加者、列 = 型）。φ_ctx = 1 - φ_person。"""
        if self.cfg.no_scope or y.shape[0] == 0:
            return np.ones_like(y)
        pi_ind = np.clip(pred.p_ind, _P_MIN, _P_MAX)
        pi_ctx = np.clip(pred.p_ctxp, _P_MIN, _P_MAX)
        bp = np.where(y > 0.5, pi_ind, 1.0 - pi_ind)
        bc = np.where(y > 0.5, pi_ctx, 1.0 - pi_ctx)
        a = pred.pi_p[None, :] * bp
        b = pred.pi_c[None, :] * bc
        return a / (a + b)

    # ------------------------------------------------------------------ 更新
    def update(self, ctx: str, participants: List[Dict[str, str]], y_person: np.ndarray, y_sw: np.ndarray,
               phi: np.ndarray, outcomes: List[Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
        """計数 → π → 型の大きさ → リプレイ緩衝 → 事前ヘッドの SGD（観測そのものへの交差エントロピー）。"""
        cfg = self.cfg
        ci = self.ctx_idx.get(ctx)
        if ci is None:
            ci = self._add_context(ctx)
        # 人の層と文脈の層（人が起こす k）
        for j, p in enumerate(participants):
            i = self.ind_idx.get(p["id"])
            if i is None:
                i = self._add_individual(p["id"], p["class"])
            self.N1[i] += phi[j] * y_person[j]
            self.N0[i] += phi[j] * (1.0 - y_person[j])
            self.CP1[ci] += (1.0 - phi[j]) * y_person[j]
            self.CP0[ci] += (1.0 - phi[j]) * (1.0 - y_person[j])
            if not cfg.no_scope:
                self.pi[:, 0] += phi[j]
                self.pi[:, 1] += 1.0 - phi[j]
        # 文脈の層（自分・世界が起こす k）
        self.CS1[ci] += y_sw
        self.CS0[ci] += 1.0 - y_sw
        # 型の大きさ（平均の移動平均）
        for o in outcomes:
            k = OUTCOME_TYPES.index(o["type"])
            rate = max(1.0 / (self.mag_n[k] + 1.0), cfg.mag_rate)
            self.mag[k] += rate * (o["magnitude"] - self.mag[k])
            self.mag_n[k] += 1.0
        # リプレイ緩衝（観測そのもの）
        ce = self.lex.ctx_emb(ctx)
        for j, p in enumerate(participants):
            self.rb_x[self.rb_ptr] = np.concatenate([self.lex.cls_emb(p["class"]), ce])
            self.rb_y[self.rb_ptr] = y_person[j]
            self.rb_ptr = (self.rb_ptr + 1) % cfg.replay_size
            self.rb_len = min(self.rb_len + 1, cfg.replay_size)
            self.n_obs += 1
        # 事前ヘッドの学習（目標は観測。事後平均ではない）
        loss = None
        if self.head is not None and self.rb_len > 0:
            for _ in range(cfg.head_batches):
                idx = rng.integers(0, self.rb_len, size=cfg.head_batch_size)
                loss = self.head.sgd_step(self.rb_x[idx], self.rb_y[idx], cfg.head_lr)
        return {"head_loss": loss}

    # ------------------------------------------------------------------ 要約（CLI show 用）
    def summary(self) -> Dict[str, Any]:
        agg = self._class_agg()
        m_gen, _ = self._gen_mean(agg)
        out: Dict[str, Any] = {"n_individuals": len(self.ind_ids), "n_contexts": len(self.ctx_names),
                               "n_obs": self.n_obs, "head_trained": self.head_trained(),
                               "pi_person": {t: float(self.pi[k, 0] / self.pi[k].sum()) for k, t in enumerate(OUTCOME_TYPES)},
                               "m_gen": {t: float(m_gen[k]) for k, t in enumerate(OUTCOME_TYPES)},
                               "class_rate": {}}
        kc = self.cfg.kappa_c
        for c, (W, WR) in agg.items():
            m_cls = (kc * m_gen + WR) / (kc + W)
            out["class_rate"][c] = {t: float(m_cls[k]) for k, t in enumerate(OUTCOME_TYPES)}
        return out

    # ------------------------------------------------------------------ 保存・復元
    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        meta = {"ind_ids": list(self.ind_ids), "ind_cls": list(self.ind_cls), "ctx_names": list(self.ctx_names),
                "n_obs": int(self.n_obs), "rb_len": int(self.rb_len), "rb_ptr": int(self.rb_ptr),
                "has_head": self.head is not None}
        arrays = {"b_N1": self.N1, "b_N0": self.N0, "b_CP1": self.CP1, "b_CP0": self.CP0, "b_CS1": self.CS1,
                  "b_CS0": self.CS0, "b_pi": self.pi, "b_mag": self.mag, "b_mag_n": self.mag_n,
                  "b_rb_x": self.rb_x, "b_rb_y": self.rb_y}
        if self.head is not None:
            arrays.update(self.head.get_arrays("b_head_"))
        return meta, arrays

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        self.ind_ids = list(meta["ind_ids"]); self.ind_cls = list(meta["ind_cls"])
        self.ind_idx = {p: i for i, p in enumerate(self.ind_ids)}
        self.ctx_names = list(meta["ctx_names"]); self.ctx_idx = {c: i for i, c in enumerate(self.ctx_names)}
        self.n_obs = int(meta["n_obs"]); self.rb_len = int(meta["rb_len"]); self.rb_ptr = int(meta["rb_ptr"])
        g = lambda k: np.array(arrays[k], dtype=np.float64)
        self.N1, self.N0 = g("b_N1"), g("b_N0")
        self.CP1, self.CP0, self.CS1, self.CS0 = g("b_CP1"), g("b_CP0"), g("b_CS1"), g("b_CS0")
        self.pi, self.mag, self.mag_n = g("b_pi"), g("b_mag"), g("b_mag_n")
        self.rb_x, self.rb_y = g("b_rb_x"), g("b_rb_y")
        if meta["has_head"]:
            self.head = MLP(2 * EMB_DIM, self.cfg.head_hidden, N_TYPES, None)
            self.head.set_arrays(arrays, "b_head_")
        else:
            self.head = None
