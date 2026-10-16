"""価値ネット: 階層カルマン回帰（MF）、モデルベース価値（MB）、調停、気分と感情の読み出し。

二つのカルマン回帰（SPEC 5.1）
  人モデル : 参加者の価値。特徴 = people + cls:<c> + ind:<id> + 類の埋め込み 8 次元
  状況モデル: 文脈と概念の価値。特徴 = world + ctx:<c> + concept:<x> + 概念埋め込みの平均 8 次元
  教師は帰属 φ で分ける。更新 K = Σz/(zᵀΣz+R)、W += K(Δu - zᵀW)、Σ ← Σ - K zᵀΣ、続いて関与した対角に q。
価値ネット自身の出力は、どの更新の目標にも入らない（docs/04 補題1）。教師は Δu（欲求の効用変化）だけ。

v3.2 の変更（SPEC-v3.2）
  1.1 欲求ごとの観測雑音 R_k = R / a_k。関心の重み a が大きい欲求ほど精度高く・速く学ぶ。利得が欲求ごとに変わるので、Σ は欲求ごとに持つ
      （形 (5, n, n)）。a の目標は |Δu|（外生）で W に依存しない。a → W の一方向だけで、W → a はない。
  1.3 Concern の方向 concern_direction ∈ {sensitize, habituate}。
  1.2 の C → Q は npn.py（Q = U_val + λ_CQ ΣC(手がかり)）。

v3.1 の追加（SPEC-v3.1）
  Concern     欲求の関心の重み a（1.1）: 目標は |Δu|（外生）。感じた価値・MB・好悪の読み出しで欲求を重み付ける。
  Conditioning 条件づけられた価値 C（1.2）: 手がかり（概念・参加者・結果型）への線形の重み。TD(0)、目標は r + γ C_frozen。
              v3.3 4.2: 手がかりごとに最大 5 個の潜在原因（価値・文脈の計数・最後に使われた時刻）。応答は事前の重みでの期待値。single_cause で v3.2 と同じ。
  felt_value  f = tanh(((Σ a_k Δu_k) + λ_C ΣC + η_ref(Σ a_k Δu_k − Q))/s_f + β m)（1.3）
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import innate
from .config import Config
from .innate import N_NEEDS, N_TYPES
from .lexicon import EMB_DIM, EmbeddingStore

Feat = Tuple[str, str, float]   # (特徴名, 層, 値)


# ======================================================================
# カルマン回帰（欲求ごとに共分散 Σ_k を持つ。v3.2）
# ======================================================================
A_FLOOR = 1e-3        # R_k = R / a_k の a_k の下限（a_k → 0 で観測雑音が発散しないように）


class KalmanModel:
    """階層カルマン回帰。W は (特徴, 5 欲求)、Σ は (5 欲求, 特徴, 特徴)。

    v3.2: 欲求 k の観測雑音を R_k = R / a_k にする（a の和は 5 のまま）ので、利得が欲求ごとに違う。
    a = None（または全部 1）なら、全欲求で同じ Σ になり v3.1 と同じ更新になる。
    """

    def __init__(self, var: Dict[str, float], q: Dict[str, float], obs_noise: float):
        self.var = dict(var)
        self.q = dict(q)
        self.R = obs_noise
        self.names: List[str] = []
        self.layers: List[str] = []
        self.idx: Dict[str, int] = {}
        self.W = np.zeros((0, N_NEEDS))
        self.S = np.zeros((N_NEEDS, 0, 0))

    def _add(self, name: str, layer: str) -> int:
        i = len(self.names)
        self.names.append(name)
        self.layers.append(layer)
        self.idx[name] = i
        self.W = np.vstack([self.W, np.zeros((1, N_NEEDS))])
        S2 = np.zeros((N_NEEDS, i + 1, i + 1))
        S2[:, :i, :i] = self.S
        S2[:, i, i] = self.var[layer]     # 新しい特徴: 事前平均 0、事前分散は層ごと
        self.S = S2
        return i

    def predict(self, feats: Sequence[Feat], a: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool]:
        """予測 zᵀW（5 欲求）、予測分散、すべての特徴が既知か。状態は変えない。
        未登録の特徴は、事前平均 0・事前分散（他の特徴とは無相関）として扱う。
        予測分散は、a で重み付けた予測 Σ_k a_k (zᵀW_k) の分散を欲求の数で割った (1/5) Σ_k a_k² zᵀΣ_k z
        （欲求の誤差は独立と見る。a = 1 かつ Σ が共通なら v3.1 の zᵀΣz と一致する）。a = None は a = 1。"""
        w2 = np.ones(N_NEEDS) if a is None else np.asarray(a, dtype=np.float64) ** 2
        scale = float(w2.mean())
        pred = np.zeros(N_NEEDS)
        var = 0.0
        ii: List[int] = []
        zz: List[float] = []
        all_known = True
        for name, layer, v in feats:
            if v == 0.0:
                continue
            i = self.idx.get(name)
            if i is None:
                var += scale * self.var[layer] * v * v
                all_known = False
            else:
                ii.append(i)
                zz.append(v)
        if ii:
            ia = np.array(ii)
            z = np.array(zz)
            pred = z @ self.W[ia]
            quad = np.einsum("i,kij,j->k", z, self.S[:, ia][:, :, ia], z)      # 欲求ごとの zᵀΣ_k z
            var += float(w2 @ quad) / N_NEEDS
        return pred, var, all_known

    def _prepare(self, feats: Sequence[Feat]) -> Tuple[np.ndarray, np.ndarray]:
        for name, layer, v in feats:
            if v != 0.0 and name not in self.idx:
                self._add(name, layer)
        ii = np.array([self.idx[name] for name, _, v in feats if v != 0.0], dtype=int)
        z = np.array([v for _, _, v in feats if v != 0.0])
        return ii, z

    def update(self, feats: Sequence[Feat], teacher: np.ndarray, a: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """1 観測のカルマン更新（欲求ごと）。K_k = Σ_k z / (zᵀΣ_k z + R/a_k)、W_k += K_k (Δu_k − zᵀW_k)、
        Σ_k ← Σ_k − K_k zᵀΣ_k、続いて関与した対角に q。K は全特徴に対して密（共分散を通じて関与しない特徴も動く）。"""
        ii, z = self._prepare(feats)
        prec = np.ones(N_NEEDS) if a is None else np.maximum(np.asarray(a, dtype=np.float64), A_FLOOR)
        Sz = np.einsum("kij,j->ki", self.S[:, :, ii], z)       # Σ_k z（欲求 k、全特徴）
        zSz = Sz[:, ii] @ z                                     # (5,)
        den = zSz + self.R / prec
        K = Sz / den[:, None]                                    # (5, n)
        pred = z @ self.W[ii]
        err = np.asarray(teacher, dtype=np.float64) - pred
        dW = (K * err[:, None]).T                                # (n, 5)
        self.W = self.W + dW
        S2 = self.S - K[:, :, None] * Sz[:, None, :]            # Σ_k − K_k zᵀ Σ_k
        S2 = 0.5 * (S2 + S2.transpose(0, 2, 1))
        for i in ii:
            S2[:, i, i] += self.q[self.layers[i]]               # 関与した特徴だけに過程雑音
        self.S = S2
        return {"dW": dW, "err": err, "pred": pred, "var": float(zSz.mean())}

    def update_fixed(self, feats: Sequence[Feat], teacher: np.ndarray, target: str, rate: float) -> Dict[str, Any]:
        """3-F 型の固定教師: W[target] += rate * Δu。予測誤差も Σ も使わない。"""
        ii, z = self._prepare(feats)
        pred = z @ self.W[ii]
        dW = np.zeros_like(self.W)
        dW[self.idx[target]] = rate * np.asarray(teacher, dtype=np.float64)
        self.W = self.W + dW
        return {"dW": dW, "err": np.asarray(teacher, dtype=np.float64) - pred, "pred": pred, "var": 0.0}

    def dW_dict(self, dW: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self.names):
            row = dW[i]
            if np.any(row != 0.0):
                out[name] = row.copy()
        return out

    def get_state(self, prefix: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        return ({"names": list(self.names), "layers": list(self.layers)},
                {f"{prefix}W": self.W, f"{prefix}S": self.S})

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray], prefix: str) -> None:
        self.names = list(meta["names"])
        self.layers = list(meta["layers"])
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.W = np.array(arrays[f"{prefix}W"], dtype=np.float64)
        self.S = np.array(arrays[f"{prefix}S"], dtype=np.float64)


# ======================================================================
# 価値ネット
# ======================================================================
class ValueNet:
    def __init__(self, cfg: Config, emb: EmbeddingStore):
        self.cfg = cfg
        self.emb = emb
        if cfg.flat_value:
            # 階層を外す: 最下層（個体・文脈・概念）の特徴だけ、事前分散 0.25
            pv = {"ind": cfg.flat_var}
            pq = {"ind": cfg.person_q["ind"]}
            sv = {"ctx": cfg.flat_var, "concept": cfg.flat_var}
            sq = {"ctx": cfg.sit_q["ctx"], "concept": cfg.sit_q["concept"]}
        else:
            pv, pq, sv, sq = cfg.person_var, cfg.person_q, cfg.sit_var, cfg.sit_q
        self.person = KalmanModel(pv, pq, cfg.obs_noise)
        self.sit = KalmanModel(sv, sq, cfg.obs_noise)

    # --- 特徴 ---
    def person_feats(self, pid: str, cls: str) -> List[Feat]:
        if self.cfg.flat_value:
            return [(f"ind:{pid}", "ind", 1.0)]
        feats: List[Feat] = [("people", "people", 1.0), (f"cls:{cls}", "cls", 1.0), (f"ind:{pid}", "ind", 1.0)]
        e = self.emb.cls(cls)
        feats += [(f"emb:{d}", "emb", float(e[d])) for d in range(EMB_DIM)]
        return feats

    def situation_feats(self, ctx: Optional[str], concepts: Sequence[str], with_ctx: bool = True) -> List[Feat]:
        feats: List[Feat] = []
        flat = self.cfg.flat_value
        if not flat:
            feats.append(("world", "world", 1.0))
        if with_ctx and ctx is not None:
            feats.append((f"ctx:{ctx}", "ctx", 1.0))
        feats += [(f"concept:{c}", "concept", 1.0) for c in concepts]
        if not flat and concepts:
            m = self.emb.concept_mean(concepts)
            feats += [(f"cemb:{d}", "cemb", float(m[d])) for d in range(EMB_DIM)]
        return feats

    # --- MF 評価（状態を変えない）---
    def mf(self, participants: Sequence[Dict[str, str]], ctx: str, concepts: Sequence[str],
           a: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """MF 価値。欲求ごとの予測を関心の重み a で合計する（a = 1 なら v3.0 と同じ）。"""
        w = np.ones(N_NEEDS) if a is None else a
        wv = self._var_weights(a)
        persons = []
        V = 0.0
        var = 0.0
        for p in participants:
            pred, v, known = self.person.predict(self.person_feats(p["id"], p["class"]), wv)
            val = float(w @ pred)
            persons.append({"id": p["id"], "value": val, "var": float(v), "pred": pred, "known": known})
            V += val
            var += float(v)
        pred_s, v_s, known_s = self.sit.predict(self.situation_feats(ctx, concepts), wv)
        vs = float(w @ pred_s)
        V += vs
        var += float(v_s)
        return {"V_MF": V, "var_MF": var, "persons": persons,
                "V_MF_person": float(sum(x["value"] for x in persons)),
                "situation": {"value": vs, "var": float(v_s), "pred": pred_s, "known": known_s}}

    def _var_weights(self, a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """予測分散の a 重み。no_precision_weighting（アブレーション）では v3.1 と同じく重みなし。"""
        return None if (a is None or self.cfg.no_precision_weighting) else a

    def like_entity(self, pid: str, cls: str, a: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray, bool]:
        pred, v, known = self.person.predict(self.person_feats(pid, cls), self._var_weights(a))
        return float((np.ones(N_NEEDS) if a is None else a) @ pred), float(v), pred, known

    def like_concept(self, name: str, a: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray, bool]:
        """概念の好悪 Like(x) = 一般 + 概念 + 埋め込みの寄与（文脈は含めない）。測定用のプローブ。"""
        pred, v, known = self.sit.predict(self.situation_feats(None, [name], with_ctx=False), self._var_weights(a))
        return float((np.ones(N_NEEDS) if a is None else a) @ pred), float(v), pred, known

    def prepare(self, participants: Sequence[Dict[str, str]], ctx: str, concepts: Sequence[str]) -> None:
        """この経験で使う特徴を（学習せずに）登録する。ヤコビアンの検査用（状態写像の定義域を揃える）。"""
        for p in participants:
            self.person._prepare(self.person_feats(p["id"], p["class"]))
        self.sit._prepare(self.situation_feats(ctx, concepts))

    # --- 学習 ---
    def learn(self, participants: Sequence[Dict[str, str]], ctx: str, concepts: Sequence[str],
              person_teachers: Sequence[np.ndarray], sit_teacher: np.ndarray,
              a: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """人モデルを参加者ごとに、状況モデルを 1 回、帰属で分けた教師で更新する。
        a: この経験を受け止める直前の関心の重み。欲求 k の観測雑音が R/a_k になる（v3.2 1.1）。None なら全欲求で R。"""
        cfg = self.cfg
        a_eff = None if cfg.no_precision_weighting else a
        dW_person: Dict[str, np.ndarray] = {}
        err_person = []
        for p, t in zip(participants, person_teachers):
            feats = self.person_feats(p["id"], p["class"])
            if cfg.fixed_teacher:
                r = self.person.update_fixed(feats, t, f"ind:{p['id']}", cfg.fixed_teacher_rate)
            else:
                r = self.person.update(feats, t, a_eff)
            for name, row in self.person.dW_dict(r["dW"]).items():      # 参加者ごとの変化量を特徴名で足し合わせる
                dW_person[name] = dW_person[name] + row if name in dW_person else row
            err_person.append(r["err"])
        feats_s = self.situation_feats(ctx, concepts)
        if cfg.fixed_teacher:
            rs = self.sit.update_fixed(feats_s, sit_teacher, f"ctx:{ctx}", cfg.fixed_teacher_rate)
        else:
            rs = self.sit.update(feats_s, sit_teacher, a_eff)
        return {"dW_person": dW_person, "dW_situation": self.sit.dW_dict(rs["dW"]),
                "err_person": err_person, "err_situation": rs["err"]}

    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        m1, a1 = self.person.get_state("v_person_")
        m2, a2 = self.sit.get_state("v_sit_")
        a1.update(a2)
        return {"person": m1, "situation": m2}, a1

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        self.person.set_state(meta["person"], arrays, "v_person_")
        self.sit.set_state(meta["situation"], arrays, "v_sit_")


# ======================================================================
# モデルベース価値と調停
# ======================================================================
def stable_seed(base_seed: int, key: str) -> int:
    """経験 id（またはプローブの正規形）から決定的に作る種。"""
    d = hashlib.sha256(f"{base_seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(d[:8], "little")


def type_effect(h: np.ndarray, mag: np.ndarray, table: Optional[np.ndarray] = None,
                a: Optional[np.ndarray] = None) -> np.ndarray:
    """型 k の結果が平均の大きさで 1 回起きたときの（a で重み付けた）効用変化の総和 g_k
    （信念の不確実性を価値の尺度に直す係数）。"""
    tab = innate.M if table is None else table
    du = innate.delta_u(h[None, :], mag[:, None] * tab)           # (K, 5)
    return du.sum(axis=1) if a is None else du @ a


def model_based(pred: Any, h: np.ndarray, mag: np.ndarray, cfg: Config, key: str,
                table: Optional[np.ndarray] = None, a: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """V_MB(s|h): 信念の予測確率で結果の集合を M_mc 回サンプルし、(a で重み付けた) U(h+Δh) - U(h) の総和を平均する。

    σ²_MB = サンプル間の分散 / M_mc + Σ_{e,k} g_k² Var_k(e)   （信念の事後の不確実性）
      Var_k(e) = 個体・類・一般・文脈・世界の Beta 事後分散の和（belief.predict が返す）、
      g_k = 型 k の結果 1 回の効用変化（デルタ法で、確率の不確実性を価値の尺度に直す）。
    乱数は経験 id（プローブでは正規形）と設定の種から作るので、同じ状態・同じ入力では決定的。
    """
    tab = innate.M if table is None else table
    w = np.ones(N_NEEDS) if a is None else a
    M_mc = cfg.mc_samples
    nP = pred.p_person.shape[0]
    rng = np.random.default_rng(stable_seed(cfg.seed, key))
    P = np.vstack([pred.p_person, pred.p_sw[None, :]])          # (nP+1, K)
    U = rng.random((M_mc, nP + 1, N_TYPES))
    Y = (U < P[None, :, :]).astype(np.float64)
    cnt = Y.sum(axis=1)                                          # (M, K) 型ごとの発生数
    dh = (cnt * mag[None, :]) @ tab                              # (M, 5)
    tot = innate.delta_u(h[None, :], dh) @ w
    V_MB = float(tot.mean())
    if nP > 0:      # 人が起こす結果型の寄与 = 全体 − 自分・世界が起こす分だけの価値（同じサンプルで差をとる）
        dh_sw = (Y[:, nP, :] * mag[None, :]) @ tab
        V_MB_person = float((tot - innate.delta_u(h[None, :], dh_sw) @ w).mean())
    else:
        V_MB_person = 0.0
    var_s = float(tot.var(ddof=1) / M_mc) if M_mc > 1 else 0.0
    g = type_effect(h, mag, tab, w)
    var_b = float(((g[None, :] ** 2) * pred.var_person).sum() + ((g ** 2) * pred.var_sw).sum())
    return {"V_MB": V_MB, "var_MB": var_s + var_b, "var_samples": var_s, "var_belief": var_b,
            "V_MB_person": V_MB_person}


def arbitrate(V_MF: float, var_MF: float, V_MB: float, var_MB: float) -> Tuple[float, float]:
    """調停 ω = σ²_MB/(σ²_MB+σ²_MF)、U_val = ω V_MF + (1-ω) V_MB。確信の高い方に重みを置く。"""
    tot = var_MB + var_MF
    omega = var_MB / tot if tot > 0 else 0.5
    return float(omega), float(omega * V_MF + (1.0 - omega) * V_MB)


# ======================================================================
# 欲求の関心の重み a（SPEC-v3.1 1.1）
# ======================================================================
class Concern:
    """人格ごとの欲求の重み a ∈ R⁵（初期値 1、和は常に 5）。

    更新: a ← normalize( a + η_a ( 目標 − a ) )。|Δu| は生得の効用の変化の大きさ（外生）。run = mean_k|Δu_k| の移動平均。
    **a の目標に価値ネットの出力は入らない**（補題1）。
      sensitize（既定）: 目標 = clip(|Δu_k| / run, 0, 5)。人生で大きく揺さぶられた欲求ほど重みが上がる（感作。正負は問わない）。
      habituate        : 目標 = clip(run / max(|Δu_k|, run / 5), 0, 5) = clip(1 / max(|Δu_k|/run, 0.2), 0, 5)。
                         大きく揺さぶられた欲求ほど重みが下がり（慣れ）、揺さぶられない欲求は最大 5。感作の逆数で、床 run/5 は感作の
                         上限 5 と対称（|Δu_k| = 0 でも 5 を超えない）。
    どちらの方向が人に近いかは主張しない（実験条件。SPEC-v3.2 1.3）。主張は「a が経験から形成され、受け止め方と学習を変える」まで。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.a = np.ones(N_NEEDS)
        self.run = 0.0        # mean_k |Δu_k| の移動平均
        self.n = 0

    def update(self, du: np.ndarray) -> np.ndarray:
        """a を更新して、更新前の a を返す。効用変化が全欲求で 0 の経験（何も起きない）では更新しない。"""
        before = self.a.copy()
        if self.cfg.no_concern:
            return before
        mag = np.abs(du)
        if float(mag.sum()) < 1e-12:
            return before
        cfg = self.cfg
        rate = max(1.0 / (self.n + 1.0), cfg.a_run_rate)
        self.run = self.run + rate * (float(mag.mean()) - self.run)
        self.n += 1
        run = max(self.run, 1e-12)
        if cfg.concern_direction == "habituate":
            target = np.clip(run / np.maximum(mag, run / cfg.habituate_floor), 0.0, float(N_NEEDS))
        else:
            target = np.clip(mag / run, 0.0, float(N_NEEDS))
        a = self.a + cfg.eta_a * (target - self.a)
        self.a = N_NEEDS * a / a.sum()
        return before

    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        return {"run": float(self.run), "n": int(self.n)}, {"concern_a": self.a}

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        self.run = float(meta["run"]); self.n = int(meta["n"])
        self.a = np.array(arrays["concern_a"], dtype=np.float64)


# ======================================================================
# 条件づけられた価値 C（SPEC-v3.1 1.2。v3.3 4.2 で潜在原因つきに拡張）
# ======================================================================
_LOG_SQRT_2PI = 0.5 * float(np.log(2.0 * np.pi))


class Conditioning:
    """手がかり（概念・参加者の個体と類・結果型）への条件づけられた価値 C。TD(0) で学ぶ。

    【v3.1 / v3.2 の形（single_cause。K = 1）】
    事象 t+1 を見たとき、事象 t の手がかり z_t を更新する:
        y_t = r_{t+1} + γ C_frozen(z_{t+1}),   C += α_C (y_t − C·z_t) z_t / (z_tᵀ z_t)
    r = Σ_k a_k Δu_k（その事象の生得の効用）。C_frozen は更新前の C（勾配を止める）。同じ文脈の連続の間だけつなぎ、
    途切れたら次の価値を 0 とする。目標の r は C に依存せず、γ < 1 のブートストラップは縮小。
    どの手がかりがどんな価値を帯びるかは、設計者ではなく経験の随伴構造が決める。

    【v3.3 4.2: 潜在原因による条件づけ価値（Gershman, Blei & Niv 2010 の簡約）】
    手がかり x ごとに、最大 K = 5 個の潜在原因 k を持つ。各原因は 価値 C_{x,k}、文脈の計数 n_{x,k}(文脈) を持つ。応答は事前の重みでの期待値:
      応答         C(x | 文脈, t) = Σ_k p(k) C_{x,k},  p(k) = w_k / (Σ_k' w_k' + α)   （新しい原因の価値は 0。Q と f に入る）
      事後         事前 × N(y_x; C_{x,k}, σ_r²)。y_x は他の手がかりの応答を引いた、この手がかりが説明すべき目標（手がかりが 1 つなら TD の目標 y そのもの）
    事前の重み w_k と、学習・復位の規則は、設計 lc_design で二通りある。

    ◆ lc_design = "revised"（既定。v3.3 第 1 部の設計の改訂。オーケストレーター承認、結果を見た後。固定した値: d = 1、α = 0.5、β_first = 5、β_later = 0.2、σ_r = 0.15、K = 5）
      原因 k は、割り当てられた試行 i の履歴（時刻 t_i と文脈）を持つ。
      事前    w_k = Σ_{i∈k} (t − t_i + 1)^(−d) · Dir予測(文脈 | n_{x,k}, β_k)
              Dir予測 = (n_{x,k}(c) + β_k)/(N_k + β_k S)（N_k = Σ_c n_{x,k}(c)、S = 手がかり x が見た文脈の数 ＋ （c が新しければ 1））。
              時間の重みは冪乗則（Gershman, Blei & Niv 2010 の距離依存の CRP; Blei & Frazier 2011）。時間が経つと最近性の差が薄れ、試行の多い原因が戻る。
              β_k は原因が作られた順序で決まる: 手がかりの最初の原因（獲得）は β_first = 5（広い）、2 番目以降（意外さで作られる）は β_later = 0.2（狭い）。
              後から学んだ情報（消去）は、最初に学んだ情報（獲得）より文脈に依存する（Bouton 2004 の簡約）。これは設計上の仮説である。
      学習    局所 MAP: 事後が最大の原因（新しい原因を含む）だけを TD の誤差で更新する: C_{x,k*} += (α_C / 手がかりの数)(y_x − C_{x,k*})。
              その原因の履歴に (t, 文脈) を加える（Anderson 1991 の合理的分類モデルの近似）。新しい原因の事後が最大なら作る（満員なら現在の質量 Σ(t−t_i+1)^(−d) が最小の原因を再利用し、その履歴を捨てる）。
      復位    手がかりなしで嫌悪の結果（大きさ r < 0）が起きたとき、その結果を「どの原因から来たか」で推論する。除外した手がかり（その事象と直前の事象の手がかり）以外の
              全ての（手がかり, 原因）について尤度 N(r; C_{x,k}, σ_r²) を求め、「どの原因でもない」（価値 0 の尤度 N(r; 0, σ_r²)）より大きい中で最大のもの 1 つに、
              (t, 文脈) の試行として加える（価値は更新しない）。尤度が同じなら事前の重みが大きい方。事前の重みを選択に使わないのは、価値 0 の消去の原因（質量が大きい）が
              嫌悪の結果を吸収して、嫌悪の原因に帰属されなくなるのを避けるため（実装前の解析による判断。事前も掛けた版は試していない）。
    ◆ lc_design = "spec_literal"（SPEC-v3.3 4.2 のとおり。改訂前の実装。K > 1 のときの対照・再現用）
      事前    w_k = N_k · Dir予測(文脈 | n_{x,k}, β = lc_beta) · exp(−(t − t_k)/τ)（t_k = 最後に使われた時刻）
      学習    ΔC_{x,k} = (α_C / 手がかりの数) · 事後_k · (y_x − C_{x,k})（事後に比例した柔らかい更新）、n_{x,k}(c) += 事後_k、t_k は事後が最大の原因だけ更新。
              新しい原因の事後が最大なら作る（満員なら N_k exp(−(t−t_k)/τ) が最小の原因を再利用）。
      復位    負の価値（C < −lc_neg_threshold）を持つ全ての原因（除外した手がかり以外のすべての手がかり）の t_k を現在に更新する。
    single_cause（K = 1）は上のすべてを使わず、v3.2 の更新と完全に同じ算術で動く（応答 = 重み C、文脈・時刻に依存しない）。

    時刻 t は NPN が数える経験（学習した ＋ 回避した）の通し番号。文脈は経験の context。文脈を渡さない読み出し（context=None）は、
    文脈の項を 1 として扱う（文脈について周辺化しない近似）。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.K = 1 if cfg.single_cause else int(cfg.lc_max_causes)
        self.names: List[str] = []
        self.idx: Dict[str, int] = {}
        self.CK = np.zeros((0, self.K))                 # 原因ごとの価値 C_{x,k}
        self.TK = np.zeros((0, self.K))                 # 原因が最後に使われた時刻 t_k
        self.AL = np.zeros((0, self.K))                 # 原因が存在するか（1.0 / 0.0）
        self.ctx_names: List[str] = []                  # 文脈の計数の列（潜在原因が見た文脈の名前）
        self.ctx_idx: Dict[str, int] = {}
        self.CNT = np.zeros((0, self.K, 0))             # 文脈の計数 n_{x,k}(文脈)
        self.revised = (self.K > 1 and cfg.lc_design == "revised")
        self.BK = np.zeros((0, self.K))                 # revised: 原因ごとの文脈の Dirichlet の集中度 β_k（作られた順序で決まる）
        self.CRE = np.zeros(0, dtype=np.int64)          # revised: 手がかりごとに、これまでに作られた原因の数
        self.hist_t: List[np.ndarray] = []              # revised: 手がかりごとの試行の履歴（時刻・原因・文脈。容量を倍にして伸ばす）
        self.hist_k: List[np.ndarray] = []
        self.hist_c: List[np.ndarray] = []
        self.hist_n: List[int] = []
        self.last_detail: Dict[str, Any] = {}           # 直前の update の内訳（台帳用。状態ではない）

    # ------------------------------------------------------------------ 互換: 手がかりごとの 1 つの値 C
    @property
    def C(self) -> np.ndarray:
        """手がかりごとの価値の要約（形 (手がかりの数,)）。K = 1 なら重みそのもの（CK[:, 0] の view）。
        K > 1 なら存在する原因の価値の、計数 N_k による加重平均（文脈を問わない要約。読み出しの応答ではない）。"""
        if self.K == 1:
            return self.CK[:, 0]
        N = self.CNT.sum(axis=2)
        w = self.AL * N
        tot = w.sum(axis=1)
        return np.where(tot > 0, (w * self.CK).sum(axis=1) / np.where(tot > 0, tot, 1.0), 0.0)

    @C.setter
    def C(self, value: np.ndarray) -> None:
        """手がかりの数と同じ長さの配列を取り込む。K = 1: 重みを置き換える（名前を足した直後の拡張にも対応）。
        K > 1: 存在する原因の価値を、要約との差だけ平行移動する（摂動の検査用）。"""
        value = np.asarray(value, dtype=np.float64)
        self._grow_rows(len(self.names))
        if self.K == 1:
            self.CK[:, 0] = value
        else:
            self.CK = self.CK + self.AL * (value - self.C)[:, None]

    def _grow_rows(self, n: int) -> None:
        """手がかり名の数 n に行を合わせる（足りない行は、K = 1 では存在する原因 1 つ、K > 1 では原因なし）。"""
        m = self.CK.shape[0]
        if n <= m:
            return
        add = n - m
        self.CK = np.vstack([self.CK, np.zeros((add, self.K))])
        self.TK = np.vstack([self.TK, np.zeros((add, self.K))])
        al = np.zeros((add, self.K))
        if self.K == 1:
            al[:, 0] = 1.0
        self.AL = np.vstack([self.AL, al])
        self.CNT = np.concatenate([self.CNT, np.zeros((add, self.K, self.CNT.shape[2]))], axis=0)
        self.BK = np.vstack([self.BK, np.full((add, self.K), self.cfg.lc_beta_first)])
        self.CRE = np.concatenate([self.CRE, np.zeros(add, dtype=np.int64)])
        for _ in range(add):
            self.hist_t.append(np.zeros(0)); self.hist_k.append(np.zeros(0, dtype=np.int64)); self.hist_c.append(np.zeros(0, dtype=np.int64))
            self.hist_n.append(0)

    def _register(self, cue: str) -> int:
        i = self.idx.get(cue)
        if i is None:
            i = len(self.names)
            self.idx[cue] = i
            self.names.append(cue)
            self._grow_rows(len(self.names))
        return i

    def _ctx_col(self, ctx: Optional[str], create: bool) -> Optional[int]:
        if ctx is None:
            return None
        j = self.ctx_idx.get(ctx)
        if j is None and create:
            j = len(self.ctx_names)
            self.ctx_idx[ctx] = j
            self.ctx_names.append(ctx)
            self.CNT = np.concatenate([self.CNT, np.zeros((self.CNT.shape[0], self.K, 1))], axis=2)
        return j

    # ------------------------------------------------------------------ 原因の事前（応答と更新で共通）
    def _ctx_pred(self, ii: np.ndarray, ctx: Optional[str], beta: Any) -> Tuple[np.ndarray, np.ndarray]:
        """文脈の Dirichlet–多項の予測 (n_{x,k}(c) + β)/(N_k + β S)（形 (m, K)）と N_k。beta は スカラーか (m, K)。文脈が None なら予測は 1。"""
        CNT = self.CNT[ii]                                   # (m, K, S)
        N = CNT.sum(axis=2)                                  # (m, K)
        pred = np.ones_like(N)
        if ctx is not None:
            j = self.ctx_idx.get(ctx)
            seen = CNT.sum(axis=1) > 0                       # (m, S) その手がかりのどれかの原因が見た文脈
            S_eff = seen.sum(axis=1).astype(np.float64)
            if j is None:
                n_c = np.zeros_like(N)
                S_eff = S_eff + 1.0
            else:
                n_c = CNT[:, :, j]
                S_eff = S_eff + (~seen[:, j]).astype(np.float64)
            pred = (n_c + beta) / (N + beta * S_eff[:, None])
        return pred, N

    def _prior_weights(self, ii: np.ndarray, ctx: Optional[str], t: float) -> np.ndarray:
        """手がかり ii の、存在する原因の事前の重み（正規化前。形 (m, K)）。新しい原因の重みは cfg.lc_alpha。"""
        if self.revised:
            pred, _ = self._ctx_pred(ii, ctx, self.BK[ii])
            mass = np.stack([self._mass_row(int(i), t) for i in ii])
            return self.AL[ii] * mass * pred
        pred, N = self._ctx_pred(ii, ctx, self.cfg.lc_beta)
        rec = self._recency(np.maximum(t - self.TK[ii], 0.0))
        return self.AL[ii] * N * pred * rec

    def _recency(self, dt: np.ndarray) -> np.ndarray:
        """（spec_literal）最後に使われてからの経過 dt（経験）の重み exp(−dt/τ)（SPEC-v3.3 4.2）。診断用の変種はこれを差し替える。"""
        return np.exp(-dt / self.cfg.lc_tau)

    # ------------------------------------------------------------------ 試行の履歴（revised）
    def _mass_row(self, i: int, t: float) -> np.ndarray:
        """手がかり i の、原因ごとの質量 Σ_{j∈k} (t − t_j + 1)^(−d)（形 (K,)）。"""
        n = self.hist_n[i]
        if n == 0:
            return np.zeros(self.K)
        dt = np.maximum(t - self.hist_t[i][:n], 0.0) + 1.0
        return np.bincount(self.hist_k[i][:n], weights=dt ** (-self.cfg.lc_decay_d), minlength=self.K)[: self.K]

    def _hist_add(self, i: int, k: int, t: float, c: int) -> None:
        n = self.hist_n[i]
        if n == len(self.hist_t[i]):
            cap = max(8, 2 * n)
            self.hist_t[i] = np.concatenate([self.hist_t[i], np.zeros(cap - n)])
            self.hist_k[i] = np.concatenate([self.hist_k[i], np.zeros(cap - n, dtype=np.int64)])
            self.hist_c[i] = np.concatenate([self.hist_c[i], np.zeros(cap - n, dtype=np.int64)])
        self.hist_t[i][n] = t
        self.hist_k[i][n] = k
        self.hist_c[i][n] = c
        self.hist_n[i] = n + 1

    def _hist_drop(self, i: int, k: int) -> None:
        """原因 k の試行を履歴から捨てる（満員のときの再利用）。"""
        n = self.hist_n[i]
        keep = self.hist_k[i][:n] != k
        m = int(keep.sum())
        self.hist_t[i] = self.hist_t[i][:n][keep].copy()
        self.hist_k[i] = self.hist_k[i][:n][keep].copy()
        self.hist_c[i] = self.hist_c[i][:n][keep].copy()
        self.hist_n[i] = m

    # ------------------------------------------------------------------ 読み出し（学習しない）
    def value_of(self, name: str, ctx: Optional[str] = None, t: Optional[float] = None) -> float:
        i = self.idx.get(name)
        if i is None:
            return 0.0
        if self.K == 1:
            return float(self.CK[i, 0])
        return float(self._response(np.array([i]), ctx, t)[0])

    def value(self, cues: Sequence[str], ctx: Optional[str] = None, t: Optional[float] = None) -> float:
        """手がかりの集合の価値 Σ_x C(x | 文脈, t)（K = 1 では C·z。z は存在の 0/1）。"""
        if self.K == 1:
            return float(sum(self.CK[self.idx[c], 0] for c in cues if c in self.idx))
        ii = np.array([self.idx[c] for c in cues if c in self.idx], dtype=np.int64)
        if ii.size == 0:
            return 0.0
        return float(self._response(ii, ctx, t).sum())

    def _response(self, ii: np.ndarray, ctx: Optional[str], t: Optional[float]) -> np.ndarray:
        t = float(self.now if t is None else t)
        w = self._prior_weights(ii, ctx, t)
        p = w / (w.sum(axis=1) + self.cfg.lc_alpha)[:, None]
        return (p * self.CK[ii]).sum(axis=1)

    now: float = 0.0       # 既定の時刻（NPN が経験の通し番号を入れる。読み出しで t を省いたとき）

    def n_causes(self, name: str) -> int:
        i = self.idx.get(name)
        return 0 if i is None else int(self.AL[i].sum())

    # ------------------------------------------------------------------ 学習
    def update(self, cues: Sequence[str], y: float, ctx: Optional[str] = None,
               t: Optional[float] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """TD の更新。戻り値は (手がかりごとの ΔC の和（台帳用）, 内訳)。ctx, t は手がかりが現れたときの文脈と時刻。"""
        cues = list(dict.fromkeys(cues))
        if not cues:
            self.last_detail = {}
            return {}, {}
        for c in cues:
            self._register(c)
        ii = np.array([self.idx[c] for c in cues])
        if self.K == 1:
            return self._update_single(cues, ii, y)
        if self.revised:
            return self._update_revised(cues, ii, y, ctx, float(self.now if t is None else t))
        return self._update_latent(cues, ii, y, ctx, float(self.now if t is None else t))

    def _update_single(self, cues: List[str], ii: np.ndarray, y: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """v3.2 の NLMS 型の更新（算術も同じ）。"""
        err = y - float(self.CK[ii, 0].sum())
        d = self.cfg.alpha_c * err / len(cues)            # z は 0/1 なので z_tᵀ z_t = 手がかりの数
        self.CK[ii, 0] += d
        detail = {c: {"dCk": [float(d)]} for c in cues}
        self.last_detail = detail
        return {c: float(d) for c in cues}, detail

    def _update_latent(self, cues: List[str], ii: np.ndarray, y: float, ctx: Optional[str],
                       t: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        cfg = self.cfg
        n = len(cues)
        K = self.K
        before = self.CK[ii].copy()
        j = self._ctx_col(ctx, create=ctx is not None)
        w = self._prior_weights(ii, ctx, t)                      # (n, K)
        total = w.sum(axis=1) + cfg.lc_alpha
        resp = ((w / total[:, None]) * self.CK[ii]).sum(axis=1)  # 応答 C(x)
        y_x = y - (resp.sum() - resp)                            # 手がかりごとの、説明すべき目標（他の手がかりの応答を引く）
        sig = cfg.lc_sigma_r
        detail: Dict[str, Any] = {}
        for a, c in enumerate(cues):
            i = int(ii[a])
            alive = self.AL[i] > 0.5
            with np.errstate(divide="ignore"):
                logp = np.where(alive, np.log(np.maximum(w[a], 1e-300)), -np.inf)
            logp = np.where(alive & (w[a] <= 0.0), -np.inf, logp)
            ll = -0.5 * ((y_x[a] - self.CK[i]) / sig) ** 2
            la = logp + ll                                       # 存在する原因の log(事前 × 尤度)
            ln = np.log(cfg.lc_alpha) - 0.5 * (y_x[a] / sig) ** 2   # 新しい原因（価値 0）
            m = max(float(la.max()), ln)
            pa = np.exp(la - m); pn = float(np.exp(ln - m))
            z = float(pa.sum()) + pn
            post = pa / z; post_new = pn / z
            created: Optional[int] = None
            reused = False
            if post_new > float(post.max()):                      # 新しい原因の事後が最大 → 作る
                free = np.nonzero(~alive)[0]
                if free.size > 0:
                    created = int(free[0])
                else:
                    N_k = self.CNT[i].sum(axis=1)
                    score = N_k * self._recency(np.maximum(t - self.TK[i], 0.0))
                    created = int(np.argmin(score)); reused = True
                self.CK[i, created] = 0.0
                self.CNT[i, created, :] = 0.0
                self.AL[i, created] = 1.0
                self.TK[i, created] = t
                post = post.copy()
                post[created] = post_new                           # 再利用なら古い原因の事後を、新しい原因のものに置き換える
                post = post / post.sum()
            err = y_x[a] - self.CK[i]                              # 原因ごとの TD の誤差
            self.CK[i] += (cfg.alpha_c / n) * post * err * self.AL[i]
            if j is not None:
                self.CNT[i, :, j] += post * self.AL[i]
            kmap = int(np.argmax(post))
            self.TK[i, kmap] = t
            detail[c] = {"post": [float(x) for x in post], "post_new": float(post_new), "map": kmap, "created": created,
                         "reused": reused, "resp_before": float(resp[a]), "y_x": float(y_x[a])}
        dC: Dict[str, float] = {}
        for a, c in enumerate(cues):
            dck = self.CK[ii[a]] - before[a]
            detail[c]["dCk"] = [float(x) for x in dck]
            dC[c] = float(dck.sum())
        self.last_detail = detail
        return dC, detail

    def _update_revised(self, cues: List[str], ii: np.ndarray, y: float, ctx: Optional[str],
                        t: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """revised: 局所 MAP。事後が最大の原因（新しい原因を含む）だけを TD の誤差で更新し、その履歴に (t, 文脈) を加える。"""
        cfg = self.cfg
        n = len(cues)
        before = self.CK[ii].copy()
        j = self._ctx_col(ctx, create=ctx is not None)
        w = self._prior_weights(ii, ctx, t)                      # (n, K)
        total = w.sum(axis=1) + cfg.lc_alpha
        resp = ((w / total[:, None]) * self.CK[ii]).sum(axis=1)  # 応答 C(x)
        y_x = y - (resp.sum() - resp)                            # 手がかりごとの、説明すべき目標（他の手がかりの応答を引く）
        sig = cfg.lc_sigma_r
        detail: Dict[str, Any] = {}
        for a, c in enumerate(cues):
            i = int(ii[a])
            alive = (self.AL[i] > 0.5) & (w[a] > 0.0)
            with np.errstate(divide="ignore"):
                logp = np.where(alive, np.log(np.maximum(w[a], 1e-300)), -np.inf)
            la = logp - 0.5 * ((y_x[a] - self.CK[i]) / sig) ** 2     # 存在する原因の log(事前 × 尤度)
            ln = np.log(cfg.lc_alpha) - 0.5 * (y_x[a] / sig) ** 2   # 新しい原因（価値 0）
            m = max(float(la.max()), ln)
            pa = np.exp(la - m); pn = float(np.exp(ln - m))
            z = float(pa.sum()) + pn
            post = pa / z; post_new = pn / z
            created: Optional[int] = None
            reused = False
            if post_new > float(post.max()):                      # 新しい原因の事後が最大 → 作る
                free = np.nonzero(self.AL[i] < 0.5)[0]
                if free.size > 0:
                    k = int(free[0])
                else:
                    k = int(np.argmin(self._mass_row(i, t))); reused = True
                    self._hist_drop(i, k)
                self.CK[i, k] = 0.0
                self.CNT[i, k, :] = 0.0
                self.AL[i, k] = 1.0
                self.BK[i, k] = cfg.lc_beta_first if int(self.CRE[i]) == 0 else cfg.lc_beta_later
                self.CRE[i] += 1
                created = k
            else:
                k = int(np.argmax(post))
            self.CK[i, k] += (cfg.alpha_c / n) * (y_x[a] - self.CK[i, k])
            self._hist_add(i, k, t, -1 if j is None else j)
            if j is not None:
                self.CNT[i, k, j] += 1.0
            self.TK[i, k] = t
            detail[c] = {"post": [float(x) for x in post], "post_new": float(post_new), "map": k, "created": created,
                         "reused": reused, "resp_before": float(resp[a]), "y_x": float(y_x[a])}
        dC: Dict[str, float] = {}
        for a, c in enumerate(cues):
            dck = self.CK[ii[a]] - before[a]
            detail[c]["dCk"] = [float(x) for x in dck]
            dC[c] = float(dck.sum())
        self.last_detail = detail
        return dC, detail

    def reinstate(self, skip: Sequence[str], t: float, ctx: Optional[str] = None,
                  r: Optional[float] = None) -> List[Tuple[str, int]]:
        """復位。手がかりなしで嫌悪の結果が起きたときに呼ぶ（skip の手がかりは除く）。戻り値は更新した (手がかり, 原因の添字)。single_cause では何もしない。
        revised（r = その事象の Σ a_k Δu_k を渡したとき）: その結果がどの原因から来たかを、価値が予測する嫌悪の大きさの尤度 N(r; C_{x,k}, σ_r²) で推論し、
        「どの原因でもない」（N(r; 0, σ_r²)）より大きい中で最大の 1 つに (t, 文脈) の試行を加える（価値は更新しない）。
        spec_literal: 負の価値（C < −lc_neg_threshold）を持つ全ての原因の最後に使われた時刻を現在に更新する。"""
        if self.K == 1 or self.CK.shape[0] == 0:
            return []
        if not (self.revised and r is not None):
            mask = (self.AL > 0.5) & (self.CK < -self.cfg.lc_neg_threshold)
            for c in skip:
                i = self.idx.get(c)
                if i is not None:
                    mask[i, :] = False
            rows, cols = np.nonzero(mask)
            self.TK[mask] = t
            return [(self.names[int(r_)], int(k)) for r_, k in zip(rows, cols)]
        sig = self.cfg.lc_sigma_r
        cand = self.AL > 0.5
        for c in skip:
            i = self.idx.get(c)
            if i is not None:
                cand[i, :] = False
        if not cand.any():
            return []
        score = np.where(cand, -0.5 * ((r - self.CK) / sig) ** 2, -np.inf)
        null = -0.5 * (r / sig) ** 2
        best = float(score.max())
        if not best > null:
            return []
        rows, cols = np.nonzero(score >= best - 1e-12)
        if rows.size > 1:                                      # 尤度が同じなら事前の重みが大きい方
            pw = [float(self._prior_weights(np.array([int(i)]), ctx, t)[0, int(k)]) for i, k in zip(rows, cols)]
            pick = int(np.argmax(pw))
            rows, cols = rows[pick:pick + 1], cols[pick:pick + 1]
        i, k = int(rows[0]), int(cols[0])
        j = self._ctx_col(ctx, create=ctx is not None)
        self._hist_add(i, k, t, -1 if j is None else j)
        if j is not None:
            self.CNT[i, k, j] += 1.0
        self.TK[i, k] = t
        return [(self.names[i], k)]

    # ------------------------------------------------------------------ 検査・摂動用の補助
    def set_single(self, name: str, value: float) -> None:
        """手がかり name を、価値 value の原因 1 つだけにする（テスト・診断用）。文脈の計数は空、時刻は now。"""
        i = self._register(name)
        self.CK[i, :] = 0.0
        self.AL[i, :] = 0.0
        self.CNT[i] = 0.0
        self.CK[i, 0] = value
        self.AL[i, 0] = 1.0
        self.TK[i, 0] = self.now
        if self.K > 1:
            self.CNT[i, 0, :] = 0.0
        if self.revised:                                       # 質量 0 では応答が 0 になるので、時刻 now に文脈なしの試行を 1 つ置く
            self.hist_t[i] = np.zeros(0); self.hist_k[i] = np.zeros(0, dtype=np.int64); self.hist_c[i] = np.zeros(0, dtype=np.int64)
            self.hist_n[i] = 0
            self.BK[i, :] = self.cfg.lc_beta_first
            self.CRE[i] = 1
            self._hist_add(i, 0, self.now, -1)

    def nudge(self, name: str, delta: float) -> None:
        """手がかり name の存在する原因すべての価値に delta を足す（テスト・診断用）。K = 1 で未登録なら登録する。"""
        i = self._register(name)
        self.CK[i] = self.CK[i] + delta * self.AL[i]

    def state_items(self) -> Dict[Any, float]:
        """摂動の検査用: 原因ごとの価値（K = 1 では手がかりごとの重み）。"""
        d: Dict[Any, float] = {}
        for i, nm in enumerate(self.names):
            for k in range(self.K):
                if self.AL[i, k] > 0.5:
                    d[("C", nm, k)] = float(self.CK[i, k])
        return d

    # ------------------------------------------------------------------ 保存・復元
    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        HI, HK, HT, HC = [], [], [], []
        for i in range(len(self.names)):
            n = self.hist_n[i]
            HI.append(np.full(n, i, dtype=np.int64)); HK.append(self.hist_k[i][:n].astype(np.int64))
            HT.append(self.hist_t[i][:n].astype(np.float64)); HC.append(self.hist_c[i][:n].astype(np.int64))
        cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)
        return ({"names": list(self.names), "ctx_names": list(self.ctx_names), "K": int(self.K), "design": self.cfg.lc_design},
                {"cond_CK": self.CK, "cond_TK": self.TK, "cond_AL": self.AL, "cond_CNT": self.CNT,
                 "cond_BK": self.BK, "cond_CRE": self.CRE,
                 "cond_HI": cat(HI, np.int64), "cond_HK": cat(HK, np.int64), "cond_HT": cat(HT, np.float64), "cond_HC": cat(HC, np.int64)})

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        if int(meta["K"]) != self.K:
            raise ValueError(f"条件づけの原因数が設定と合いません: 保存 {meta['K']}, 設定 {self.K}")
        if meta.get("design") != self.cfg.lc_design:
            raise ValueError(f"条件づけの設計が設定と合いません: 保存 {meta.get('design')!r}, 設定 {self.cfg.lc_design!r}")
        self.names = list(meta["names"])
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.ctx_names = list(meta["ctx_names"])
        self.ctx_idx = {c: j for j, c in enumerate(self.ctx_names)}
        self.CK = np.array(arrays["cond_CK"], dtype=np.float64)
        self.TK = np.array(arrays["cond_TK"], dtype=np.float64)
        self.AL = np.array(arrays["cond_AL"], dtype=np.float64)
        self.CNT = np.array(arrays["cond_CNT"], dtype=np.float64)
        self.BK = np.array(arrays["cond_BK"], dtype=np.float64)
        self.CRE = np.array(arrays["cond_CRE"], dtype=np.int64)
        HI, HK = np.array(arrays["cond_HI"], dtype=np.int64), np.array(arrays["cond_HK"], dtype=np.int64)
        HT, HC = np.array(arrays["cond_HT"], dtype=np.float64), np.array(arrays["cond_HC"], dtype=np.int64)
        self.hist_t, self.hist_k, self.hist_c, self.hist_n = [], [], [], []
        for i in range(len(self.names)):
            m = HI == i
            self.hist_t.append(HT[m].copy()); self.hist_k.append(HK[m].copy()); self.hist_c.append(HC[m].copy())
            self.hist_n.append(int(m.sum()))


# ======================================================================
# 感じた価値・気分・感情（読み出し。学習の目標にはしない）
# ======================================================================
def felt_parts(inn: float, cond: float, Q: float, cfg: Config) -> Dict[str, float]:
    """f の内訳。inn = Σ_k a_k Δu_k（生得の効用）、cond = λ_C ΣC（条件づけられた価値）、
    ref = η_ref (inn − Q)（予期との差。参照点依存、Rutledge et al. 2014）。"""
    ref = 0.0 if cfg.no_reference else cfg.eta_ref * (inn - Q)
    return {"innate": float(inn), "conditioned": float(cond), "reference": float(ref)}


def felt_value(parts: Dict[str, float], mood: float, cfg: Config) -> float:
    """f = tanh( (Σ a_k Δu_k + λ_C ΣC + η_ref (Σ a_k Δu_k − Q)) / s_f + β m )。f は読み出しで、どの学習の目標にも入らない。"""
    return float(np.tanh((parts["innate"] + parts["conditioned"] + parts["reference"]) / cfg.s_f + cfg.beta_mood * mood))


def surprise(f: float, Q: float, cfg: Config) -> float:
    """驚き δ = f - tanh(Q / s_f)。"""
    return float(f - np.tanh(Q / cfg.s_f))


def next_mood(mood: float, delta: float, cfg: Config) -> float:
    """気分 m ← 0.8 m + 0.2 δ。"""
    return float(cfg.mood_keep * mood + (1.0 - cfg.mood_keep) * delta)


def emotions(f: float, delta: float, Q: float, contrib: np.ndarray, person_phi: np.ndarray,
             is_person: np.ndarray, cfg: Config) -> Dict[str, float]:
    """感情の読み出し（言語化用の手がかり）。強度はすべて 0〜1。学習には使わない。

      喜び   = max(f, 0)
      失望   = min(1, -δ)   （δ < 0 かつ Q > 0 のとき）
      安堵   = min(1, δ)    （δ > 0 かつ Q < 0 のとき）
      怒り   = tanh(a / s_f), a = Σ_{o: 行為者が参加者} φ_person(o) · max(-Σ_k contrib[o,k], 0)
      悲しみ = tanh(b / s_f), b = Σ_o max(-contrib[o, 関係性], 0)
      興味   = tanh(c / s_f), c = Σ_o max(contrib[o, 刺激], 0)
    """
    n = contrib.shape[0]
    if n:
        a = float((person_phi * is_person * np.maximum(-contrib.sum(axis=1), 0.0)).sum())
        b = float(np.maximum(-contrib[:, 0], 0.0).sum())
        c = float(np.maximum(contrib[:, 3], 0.0).sum())
    else:
        a = b = c = 0.0
    return {
        "joy": float(max(f, 0.0)),
        "disappointment": float(min(1.0, -delta)) if (delta < 0 and Q > 0) else 0.0,
        "relief": float(min(1.0, delta)) if (delta > 0 and Q < 0) else 0.0,
        "anger": float(np.tanh(a / cfg.s_f)),
        "sadness": float(np.tanh(b / cfg.s_f)),
        "interest": float(np.tanh(c / cfg.s_f)),
    }
