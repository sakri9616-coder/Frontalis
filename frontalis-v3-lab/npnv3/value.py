"""価値ネット: 階層カルマン回帰（MF）、モデルベース価値（MB）、調停、気分と感情の読み出し。

二つのカルマン回帰（SPEC 5.1）
  人モデル : 参加者の価値。特徴 = people + cls:<c> + ind:<id> + 類の埋め込み 8 次元
  状況モデル: 文脈と概念の価値。特徴 = world + ctx:<c> + concept:<x> + 概念埋め込みの平均 8 次元
  教師は帰属 φ で分ける。更新 K = Σz/(zᵀΣz+R)、W += K(Δu - zᵀW)、Σ ← Σ - K zᵀΣ、続いて関与した対角に q。
価値ネット自身の出力は、どの更新の目標にも入らない（docs/04 補題1）。教師は Δu（欲求の効用変化）だけ。

v3.1 の追加（SPEC-v3.1）
  Concern     欲求の関心の重み a（1.1）: 目標は |Δu|（外生）。感じた価値・MB・好悪の読み出しで欲求を重み付ける。
  Conditioning 条件づけられた価値 C（1.2）: 手がかり（概念・参加者・結果型）への線形の重み。TD(0)、目標は r + γ C_frozen。
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
# カルマン回帰（欲求 5 次元で共分散を共有）
# ======================================================================
class KalmanModel:
    def __init__(self, var: Dict[str, float], q: Dict[str, float], obs_noise: float):
        self.var = dict(var)
        self.q = dict(q)
        self.R = obs_noise
        self.names: List[str] = []
        self.layers: List[str] = []
        self.idx: Dict[str, int] = {}
        self.W = np.zeros((0, N_NEEDS))
        self.S = np.zeros((0, 0))

    def _add(self, name: str, layer: str) -> int:
        i = len(self.names)
        self.names.append(name)
        self.layers.append(layer)
        self.idx[name] = i
        self.W = np.vstack([self.W, np.zeros((1, N_NEEDS))])
        S2 = np.zeros((i + 1, i + 1))
        S2[:i, :i] = self.S
        S2[i, i] = self.var[layer]     # 新しい特徴: 事前平均 0、事前分散は層ごと
        self.S = S2
        return i

    def predict(self, feats: Sequence[Feat]) -> Tuple[np.ndarray, float, bool]:
        """予測 zᵀW（5 欲求）、予測分散 zᵀΣz、すべての特徴が既知か。状態は変えない。
        未登録の特徴は、事前平均 0・事前分散（他の特徴とは無相関）として扱う。"""
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
                var += self.var[layer] * v * v
                all_known = False
            else:
                ii.append(i)
                zz.append(v)
        if ii:
            ia = np.array(ii)
            z = np.array(zz)
            pred = z @ self.W[ia]
            var += float(z @ self.S[np.ix_(ia, ia)] @ z)
        return pred, var, all_known

    def _prepare(self, feats: Sequence[Feat]) -> Tuple[np.ndarray, np.ndarray]:
        for name, layer, v in feats:
            if v != 0.0 and name not in self.idx:
                self._add(name, layer)
        ii = np.array([self.idx[name] for name, _, v in feats if v != 0.0], dtype=int)
        z = np.array([v for _, _, v in feats if v != 0.0])
        return ii, z

    def update(self, feats: Sequence[Feat], teacher: np.ndarray) -> Dict[str, Any]:
        """1 観測のカルマン更新。K は全特徴に対して密（共分散を通じて関与しない特徴も動く）。"""
        ii, z = self._prepare(feats)
        Sz = self.S[:, ii] @ z                      # Σ z（全特徴）
        den = float(z @ Sz[ii]) + self.R
        K = Sz / den
        pred = z @ self.W[ii]
        err = np.asarray(teacher, dtype=np.float64) - pred
        dW = np.outer(K, err)
        self.W = self.W + dW
        S2 = self.S - np.outer(K, Sz)               # Σ - K zᵀ Σ
        S2 = 0.5 * (S2 + S2.T)
        for i in ii:
            S2[i, i] += self.q[self.layers[i]]      # 関与した特徴だけに過程雑音
        self.S = S2
        return {"dW": dW, "err": err, "pred": pred, "var": den - self.R}

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
        persons = []
        V = 0.0
        var = 0.0
        for p in participants:
            pred, v, known = self.person.predict(self.person_feats(p["id"], p["class"]))
            val = float(w @ pred)
            persons.append({"id": p["id"], "value": val, "var": float(v), "pred": pred, "known": known})
            V += val
            var += float(v)
        pred_s, v_s, known_s = self.sit.predict(self.situation_feats(ctx, concepts))
        vs = float(w @ pred_s)
        V += vs
        var += float(v_s)
        return {"V_MF": V, "var_MF": var, "persons": persons,
                "situation": {"value": vs, "var": float(v_s), "pred": pred_s, "known": known_s}}

    def like_entity(self, pid: str, cls: str, a: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray, bool]:
        pred, v, known = self.person.predict(self.person_feats(pid, cls))
        return float((np.ones(N_NEEDS) if a is None else a) @ pred), float(v), pred, known

    def like_concept(self, name: str, a: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray, bool]:
        """概念の好悪 Like(x) = 一般 + 概念 + 埋め込みの寄与（文脈は含めない）。測定用のプローブ。"""
        pred, v, known = self.sit.predict(self.situation_feats(None, [name], with_ctx=False))
        return float((np.ones(N_NEEDS) if a is None else a) @ pred), float(v), pred, known

    def prepare(self, participants: Sequence[Dict[str, str]], ctx: str, concepts: Sequence[str]) -> None:
        """この経験で使う特徴を（学習せずに）登録する。ヤコビアンの検査用（状態写像の定義域を揃える）。"""
        for p in participants:
            self.person._prepare(self.person_feats(p["id"], p["class"]))
        self.sit._prepare(self.situation_feats(ctx, concepts))

    # --- 学習 ---
    def learn(self, participants: Sequence[Dict[str, str]], ctx: str, concepts: Sequence[str],
              person_teachers: Sequence[np.ndarray], sit_teacher: np.ndarray) -> Dict[str, Any]:
        """人モデルを参加者ごとに、状況モデルを 1 回、帰属で分けた教師で更新する。"""
        cfg = self.cfg
        dW_person: Dict[str, np.ndarray] = {}
        err_person = []
        for p, t in zip(participants, person_teachers):
            feats = self.person_feats(p["id"], p["class"])
            if cfg.fixed_teacher:
                r = self.person.update_fixed(feats, t, f"ind:{p['id']}", cfg.fixed_teacher_rate)
            else:
                r = self.person.update(feats, t)
            for name, row in self.person.dW_dict(r["dW"]).items():      # 参加者ごとの変化量を特徴名で足し合わせる
                dW_person[name] = dW_person[name] + row if name in dW_person else row
            err_person.append(r["err"])
        feats_s = self.situation_feats(ctx, concepts)
        if cfg.fixed_teacher:
            rs = self.sit.update_fixed(feats_s, sit_teacher, f"ctx:{ctx}", cfg.fixed_teacher_rate)
        else:
            rs = self.sit.update(feats_s, sit_teacher)
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
    var_s = float(tot.var(ddof=1) / M_mc) if M_mc > 1 else 0.0
    g = type_effect(h, mag, tab, w)
    var_b = float(((g[None, :] ** 2) * pred.var_person).sum() + ((g ** 2) * pred.var_sw).sum())
    return {"V_MB": V_MB, "var_MB": var_s + var_b, "var_samples": var_s, "var_belief": var_b}


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

    更新: a ← normalize( a + η_a ( |Δu| / mean(|Δu|_run) − a ) )。|Δu| は生得の効用の変化の大きさ（外生）。
    **a の目標に価値ネットの出力は入らない**（補題1）。意味: 人生で大きく揺さぶられた欲求ほど、その人格にとって重要になる
    （感作。正負は問わない。この方向は設計上の仮説）。
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
        target = np.clip(mag / max(self.run, 1e-12), 0.0, float(N_NEEDS))
        a = self.a + cfg.eta_a * (target - self.a)
        self.a = N_NEEDS * a / a.sum()
        return before

    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        return {"run": float(self.run), "n": int(self.n)}, {"concern_a": self.a}

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        self.run = float(meta["run"]); self.n = int(meta["n"])
        self.a = np.array(arrays["concern_a"], dtype=np.float64)


# ======================================================================
# 条件づけられた価値 C（SPEC-v3.1 1.2）
# ======================================================================
class Conditioning:
    """手がかり（概念・参加者の個体と類・結果型）への線形の重み C。TD(0)で学ぶ。

    事象 t+1 を見たとき、事象 t の手がかり z_t を更新する:
        y_t = r_{t+1} + γ C_frozen(z_{t+1}),   C += α_C (y_t − C·z_t) z_t / (z_tᵀ z_t)
    r = Σ_k a_k Δu_k（その事象の生得の効用）。C_frozen は更新前の C（勾配を止める）。同じ文脈の連続の間だけつなぎ、
    途切れたら次の価値を 0 とする。目標の r は C に依存せず、γ < 1 のブートストラップは縮小。
    どの手がかりがどんな価値を帯びるかは、設計者ではなく経験の随伴構造が決める。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.names: List[str] = []
        self.idx: Dict[str, int] = {}
        self.C = np.zeros(0)

    def value_of(self, name: str) -> float:
        i = self.idx.get(name)
        return 0.0 if i is None else float(self.C[i])

    def value(self, cues: Sequence[str]) -> float:
        """手がかりの集合の価値 C·z（z は存在の 0/1）。"""
        return float(sum(self.C[self.idx[c]] for c in cues if c in self.idx))

    def update(self, cues: Sequence[str], y: float) -> Dict[str, float]:
        """NLMS 型の更新。戻り値は手がかりごとの ΔC（台帳用）。"""
        cues = list(dict.fromkeys(cues))
        if not cues:
            return {}
        for c in cues:
            if c not in self.idx:
                self.idx[c] = len(self.names)
                self.names.append(c)
                self.C = np.append(self.C, 0.0)
        ii = np.array([self.idx[c] for c in cues])
        err = y - float(self.C[ii].sum())
        d = self.cfg.alpha_c * err / len(cues)            # z は 0/1 なので z_tᵀ z_t = 手がかりの数
        self.C[ii] += d
        return {c: float(d) for c in cues}

    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        return {"names": list(self.names)}, {"cond_C": self.C}

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        self.names = list(meta["names"])
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.C = np.array(arrays["cond_C"], dtype=np.float64)


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
