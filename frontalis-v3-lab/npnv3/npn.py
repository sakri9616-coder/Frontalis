"""統合: experience() / appraise() / probe() / save() / load()。

experience(exp) の順序（SPEC 6、v3.1・v3.2 で拡張）
  1. schema 検証
  2. 予期: 信念の予測 → V_MB, V_MF, ω, U_val、Q = U_val + λ_CQ ΣC(手がかり)（結果を見る前。v3.2 1.2）。
     曖昧な結果は、信念で解釈の重みを決める（1.4）
  3. optional なら接近・回避を選ぶ。回避なら記録して終了
  4. Δh, Δu（ベクトル）, 寄与の分解, 感じた価値 f（a・C・参照点、1.3）, δ, 感情
  5. 信念の更新（責任 φ → 計数、π、型の大きさ、リプレイ緩衝 → 事前ヘッドの SGD）。曖昧な型は計数しない（既定）
  6. 価値の更新（φ で教師を分け、人モデル・状況モデルのカルマン。欲求 k の観測雑音は R/a_k、v3.2 1.1）。条件づけ C の TD(0)（1.2。v3.3 4.2: 潜在原因ごと、
     文脈と時刻つき。手がかりなしの嫌悪の結果は、負の価値を持つ原因の最後に使われた時刻を更新する = 復位）
  7. 欲求の関心の重み a（1.1）、欲求の状態 h、気分 m の更新
  8. 台帳に記録

v3.3 第 2 部: experience(exp) は kind で分岐する。"experience"（既定）は上の順序。"correction"（訂正）は参照先の経験の観測を取り消して直し（信念）、当時の記録は変えず、
人モデルの価値を再評価する（_correct）。"choice"（二択）は 2 つの選択肢の Q（C と J を含む）から softmax で選び、選んだ側を通常の経験として学ぶ（自分で選んだ経験: a の更新が速い）。
選んだ側の手がかりの正当化 J を +、選ばなかった側を − に動かす（_choose）。J は Q にだけ入り、経験の教師にならない。
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from . import innate
from . import value as V
from .belief import BeliefNet
from .config import Config
from .innate import NEEDS, N_NEEDS, N_TYPES, OUTCOME_TYPES, TYPE_INDEX
from .ledger import Ledger
from .lexicon import EmbeddingStore, Lexicon
from .mlp import sigmoid
from .schema import (KIND_CHOICE, KIND_CORRECTION, RESERVED_IDS, SchemaError, experience_kind, validate_choice, validate_correction,
                     validate_experience, validate_probe)

SCHEMA_VERSION = 7          # v3.0 は 1、v3.1 は 2（a, C, 所有する埋め込み, 効果表）。改訂 2 で観測の記録（想起）、v3.2 で欲求ごとの Σ とヘッドの対数尤度が加わった(3 → 4)。
                            # v3.3 で、類ごとの生の計数・類を除いたヘッドのモデル（leave-class-out）・潜在原因つきの条件づけ（原因の価値・文脈の計数・時刻）・pending の文脈と時刻が加わった(4 → 5)。
                            # v3.3 第 1 部の設計の改訂で、潜在原因の試行の履歴（時刻・原因・文脈）と、原因ごとの文脈の集中度が加わった(5 → 6)
                            # v3.3 第 2 部で、選択の正当化 J・学習した経験の記録（exp_log）・訂正の回数、信念の観測の通し番号（rec_exp, rec_ser, rb_ser）と
                            # 自分・世界の観測の記録（sw_*）が加わった(6 → 7)
STATE_FILE = "state.json"
ARRAY_FILE = "arrays.npz"


class StateIntegrityError(ValueError):
    """保存状態の内容ハッシュが一致しない（改ざん・破損）。"""


class StateVersionError(ValueError):
    """保存状態のスキーマ版が合わない。"""


# ----------------------------------------------------------------------
# 補助
# ----------------------------------------------------------------------
def _py(x: Any) -> Any:
    """numpy の値を JSON 化できる Python の値に直す（再帰）。"""
    t = type(x)
    if t is float or t is int or t is str or t is bool or x is None:       # 速い経路（呼び出しが多い）
        return x
    if isinstance(x, np.ndarray):
        return [_py(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    return x


def _need_dict(v: np.ndarray) -> Dict[str, float]:
    return {n: float(a) for n, a in zip(NEEDS, v)}


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(_canonical(meta))
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def probe_key(p: Dict[str, Any]) -> str:
    return _canonical(p).decode("utf-8")


def _dedupe(xs: List[str]) -> List[str]:
    return list(dict.fromkeys(xs))


# ----------------------------------------------------------------------
class NPN:
    """経験で人格が形成される外部ニューラルネットワーク（最小実装 v3.2）。"""

    def __init__(self, seed: int = 0, lexicon: Optional[Lexicon] = None, config: Optional[Config] = None,
                 effect_table: Optional[np.ndarray] = None, **overrides: Any):
        cfg = config if config is not None else Config(seed=seed)
        if config is not None and "seed" not in overrides and seed != 0:
            overrides["seed"] = seed
        if overrides:
            cfg = cfg.replace(**overrides)
        self.cfg = cfg
        self.emb = EmbeddingStore(lexicon if lexicon is not None else Lexicon.load())
        self.M = np.array(innate.M if effect_table is None else effect_table, dtype=np.float64)
        self.rng = np.random.default_rng(cfg.seed)
        self.belief = BeliefNet(cfg, self.emb, self.rng)
        self.value = V.ValueNet(cfg, self.emb)
        self.concern = V.Concern(cfg)
        self.cond = V.Conditioning(cfg)
        self.h = np.full(N_NEEDS, innate.H0)
        self.mood = 0.0
        self.ledger = Ledger()
        self.n_experiences = 0
        self.n_avoided = 0
        self.pending: Optional[Dict[str, Any]] = None      # 条件づけの TD: 直前の事象の手がかりと文脈の連続
        self.warnings: List[str] = []
        self.just: Dict[str, float] = {}                   # v3.3 第2部 3: 選択の正当化 J（手がかり concept:/cls:/ctx: → 値。経験の教師にならないチャンネル）
        self.exp_log: List[Dict[str, Any]] = []            # v3.3 第2部 2: 学習した経験の記録（訂正が参照する）
        self._exp_pos: Dict[str, int] = {}                 # 経験 id → exp_log の位置（記録から作り直せる。状態ではない）
        self.n_corrections = 0

    @property
    def a(self) -> np.ndarray:
        return self.concern.a

    # ==================================================================
    # 評価の核（状態を変えない）
    # ==================================================================
    def _interp_weights(self, interp: Dict[str, float], pvec: np.ndarray) -> np.ndarray:
        """w_I ∝ q_LLM(I) · P_belief(I | actor, context)（正規化）。no_interpretation では信念を使わず w_I = q_LLM(I)。"""
        w = np.zeros(N_TYPES)
        for t, q in interp.items():
            k = TYPE_INDEX[t]
            w[k] = q if self.cfg.no_interpretation else q * pvec[k]
        return w / w.sum()

    def _j_cues(self, ctx: str, concepts: List[str], participants: List[Dict[str, str]]) -> List[str]:
        """正当化 J の手がかり（SPEC-v3.3 3.3: 概念・類・文脈）。個体は含めない。"""
        return _dedupe([f"concept:{c}" for c in concepts] + [f"cls:{p['class']}" for p in participants] + [f"ctx:{ctx}"])

    def _j_q(self, cues: List[str]) -> float:
        """Q に入る正当化 Σ J(手がかり)。no_justification では 0。"""
        if self.cfg.no_justification:
            return 0.0
        return float(sum(self.just.get(c, 0.0) for c in cues))

    def _expectation(self, ctx: str, concepts: List[str], P: List[Dict[str, str]], announced: List[str], key: str,
                     t_now: float) -> Dict[str, Any]:
        """予期（結果を見る前）: 信念の予測 → V_MB, V_MF, ω, U_val、Q = U_val + λ_CQ ΣC(手がかり) + ΣJ(手がかり)。状態を変えない。key は MB の乱数の種（同じ key なら同じ標本）。"""
        cfg = self.cfg
        a = self.concern.a
        pred = self.belief.predict(ctx, P)
        mf = self.value.mf(P, ctx, concepts, a)
        mb = V.model_based(pred, self.h, self.belief.mag, cfg, key, self.M, a)
        omega, Q_val = V.arbitrate(mf["V_MF"], mf["var_MF"], mb["V_MB"], mb["var_MB"])
        cues_q = self._cues_q(concepts, P, announced)
        C_Q = self._c_q(cues_q, ctx, t_now)
        J_Q = self._j_q(self._j_cues(ctx, concepts, P))
        Q = Q_val + C_Q + J_Q
        return {"pred": pred, "mf": mf, "mb": mb, "omega": omega, "Q_val": Q_val, "C_Q": C_Q, "J_Q": J_Q, "Q": Q, "cues_q": cues_q}

    def _parse_outcomes(self, P: List[Dict[str, str]], outs: List[Dict[str, Any]], pred: Any) -> Dict[str, Any]:
        """結果の列を、解釈の重み W_out、観測 y とマスク、責任 φ、結果ごとの φ_o に直す（経験の評価と訂正で共通）。状態を変えない。"""
        cfg = self.cfg
        nP = len(P)
        pidx = {p["id"]: j for j, p in enumerate(P)}
        n_o = len(outs)
        W_out = np.zeros((n_o, N_TYPES))
        ambiguous = [False] * n_o
        actor_j = np.full(n_o, -1, dtype=int)
        for i, o in enumerate(outs):
            actor_j[i] = pidx.get(o["actor"], -1)
            if "type" in o:
                W_out[i, TYPE_INDEX[o["type"]]] = 1.0
            else:
                ambiguous[i] = True
                pvec = pred.p_person[actor_j[i]] if actor_j[i] >= 0 else pred.p_sw
                W_out[i] = self._interp_weights(o["interpretations"], pvec)
        # 観測 y とマスク（曖昧な型は観測なし。interpretation_counts のときは重みで計数）
        y_person = np.zeros((nP, N_TYPES)); y_mask = np.ones((nP, N_TYPES))
        y_sw = np.zeros(N_TYPES); sw_mask = np.ones(N_TYPES)
        for i, o in enumerate(outs):
            if not ambiguous[i]:
                continue
            for k in np.nonzero(W_out[i] > 0)[0]:
                if actor_j[i] >= 0:
                    if cfg.interpretation_counts:
                        y_person[actor_j[i], k] = max(y_person[actor_j[i], k], W_out[i, k])
                    else:
                        y_mask[actor_j[i], k] = 0.0
                else:
                    if cfg.interpretation_counts:
                        y_sw[k] = max(y_sw[k], W_out[i, k])
                    else:
                        sw_mask[k] = 0.0
        for i, o in enumerate(outs):
            if ambiguous[i]:
                continue
            k = TYPE_INDEX[o["type"]]
            if actor_j[i] >= 0:
                y_person[actor_j[i], k] = 1.0; y_mask[actor_j[i], k] = 1.0
            else:
                y_sw[k] = 1.0; sw_mask[k] = 1.0
        phi = self.belief.responsibility(pred, y_person)
        phi1 = self.belief.responsibility(pred, np.ones((nP, N_TYPES))) if (nP and any(ambiguous)) else phi
        phi_o = np.ones(n_o)
        is_person = np.zeros(n_o)
        for i in range(n_o):
            j = actor_j[i]
            if j >= 0:
                is_person[i] = 1.0
                phi_o[i] = float(W_out[i] @ phi1[j]) if ambiguous[i] else phi[j, int(np.argmax(W_out[i]))]
        return {"W_out": W_out, "ambiguous": ambiguous, "actor_j": actor_j, "y_person": y_person, "y_mask": y_mask, "y_sw": y_sw,
                "sw_mask": sw_mask, "phi": phi, "phi1": phi1, "phi_o": phi_o, "is_person": is_person}

    def _split_teachers(self, contrib: np.ndarray, phi_o: np.ndarray, is_person: np.ndarray, actor_j: np.ndarray, nP: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """教師（帰属で分ける）。fixed_teacher では帰属を使わない（3-F 型）。"""
        cfg = self.cfg
        person_teachers = [np.zeros(N_NEEDS) for _ in range(nP)]
        sit_teacher = np.zeros(N_NEEDS)
        for i in range(len(phi_o)):
            if is_person[i]:
                w = 1.0 if cfg.fixed_teacher else phi_o[i]
                person_teachers[actor_j[i]] = person_teachers[actor_j[i]] + w * contrib[i]
                if not cfg.fixed_teacher:
                    sit_teacher = sit_teacher + (1.0 - phi_o[i]) * contrib[i]
            else:
                sit_teacher = sit_teacher + contrib[i]
        return person_teachers, sit_teacher

    def _core(self, e: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.cfg
        a = self.concern.a
        P = e["participants"]
        nP = len(P)
        outs = e["outcomes"]
        n_o = len(outs)
        warnings = self.emb.unknown_words([("cls", p["class"]) for p in P] + [("ctx", e["context"])]
                                          + [("concept", c) for c in e["concepts"]])
        t_now = self._clock()
        # 2. 予期
        ex = self._expectation(e["context"], e["concepts"], P, e["announced"], e["id"], t_now)
        pred, mf, mb, omega = ex["pred"], ex["mf"], ex["mb"], ex["omega"]
        Q_val, C_Q, J_Q, Q, cues_q = ex["Q_val"], ex["C_Q"], ex["J_Q"], ex["Q"], ex["cues_q"]
        Q_person = float(omega * mf["V_MF_person"] + (1.0 - omega) * mb["V_MB_person"])
        # 結果の解釈の重み（曖昧な結果は信念で決める）、観測 y、責任 φ
        po = self._parse_outcomes(P, outs, pred)
        W_out, ambiguous, actor_j, phi, phi_o, is_person = po["W_out"], po["ambiguous"], po["actor_j"], po["phi"], po["phi_o"], po["is_person"]
        y_person, y_mask, y_sw, sw_mask = po["y_person"], po["y_mask"], po["y_sw"], po["sw_mask"]
        # Δh, Δu, 寄与
        rows = innate.mixed_rows(W_out, [o["magnitude"] for o in outs], self.M)
        dh, du, contrib = innate.contributions(self.h, rows)
        # 教師（帰属で分ける）。fixed_teacher では帰属を使わない（3-F 型）。
        person_teachers, sit_teacher = self._split_teachers(contrib, phi_o, is_person, actor_j, nP)
        # 手がかりと感じた価値
        types_in = []
        for i, o in enumerate(outs):
            types_in += [f"type:{o['type']}"] if not ambiguous[i] else [f"type:{t}" for t in o["interpretations"]]
        cues = _dedupe([f"concept:{c}" for c in e["concepts"]]
                       + [x for p in P for x in (f"ind:{p['id']}", f"cls:{p['class']}")] + types_in)
        inn = float(a @ du)
        cond = 0.0 if cfg.no_conditioning else cfg.lambda_c * self.cond.value(cues, e["context"], t_now)
        parts = V.felt_parts(inn, cond, Q, cfg)
        f = V.felt_value(parts, self.mood, cfg)
        delta = V.surprise(f, Q, cfg)
        emo = V.emotions(f, delta, Q, contrib, phi_o, is_person, cfg)
        # --- 辞書にまとめる ---
        contrib_list = []
        for i, o in enumerate(outs):
            k = int(np.argmax(W_out[i]))
            pp = float(pred.p_person[actor_j[i], k]) if actor_j[i] >= 0 else float(pred.p_sw[k])
            contrib_list.append({
                "type": o["type"] if not ambiguous[i] else "ambiguous", "actor": o["actor"], "magnitude": o["magnitude"],
                "contrib": _need_dict(contrib[i]),
                "phi_person": float(phi_o[i]) if is_person[i] else None,
                "phi_ctx": float(1.0 - phi_o[i]) if is_person[i] else None,
                "p_predicted": pp,
            })
        phi_list = [{"participant": p["id"], "type": t, "phi_person": float(phi[j, k]),
                     "phi_ctx": float(1.0 - phi[j, k])}
                    for j, p in enumerate(P) for k, t in enumerate(OUTCOME_TYPES)
                    if y_person[j, k] > 0.5 and y_mask[j, k] > 0.5]
        interp_list = [{"outcome_index": i, "actor": o["actor"], "llm": dict(o["interpretations"]),
                        "weights": {t: float(W_out[i, TYPE_INDEX[t]]) for t in o["interpretations"]}}
                       for i, o in enumerate(outs) if ambiguous[i]]
        ap: Dict[str, Any] = {
            "kind": "experience", "id": e["id"], "context": e["context"],
            "Q": float(Q), "Q_val": float(Q_val), "C_Q": float(C_Q), "J_Q": float(J_Q), "Q_person": Q_person, "cues_Q": cues_q,
            "V_MB": mb["V_MB"], "V_MF": mf["V_MF"], "omega": omega,
            "sigma2_MB": mb["var_MB"], "sigma2_MF": mf["var_MF"],
            "dh": _need_dict(dh), "du": _need_dict(du), "du_total": float(du.sum()), "du_weighted": inn,
            "a": a.copy(), "contributions": contrib_list, "phi": phi_list, "interpretations": interp_list,
            "teachers": {"person": {p["id"]: _need_dict(person_teachers[j]) for j, p in enumerate(P)},
                         "situation": _need_dict(sit_teacher)},
            "f": f, "f_parts": parts, "delta": delta, "mood": float(self.mood), "emotions": emo,
            "cues": cues, "belief_head_used": bool(pred.head_used), "warnings": warnings,
        }
        internals = {"pred": pred, "mf": mf, "mb": mb, "y_person": y_person, "y_mask": y_mask, "y_sw": y_sw,
                     "sw_mask": sw_mask, "phi": phi, "dh": dh, "du": du, "contrib": contrib,
                     "person_teachers": person_teachers, "sit_teacher": sit_teacher, "cues": cues, "inn": inn,
                     "W_out": W_out, "phi_o": phi_o, "is_person": is_person, "actor_j": actor_j, "ambiguous": ambiguous}
        return ap, internals

    @staticmethod
    def _cues_q(concepts: List[str], participants: List[Dict[str, str]], announced: List[str]) -> List[str]:
        """選択の前に分かる手がかり: 概念、参加者の個体と類、申し出に付いている結果の型（announced）。"""
        return _dedupe([f"concept:{c}" for c in concepts]
                       + [x for p in participants for x in (f"ind:{p['id']}", f"cls:{p['class']}")]
                       + [f"type:{t}" for t in announced])

    def _clock(self) -> float:
        """経験の通し番号（学習した ＋ 回避した）。条件づけの潜在原因の時間（τ = 30 経験）はこれで数える。状態から決まる。"""
        return float(self.n_experiences + self.n_avoided)

    def _c_q(self, cues: List[str], ctx: Optional[str], t: float) -> float:
        """Q に入る条件づけられた価値 λ_CQ ΣC(手がかり | 文脈, t)。no_conditioning / no_cq では 0。"""
        if self.cfg.no_conditioning or self.cfg.no_cq:
            return 0.0
        return self.cfg.lambda_cq * self.cond.value(cues, ctx, t)

    def _p_approach(self, Q: float) -> float:
        return float(sigmoid(np.array([self.cfg.beta_choice * (Q + self.cfg.explore_bonus)]))[0])

    @staticmethod
    def _continues(prev: Optional[str], cur: Optional[str]) -> bool:
        """文脈の連続: session が同じ（どちらも未指定なら連続する事象）。"""
        return prev == cur

    # ==================================================================
    # 公開 API
    # ==================================================================
    def appraise(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """予期・評価（2〜4）を計算する。状態（乱数を含む）を一切変えない。二択は、選択肢ごとの Q・選ぶ確率・葛藤の強さを返す（選ばない）。
        訂正は状態を変える操作なので、appraise の対象ではない。"""
        kind = experience_kind(exp)
        if kind == KIND_CORRECTION:
            raise SchemaError("訂正は状態を変える操作です。appraise の対象は経験と二択です")
        if kind == KIND_CHOICE:
            return _py(self._choose(validate_choice(exp), commit=False))
        e = validate_experience(exp)
        ap, _ = self._core(e)
        ap["optional"] = e["optional"]
        if e["optional"]:
            ap["p_approach"] = self._p_approach(ap["Q"])
            ap["approached"] = None
        return _py(ap)

    def experience(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """経験を評価し、学習する。戻り値は appraisal dict。kind = "correction"（訂正。2.2）と "choice"（二択。3.2）も受ける。"""
        kind = experience_kind(exp)
        if kind == KIND_CORRECTION:
            return self._correct(validate_correction(exp))
        if kind == KIND_CHOICE:
            return self._choose(validate_choice(exp), commit=True)
        return self._experience(validate_experience(exp))

    # ------------------------------------------------------------------ 正当化 J（v3.3 第2部 3.3）
    def _j_apply(self, dJ: Dict[str, float]) -> Dict[str, float]:
        """J の 1 経験分の更新（保存はしない）: 動かした手がかり J ← J + dJ、J ← J_max tanh(J/J_max)（有界）、続いて毎経験の減衰 J ← (1 − λ_J) J。
        動かしていない手がかりは減衰だけ（dJ = 0 の手がかり、つまり両方の選択肢に共通の手がかりも動かさない）。"""
        cfg = self.cfg
        out: Dict[str, float] = {}
        for k, v in self.just.items():
            if dJ.get(k, 0.0) == 0.0:
                out[k] = (1.0 - cfg.lambda_j) * v
        for k, d in dJ.items():
            if d != 0.0:
                v = cfg.j_max * float(np.tanh((self.just.get(k, 0.0) + d) / cfg.j_max))
                out[k] = (1.0 - cfg.lambda_j) * v
        return {k: out[k] for k in sorted(out)}

    # ------------------------------------------------------------------ 二択（v3.3 第2部 3.2〜3.3）
    def _choose(self, ch: Dict[str, Any], commit: bool) -> Dict[str, Any]:
        """二択。各選択肢の Q（C と J を含む）から softmax(β_c Q) で選ぶ。選んだ側の結果だけを観測して学ぶ（通常の経験として）。
        葛藤の強さ c = exp(−(ΔQ/σ_c)²)（ΔQ = 選ぶ前の Q の差）。選んだ側の手がかり（概念・類・文脈）の J を +η_J c、選ばなかった側の J を −η_J c 動かす
        （両方に共通の手がかりは相殺）。J の有界化と減衰は経験の最後（_j_apply）。commit=False は選ばずに数値だけ返す（appraise）。"""
        cfg = self.cfg
        opts = ch["options"]
        for o in opts:
            for pt in o["participants"]:
                self.belief.check_class(pt["id"], pt["class"])
        t_now = self._clock()
        # 2 つの選択肢は、同じ種（二択の id）で MB の標本を引く: 信念が同じなら Q の差は 0 になる（標本の雑音で ΔQ が揺れない）
        exs = [self._expectation(o["context"], o["concepts"], o["participants"], o["announced"], ch["id"], t_now) for o in opts]
        Q = np.array([x["Q"] for x in exs])
        z = cfg.beta_choice * Q
        ez = np.exp(z - z.max())
        p = ez / ez.sum()
        dQ = float(Q[0] - Q[1])
        conflict = float(np.exp(-((dQ / cfg.sigma_conflict) ** 2)))
        jcues = [self._j_cues(o["context"], o["concepts"], o["participants"]) for o in opts]
        info: Dict[str, Any] = {
            "kind": "choice", "id": ch["id"], "chosen": None, "p_choose": [float(p[0]), float(p[1])], "dQ": dQ, "conflict": conflict,
            "options": [{"Q": float(x["Q"]), "Q_val": float(x["Q_val"]), "C_Q": float(x["C_Q"]), "J_Q": float(x["J_Q"]), "j_cues": jcues[i]} for i, x in enumerate(exs)],
        }
        if not commit:
            return info
        chosen = int(self.rng.random() < p[1])
        dJ: Dict[str, float] = {}
        if not cfg.no_justification:
            step = cfg.eta_j * conflict
            for cue in jcues[chosen]:
                dJ[cue] = dJ.get(cue, 0.0) + step
            for cue in jcues[1 - chosen]:
                dJ[cue] = dJ.get(cue, 0.0) - step
            dJ = {k: v for k, v in dJ.items() if v != 0.0}
        j_after = self._j_apply(dJ)
        info["chosen"] = chosen
        info["dJ"] = dict(dJ)
        info["J_before"] = {k: float(self.just.get(k, 0.0)) for k in dJ}
        info["J_after"] = {k: float(j_after.get(k, 0.0)) for k in dJ}
        if cfg.record_ledger:
            self.ledger.add({"kind": "choice", "id": ch["id"], "chosen": chosen, "Q": [float(x) for x in Q], "p_choose": [float(p[0]), float(p[1])],
                             "dQ": dQ, "conflict": conflict, "dJ": {k: float(v) for k, v in dJ.items()},
                             "J_before": dict(info["J_before"]), "J_after": dict(info["J_after"]),
                             "options": [{"Q_val": float(x["Q_val"]), "C_Q": float(x["C_Q"]), "J_Q": float(x["J_Q"])} for x in exs]})
        o = opts[chosen]
        e = {"id": ch["id"], "context": o["context"], "concepts": list(o["concepts"]), "participants": list(o["participants"]),
             "outcomes": list(ch["outcomes_by_option"][str(chosen)]), "optional": False, "session": o["session"], "announced": list(o["announced"])}
        return self._experience(e, choice=info, dJ=dJ)

    # ------------------------------------------------------------------ 経験
    def _experience(self, e: Dict[str, Any], choice: Optional[Dict[str, Any]] = None, dJ: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """検証済みの経験を評価して学習する（通常の経験と、二択で選んだ側）。choice は二択の情報（あれば自分で選んだ経験: chosen = 1）。"""
        cfg = self.cfg
        for p in e["participants"]:
            self.belief.check_class(p["id"], p["class"])
        ap, it = self._core(e)
        # 3. optional なら接近・回避を選ぶ
        ap["optional"] = e["optional"]
        if e["optional"]:
            p_app = self._p_approach(ap["Q"])
            approached = bool(self.rng.random() < p_app)
            ap["p_approach"] = p_app
            ap["approached"] = approached
            if not approached:
                if cfg.record_ledger:
                    self.ledger.add({"kind": "avoid", "id": e["id"], "context": e["context"],
                                     "Q": float(ap["Q"]), "C_Q": float(ap["C_Q"]), "J_Q": float(ap["J_Q"]), "p_approach": p_app})
                self.n_avoided += 1
                self.cond.now = self._clock()
                self.just = self._j_apply({})                           # 時間は経つ（毎経験の減衰）
                for key in ("f", "f_parts", "delta", "du", "du_total", "du_weighted", "dh", "contributions", "phi",
                            "teachers", "emotions", "interpretations"):
                    ap[key] = None
                ap["value_change"] = None
                return _py(ap)
        chosen = bool(choice is not None or e["optional"])              # 主体性: 自分で選んだ経験（optional を受けた、または二択で選んだ）
        serial = self.n_experiences
        h_before = self.h.copy()
        # 埋め込みの登録（学習時のみ。以後は状態の値を使う）と警告
        P = e["participants"]
        for p in P:
            self.emb.register("cls", p["class"])
        self.emb.register("ctx", e["context"])
        for c in e["concepts"]:
            self.emb.register("concept", c)
        self.warnings += ap["warnings"]
        # 5. 信念の更新
        bu = self.belief.update(e["context"], P, it["y_person"], it["y_mask"], it["y_sw"], it["sw_mask"], it["phi"],
                                e["outcomes"], self.rng, exp_serial=serial)
        if bu.get("retrained") and cfg.record_ledger:             # v3.3: 類を除いたヘッドの学び直し。使った類・観測の数・混合の重みを残す
            self.ledger.add({"kind": "head_retrain", "after": e["id"], "classes": list(self.belief.rawcls),
                             "rows": int(self.belief.rb_len), "trained_heads": list(self.belief.trained_names()),
                             "weights": self.belief.head_weights_dict()})
        if bu["recall"] is not None and cfg.record_ledger:        # 想起（帰属のやり直し）。台帳には掃引の要約を残す
            self.ledger.add({"kind": "recall", "after": e["id"], **{k: float(v) for k, v in bu["recall"].items()}})
        # 6. 価値の更新
        a = self.concern.a.copy()
        mf_before = it["mf"]
        lr = self.value.learn(P, e["context"], e["concepts"], it["person_teachers"], it["sit_teacher"], a)
        mf_after = self.value.mf(P, e["context"], e["concepts"], a)
        # 6b. 条件づけ C の TD(0): 直前の事象の手がかりを、この事象の r + γ C_frozen で更新する
        #     潜在原因は、手がかりが現れたときの文脈と時刻（pending）で事後を求める。この事象の C_frozen は、この事象の文脈・時刻での応答
        dC: Dict[str, float] = {}
        td_y: Optional[float] = None
        cdet: Dict[str, Any] = {}
        reinst: List[Any] = []
        t_now = self._clock()
        if not cfg.no_conditioning:
            r = it["inn"]
            if self.pending is not None:
                if self._continues(self.pending["session"], e["session"]):
                    td_y = r + cfg.gamma_c * self.cond.value(it["cues"], e["context"], t_now)
                else:
                    td_y = 0.0                                  # 文脈が途切れたら、次の価値は 0
                dC, cdet = self.cond.update(self.pending["cues"], td_y, self.pending["context"], self.pending["t"])
                if cfg.record_ledger and self.cond.K > 1:                  # 割当の記録（どちらの設計でも）: 手がかりの経験ごとに、事前の内訳・事後・MAP の原因・学習前後の C_k・各原因の ΔC
                    for cue_name, d in cdet.items():
                        self.ledger.add({"kind": "assignment", "after": e["id"], "cue_event": self.pending.get("id"), "cue": cue_name,
                                         "t": float(self.pending["t"]), "context": self.pending["context"], "td_target": float(td_y),
                                         "y_x": d["y_x"], "post": d["post"], "post_new": d["post_new"], "map": d["map"],
                                         "created": d["created"], "reused": d["reused"], **d["trace"]})
            # 復位: 手がかりなしで嫌悪の結果が起きたとき、負の価値を持つ原因の最後に使われた時刻を更新する（この事象と直前の事象の手がかりは除く）
            if r <= -cfg.lc_aversive_threshold:
                skip = list(it["cues"]) + (list(self.pending["cues"]) if self.pending is not None else [])
                reinst = self.cond.reinstate(skip, t_now, e["context"], r)
                if cfg.record_ledger and self.cond.last_reinstate:                            # 復位: revised は推論した原因とその事後、spec_literal は時刻を更新した原因の一覧
                    self.ledger.add({"kind": "reinstate", "after": e["id"], **self.cond.last_reinstate})
            self.pending = {"cues": list(it["cues"]), "session": e["session"], "context": e["context"], "t": t_now, "id": e["id"]}
        # 7. 関心の重み a（主体性の重み: 自分で選んだ経験は速く）、欲求の状態 h、気分 m
        self.concern.update(it["du"], chosen)
        mood_before = self.mood
        self.mood = V.next_mood(self.mood, ap["delta"], cfg)
        self.h = innate.next_h(self.h, it["dh"])
        # 要約
        ent = {}
        for pb, pa in zip(mf_before["persons"], mf_after["persons"]):
            ent[pb["id"]] = {"before": pb["value"], "after": pa["value"], "delta": pa["value"] - pb["value"]}
        sit = {"before": mf_before["situation"]["value"], "after": mf_after["situation"]["value"],
               "delta": mf_after["situation"]["value"] - mf_before["situation"]["value"]}
        tops = []
        for model, dW in (("person", lr["dW_person"]), ("situation", lr["dW_situation"])):
            for name, row in dW.items():
                tops.append((float(np.abs(row).sum()), model, name, row))
        tops.sort(key=lambda t: -t[0])
        ap["value_change"] = {
            "entities": ent, "situation": sit,
            "prediction_error": {"person": {p["id"]: _need_dict(er) for p, er in zip(P, lr["err_person"])},
                                 "situation": _need_dict(lr["err_situation"])},
            "top_features": [{"model": m, "feature": n, "dW": _need_dict(r_), "dW_sum": float(r_.sum())}
                             for _, m, n, r_ in tops[:6]],
            "conditioned": {"td_target": td_y, "dC": dC, "reinstated": [[c, int(k)] for c, k in reinst]},
            "a_after": self.concern.a.copy(),
        }
        ap["mood_after"] = float(self.mood)
        if choice is not None:
            ap["kind"] = "choice"
            ap["choice"] = choice
        # 8. 台帳
        if cfg.record_ledger:
            self.ledger.add({
                "kind": "experience", "id": e["id"], "context": e["context"],
                "Q": float(ap["Q"]), "C_Q": float(ap["C_Q"]), "J_Q": float(ap["J_Q"]), "f": ap["f"], "delta": ap["delta"], "emotions": dict(ap["emotions"]),
                "du": [float(x) for x in it["du"]], "dh": [float(x) for x in it["dh"]],
                "phi": [[x["participant"], x["type"], x["phi_person"]] for x in ap["phi"]],
                "mood": [float(mood_before), float(self.mood)],
                "a": [[float(x) for x in a], [float(x) for x in self.concern.a]],
                "td": {"y": td_y, "dC": {k: float(v) for k, v in dC.items()}, "lc": cdet,
                       "reinstated": [[c, int(k)] for c, k in reinst]},
                "teachers": {"person": {p["id"]: [float(x) for x in t] for p, t in zip(P, it["person_teachers"])},
                             "situation": [float(x) for x in it["sit_teacher"]]},
                "dW": {"person": {n: [float(x) for x in r_] for n, r_ in lr["dW_person"].items()},
                       "situation": {n: [float(x) for x in r_] for n, r_ in lr["dW_situation"].items()}},
            })
        self._log_experience(e, it, serial, t_now, h_before, a, "choice" if choice is not None else "experience")
        self.n_experiences += 1
        self.cond.now = self._clock()
        self.just = self._j_apply(dJ or {})                              # 正当化 J: 選んだ側 +、選ばなかった側 −、有界化、毎経験の減衰（二択の最後）
        return _py(ap)

    # ------------------------------------------------------------------ 経験記録（訂正が参照する。台帳とは別）
    def _log_experience(self, e: Dict[str, Any], it: Dict[str, Any], serial: int, t_now: float, h_before: np.ndarray,
                        a_before: np.ndarray, how: str) -> None:
        """学習した経験の記録: 参加者・結果（現在の事実）・解釈の重み・責任 φ_o・欲求の状態と a・参加者ごとの寄与。訂正（kind = "correction"）が、
        当時の寄与を引き、価値の再生（retroactive_value）が使う。当時の記録（f, φ, Δu, 感情）は台帳に残り、訂正では変えない。"""
        P = e["participants"]
        outs = copy.deepcopy(e["outcomes"])
        self.exp_log.append({
            "id": e["id"], "serial": int(serial), "how": how, "t": float(t_now), "context": e["context"], "concepts": list(e["concepts"]),
            "participants": [dict(p) for p in P], "announced": list(e["announced"]), "session": e["session"],
            "outcomes": outs, "original_outcomes": copy.deepcopy(outs),
            "w_out": [[float(x) for x in row] for row in it["W_out"]], "phi_o": [float(x) for x in it["phi_o"]],
            "actor_j": [int(x) for x in it["actor_j"]], "h_before": [float(x) for x in h_before], "a_before": [float(x) for x in a_before],
            "teacher_by_participant": [[float(x) for x in v] for v in it["person_teachers"]], "corrections": []})
        self._exp_pos[e["id"]] = len(self.exp_log) - 1

    def _replay_values(self) -> V.ValueNet:
        """価値ネットを、経験記録の現在の事実で最初から再生する（retroactive_value）。欲求の状態 h は H0 から、各経験の結果で進め直し、
        観測雑音の a と責任 φ_o は当時のもの（訂正された経験の φ_o は、訂正のときに求めたもの）。条件づけ C・関心の重み a・信念は触れない。"""
        cfg = self.cfg
        nv = V.ValueNet(cfg, self.emb)
        h = np.full(N_NEEDS, innate.H0)
        for rec in self.exp_log:
            P = rec["participants"]
            n_o = len(rec["outcomes"])
            W_out = np.array(rec["w_out"], dtype=np.float64).reshape(n_o, N_TYPES)
            rows = innate.mixed_rows(W_out, [o["magnitude"] for o in rec["outcomes"]], self.M)
            dh, du, contrib = innate.contributions(h, rows)
            actor_j = np.array(rec["actor_j"], dtype=int)
            pt, st = self._split_teachers(contrib, np.array(rec["phi_o"], dtype=np.float64), (actor_j >= 0).astype(np.float64), actor_j, len(P))
            nv.learn(P, rec["context"], rec["concepts"], pt, st, np.array(rec["a_before"], dtype=np.float64))
            h = innate.next_h(h, dh)
        return nv

    # ------------------------------------------------------------------ 訂正（v3.3 第2部 2.2〜2.3）
    def _correct(self, c: Dict[str, Any]) -> Dict[str, Any]:
        """訂正の経験。事実（信念）は訂正し、当時の主観的な記録（台帳の f, φ, Δu, 感情）は変えず、価値は自動では消さずに再評価する。
        1. 信念: refers_to の経験の観測を取り消し、訂正後の観測を加える（責任 φ は取り消した後の信念で求める）。
        2. 当時の記録: 変更しない。
        3. 価値: 参加者 e ごとに Δ_e = 寄与_訂正後(e) − 寄与_当時(e)（人モデルが学んだ量 = φ で重み付けた寄与）を求め、人モデルの e の特徴に、観測を「現在の予測 + Δ_e」とするカルマン更新を
           観測雑音 R_corr = R / w_reapp（実際は欲求ごとに R/(a_k w_reapp)）で 1 回。確信の高い古い価値ほど動きにくい。C と a は変えない。
        4. 訂正そのものの感じた価値（安堵・罪悪感の手がかり）を読み出して台帳に記録する（学習には使わない）。
        アブレーション: retroactive_value（価値を訂正後の事実で最初から再生。人モデルも状況モデルも作り直すので、台帳の ΔW は両方の差）、no_reappraisal（価値を変えない）。"""
        cfg = self.cfg
        pos = self._exp_pos.get(c["refers_to"])
        if pos is None:
            raise ValueError(f"訂正の参照先が見つかりません（学習した経験の id のみ。回避した経験には観測がありません）: {c['refers_to']!r}")
        if not self.belief.can_correct():
            raise ValueError("訂正には観測の記録が必要です（no_scope / no_reattribution の設定では使えません）")
        rec = self.exp_log[pos]
        P = rec["participants"]
        nP = len(P)
        pids = {p["id"] for p in P}
        for o in c["corrected_outcomes"]:
            if not (o["actor"] in RESERVED_IDS or o["actor"] in pids):
                raise ValueError(f"訂正後の結果の actor は、参照先の経験の参加者か self / world です: {o['actor']!r}（参加者 {sorted(pids)}）")
        t_now = self._clock()
        a_now = self.concern.a.copy()
        key = "correction:" + c["id"]
        # 読み出し用: 訂正前の予期と、参加者への好悪
        ex0 = self._expectation(rec["context"], rec["concepts"], P, rec["announced"], key, t_now)
        like0 = [float(self.value.like_entity(p["id"], p["class"], a_now)[0]) for p in P]
        p0 = np.array(ex0["pred"].p_person)
        # 1. 信念
        old = self.belief.withdraw_experience(rec["serial"])
        pred = self.belief.predict(rec["context"], P)
        po = self._parse_outcomes(P, c["corrected_outcomes"], pred)
        self.belief.install_experience(old, po["y_person"], po["y_mask"], po["phi"], po["y_sw"], po["sw_mask"])
        if cfg.recall_after_correction:
            self.belief.reattribute()                      # 実験設定（既定 OFF）: 訂正の後に想起の掃引を 1 回
        # 3. 価値: 訂正後の寄与（当時の欲求の状態 h_before で）。Δ_e は、人モデルの e が学んだ量（寄与を責任 φ で分けた教師）の差:
        #    当時の教師 Σ_o φ_o contrib_o（学習時に人モデルに与えたもの）と、訂正後の教師（訂正後の結果と、取り消した後の信念で求めた φ）。
        #    重みなしの寄与どうしの差だと、人モデルが φ の分しか学んでいないのに全部を引いてしまい、過剰に訂正する（CHANGELOG の D 節）。
        rows = innate.mixed_rows(po["W_out"], [o["magnitude"] for o in c["corrected_outcomes"]], self.M)
        dh, du, contrib = innate.contributions(np.array(rec["h_before"], dtype=np.float64), rows)
        new_teach, _ = self._split_teachers(contrib, po["phi_o"], po["is_person"], po["actor_j"], nP)
        old_teach = [np.array(v) for v in rec["teacher_by_participant"]]
        shifts = [new_teach[j] - old_teach[j] for j in range(nP)]
        # 経験記録を現在の事実に直す（当時の結果は original_outcomes に残る）
        rec["outcomes"] = copy.deepcopy(c["corrected_outcomes"])
        rec["w_out"] = [[float(x) for x in row] for row in po["W_out"]]
        rec["phi_o"] = [float(x) for x in po["phi_o"]]
        rec["actor_j"] = [int(x) for x in po["actor_j"]]
        rec["teacher_by_participant"] = [[float(x) for x in v] for v in new_teach]
        rec["corrections"].append(c["id"])
        mode = "none" if cfg.no_reappraisal else ("retroactive" if cfg.retroactive_value else "kalman")
        dW_person: Dict[str, np.ndarray] = {}
        dW_sit: Dict[str, np.ndarray] = {}
        if mode == "kalman":
            prec = (np.ones(N_NEEDS) if cfg.no_precision_weighting else a_now) * cfg.w_reapp      # R_corr,k = R / (a_k w_reapp)
            dW_person = self.value.reappraise(P, shifts, prec)["dW_person"]
        elif mode == "retroactive":
            nv = self._replay_values()
            for new_m, old_m, dst in ((nv.person, self.value.person, dW_person), (nv.sit, self.value.sit, dW_sit)):
                old_w = dict(zip(old_m.names, old_m.W))
                for i, nm in enumerate(new_m.names):
                    d = new_m.W[i] - old_w.get(nm, np.zeros(N_NEEDS))
                    if np.any(d != 0.0):
                        dst[nm] = d
            self.value = nv
        # 4. 読み出し: 訂正後の予期、好悪の変化、安堵・罪悪感の手がかり
        ex1 = self._expectation(rec["context"], rec["concepts"], P, rec["announced"], key, t_now)
        like1 = [float(self.value.like_entity(p["id"], p["class"], a_now)[0]) for p in P]
        gain = float(ex1["Q"] - ex0["Q"])
        f = V.felt_value({"innate": gain, "conditioned": 0.0, "reference": 0.0}, self.mood, cfg)
        relief = float(min(1.0, f)) if (f > 0.0 and ex0["Q"] < 0.0) else 0.0
        guilt = float(np.tanh(sum(max(l1 - l0, 0.0) for l0, l1 in zip(like0, like1) if l0 < 0.0) / cfg.s_f))
        p1 = np.array(self.belief.predict(rec["context"], P).p_person)
        types_changed = sorted({o["type"] for o in rec["original_outcomes"] if "type" in o} | {o["type"] for o in c["corrected_outcomes"] if "type" in o})
        belief_change = {p["id"]: {t: [float(p0[j, TYPE_INDEX[t]]), float(p1[j, TYPE_INDEX[t]])] for t in types_changed} for j, p in enumerate(P)}
        ap = {"kind": "correction", "id": c["id"], "refers_to": c["refers_to"], "source": c["source"], "reappraisal": mode,
              "original_outcomes": copy.deepcopy(rec["original_outcomes"]), "corrected_outcomes": copy.deepcopy(c["corrected_outcomes"]),
              "shifts": {p["id"]: _need_dict(shifts[j]) for j, p in enumerate(P)},
              "like": {p["id"]: [like0[j], like1[j]] for j, p in enumerate(P)},
              "Q": [float(ex0["Q"]), float(ex1["Q"])], "f": f, "emotions": {"relief": relief, "guilt": guilt},
              "belief_change": belief_change}
        if cfg.record_ledger:
            self.ledger.add({**{k: v for k, v in ap.items() if k != "kind"}, "kind": "correction", "t": float(t_now),
                             "dW": {"person": {n: [float(x) for x in r_] for n, r_ in dW_person.items()},
                                   "situation": {n: [float(x) for x in r_] for n, r_ in dW_sit.items()}}})
        self.n_corrections += 1
        return _py(ap)

    def probe(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """学習しない評価。状況 / エンティティ / 概念 / 手がかり。"""
        q = validate_probe(p)
        kind = q["kind"]
        cfg = self.cfg
        a = self.concern.a
        if kind == "situation":
            pred = self.belief.predict(q["context"], q["participants"])
            mf = self.value.mf(q["participants"], q["context"], q["concepts"], a)
            mb = V.model_based(pred, self.h, self.belief.mag, cfg, probe_key(q), self.M, a)
            omega, Q_val = V.arbitrate(mf["V_MF"], mf["var_MF"], mb["V_MB"], mb["var_MB"])
            cues_q = self._cues_q(q["concepts"], q["participants"], q["announced"])
            C_Q = self._c_q(cues_q, q["context"], self._clock())
            J_Q = self._j_q(self._j_cues(q["context"], q["concepts"], q["participants"]))
            out = {"kind": kind, "Q": Q_val + C_Q + J_Q, "Q_val": Q_val, "C_Q": C_Q, "J_Q": J_Q,
                   "V_MB": mb["V_MB"], "V_MF": mf["V_MF"], "omega": omega,
                   "V_MF_person": mf["V_MF_person"], "V_MB_person": mb["V_MB_person"],
                   "Q_person": float(omega * mf["V_MF_person"] + (1.0 - omega) * mb["V_MB_person"]),
                   "sigma2_MB": mb["var_MB"], "sigma2_MF": mf["var_MF"],
                   "persons": [{"id": x["id"], "value": x["value"], "known": x["known"]} for x in mf["persons"]],
                   "situation_value": mf["situation"]["value"],
                   "p_person": {pp["id"]: {t: float(pred.p_person[j, k]) for k, t in enumerate(OUTCOME_TYPES)}
                                for j, pp in enumerate(q["participants"])},
                   "p_sw": {t: float(pred.p_sw[k]) for k, t in enumerate(OUTCOME_TYPES)}}
            return _py(out)
        if kind == "entity":
            like, var_mf, pred5, known = self.value.like_entity(q["entity"], q["class"], a)
            part = [{"id": q["entity"], "class": q["class"]}]
            pred = self.belief.predict(q["context"], part)
            g = V.type_effect(self.h, self.belief.mag, self.M, a)
            pp = pred.p_person[0]
            V_MB = float((g * pp).sum())
            var_MB = float(((g ** 2) * (pred.var_person[0] + pp * (1 - pp) / cfg.mc_samples)).sum())
            omega, Q_val = V.arbitrate(like, var_mf, V_MB, var_MB)
            C_Q = self._c_q([f"ind:{q['entity']}", f"cls:{q['class']}"], q["context"], self._clock())
            J_Q = self._j_q([f"cls:{q['class']}"] + ([f"ctx:{q['context']}"] if q["context"] is not None else []))
            out = {"kind": kind, "entity": q["entity"], "class": q["class"], "context": q["context"],
                   "like": like, "like_by_need": _need_dict(pred5), "known": known,
                   "V_MF": like, "V_MB": V_MB, "omega": omega, "Q": Q_val + C_Q + J_Q, "Q_val": Q_val, "C_Q": C_Q, "J_Q": J_Q,
                   "sigma2_MF": var_mf, "sigma2_MB": var_MB,
                   "conditioned": self.cond.value([f"ind:{q['entity']}", f"cls:{q['class']}"], q["context"], self._clock()),
                   "p_person": {t: float(pp[k]) for k, t in enumerate(OUTCOME_TYPES)},
                   "p_ind": {t: float(pred.p_ind[0, k]) for k, t in enumerate(OUTCOME_TYPES)},
                   "m_cls": {t: float(pred.m_cls[0, k]) for k, t in enumerate(OUTCOME_TYPES)}}
            return _py(out)
        if kind == "cue":
            return _py({"kind": kind, "cue": q["cue"], "context": q["context"],
                        "C": self.cond.value_of(q["cue"], q["context"], self._clock()),
                        "J": 0.0 if self.cfg.no_justification else float(self.just.get(q["cue"], 0.0)),
                        "n_causes": self.cond.n_causes(q["cue"]), "known": q["cue"] in self.cond.idx,
                        "breakdown": self.cond.explain(q["cue"], q["context"], self._clock())})
        like, var_mf, pred5, known = self.value.like_concept(q["concept"], a)
        return _py({"kind": kind, "concept": q["concept"], "like": like, "like_by_need": _need_dict(pred5),
                    "known": known, "sigma2_MF": var_mf,
                    "J": 0.0 if self.cfg.no_justification else float(self.just.get(f"concept:{q['concept']}", 0.0)),
                    "conditioned": self.cond.value_of(f"concept:{q['concept']}", None, self._clock())})

    def explain_response(self, cue: str, context: Optional[str] = None, t: Optional[float] = None) -> Dict[str, Any]:
        """条件づけられた価値の応答 C(cue | 文脈, t) が、なぜその値なのかの内訳（学習しない。状態は変えない）。
        原因ごとの重み p(k)・価値 C_k・寄与 p(k)·C_k と、p(k) の要因（時間の項・文脈の項・計数の項と、試行数・文脈での計数）。t を省くと現在の時刻。
        「なぜ嫌悪が戻ったか」は、消去の前後や文脈・時間を変えて呼び、原因の重み p(k) の移り変わり（と価値 C_k が動いていないこと）を見る。
        割当の記録（どの試行がどの原因に入ったか）は assignment_trace(cue)。"""
        return _py(self.cond.explain(cue, context, self._clock() if t is None else float(t)))

    def assignment_trace(self, cue: str) -> List[Dict[str, Any]]:
        """手がかり cue の割当の記録（台帳の kind = "assignment"、および cue に試行を加えた kind = "reinstate"）を時間順に返す。"""
        out = []
        for en in self.ledger.entries:
            if en["kind"] == "assignment" and en["cue"] == cue:
                out.append(en)
            elif en["kind"] == "reinstate" and ((en.get("chosen") and en["chosen"]["cue"] == cue)
                                                or any(x["cue"] == cue for x in en.get("refreshed", []))):
                out.append(en)
        return out

    # ==================================================================
    # 保存・復元
    # ==================================================================
    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        bmeta, barr = self.belief.get_state()
        vmeta, varr = self.value.get_state()
        cmeta, carr = self.concern.get_state()
        dmeta, darr = self.cond.get_state()
        emeta, earr = self.emb.get_state()
        meta = {
            "config": self.cfg.to_dict(),
            "rng": _py(self.rng.bit_generator.state),
            "mood": float(self.mood),
            "n_experiences": int(self.n_experiences),
            "n_avoided": int(self.n_avoided),
            "belief": bmeta,
            "value": vmeta,
            "concern": cmeta,
            "conditioning": dmeta,
            "pending": _py(self.pending),
            "embeddings": emeta,
            "warnings": list(self.warnings),
            "ledger": _py(self.ledger.to_list()),
            "justification": _py(self.just),
            "exp_log": _py(self.exp_log),
            "n_corrections": int(self.n_corrections),
        }
        arrays = {"h": self.h, "M": self.M}
        for d in (barr, varr, carr, darr, earr):
            arrays.update(d)
        return meta, arrays

    def save(self, directory: Union[str, Path]) -> str:
        """state.json（設定・計数・特徴の索引・乱数の状態・所有する埋め込みの名前・台帳など）と
        arrays.npz（W, Σ, C, a, 重み、所有する埋め込みの値など）。スキーマ版と内容のハッシュを持つ。"""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta, arrays = self.get_state()
        digest = _digest(meta, arrays)
        with open(d / ARRAY_FILE, "wb") as f:
            np.savez(f, **arrays)
        with open(d / STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"schema_version": SCHEMA_VERSION, "content_hash": digest, "meta": meta}, f, ensure_ascii=False)
        return digest

    @classmethod
    def load(cls, directory: Union[str, Path], lexicon: Optional[Lexicon] = None) -> "NPN":
        """lexicon は、これから初めて現れる語の供給源。登録済みの語の埋め込みは状態の値を使うので、
        lexicon を別の座標に替えても、既知の語の評価は変わらない（U9）。"""
        d = Path(directory)
        with open(d / STATE_FILE, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise StateVersionError(f"スキーマ版が合いません: 保存 {doc.get('schema_version')!r}, 実装 {SCHEMA_VERSION}")
        with np.load(d / ARRAY_FILE, allow_pickle=False) as z:
            arrays = {k: np.array(z[k]) for k in z.files}
        meta = doc["meta"]
        if _digest(meta, arrays) != doc.get("content_hash"):
            raise StateIntegrityError("内容ハッシュが一致しません（改ざんまたは破損）")
        cfg = Config.from_dict(meta["config"])
        obj = cls(lexicon=lexicon, config=cfg, effect_table=arrays["M"])
        obj.rng = np.random.Generator(np.random.PCG64())
        obj.rng.bit_generator.state = meta["rng"]
        obj.belief.set_state(meta["belief"], arrays)
        obj.value.set_state(meta["value"], arrays)
        obj.concern.set_state(meta["concern"], arrays)
        obj.cond.set_state(meta["conditioning"], arrays)
        obj.emb.set_state(meta["embeddings"], arrays)
        obj.pending = meta["pending"]
        obj.warnings = list(meta["warnings"])
        obj.h = np.array(arrays["h"], dtype=np.float64)
        obj.mood = float(meta["mood"])
        obj.ledger = Ledger.from_list(meta["ledger"])
        obj.n_experiences = int(meta["n_experiences"])
        obj.n_avoided = int(meta["n_avoided"])
        obj.just = {k: float(v) for k, v in meta["justification"].items()}
        obj.exp_log = meta["exp_log"]
        obj._exp_pos = {r["id"]: i for i, r in enumerate(obj.exp_log)}
        obj.n_corrections = int(meta["n_corrections"])
        obj.cond.now = obj._clock()
        return obj

    # ==================================================================
    # 便利
    # ==================================================================
    def summary(self) -> Dict[str, Any]:
        """CLI show 用: 価値の上位と信念の要約。"""
        def top(model: V.KalmanModel, n: int = 8) -> List[Dict[str, Any]]:
            rows = [(float(self.concern.a @ model.W[i]), nm, model.W[i]) for i, nm in enumerate(model.names)
                    if not nm.startswith(("emb:", "cemb:"))]
            rows.sort(key=lambda t: -abs(t[0]))
            return [{"feature": nm, "value": v, "by_need": _need_dict(w)} for v, nm, w in rows[:n]]
        conds = sorted(zip(self.cond.names, self.cond.C, [self.cond.n_causes(n) for n in self.cond.names]),
                       key=lambda t: -abs(t[1]))[:8]
        jtop = sorted(self.just.items(), key=lambda kv: -abs(kv[1]))[:8]
        return _py({
            "justification_top": [{"cue": k, "J": float(v)} for k, v in jtop if abs(v) > 1e-9], "n_corrections": self.n_corrections,
            "n_experiences": self.n_experiences, "n_avoided": self.n_avoided,
            "h": _need_dict(self.h), "mood": self.mood, "a": _need_dict(self.concern.a),
            "person_top": top(self.value.person), "situation_top": top(self.value.sit),
            "conditioned_top": [{"cue": n, "C": float(c), "n_causes": int(k)} for n, c, k in conds],
            "belief": self.belief.summary(), "warnings": list(self.warnings),
        })
