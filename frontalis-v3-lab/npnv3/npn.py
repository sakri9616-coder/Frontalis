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
import math
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

SCHEMA_VERSION = 8          # v3.0 は 1、v3.1 は 2（a, C, 所有する埋め込み, 効果表）。改訂 2 で観測の記録（想起）、v3.2 で欲求ごとの Σ とヘッドの対数尤度が加わった(3 → 4)。
                            # v3.3 で、類ごとの生の計数・類を除いたヘッドのモデル（leave-class-out）・潜在原因つきの条件づけ（原因の価値・文脈の計数・時刻）・pending の文脈と時刻が加わった(4 → 5)。
                            # v3.3 第 1 部の設計の改訂で、潜在原因の試行の履歴（時刻・原因・文脈）と、原因ごとの文脈の集中度が加わった(5 → 6)
                            # v3.3 第 2 部で、選択の正当化 J・学習した経験の記録（exp_log）・訂正の回数、信念の観測の通し番号（rec_exp, rec_ser, rb_ser）と
                            # 自分・世界の観測の記録（sw_*）が加わった(6 → 7)
                            # v3.4 Stage 2 で、情報源の信用 Beta（src_beta）・検証待ちの訂正（ver_pending）・C-OR の累積 G・空の刻みの数（n_ticks）・検証の回数が加わった(7 → 8)。
                            # 7 の状態は既定（reappraisal_mode = "v33"）で読める（新しい項目は空として扱う）。6 以前は読めない。
READABLE_SCHEMAS = (7, 8)
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
        # --- v3.4 Stage 2（M1・M2）。reappraisal_mode = "v33" では一切使われない（空のまま）---
        self.src_beta: Dict[str, List[float]] = {}         # 情報源の信用 Beta(α_s, β_s)。更新は検証の出来事だけ（SPEC-v3.4 4.3, H3）。無い情報源は Beta(α0, β0)
        self.ver_pending: List[Dict[str, Any]] = []        # 検証待ちの訂正（訂正 1 回につき検証は最大 1 回）
        self.G: Dict[str, float] = {}                      # C-OR: 対象 e についての、受け入れた訂正の罪悪感と後悔の累積（減らさない。M3 は Stage 3。_m3_guilt_regret）
        self.n_ticks = 0                                   # 空の刻み（advance_clock）の数。時計に足される
        self.n_verifications = 0
        # --- v3.4 Stage 3（M3）---
        self.log_tau: Dict[str, float] = {}                # 自分の解釈への信頼 log τ（鍵 = "類|文脈"。無ければ 0 = τ 1）。変わるのは訂正の出来事だけ（H3・H9）
        self.tau_yoke: Optional[List[Dict[str, float]]] = None   # τ-yoke（試験用）: 訂正の通し番号ごとの log τ の全体（他の人格の軌跡）
        self.last_ev: Dict[str, str] = {}                  # "kappa:<源>" / "tau:<鍵>" → その量を最後に動かした出来事の id（台帳の因果の連鎖用）
        self.just_src: Dict[str, Dict[str, float]] = {}    # 正当化 J の源ごとの和（手がかり → 源 → 和。Σ 源 = J。P1）
        self.chosen_before: List[str] = []                 # 以前に選んだ選択肢の手がかり（persist の reaffirm）
        self.decisions: List[Dict[str, Any]] = []          # record_decisions のとき、経験ごとの離散的な決定（B1。状態ではない。保存しない）
        self.force_decisions: Optional[List[Dict[str, Any]]] = None
        self._force_i = 0
        self._dec: Dict[str, Any] = {}

    @property
    def a(self) -> np.ndarray:
        return self.concern.a

    # ==================================================================
    # 評価の核（状態を変えない）
    # ==================================================================
    def _interp_weights(self, interp: Dict[str, float], pvec: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """w_I ∝ q_LLM(I) · P_belief(I | actor, context)^τ（正規化。SPEC-v3.4 4.4。τ = 1 では v3.3 と同じ式そのもの）。no_interpretation では信念を使わず w_I = q_LLM(I)。"""
        w = np.zeros(N_TYPES)
        for t, q in interp.items():
            k = TYPE_INDEX[t]
            w[k] = q if self.cfg.no_interpretation else (q * pvec[k] if tau == 1.0 else q * pvec[k] ** tau)
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

    @staticmethod
    def tau_key(cls_or_actor: str, ctx: Optional[str]) -> str:
        return f"{cls_or_actor}|{ctx}"

    def tau_of(self, cls_or_actor: str, ctx: Optional[str], log_tau: Optional[Dict[str, float]] = None) -> float:
        """自分の解釈への信頼 τ（類 × 文脈。無ければ 1）。v33 では常に 1。"""
        if self.cfg.reappraisal_mode == "v33" or ctx is None:
            return 1.0
        lt = self.log_tau if log_tau is None else log_tau
        v = lt.get(self.tau_key(cls_or_actor, ctx), 0.0)
        return 1.0 if v == 0.0 else math.exp(v)

    def _parse_outcomes(self, P: List[Dict[str, str]], outs: List[Dict[str, Any]], pred: Any, belief: Optional[BeliefNet] = None,
                        ctx: Optional[str] = None, log_tau: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """結果の列を、解釈の重み W_out、観測 y とマスク、責任 φ、結果ごとの φ_o に直す（経験の評価と訂正で共通）。状態を変えない。
        belief は責任 φ を求める信念（既定は自分の信念。v3.4 M2 が、写しの信念で当時の事実を作り直すときに渡す）。
        ctx を渡すと、曖昧な結果の解釈の重みに v3.4 M3 の τ（行為者の類 × 文脈。log_tau を渡せば当時の値）が効く。"""
        cfg = self.cfg
        bel = self.belief if belief is None else belief
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
                tau_i = 1.0
                if ctx is not None and self.cfg.reappraisal_mode != "v33":
                    tau_i = self.tau_of(P[actor_j[i]]["class"] if actor_j[i] >= 0 else o["actor"], ctx, log_tau)
                W_out[i] = self._interp_weights(o["interpretations"], pvec, tau_i)
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
        phi = bel.responsibility(pred, y_person)
        phi1 = bel.responsibility(pred, np.ones((nP, N_TYPES))) if (nP and any(ambiguous)) else phi
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
        po = self._parse_outcomes(P, outs, pred, ctx=e["context"])
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
        return float(self.n_experiences + self.n_avoided + self.n_ticks)

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
        v = validate_choice(exp) if kind == KIND_CHOICE else validate_experience(exp)
        forced = self._next_forced()
        self._dec = {"id": v["id"], "approach": None, "choice": None, "assign": {}, "reinstate_called": False, "reinstate": []}
        ap = self._choose(v, commit=True, forced=forced) if kind == KIND_CHOICE else self._experience(v, forced=forced)
        if self.cfg.record_decisions:
            self.decisions.append(self._dec)
        return ap

    def _next_forced(self) -> Optional[Dict[str, Any]]:
        """B1 の強制再生: force_decisions（元の実行の離散的な決定の列。npn.decisions）が与えられていれば、経験ごとに次の決定を返す。"""
        if self.force_decisions is None:
            return None
        d = self.force_decisions[self._force_i]
        self._force_i += 1
        return d             # 一部の項目だけでもよい（省いた項目は自然な決定。例: {"choice": 0} だけで二択の選択を強制）

    # ------------------------------------------------------------------ 正当化 J（v3.3 第2部 3.3、第8節）
    # J の変化はすべて源つきの出来事 (手がかり, 源, 変化量) として記録する（just_src。台帳の kind = "justification"）。Σ 源 = J（P1）。
    #   orig（Z-orig。第3節そのまま）: spread（迷った選択）、bound（J_max tanh の有界化で変わった分）、decay（毎経験の減衰 λ_J）
    #   persist（Z-persist）: spread、reaffirm（再び選ぶ）、counter（反する選択）、overwhelm（圧倒的な反証）、cap（数値の安全の上限 J_num）。減衰なし
    #   persist_with_decay: persist に decay
    def _j_step_orig(self, dJ: Dict[str, float]) -> Tuple[Dict[str, float], List[Tuple[str, str, float]]]:
        """Z-orig の 1 経験分の更新（保存はしない）: 動かした手がかり J ← J + dJ、J ← J_max tanh(J/J_max)（有界）、続いて毎経験の減衰 J ← (1 − λ_J) J。
        動かしていない手がかりは減衰だけ。出来事の列（spread・bound・decay）も返す。"""
        cfg = self.cfg
        out: Dict[str, float] = {}
        events: List[Tuple[str, str, float]] = []
        for k, v in self.just.items():
            if dJ.get(k, 0.0) == 0.0:
                nv = (1.0 - cfg.lambda_j) * v
                out[k] = nv
                events.append((k, "decay", nv - v))
        for k, d in dJ.items():
            if d != 0.0:
                j0 = self.just.get(k, 0.0)
                raw = j0 + d
                v = cfg.j_max * float(np.tanh(raw / cfg.j_max))
                nv = (1.0 - cfg.lambda_j) * v
                out[k] = nv
                events += [(k, "spread", d), (k, "bound", v - raw), (k, "decay", nv - v)]
        return {k: out[k] for k in sorted(out)}, events

    def _j_step_persist(self, events_in: List[Tuple[str, str, float]]) -> Tuple[Dict[str, float], List[Tuple[str, str, float]]]:
        """Z-persist の更新: 記録された出来事だけを足す。数値の安全のための上限（cap）。persist_with_decay は続いて毎経験の減衰。出来事がなければ J は一切変わらない（P2）。"""
        cfg = self.cfg
        out = dict(self.just)
        events: List[Tuple[str, str, float]] = []
        for k, src, d in events_in:
            out[k] = out.get(k, 0.0) + d
            events.append((k, src, d))
        for k in sorted(out):
            if abs(out[k]) > cfg.j_num:
                nv = float(np.sign(out[k])) * cfg.j_num
                events.append((k, "cap", nv - out[k]))
                out[k] = nv
        if cfg.justification_mode == "persist_with_decay":
            for k in sorted(out):
                nv = (1.0 - cfg.lambda_j) * out[k]
                events.append((k, "decay", nv - out[k]))
                out[k] = nv
        if not events:
            return dict(self.just), events
        return {k: out[k] for k in sorted(out)}, events

    def _j_commit(self, exp_id: str, dJ: Optional[Dict[str, float]] = None, events_in: Optional[List[Tuple[str, str, float]]] = None,
                  discount: Optional[Dict[str, Any]] = None, chosen_cues: Optional[List[str]] = None) -> None:
        """1 経験分の J の更新を確定する（orig は dJ、persist は出来事の列）。源ごとの和 just_src を更新し、変化があれば台帳に kind = "justification" を残す。"""
        cfg = self.cfg
        if cfg.justification_mode == "orig":
            new, events = self._j_step_orig(dJ or {})
        else:
            new, events = self._j_step_persist(events_in or [])
        self.just = new
        for k, src, d in events:
            row = self.just_src.setdefault(k, {})
            row[src] = row.get(src, 0.0) + d
        if chosen_cues:
            self.chosen_before = sorted(set(self.chosen_before) | set(chosen_cues))
        if cfg.record_ledger and (events or discount):
            ent: Dict[str, Any] = {"kind": "justification", "after": exp_id, "mode": cfg.justification_mode,
                                   "events": [{"cue": k, "source": src, "delta": float(d)} for k, src, d in events]}
            if discount:
                ent["discount"] = discount
            self.ledger.add(ent)

    @staticmethod
    def _apply_discount(person_teachers: List[np.ndarray], sit_teacher: np.ndarray, q: float) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
        """反証への抵抗（Z-persist。SPEC-v3.3 8.3）: 人モデルの教師の負の部分の割合 q = ρ_dis tanh(J) を「自分の選択のせいではない」として割り引き、
        状況モデル（場面のせい）の教師へ回す。割り引いた量 moved_j = q · min(教師_j, 0)（負）を返す。教師の合計（人 + 状況）は変わらない。価値ネットの出力は使わない。"""
        moved = [q * np.minimum(t, 0.0) for t in person_teachers]
        pt = [t - m for t, m in zip(person_teachers, moved)]
        st = sit_teacher
        for m in moved:
            st = st + m
        return pt, st, moved

    # ------------------------------------------------------------------ 選好の説明（Z8）
    def explain_preference(self, options: List[Dict[str, Any]], key: str = "preference") -> Dict[str, Any]:
        """予期 Q を「経験に根拠のある部分」と「正当化 J の部分」に分けて返す（学習しない。状態を変えない）。
        options は 1 つか 2 つの状況プローブ（context・concepts・participants・announced）。各選択肢について:
          Q = Q_grounded + J_Q、Q_grounded = Q_val + C_Q（価値・信念・条件づけ。経験に根拠のある部分）、J_Q = Σ J(手がかり)。
          手がかりごとの J と、その源ごとの和（spread・reaffirm・counter・overwhelm・bound・cap・decay。Σ 源 = J）。
        2 つなら、差 ΔQ = Δ根拠 + Δ正当化、選ぶ確率、どちらを好むか（Q・根拠のみ・正当化のみ）、正当化が選好を決めているか（Q と根拠のみで向きが違う）。
        MB の標本は key で決まる（二択の id を渡すと、その二択の選択と同じ Q）。"""
        if len(options) not in (1, 2):
            raise SchemaError("explain_preference の options は 1 つか 2 つの状況です")
        qs = [validate_probe(o) for o in options]
        for q in qs:
            if q["kind"] != "situation":
                raise SchemaError("explain_preference の options は状況（context を持つ）です")
        t_now = self._clock()
        out_opts = []
        for q in qs:
            ex = self._expectation(q["context"], q["concepts"], q["participants"], q["announced"], key, t_now)
            cues = self._j_cues(q["context"], q["concepts"], q["participants"])
            jd = {}
            for c in cues:
                v = 0.0 if self.cfg.no_justification else float(self.just.get(c, 0.0))
                if v != 0.0 or c in self.just_src:
                    jd[c] = {"J": v, "sources": {s_: float(d) for s_, d in sorted(self.just_src.get(c, {}).items()) if d != 0.0}}
            out_opts.append({"Q": float(ex["Q"]), "Q_grounded": float(ex["Q_val"] + ex["C_Q"]), "Q_val": float(ex["Q_val"]), "C_Q": float(ex["C_Q"]),
                             "J_Q": float(ex["J_Q"]), "justification": jd})
        out: Dict[str, Any] = {"mode": self.cfg.justification_mode, "options": out_opts}
        if len(out_opts) == 2:
            a_, b_ = out_opts
            dQ, dG, dJ = a_["Q"] - b_["Q"], a_["Q_grounded"] - b_["Q_grounded"], a_["J_Q"] - b_["J_Q"]
            z = self.cfg.beta_choice * np.array([a_["Q"], b_["Q"]])
            e_ = np.exp(z - z.max())
            pr = e_ / e_.sum()
            pick = lambda d: (0 if d > 0 else 1) if d != 0 else None
            out["delta"] = {"Q": float(dQ), "grounded": float(dG), "justification": float(dJ)}
            out["p_choose"] = [float(pr[0]), float(pr[1])]
            out["preferred"] = {"by_Q": pick(dQ), "by_grounded": pick(dG), "by_justification": pick(dJ)}
            out["justification_decides"] = bool(pick(dQ) != pick(dG))
        return _py(out)

    # ------------------------------------------------------------------ 二択（v3.3 第2部 3.2〜3.3）
    def _choose(self, ch: Dict[str, Any], commit: bool, forced: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """二択。各選択肢の Q（C と J を含む）から softmax(β_c Q) で選ぶ。選んだ側の結果だけを観測して学ぶ（通常の経験として）。
        葛藤の強さ c = exp(−(ΔQ/σ_c)²)（ΔQ = 選ぶ前の Q の差）。選んだ側の手がかり（概念・類・文脈）の J を +η_J c、選ばなかった側の J を −η_J c 動かす
        （両方に共通の手がかりは相殺）。orig: J の有界化と減衰は経験の最後（_j_step_orig）。persist: さらに reaffirm（再び選ぶ）・counter（反する選択）の出来事を足す。
        commit=False は選ばずに数値だけ返す（appraise）。forced は B1 の強制再生用（選んだ側を記録された決定に合わせる。乱数は同じだけ引く）。"""
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
        if forced is not None and forced.get("choice") is not None:
            chosen = int(forced["choice"])
        ch_c = [x for x in jcues[chosen] if x not in jcues[1 - chosen]]           # 選んだ側だけの手がかり（共通の手がかりは相殺するので動かさない）
        un_c = [x for x in jcues[1 - chosen] if x not in jcues[chosen]]
        step = cfg.eta_j * conflict
        dJ: Dict[str, float] = {}
        pre_events: List[Tuple[str, str, float]] = []
        if not cfg.no_justification:
            for cue in ch_c:
                dJ[cue] = step
            for cue in un_c:
                dJ[cue] = -step
            dJ = {k: v for k, v in dJ.items() if v != 0.0}
            if cfg.justification_mode != "orig":
                pre_events = [(k, "spread", v) for k, v in dJ.items()]
                for cue in ch_c:                                                   # reaffirm: 以前に選んだ手がかりを再び選ぶ（葛藤がなくても起きる）
                    if cue in self.chosen_before:
                        pre_events.append((cue, "reaffirm", cfg.eta_r * (1.0 - abs(self.just.get(cue, 0.0)) / cfg.j_num)))
                for cue in un_c:                                                   # counter: J > 0 の手がかりを持つ選択肢を選ばなかった
                    jv = self.just.get(cue, 0.0)
                    if jv > 0.0:
                        pre_events.append((cue, "counter", -cfg.eta_r * jv))
        if cfg.justification_mode == "orig":
            j_after, _ = self._j_step_orig(dJ)
        else:
            sim = dict(self.just)
            for k, _src, d in pre_events:
                sim[k] = sim.get(k, 0.0) + d
            j_after = {k: max(-cfg.j_num, min(cfg.j_num, v)) for k, v in sim.items()}      # overwhelm の前（出来事の全体は台帳の kind = "justification"）
        touched = sorted({k for k, _s, _d in pre_events} | set(dJ))
        info["chosen"] = chosen
        info["dJ"] = dict(dJ)
        info["pre_events"] = [{"cue": k, "source": s_, "delta": float(d)} for k, s_, d in pre_events]
        info["J_before"] = {k: float(self.just.get(k, 0.0)) for k in touched}
        info["J_after"] = {k: float(j_after.get(k, 0.0)) for k in touched}
        if cfg.record_ledger:
            self.ledger.add({"kind": "choice", "id": ch["id"], "chosen": chosen, "Q": [float(x) for x in Q], "p_choose": [float(p[0]), float(p[1])],
                             "dQ": dQ, "conflict": conflict, "dJ": {k: float(v) for k, v in dJ.items()}, "mode": cfg.justification_mode,
                             "pre_events": list(info["pre_events"]), "J_before": dict(info["J_before"]), "J_after": dict(info["J_after"]),
                             "options": [{"Q_val": float(x["Q_val"]), "C_Q": float(x["C_Q"]), "J_Q": float(x["J_Q"])} for x in exs]})
        o = opts[chosen]
        e = {"id": ch["id"], "context": o["context"], "concepts": list(o["concepts"]), "participants": list(o["participants"]),
             "outcomes": list(ch["outcomes_by_option"][str(chosen)]), "optional": False, "session": o["session"], "announced": list(o["announced"])}
        jctx = {"dJ": dJ, "pre_events": pre_events, "chosen_cues": ch_c}
        return self._experience(e, choice=info, jctx=jctx, forced=forced)

    # ------------------------------------------------------------------ 経験
    def _experience(self, e: Dict[str, Any], choice: Optional[Dict[str, Any]] = None, jctx: Optional[Dict[str, Any]] = None,
                    forced: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """検証済みの経験を評価して学習する（通常の経験と、二択で選んだ側）。choice は二択の情報（あれば自分で選んだ経験: chosen = 1）。
        jctx は二択の正当化の出来事（dJ・事前の出来事・選んだ側だけの手がかり）。forced は B1 の強制再生（離散的な決定を記録に合わせる）。"""
        cfg = self.cfg
        for p in e["participants"]:
            self.belief.check_class(p["id"], p["class"])
        ap, it = self._core(e)
        # 3. optional なら接近・回避を選ぶ
        ap["optional"] = e["optional"]
        if e["optional"]:
            p_app = self._p_approach(ap["Q"])
            approached = bool(self.rng.random() < p_app)
            if forced is not None and forced.get("approach") is not None:
                approached = bool(forced["approach"])                   # 強制再生: 乱数は同じだけ引いた上で、記録された決定に合わせる
            ap["p_approach"] = p_app
            ap["approached"] = approached
            self._dec["approach"] = approached
            if not approached:
                if cfg.record_ledger:
                    self.ledger.add({"kind": "avoid", "id": e["id"], "context": e["context"],
                                     "Q": float(ap["Q"]), "C_Q": float(ap["C_Q"]), "J_Q": float(ap["J_Q"]), "p_approach": p_app})
                self.n_avoided += 1
                self.cond.now = self._clock()
                self._j_commit(e["id"])                                 # 時間は経つ（orig と persist_with_decay は毎経験の減衰。persist は何も変えない）
                for key in ("f", "f_parts", "delta", "du", "du_total", "du_weighted", "dh", "contributions", "phi",
                            "teachers", "emotions", "interpretations"):
                    ap[key] = None
                ap["value_change"] = None
                return _py(ap)
        chosen = bool(choice is not None or e["optional"])              # 主体性: 自分で選んだ経験（optional を受けた、または二択で選んだ）
        if choice is not None:
            self._dec["choice"] = int(choice["chosen"])
        # 反証への抵抗（Z-persist。SPEC-v3.3 8.3）: J > 0 の選択肢を選んだ経験で、人モデルの負の教師の ρ_dis tanh(J) を場面へ回す。教師は結果と J だけから決まる（価値ネットの出力を使わない）
        disc: Optional[Dict[str, Any]] = None
        if choice is not None and jctx is not None and cfg.justification_mode != "orig" and not cfg.no_justification:
            jt = float(sum(self.just.get(x, 0.0) for x in jctx["chosen_cues"]))
            if jt > 0.0:
                q_dis = cfg.rho_dis * float(np.tanh(jt))
                it["person_teachers"], it["sit_teacher"], moved = self._apply_discount(it["person_teachers"], it["sit_teacher"], q_dis)
                ap["teachers"] = {"person": {p["id"]: _need_dict(it["person_teachers"][j]) for j, p in enumerate(e["participants"])},
                                  "situation": _need_dict(it["sit_teacher"])}
                disc = {"q": q_dis, "J_chosen": jt, "moved": {p["id"]: [float(x) for x in m] for p, m in zip(e["participants"], moved)}}
                ap["discount"] = disc
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
        # 4b. v3.4 M2: 訂正の後に初めて、その対象が曖昧でない結果を伴って現れたら、情報源の信用を検証する（信念を更新する前の予測で。検証待ちがなければ何もしない）
        if self.ver_pending:
            self._verify_pending(e, it)
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
                dC, cdet = self.cond.update(self.pending["cues"], td_y, self.pending["context"], self.pending["t"],
                                            force=None if forced is None else forced.get("assign"))
                self._dec["assign"] = {c_: {"created": d_["created"] is not None, "map": int(d_["map"])} for c_, d_ in cdet.items() if "created" in d_}
                if cfg.record_ledger and self.cond.K > 1:                  # 割当の記録（どちらの設計でも）: 手がかりの経験ごとに、事前の内訳・事後・MAP の原因・学習前後の C_k・各原因の ΔC
                    for cue_name, d in cdet.items():
                        self.ledger.add({"kind": "assignment", "after": e["id"], "cue_event": self.pending.get("id"), "cue": cue_name,
                                         "t": float(self.pending["t"]), "context": self.pending["context"], "td_target": float(td_y),
                                         "y_x": d["y_x"], "post": d["post"], "post_new": d["post_new"], "map": d["map"],
                                         "created": d["created"], "reused": d["reused"], **d["trace"]})
            # 復位: 手がかりなしで嫌悪の結果が起きたとき、負の価値を持つ原因の最後に使われた時刻を更新する（この事象と直前の事象の手がかりは除く）
            called = (r <= -cfg.lc_aversive_threshold) if (forced is None or forced.get("reinstate_called") is None) else bool(forced["reinstate_called"])
            self._dec["reinstate_called"] = bool(called)
            if called:
                skip = list(it["cues"]) + (list(self.pending["cues"]) if self.pending is not None else [])
                reinst = self.cond.reinstate(skip, t_now, e["context"], r, force=None if (forced is None or forced.get("reinstate_called") is None) else forced.get("reinstate"))
                self._dec["reinstate"] = [[c_, int(k_)] for c_, k_ in reinst]
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
                "kind": "experience", "id": e["id"], "context": e["context"], "chosen": bool(chosen),
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
        self._log_experience(e, it, serial, t_now, h_before, a, "choice" if choice is not None else "experience", 0.0 if disc is None else disc["q"])
        self.n_experiences += 1
        self.cond.now = self._clock()
        # 正当化 J の確定（二択の最後）。orig: 選んだ側 +、選ばなかった側 −、有界化、毎経験の減衰。persist: 記録された出来事（spread・reaffirm・counter・overwhelm）だけ
        events: List[Tuple[str, str, float]] = list(jctx["pre_events"]) if jctx else []
        if choice is not None and jctx is not None and cfg.justification_mode != "orig" and not cfg.no_justification:
            pos = {x: self.just.get(x, 0.0) for x in jctx["chosen_cues"] if self.just.get(x, 0.0) > 0.0}
            jp = float(sum(pos.values()))
            if jp > 0.0:                                                 # overwhelm: 負の予測誤差 ε = max(0, −δ) が、正当化で割り引ける量 tanh(J) を超えた分に比例して J を減らす
                excess = max(max(0.0, -float(ap["delta"])) - float(np.tanh(jp)), 0.0)
                if excess > 0.0:
                    frac = min(cfg.eta_ov * excess, 1.0)                  # 超えた分に比例して、正の J の手がかりを割合 frac だけ減らす（乗算型。excess ≥ 1/η_ov ですべて消える）
                    events += [(x, "overwhelm", -frac * v) for x, v in pos.items()]
        self._j_commit(e["id"], dJ=(jctx or {}).get("dJ"), events_in=events, discount=disc,
                       chosen_cues=jctx["chosen_cues"] if (choice is not None and jctx is not None) else None)
        return _py(ap)

    # ------------------------------------------------------------------ 経験記録（訂正が参照する。台帳とは別）
    def _log_experience(self, e: Dict[str, Any], it: Dict[str, Any], serial: int, t_now: float, h_before: np.ndarray,
                        a_before: np.ndarray, how: str, discount_q: float = 0.0) -> None:
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
            "teacher_by_participant": [[float(x) for x in v] for v in it["person_teachers"]], "discount_q": float(discount_q), "corrections": []})
        self._exp_pos[e["id"]] = len(self.exp_log) - 1

    def _replay_values(self) -> V.ValueNet:
        """価値ネットを、経験記録の現在の事実で最初から再生する（retroactive_value）。欲求の状態 h は H0 から、各経験の結果で進め直し、
        観測雑音の a と責任 φ_o は当時のもの（訂正された経験の φ_o は、訂正のときに求めたもの）。条件づけ C・関心の重み a・信念は触れない。"""
        cfg = self.cfg
        nv = V.ValueNet(cfg, self.emb)
        h = np.full(N_NEEDS, innate.H0)
        for rec in self.exp_log:
            if rec.get("how") == "correction":
                continue                                          # v3.4 M3: 訂正の経験の記録は価値の教師を持たない（kind = correction_experience）
            P = rec["participants"]
            n_o = len(rec["outcomes"])
            W_out = np.array(rec["w_out"], dtype=np.float64).reshape(n_o, N_TYPES)
            rows = innate.mixed_rows(W_out, [o["magnitude"] for o in rec["outcomes"]], self.M)
            dh, du, contrib = innate.contributions(h, rows)
            actor_j = np.array(rec["actor_j"], dtype=int)
            pt, st = self._split_teachers(contrib, np.array(rec["phi_o"], dtype=np.float64), (actor_j >= 0).astype(np.float64), actor_j, len(P))
            if rec.get("discount_q", 0.0) > 0.0:
                pt, st, _ = self._apply_discount(pt, st, rec["discount_q"])           # Z-persist: 当時の割り引きを同じ割合で再現する
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
        アブレーション: retroactive_value（価値を訂正後の事実で最初から再生。人モデルも状況モデルも作り直すので、台帳の ΔW は両方の差）、no_reappraisal（価値を変えない）。
        v3.4（reappraisal_mode = "v34" | "v34_or"）: 手順 3 を M1 に置き換える（_m1_absorb。含意された変化 D_i を g_i で重み付けて、事後共分散に沿って W に吸収する）。
        手順 1・2・4 は同じ。検証待ちの登録は M2（_verify_pending）。
        v3.4 M3（Stage 3。absorb_on のときだけ）: 訂正の主観的評価（f_c・安堵・own_i・罪悪感・後悔・自己正当化。_m3_evaluate）→ 双方向で有界の τ の更新 →
        訂正を時間の出来事にする（時計 +1、気分 m ← 0.8 m + 0.2 δ_c）。f_c と安堵は気分にだけ入る。台帳 kind = "correction_experience" と exp_log に追記する。"""
        cfg = self.cfg
        absorb_on = cfg.reappraisal_mode != "v33" and not cfg.no_reappraisal and not cfg.retroactive_value
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
        if absorb_on:
            # M1 手順 1（前半）: 訂正 i の前の事実で価値ネットを再生し、参加者と状況の予測を取る（V⁽ⁱ⁻¹⁾）。当時の事実（w_out・outcomes）も取っておく
            nv_prev = self._replay_values()
            prev_person = [nv_prev.predict_person(p) for p in P]
            prev_sit = nv_prev.predict_situation(rec["context"], rec["concepts"])
            prev_outcomes = copy.deepcopy(rec["outcomes"])
            prev_w_out = np.array(rec["w_out"], dtype=np.float64).reshape(len(prev_outcomes), N_TYPES)
            prev_actor = np.array(rec["actor_j"], dtype=int)
        # 読み出し用: 訂正前の予期と、参加者への好悪
        ex0 = self._expectation(rec["context"], rec["concepts"], P, rec["announced"], key, t_now)
        like0 = [float(self.value.like_entity(p["id"], p["class"], a_now)[0]) for p in P]
        p0 = np.array(ex0["pred"].p_person)
        # 1. 信念
        old = self.belief.withdraw_experience(rec["serial"])
        pred = self.belief.predict(rec["context"], P)
        po = self._parse_outcomes(P, c["corrected_outcomes"], pred, ctx=rec["context"])
        if absorb_on:
            llr = self._fact_llr(pred, nP, prev_w_out, prev_actor, po["W_out"], po["actor_j"])      # M1 手順 2: 取り消した後の信念 B₋ⱼ での、訂正後の事実と当時の事実の対数尤度の差
        self.belief.install_experience(old, po["y_person"], po["y_mask"], po["phi"], po["y_sw"], po["sw_mask"])
        if cfg.recall_after_correction:
            self.belief.reattribute()                      # 実験設定（既定 OFF）: 訂正の後に想起の掃引を 1 回
        # 3. 価値: 訂正後の寄与（当時の欲求の状態 h_before で）。Δ_e は、人モデルの e が学んだ量（寄与を責任 φ で分けた教師）の差:
        #    当時の教師 Σ_o φ_o contrib_o（学習時に人モデルに与えたもの）と、訂正後の教師（訂正後の結果と、取り消した後の信念で求めた φ）。
        #    重みなしの寄与どうしの差だと、人モデルが φ の分しか学んでいないのに全部を引いてしまい、過剰に訂正する（CHANGELOG の D 節）。
        rows = innate.mixed_rows(po["W_out"], [o["magnitude"] for o in c["corrected_outcomes"]], self.M)
        dh, du, contrib = innate.contributions(np.array(rec["h_before"], dtype=np.float64), rows)
        new_teach, new_sit = self._split_teachers(contrib, po["phi_o"], po["is_person"], po["actor_j"], nP)
        if rec.get("discount_q", 0.0) > 0.0:
            new_teach, new_sit, _ = self._apply_discount(new_teach, new_sit, rec["discount_q"])      # 当時の抵抗（Z-persist）と同じ割合で割り引く
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
        if absorb_on:
            mode = "absorb_or" if cfg.reappraisal_mode == "v34_or" else "absorb"
        dW_person: Dict[str, np.ndarray] = {}
        dW_sit: Dict[str, np.ndarray] = {}
        m1: Optional[Dict[str, Any]] = None
        if absorb_on:
            dW_person, dW_sit, m1 = self._m1_absorb(c, rec, P, prev_person, prev_sit, llr, prev_outcomes)
        elif mode == "kalman":
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
        m3: Optional[Dict[str, Any]] = None
        if absorb_on:
            m3 = self._m3_evaluate(c, rec, P, prev_outcomes, prev_w_out, prev_actor, po, m1, like0, like1, ex0, ex1, f, relief)
            guilt = m3["guilt"]
        types_changed = sorted({o["type"] for o in rec["original_outcomes"] if "type" in o} | {o["type"] for o in c["corrected_outcomes"] if "type" in o})
        belief_change = {p["id"]: {t: [float(p0[j, TYPE_INDEX[t]]), float(p1[j, TYPE_INDEX[t]])] for t in types_changed} for j, p in enumerate(P)}
        ap = {"kind": "correction", "id": c["id"], "refers_to": c["refers_to"], "source": c["source"], "reappraisal": mode,
              "original_outcomes": copy.deepcopy(rec["original_outcomes"]), "corrected_outcomes": copy.deepcopy(c["corrected_outcomes"]),
              "shifts": {p["id"]: _need_dict(shifts[j]) for j, p in enumerate(P)},
              "like": {p["id"]: [like0[j], like1[j]] for j, p in enumerate(P)},
              "Q": [float(ex0["Q"]), float(ex1["Q"])], "f": f, "emotions": {"relief": relief, "guilt": guilt},
              "belief_change": belief_change}
        if m1 is not None:
            ap["m1"] = m1
        if m3 is not None:
            ap["emotions"] = {"relief": relief, "guilt": m3["guilt"], "regret": m3["regret"], "defend": m3["defend"]}
            ap["m3"] = m3
        if cfg.record_ledger:
            self.ledger.add({**{k: v for k, v in ap.items() if k != "kind"}, "kind": "correction", "t": float(t_now),
                             "dW": {"person": {n: [float(x) for x in r_] for n, r_ in dW_person.items()},
                                   "situation": {n: [float(x) for x in r_] for n, r_ in dW_sit.items()}}})
        self.n_corrections += 1
        return _py(ap)

    # ------------------------------------------------------------------ v3.4 M3: 訂正の主観的評価と τ（SPEC-v3.4 4.4）
    @staticmethod
    def _same_outcome(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def _own_units(self, P: List[Dict[str, str]], prev_outcomes: List[Dict[str, Any]], prev_w_out: np.ndarray, corrected: List[Dict[str, Any]],
                   W_new: np.ndarray) -> List[Dict[str, Any]]:
        """訂正で変わった結果の単位。取り除かれた結果（当時の版にあり、訂正後の版に同じものがない）を、同じ行為者の追加された結果と対にする
        （対がなければ単独）。追加だけの結果も 1 単位。misread = 当時の結果が曖昧で（型を人格自身の解釈が決めていた）、解釈の最大の型が対の訂正後の型と違う。
        報告の誤り（曖昧でない誤った結果）、対のない結果、追加だけの結果は misread にならない（own への寄与 0）。"""
        removed = [i for i, o in enumerate(prev_outcomes) if not any(self._same_outcome(o, q) for q in corrected)]
        added = [k for k, q in enumerate(corrected) if not any(self._same_outcome(p, q) for p in prev_outcomes)]
        used: set = set()
        units: List[Dict[str, Any]] = []
        pcls = {p["id"]: p["class"] for p in P}
        for i in removed:
            o = prev_outcomes[i]
            k = next((kk for kk in added if kk not in used and corrected[kk]["actor"] == o["actor"]), None)
            if k is not None:
                used.add(k)
            amb = "type" not in o
            prev_t = OUTCOME_TYPES[int(np.argmax(prev_w_out[i]))]
            new_t = OUTCOME_TYPES[int(np.argmax(W_new[k]))] if k is not None else None
            mis = bool(amb and new_t is not None and prev_t != new_t)
            units.append({"prev_index": int(i), "added_index": None if k is None else int(k), "actor": o["actor"], "ambiguous": amb,
                          "prev_type": prev_t, "new_type": new_t, "misread": mis, "class": pcls.get(o["actor"], o["actor"]),
                          "interpretations": dict(o["interpretations"]) if amb else None})
        for k in added:
            if k not in used:
                units.append({"prev_index": None, "added_index": int(k), "actor": corrected[k]["actor"], "ambiguous": False, "prev_type": None,
                              "new_type": OUTCOME_TYPES[int(np.argmax(W_new[k]))], "misread": False,
                              "class": pcls.get(corrected[k]["actor"], corrected[k]["actor"]), "interpretations": None})
        return units

    @staticmethod
    def _tau_step(defend: float, guilt: float, regret: float, eta: float) -> float:
        """1 回の Δ log τ = η_τ · clip(defend − guilt − regret, −1, 1)。SPEC の式（clip なし）は guilt + regret > 1 のとき |Δ| > η_τ になり、
        同じ節の「1 回の変化は |Δ log τ| ≤ η_τ」と食い違う。後者（不変量 H9）を守るため符号つきの和を [−1, 1] に切る（CHANGELOG-v3.4 S3-3）。"""
        return float(eta * max(-1.0, min(1.0, defend - guilt - regret)))

    def set_tau_yoke(self, trajectory: Optional[List[Dict[str, float]]]) -> None:
        """【試験用の入口】τ-yoke: 訂正の通し番号 i の評価の後の log τ の全体として、与えた軌跡の i 番目をそのまま使う（自分の評価では動かさない）。
        軌跡は、別の人格の台帳から取る（npnv3.trace.tau_trajectory）。cfg.m3_variant = "tau_yoke" のときだけ使われる。"""
        self.tau_yoke = None if trajectory is None else [{str(k): float(v) for k, v in d.items()} for d in trajectory]

    def _tick(self) -> None:
        """出来事の時計を 1 進める（条件づけの時刻 now、J の減衰）。信念・価値・a・C・h・気分は変えない。台帳の tick は advance_clock が足す。"""
        self.n_ticks += 1
        self.cond.now = self._clock()
        self._j_commit("tick:" + str(self.n_ticks))

    def _m3_evaluate(self, c: Dict[str, Any], rec: Dict[str, Any], P: List[Dict[str, str]], prev_outcomes: List[Dict[str, Any]], prev_w_out: np.ndarray,
                     prev_actor: np.ndarray, po: Dict[str, Any], m1: Dict[str, Any], like0: List[float], like1: List[float], ex0: Dict[str, Any],
                     ex1: Dict[str, Any], f: float, relief: float) -> Dict[str, Any]:
        """M3 の本体（吸収の後に呼ぶ）。own_i・罪悪感・後悔・自己正当化を求め、τ を更新し、訂正を時間の出来事にして気分を動かし、
        kind = "correction_experience" を exp_log と台帳に追記する。f_c と安堵は気分にだけ入る（学習の目標に入る経路を作らない）。"""
        cfg = self.cfg
        variant = cfg.m3_variant
        ctx = rec["context"]
        units = self._own_units(P, prev_outcomes, prev_w_out, c["corrected_outcomes"], po["W_out"])
        own = float(sum(1 for u in units if u["misread"])) / len(units) if units else 0.0
        s_f = cfg.s_f
        guilt = own * float(np.tanh(sum(max(l1 - l0, 0.0) for l0, l1 in zip(like0, like1) if l0 < 0.0) / s_f))
        regret = own * float(np.tanh(sum(max(l0 - l1, 0.0) for l0, l1 in zip(like0, like1) if l0 > 0.0) / s_f))
        a_now = self.concern.a
        pids = [p["id"] for p in P]
        dmag = float(sum(float(a_now @ np.abs(np.array([m1["D"][pid][nm] for nm in NEEDS], dtype=np.float64))) for pid in pids))
        g_i = float(m1["g"])                                        # 基底の g_i = σ(logit κ̄ + LLR)。C-OR の (1 + λ_g G_e) は掛けない（CHANGELOG-v3.4 S3-4）
        defend = own * (1.0 - g_i) * float(np.tanh(dmag / s_f))
        # --- τ ---
        mis_keys: List[str] = []
        amb_keys: List[str] = []
        for u in units:
            k = self.tau_key(u["class"], ctx)
            if u["misread"] and k not in mis_keys:
                mis_keys.append(k)
            if u["ambiguous"] and k not in amb_keys:
                amb_keys.append(k)
        raw = defend - guilt - regret
        step = self._tau_step(defend, guilt, regret, cfg.eta_tau)
        lt_before = dict(self.log_tau)
        lim = math.log(cfg.tau_max)
        tau_source = {"clamp_tau": "clamp", "clamp_both": "clamp", "tau_off": "off", "tau_yoke": "yoke"}.get(variant, "own")
        ordinal = int(self.n_corrections)
        if tau_source == "own":
            if step != 0.0:
                for k in mis_keys:
                    self.log_tau[k] = max(-lim, min(lim, self.log_tau.get(k, 0.0) + step))
        elif tau_source == "yoke":
            if self.tau_yoke is not None and ordinal < len(self.tau_yoke):
                self.log_tau = dict(self.tau_yoke[ordinal])
            else:
                tau_source = "yoke_missing"
        changed = sorted(k for k in set(lt_before) | set(self.log_tau) if lt_before.get(k, 0.0) != self.log_tau.get(k, 0.0))
        eid = "corr:" + str(c["id"])
        # τ が効いた解釈の重みの変化（誤読された結果の解釈の重みを、同じ信念で τ の前後に求める）
        tau_effect: Dict[str, Any] = {}
        if changed:
            predb = self.belief.predict(ctx, P)
            for u in units:
                if not u["ambiguous"]:
                    continue
                k = self.tau_key(u["class"], ctx)
                if k in changed and k not in tau_effect:
                    j = next((jj for jj, p in enumerate(P) if p["id"] == u["actor"]), -1)
                    pv = predb.p_person[j] if j >= 0 else predb.p_sw
                    wb = self._interp_weights(u["interpretations"], pv, math.exp(lt_before.get(k, 0.0)))
                    wa = self._interp_weights(u["interpretations"], pv, math.exp(self.log_tau.get(k, 0.0)))
                    ts = sorted(u["interpretations"])
                    tau_effect[k] = {"types": ts, "w_before": [float(wb[TYPE_INDEX[t]]) for t in ts], "w_after": [float(wa[TYPE_INDEX[t]]) for t in ts]}
        # --- 因果の連鎖（既にある経路だけ: κ → g、τ → 解釈の重み → LLR）---
        skey = m1["source"]
        refs: List[List[str]] = []
        if ("kappa:" + skey) in self.last_ev:
            refs.append([self.last_ev["kappa:" + skey], "kappa->g"])
        for k in amb_keys:
            if ("tau:" + k) in self.last_ev:
                refs.append([self.last_ev["tau:" + k], "tau->interpretation->llr"])
        for k in changed:
            self.last_ev["tau:" + k] = eid
        # --- 時間の出来事と気分（f_c・安堵は気分にだけ入る）---
        delta_c = float(V.surprise(f, ex0["Q"], cfg))
        mood_before = float(self.mood)
        clock_on = variant != "clock_off"
        mood_on = variant not in ("clamp_mood", "clamp_both", "clock_off")
        t_before = self._clock()
        if clock_on:
            self._tick()
        if mood_on:
            self.mood = V.next_mood(self.mood, delta_c, cfg)
        inc = float(guilt) + float(regret)
        if inc != 0.0:
            for pid in pids:
                self.G[pid] = self.G.get(pid, 0.0) + inc
        out = {"eid": eid, "own": own, "units": units, "guilt": float(guilt), "regret": float(regret), "defend": float(defend),
               "f_c": float(f), "relief": float(relief), "delta_c": delta_c, "g": g_i, "llr": float(m1["llr"]),
               "kappa_bar": [float(m1["kappa_bar"]), float(m1["kappa_bar"])],
               "tau_raw": float(raw), "tau_step": float(step), "tau_source": tau_source, "tau_clamped": tau_source == "clamp",
               "log_tau": {k: [float(lt_before.get(k, 0.0)), float(self.log_tau.get(k, 0.0))] for k in changed},
               "tau_effect": tau_effect, "mood": [mood_before, float(self.mood)], "mood_on": mood_on, "clock_on": clock_on,
               "t": [float(t_before), float(self._clock())], "G_added": inc, "refs": refs, "variant": variant}
        out = _py(out)
        if cfg.record_ledger:
            self.ledger.add(_py({"kind": "correction_experience", "id": c["id"], "refers_to": c["refers_to"], "source": skey, **out,
                                 "like": {p["id"]: [like0[j], like1[j]] for j, p in enumerate(P)}, "Q": [float(ex0["Q"]), float(ex1["Q"])]}))
        self.exp_log.append({"id": c["id"], "how": "correction", "serial": None, "original_outcomes": [], "refers_to": c["refers_to"], "source": skey, "t": float(t_before),
                             "own": own, "guilt": float(guilt), "regret": float(regret), "defend": float(defend), "f_c": float(f), "g": g_i,
                             "log_tau_after": {k: float(v) for k, v in self.log_tau.items()}})
        return out

    # ------------------------------------------------------------------ v3.4 M1: 含意された変化の吸収（SPEC-v3.4 4.2）
    @staticmethod
    def _src_key(source: Optional[str]) -> str:
        return "(none)" if source is None else str(source)

    def _kappa_ab(self, key: str) -> Tuple[float, float]:
        ab = self.src_beta.get(key)
        return (float(self.cfg.kappa_alpha0), float(self.cfg.kappa_beta0)) if ab is None else (float(ab[0]), float(ab[1]))

    def kappa_bar(self, source: Optional[str]) -> float:
        """情報源の信用 κ̄_s = α_s / (α_s + β_s)（SPEC-v3.4 4.2 手順 3）。訂正の source を指定する。"""
        a, b = self._kappa_ab(self._src_key(source))
        return a / (a + b)

    def _sigma(self, z: float) -> float:
        zc = max(-self.cfg.g_logit_clip, min(self.cfg.g_logit_clip, float(z)))
        return 1.0 / (1.0 + math.exp(-zc))

    @staticmethod
    def _fact_indicators(nP: int, W_out: np.ndarray, actor_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """事実の版の指示変数（参加者 × 型、自分・世界 × 型）。明確な結果は 0/1、曖昧な結果は解釈の重み w_out（柔らかい指示変数）。同じ主体・型の複数の結果は最大をとる。"""
        y_p = np.zeros((nP, N_TYPES))
        y_s = np.zeros(N_TYPES)
        for i in range(len(actor_j)):
            j = int(actor_j[i])
            if j >= 0:
                y_p[j] = np.maximum(y_p[j], W_out[i])
            else:
                y_s = np.maximum(y_s, W_out[i])
        return y_p, y_s

    def _fact_llr(self, pred: Any, nP: int, W_old: np.ndarray, actor_old: np.ndarray, W_new: np.ndarray, actor_new: np.ndarray) -> float:
        """LLR_i = log p(y_訂正後 | B₋ⱼ) − log p(y_当時 | B₋ⱼ)（SPEC-v3.4 4.2 手順 2）。pred は経験 j を取り消した後の信念の予測。
        どちらかの版に現れた結果型 k ごとに、参加者 j（p_person[j, k]）と自分・世界（p_sw[k]）の発生のベルヌーイ。価値ネットの出力は使わない。"""
        yp_o, ys_o = self._fact_indicators(nP, W_old, actor_old)
        yp_n, ys_n = self._fact_indicators(nP, W_new, actor_new)
        T = (ys_o > 0.0) | (ys_n > 0.0)
        if nP:
            T = T | (yp_o.max(axis=0) > 0.0) | (yp_n.max(axis=0) > 0.0)
        p_p = np.asarray(pred.p_person, dtype=np.float64).reshape(nP, N_TYPES)
        p_s = np.asarray(pred.p_sw, dtype=np.float64)

        def ll(yp: np.ndarray, ys: np.ndarray) -> float:
            tot = float((ys * np.log(p_s) + (1.0 - ys) * np.log(1.0 - p_s))[T].sum())
            if nP:
                tot += float((yp * np.log(p_p) + (1.0 - yp) * np.log(1.0 - p_p))[:, T].sum())
            return tot
        return ll(yp_n, ys_n) - ll(yp_o, ys_o)

    def _seed_G(self, pid: str, value: float) -> None:
        """【試験用の入口】対象 pid の C-OR の累積 G_e を設定する（Stage 2 は M3 がないので G_e が自然には動かない。g > 1 の経路と RA5 の検査を単体で動かすため）。"""
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError("G_e は有限で 0 以上です")
        self.G[str(pid)] = float(value)

    def _m1_absorb(self, c: Dict[str, Any], rec: Dict[str, Any], P: List[Dict[str, str]], prev_person: List[np.ndarray], prev_sit: np.ndarray,
                   llr: float, prev_outcomes: List[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
        """M1（SPEC-v3.4 4.2 手順 1・3〜6・8）。rec は現在の事実に直した後の経験記録。prev_person・prev_sit は訂正の前の事実での再生（V⁽ⁱ⁻¹⁾）の予測。
        D_i = pred_{V⁽ⁱ⁾} − pred_{V⁽ⁱ⁻¹⁾}（参加者ごと・状況）。g_i = σ(logit κ̄_s + LLR_i)（C-OR は人の対象ごとに (1 + λ_g G_e) を掛ける。上限なし）。
        Δ = g_i D_i を W に吸収し（Σ は変えない）、残り (1 − g_i) D_i を台帳に記録する。条件づけ C と関心の重み a は変えない。M2 の検証待ちを登録する。"""
        cfg = self.cfg
        nv = self._replay_values()
        D_person = [nv.predict_person(p) - prev_person[j] for j, p in enumerate(P)]
        D_sit = nv.predict_situation(rec["context"], rec["concepts"]) - prev_sit
        skey = self._src_key(c.get("source"))
        al, be = (1.0, 1.0) if cfg.kappa_fixed else self._kappa_ab(skey)       # A-κ-fixed（Stage 4）: κ̄ ≡ 0.5
        kbar = al / (al + be)
        z = math.log(al / be) + float(llr)
        g = self._sigma(z) if cfg.g_const is None else float(cfg.g_const)         # A-g-const（Stage 4）: g_i ≡ g_const。z と LLR は記録する
        or_mode = cfg.reappraisal_mode == "v34_or"
        pids = [p["id"] for p in P]
        G_before = {pid: float(self.G.get(pid, 0.0)) for pid in pids}
        g_e = [g * (1.0 + cfg.lambda_g * G_before[pid]) if or_mode else g for pid in pids]       # C-OR: 上限なし。状況の対象は人ではないので g のまま
        delta_p = [g_e[j] * D_person[j] for j in range(len(P))]
        delta_s = g * D_sit
        before_p = [self.value.predict_person(p) for p in P]
        before_s = self.value.predict_situation(rec["context"], rec["concepts"])
        res = self.value.absorb(P, rec["context"], rec["concepts"], delta_p, delta_s)
        after_p = [self.value.predict_person(p) for p in P]
        after_s = self.value.predict_situation(rec["context"], rec["concepts"])
        # G_e の累積（受け入れた訂正の罪悪感と後悔）は、like_後が要るので _m3_evaluate が吸収の後に足す（Stage 3）
        if not cfg.kappa_fixed:                                                  # A-κ-fixed: 検証なし
            self.ver_pending.append({"correction": c["id"], "source": skey, "refers_to": c["refers_to"], "serial": int(rec["serial"]),
                                     "targets": list(pids), "prev_outcomes": prev_outcomes,
                                     "correction_eid": "corr:" + str(c["id"]), "log_tau": dict(self.log_tau),           # 当時の読み（この訂正の τ の更新の前）
                                     "tau_refs": {k: self.last_ev["tau:" + k] for k in sorted(self.log_tau) if ("tau:" + k) in self.last_ev}})
        m1 = _py({
            "source": skey, "kappa_bar": kbar, "kappa_alpha_beta": [al, be], "llr": float(llr), "z": z, "z_clipped": abs(z) > cfg.g_logit_clip,
            "g": g, "g_e": {pid: g_e[j] for j, pid in enumerate(pids)}, "G_e": G_before, "or": or_mode,
            "D": {pid: _need_dict(D_person[j]) for j, pid in enumerate(pids)}, "D_sit": _need_dict(D_sit),
            "delta": {pid: _need_dict(delta_p[j]) for j, pid in enumerate(pids)}, "delta_sit": _need_dict(delta_s),
            "remainder": {pid: _need_dict(D_person[j] - delta_p[j]) for j, pid in enumerate(pids)}, "remainder_sit": _need_dict(D_sit - delta_s),
            "pred_change": {pid: _need_dict(after_p[j] - before_p[j]) for j, pid in enumerate(pids)}, "pred_change_sit": _need_dict(after_s - before_s),
        })
        return res["dW_person"], res["dW_situation"], m1

    # ------------------------------------------------------------------ v3.4 M2: 情報源の信用の形成（SPEC-v3.4 4.3）
    def _reverted_belief(self, pend: Dict[str, Any]) -> BeliefNet:
        """訂正 i の観測を当時の事実に戻した信念。信念の写しの上で作り、本体は変えない（埋め込みの表は共有する。ここでは読むだけ）。"""
        rec = self.exp_log[self._exp_pos[pend["refers_to"]]]
        P = rec["participants"]
        bel = copy.deepcopy(self.belief, {id(self.emb): self.emb})
        old = bel.withdraw_experience(rec["serial"])
        pred = bel.predict(rec["context"], P)
        po = self._parse_outcomes(P, pend["prev_outcomes"], pred, belief=bel, ctx=rec["context"], log_tau=pend.get("log_tau", {}))
        bel.install_experience(old, po["y_person"], po["y_mask"], po["phi"], po["y_sw"], po["sw_mask"])
        return bel

    @staticmethod
    def _row_ll(p: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
        return float((m * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).sum())

    def _verify_pending(self, e: Dict[str, Any], it: Dict[str, Any]) -> None:
        """検証待ちの訂正ごとに、その対象 e が曖昧でない結果（行為者が e）を伴って学習した経験に初めて現れたとき、1 回だけ情報源の信用を更新する。
        LLR_ver = log p(y_新 | 現在の信念) − log p(y_新 | 訂正 i を当時の事実に戻した信念)。y_新 = この経験での e の観測（マスク済み）。現在の信念 = この経験が更新する前の信念。"""
        P = e["participants"]
        amb, actor_j = it["ambiguous"], it["actor_j"]
        direct = {int(actor_j[i]) for i in range(len(amb)) if not amb[i] and int(actor_j[i]) >= 0}
        keep: List[Dict[str, Any]] = []
        for pend in self.ver_pending:
            tj = next((jj for jj, p in enumerate(P) if jj in direct and p["id"] in pend["targets"]), None)
            if tj is None:
                keep.append(pend)
                continue
            rev = self._reverted_belief(pend)
            p_rev = np.asarray(rev.predict(e["context"], P).p_person, dtype=np.float64)[tj]
            p_cur = np.asarray(it["pred"].p_person, dtype=np.float64)[tj]
            y, m = it["y_person"][tj], it["y_mask"][tj]
            llr_ver = self._row_ll(p_cur, y, m) - self._row_ll(p_rev, y, m)
            refs = [[pend["correction_eid"], "verifies"]] + [[v, "tau->interpretation->llr_ver"] for v in pend.get("tau_refs", {}).values()]
            self._verify_source(pend["source"], llr_ver, {"correction": pend["correction"], "experience": e["id"], "target": P[tj]["id"], "synthetic": False,
                                                           "eid": "ver:" + str(pend["correction"]), "refs": refs, "tau_snapshot": dict(pend.get("log_tau", {}))})
        self.ver_pending = keep

    def _verify_source(self, skey: str, llr_ver: float, info: Dict[str, Any]) -> Dict[str, Any]:
        """情報源 s の Beta を α_s += σ(LLR_ver)、β_s += 1 − σ(LLR_ver) で更新する。κ̄_s が変わるのはこの出来事だけ（H3）。台帳に kind = "verification" を残す。"""
        s_ = self._sigma(llr_ver)
        al, be = self._kappa_ab(skey)
        al2, be2 = al + s_, be + (1.0 - s_)
        self.src_beta[skey] = [al2, be2]
        self.n_verifications += 1
        info = dict(info)
        eid = info.pop("eid", None) or f"ver:syn{self.n_verifications}"
        refs = list(info.pop("refs", []))
        if ("kappa:" + skey) in self.last_ev:
            refs.append([self.last_ev["kappa:" + skey], "kappa(prev)"])
        self.last_ev["kappa:" + skey] = eid
        ent = {"kind": "verification", "eid": eid, "refs": refs, "source": skey, "llr_ver": float(llr_ver), "d_alpha": float(s_), "d_beta": float(1.0 - s_),
               "alpha": [al, al2], "beta": [be, be2], "kappa": [al / (al + be), al2 / (al2 + be2)], **info}
        if self.cfg.record_ledger:
            self.ledger.add(ent)
        return ent

    def verify_source(self, source: Optional[str], llr_ver: float) -> Dict[str, Any]:
        """【試験用の入口】与えた LLR_ver で、情報源の信用を 1 回更新する（RA8(d): 反対向きの検証で κ̄ が動くかの確認。H3 の対照）。台帳には synthetic = True で残る。"""
        return self._verify_source(self._src_key(source), float(llr_ver), {"synthetic": True})

    def advance_clock(self, k: int = 1) -> None:
        """【試験用の入口】空の刻み: 出来事の時計を k 進めるだけ（SPEC-v3.4 第 5.2 節「訂正なしの枝」）。信念・価値・気分・a・C は変えない。
        J は、その刻みで orig の毎経験の減衰を受ける（persist は変わらない）。条件づけの時刻 now を更新する。Stage 3 が「訂正を時計の出来事にする」ときの時計と同じ量。"""
        for _ in range(int(k)):
            self._tick()
            if self.cfg.record_ledger:
                self.ledger.add({"kind": "tick", "t": self._clock()})

    def trace_loop(self) -> Dict[str, Any]:
        """κ と τ の変化の連鎖を台帳だけから再構成する（npnv3.trace.trace_loop）。"""
        from .trace import trace_loop
        return trace_loop(self.ledger.entries)

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
            "justification_sources": _py(self.just_src),
            "chosen_before": list(self.chosen_before),
            "exp_log": _py(self.exp_log),
            "n_corrections": int(self.n_corrections),
            "reappraisal": {"src_beta": _py(self.src_beta), "ver_pending": _py(self.ver_pending), "G": _py(self.G),
                            "n_ticks": int(self.n_ticks), "n_verifications": int(self.n_verifications),
                            "log_tau": _py(self.log_tau), "tau_yoke": _py(self.tau_yoke), "last_ev": _py(self.last_ev)},
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
        if doc.get("schema_version") not in READABLE_SCHEMAS:
            raise StateVersionError(f"スキーマ版が合いません: 保存 {doc.get('schema_version')!r}, 実装 {SCHEMA_VERSION}（読めるのは {READABLE_SCHEMAS}）")
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
        obj.just_src = {k: {s_: float(d) for s_, d in row.items()} for k, row in meta["justification_sources"].items()}
        obj.chosen_before = list(meta["chosen_before"])
        obj.exp_log = meta["exp_log"]
        obj._exp_pos = {r["id"]: i for i, r in enumerate(obj.exp_log) if r.get("how") != "correction"}
        obj.n_corrections = int(meta["n_corrections"])
        ra = meta.get("reappraisal") or {}                      # スキーマ 7 の状態には無い（空として扱う = v3.3 と同じ）
        obj.src_beta = {k: [float(v[0]), float(v[1])] for k, v in ra.get("src_beta", {}).items()}
        obj.ver_pending = copy.deepcopy(ra.get("ver_pending", []))
        obj.G = {k: float(v) for k, v in ra.get("G", {}).items()}
        obj.n_ticks = int(ra.get("n_ticks", 0))
        obj.n_verifications = int(ra.get("n_verifications", 0))
        obj.log_tau = {k: float(v) for k, v in (ra.get("log_tau") or {}).items()}
        ty = ra.get("tau_yoke")
        obj.tau_yoke = None if ty is None else [{k: float(v) for k, v in d.items()} for d in ty]
        obj.last_ev = {k: str(v) for k, v in (ra.get("last_ev") or {}).items()}
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
