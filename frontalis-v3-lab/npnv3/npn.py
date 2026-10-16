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
"""
from __future__ import annotations

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
from .schema import validate_experience, validate_probe

SCHEMA_VERSION = 6          # v3.0 は 1、v3.1 は 2（a, C, 所有する埋め込み, 効果表）。改訂 2 で観測の記録（想起）、v3.2 で欲求ごとの Σ とヘッドの対数尤度が加わった(3 → 4)。
                            # v3.3 で、類ごとの生の計数・類を除いたヘッドのモデル（leave-class-out）・潜在原因つきの条件づけ（原因の価値・文脈の計数・時刻）・pending の文脈と時刻が加わった(4 → 5)。
                            # v3.3 第 1 部の設計の改訂で、潜在原因の試行の履歴（時刻・原因・文脈）と、原因ごとの文脈の集中度が加わった(5 → 6)
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

    def _core(self, e: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.cfg
        a = self.concern.a
        P = e["participants"]
        nP = len(P)
        pidx = {p["id"]: j for j, p in enumerate(P)}
        outs = e["outcomes"]
        n_o = len(outs)
        warnings = self.emb.unknown_words([("cls", p["class"]) for p in P] + [("ctx", e["context"])]
                                          + [("concept", c) for c in e["concepts"]])
        t_now = self._clock()
        # 2. 予期
        pred = self.belief.predict(e["context"], P)
        mf = self.value.mf(P, e["context"], e["concepts"], a)
        mb = V.model_based(pred, self.h, self.belief.mag, cfg, e["id"], self.M, a)
        omega, Q_val = V.arbitrate(mf["V_MF"], mf["var_MF"], mb["V_MB"], mb["var_MB"])
        cues_q = self._cues_q(e["concepts"], P, e["announced"])
        C_Q = self._c_q(cues_q, e["context"], t_now)
        Q = Q_val + C_Q
        Q_person = float(omega * mf["V_MF_person"] + (1.0 - omega) * mb["V_MB_person"])
        # 結果の解釈の重み（曖昧な結果は信念で決める）
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
        # Δh, Δu, 寄与
        rows = innate.mixed_rows(W_out, [o["magnitude"] for o in outs], self.M)
        dh, du, contrib = innate.contributions(self.h, rows)
        phi_o = np.ones(n_o)
        is_person = np.zeros(n_o)
        for i in range(n_o):
            j = actor_j[i]
            if j >= 0:
                is_person[i] = 1.0
                phi_o[i] = float(W_out[i] @ phi1[j]) if ambiguous[i] else phi[j, int(np.argmax(W_out[i]))]
        # 教師（帰属で分ける）。fixed_teacher では帰属を使わない（3-F 型）。
        person_teachers = [np.zeros(N_NEEDS) for _ in range(nP)]
        sit_teacher = np.zeros(N_NEEDS)
        for i in range(n_o):
            if is_person[i]:
                w = 1.0 if cfg.fixed_teacher else phi_o[i]
                person_teachers[actor_j[i]] = person_teachers[actor_j[i]] + w * contrib[i]
                if not cfg.fixed_teacher:
                    sit_teacher = sit_teacher + (1.0 - phi_o[i]) * contrib[i]
            else:
                sit_teacher = sit_teacher + contrib[i]
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
            "Q": float(Q), "Q_val": float(Q_val), "C_Q": float(C_Q), "Q_person": Q_person, "cues_Q": cues_q,
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
                     "person_teachers": person_teachers, "sit_teacher": sit_teacher, "cues": cues, "inn": inn}
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
        """予期・評価（2〜4）を計算する。状態（乱数を含む）を一切変えない。"""
        e = validate_experience(exp)
        ap, _ = self._core(e)
        ap["optional"] = e["optional"]
        if e["optional"]:
            ap["p_approach"] = self._p_approach(ap["Q"])
            ap["approached"] = None
        return _py(ap)

    def experience(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """経験を評価し、学習する。戻り値は appraisal dict。"""
        cfg = self.cfg
        e = validate_experience(exp)
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
                                     "Q": float(ap["Q"]), "C_Q": float(ap["C_Q"]), "p_approach": p_app})
                self.n_avoided += 1
                self.cond.now = self._clock()
                for key in ("f", "f_parts", "delta", "du", "du_total", "du_weighted", "dh", "contributions", "phi",
                            "teachers", "emotions", "interpretations"):
                    ap[key] = None
                ap["value_change"] = None
                return _py(ap)
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
                                e["outcomes"], self.rng)
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
            # 復位: 手がかりなしで嫌悪の結果が起きたとき、負の価値を持つ原因の最後に使われた時刻を更新する（この事象と直前の事象の手がかりは除く）
            if r <= -cfg.lc_aversive_threshold:
                skip = list(it["cues"]) + (list(self.pending["cues"]) if self.pending is not None else [])
                reinst = self.cond.reinstate(skip, t_now, e["context"], r)
            self.pending = {"cues": list(it["cues"]), "session": e["session"], "context": e["context"], "t": t_now}
        # 7. 関心の重み a、欲求の状態 h、気分 m
        self.concern.update(it["du"])
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
        # 8. 台帳
        if cfg.record_ledger:
            self.ledger.add({
                "kind": "experience", "id": e["id"], "context": e["context"],
                "Q": float(ap["Q"]), "C_Q": float(ap["C_Q"]), "f": ap["f"], "delta": ap["delta"],
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
        self.n_experiences += 1
        self.cond.now = self._clock()
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
            out = {"kind": kind, "Q": Q_val + C_Q, "Q_val": Q_val, "C_Q": C_Q,
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
            out = {"kind": kind, "entity": q["entity"], "class": q["class"], "context": q["context"],
                   "like": like, "like_by_need": _need_dict(pred5), "known": known,
                   "V_MF": like, "V_MB": V_MB, "omega": omega, "Q": Q_val + C_Q, "Q_val": Q_val, "C_Q": C_Q,
                   "sigma2_MF": var_mf, "sigma2_MB": var_MB,
                   "conditioned": self.cond.value([f"ind:{q['entity']}", f"cls:{q['class']}"], q["context"], self._clock()),
                   "p_person": {t: float(pp[k]) for k, t in enumerate(OUTCOME_TYPES)},
                   "p_ind": {t: float(pred.p_ind[0, k]) for k, t in enumerate(OUTCOME_TYPES)},
                   "m_cls": {t: float(pred.m_cls[0, k]) for k, t in enumerate(OUTCOME_TYPES)}}
            return _py(out)
        if kind == "cue":
            return _py({"kind": kind, "cue": q["cue"], "context": q["context"],
                        "C": self.cond.value_of(q["cue"], q["context"], self._clock()),
                        "n_causes": self.cond.n_causes(q["cue"]), "known": q["cue"] in self.cond.idx})
        like, var_mf, pred5, known = self.value.like_concept(q["concept"], a)
        return _py({"kind": kind, "concept": q["concept"], "like": like, "like_by_need": _need_dict(pred5),
                    "known": known, "sigma2_MF": var_mf,
                    "conditioned": self.cond.value_of(f"concept:{q['concept']}", None, self._clock())})

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
        return _py({
            "n_experiences": self.n_experiences, "n_avoided": self.n_avoided,
            "h": _need_dict(self.h), "mood": self.mood, "a": _need_dict(self.concern.a),
            "person_top": top(self.value.person), "situation_top": top(self.value.sit),
            "conditioned_top": [{"cue": n, "C": float(c), "n_causes": int(k)} for n, c, k in conds],
            "belief": self.belief.summary(), "warnings": list(self.warnings),
        })
