"""統合: experience() / appraise() / probe() / save() / load()。

experience(exp) の順序（SPEC 6）
  1. schema 検証
  2. 予期: 信念の予測 → V_MB, V_MF, ω, Q（結果を見る前）
  3. optional なら接近・回避を選ぶ。回避なら記録して終了
  4. Δh, Δu（ベクトル）, 寄与の分解, f, δ, 感情
  5. 信念の更新（責任 φ → 計数、π、型の大きさ、リプレイ緩衝 → 事前ヘッドの SGD）
  6. 価値の更新（φ で教師を分け、人モデル・状況モデルのカルマン）
  7. 欲求の状態 h と気分 m の更新
  8. 台帳に記録
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from . import innate
from . import value as V
from .belief import BeliefNet
from .config import Config
from .innate import NEEDS, N_NEEDS, N_TYPES, OUTCOME_TYPES, TYPE_INDEX
from .ledger import Ledger
from .lexicon import Lexicon
from .mlp import sigmoid
from .schema import ACTOR_SELF, ACTOR_WORLD, validate_experience, validate_probe

SCHEMA_VERSION = 1
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


# ----------------------------------------------------------------------
class NPN:
    """経験で人格が形成される外部ニューラルネットワーク（最小実装 v3.0）。"""

    def __init__(self, seed: int = 0, lexicon: Optional[Lexicon] = None, config: Optional[Config] = None,
                 **overrides: Any):
        cfg = config if config is not None else Config(seed=seed)
        if config is not None and "seed" not in overrides and seed != 0:
            overrides["seed"] = seed
        if overrides:
            cfg = cfg.replace(**overrides)
        self.cfg = cfg
        self.lex = lexicon if lexicon is not None else Lexicon.load()
        self.rng = np.random.default_rng(cfg.seed)
        self.belief = BeliefNet(cfg, self.lex, self.rng)
        self.value = V.ValueNet(cfg, self.lex)
        self.h = np.full(N_NEEDS, innate.H0)
        self.mood = 0.0
        self.ledger = Ledger()
        self.n_experiences = 0
        self.n_avoided = 0

    # ==================================================================
    # 評価の核（状態を変えない）
    # ==================================================================
    def _core(self, e: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cfg = self.cfg
        P = e["participants"]
        nP = len(P)
        pidx = {p["id"]: j for j, p in enumerate(P)}
        outs = e["outcomes"]
        # 2. 予期
        pred = self.belief.predict(e["context"], P)
        mf = self.value.mf(P, e["context"], e["concepts"])
        mb = V.model_based(pred, self.h, self.belief.mag, cfg, e["id"])
        omega, Q = V.arbitrate(mf["V_MF"], mf["var_MF"], mb["V_MB"], mb["var_MB"])
        # 4. 観測、責任 φ、Δh, Δu, 寄与
        y_person = np.zeros((nP, N_TYPES))
        y_sw = np.zeros(N_TYPES)
        for o in outs:
            k = TYPE_INDEX[o["type"]]
            if o["actor"] in pidx:
                y_person[pidx[o["actor"]], k] = 1.0
            else:
                y_sw[k] = 1.0
        phi = self.belief.responsibility(pred, y_person)
        rows = innate.outcome_rows([o["type"] for o in outs], [o["magnitude"] for o in outs])
        dh, du, contrib = innate.contributions(self.h, rows)
        n_o = len(outs)
        phi_o = np.ones(n_o)
        is_person = np.zeros(n_o)
        actor_j = np.full(n_o, -1, dtype=int)
        for i, o in enumerate(outs):
            if o["actor"] in pidx:
                j = pidx[o["actor"]]
                actor_j[i] = j
                is_person[i] = 1.0
                phi_o[i] = phi[j, TYPE_INDEX[o["type"]]]
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
        f = V.felt_value(du, self.mood, cfg)
        delta = V.surprise(f, Q, cfg)
        emo = V.emotions(f, delta, Q, contrib, phi_o, is_person, cfg)
        # --- 辞書にまとめる ---
        contrib_list = []
        for i, o in enumerate(outs):
            contrib_list.append({
                "type": o["type"], "actor": o["actor"], "magnitude": o["magnitude"],
                "contrib": _need_dict(contrib[i]),
                "phi_person": float(phi_o[i]) if is_person[i] else None,
                "phi_ctx": float(1.0 - phi_o[i]) if is_person[i] else None,
                "p_predicted": (float(pred.p_person[actor_j[i], TYPE_INDEX[o["type"]]]) if is_person[i]
                                else float(pred.p_sw[TYPE_INDEX[o["type"]]])),
            })
        phi_list = [{"participant": p["id"], "type": t, "phi_person": float(phi[j, k]),
                     "phi_ctx": float(1.0 - phi[j, k])}
                    for j, p in enumerate(P) for k, t in enumerate(OUTCOME_TYPES) if y_person[j, k] > 0.5]
        ap: Dict[str, Any] = {
            "kind": "experience", "id": e["id"], "context": e["context"],
            "Q": float(Q), "V_MB": mb["V_MB"], "V_MF": mf["V_MF"], "omega": omega,
            "sigma2_MB": mb["var_MB"], "sigma2_MF": mf["var_MF"],
            "dh": _need_dict(dh), "du": _need_dict(du), "du_total": float(du.sum()),
            "contributions": contrib_list, "phi": phi_list,
            "teachers": {"person": {p["id"]: _need_dict(person_teachers[j]) for j, p in enumerate(P)},
                         "situation": _need_dict(sit_teacher)},
            "f": f, "delta": delta, "mood": float(self.mood), "emotions": emo,
            "belief_head_used": bool(pred.head_used),
        }
        internals = {"pred": pred, "mf": mf, "mb": mb, "y_person": y_person, "y_sw": y_sw, "phi": phi,
                     "dh": dh, "du": du, "contrib": contrib, "person_teachers": person_teachers,
                     "sit_teacher": sit_teacher}
        return ap, internals

    def _p_approach(self, Q: float) -> float:
        return float(sigmoid(np.array([self.cfg.beta_choice * (Q + self.cfg.explore_bonus)]))[0])

    # ==================================================================
    # 公開 API
    # ==================================================================
    def appraise(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """予期・評価（2〜4）を計算する。状態（乱数を含む）を一切変えない。"""
        e = validate_experience(exp, self.lex)
        ap, _ = self._core(e)
        ap["optional"] = e["optional"]
        if e["optional"]:
            ap["p_approach"] = self._p_approach(ap["Q"])
            ap["approached"] = None
        return _py(ap)

    def experience(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """経験を評価し、学習する。戻り値は appraisal dict。"""
        e = validate_experience(exp, self.lex)
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
                self.ledger.add({"kind": "avoid", "id": e["id"], "context": e["context"],
                                 "Q": float(ap["Q"]), "p_approach": p_app})
                self.n_avoided += 1
                for key in ("f", "delta", "du", "du_total", "dh", "contributions", "phi", "teachers", "emotions"):
                    ap[key] = None
                ap["value_change"] = None
                return _py(ap)
        # 5. 信念の更新
        P = e["participants"]
        self.belief.update(e["context"], P, it["y_person"], it["y_sw"], it["phi"], e["outcomes"], self.rng)
        # 6. 価値の更新
        mf_before = it["mf"]
        lr = self.value.learn(P, e["context"], e["concepts"], it["person_teachers"], it["sit_teacher"])
        mf_after = self.value.mf(P, e["context"], e["concepts"])
        # 7. 欲求の状態と気分
        mood_before = self.mood
        self.mood = V.next_mood(self.mood, ap["delta"], self.cfg)
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
            "top_features": [{"model": m, "feature": n, "dW": _need_dict(r), "dW_sum": float(r.sum())}
                             for _, m, n, r in tops[:6]],
        }
        ap["mood_after"] = float(self.mood)
        # 8. 台帳
        self.ledger.add({
            "kind": "experience", "id": e["id"], "context": e["context"],
            "Q": float(ap["Q"]), "f": ap["f"], "delta": ap["delta"],
            "du": [float(x) for x in it["du"]], "dh": [float(x) for x in it["dh"]],
            "phi": [[x["participant"], x["type"], x["phi_person"]] for x in ap["phi"]],
            "mood": [float(mood_before), float(self.mood)],
            "teachers": {"person": {p["id"]: [float(x) for x in t] for p, t in zip(P, it["person_teachers"])},
                         "situation": [float(x) for x in it["sit_teacher"]]},
            "dW": {"person": {n: [float(x) for x in r] for n, r in lr["dW_person"].items()},
                   "situation": {n: [float(x) for x in r] for n, r in lr["dW_situation"].items()}},
        })
        self.n_experiences += 1
        return _py(ap)

    def probe(self, p: Dict[str, Any]) -> Dict[str, Any]:
        """学習しない評価。状況 / エンティティ / 概念。"""
        q = validate_probe(p, self.lex)
        kind = q["kind"]
        cfg = self.cfg
        if kind == "situation":
            pred = self.belief.predict(q["context"], q["participants"])
            mf = self.value.mf(q["participants"], q["context"], q["concepts"])
            mb = V.model_based(pred, self.h, self.belief.mag, cfg, probe_key(q))
            omega, Q = V.arbitrate(mf["V_MF"], mf["var_MF"], mb["V_MB"], mb["var_MB"])
            out = {"kind": kind, "Q": Q, "V_MB": mb["V_MB"], "V_MF": mf["V_MF"], "omega": omega,
                   "sigma2_MB": mb["var_MB"], "sigma2_MF": mf["var_MF"],
                   "persons": [{"id": x["id"], "value": x["value"], "known": x["known"]} for x in mf["persons"]],
                   "situation_value": mf["situation"]["value"],
                   "p_person": {pp["id"]: {t: float(pred.p_person[j, k]) for k, t in enumerate(OUTCOME_TYPES)}
                                for j, pp in enumerate(q["participants"])},
                   "p_sw": {t: float(pred.p_sw[k]) for k, t in enumerate(OUTCOME_TYPES)}}
            return _py(out)
        if kind == "entity":
            like, var_mf, pred5, known = self.value.like_entity(q["entity"], q["class"])
            part = [{"id": q["entity"], "class": q["class"]}]
            pred = self.belief.predict(q["context"], part)
            g = V.type_effect(self.h, self.belief.mag)
            pp = pred.p_person[0]
            V_MB = float((g * pp).sum())
            var_MB = float(((g ** 2) * (pred.var_person[0] + pp * (1 - pp) / cfg.mc_samples)).sum())
            omega, Q = V.arbitrate(like, var_mf, V_MB, var_MB)
            out = {"kind": kind, "entity": q["entity"], "class": q["class"], "context": q["context"],
                   "like": like, "like_by_need": _need_dict(pred5), "known": known,
                   "V_MF": like, "V_MB": V_MB, "omega": omega, "Q": Q,
                   "sigma2_MF": var_mf, "sigma2_MB": var_MB,
                   "p_person": {t: float(pp[k]) for k, t in enumerate(OUTCOME_TYPES)},
                   "p_ind": {t: float(pred.p_ind[0, k]) for k, t in enumerate(OUTCOME_TYPES)},
                   "m_cls": {t: float(pred.m_cls[0, k]) for k, t in enumerate(OUTCOME_TYPES)}}
            return _py(out)
        like, var_mf, pred5, known = self.value.like_concept(q["concept"])
        return _py({"kind": kind, "concept": q["concept"], "like": like, "like_by_need": _need_dict(pred5),
                    "known": known, "sigma2_MF": var_mf})

    # ==================================================================
    # 保存・復元
    # ==================================================================
    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        bmeta, barr = self.belief.get_state()
        vmeta, varr = self.value.get_state()
        meta = {
            "config": self.cfg.to_dict(),
            "lexicon": self.lex.to_dict(),
            "rng": _py(self.rng.bit_generator.state),
            "mood": float(self.mood),
            "n_experiences": int(self.n_experiences),
            "n_avoided": int(self.n_avoided),
            "belief": bmeta,
            "value": vmeta,
            "ledger": _py(self.ledger.to_list()),
        }
        arrays = {"h": self.h}
        arrays.update(barr)
        arrays.update(varr)
        return meta, arrays

    def save(self, directory: Union[str, Path]) -> str:
        """state.json（設定・計数・特徴の索引・乱数の状態・台帳など）と arrays.npz（W, Σ, 重みなど）。
        スキーマ版と内容のハッシュを持つ。"""
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
    def load(cls, directory: Union[str, Path]) -> "NPN":
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
        lex = Lexicon(meta["lexicon"])
        obj = cls(lexicon=lex, config=cfg)
        # 乱数
        obj.rng = np.random.Generator(np.random.PCG64())
        obj.rng.bit_generator.state = meta["rng"]
        obj.belief.set_state(meta["belief"], arrays)
        obj.value.set_state(meta["value"], arrays)
        obj.h = np.array(arrays["h"], dtype=np.float64)
        obj.mood = float(meta["mood"])
        obj.ledger = Ledger.from_list(meta["ledger"])
        obj.n_experiences = int(meta["n_experiences"])
        obj.n_avoided = int(meta["n_avoided"])
        return obj

    # ==================================================================
    # 便利
    # ==================================================================
    def summary(self) -> Dict[str, Any]:
        """CLI show 用: 価値の上位と信念の要約。"""
        def top(model: V.KalmanModel, n: int = 8) -> List[Dict[str, Any]]:
            rows = [(float(model.W[i].sum()), nm, model.W[i]) for i, nm in enumerate(model.names)
                    if not nm.startswith(("emb:", "cemb:"))]
            rows.sort(key=lambda t: -abs(t[0]))
            return [{"feature": nm, "value": v, "by_need": _need_dict(w)} for v, nm, w in rows[:n]]
        return _py({
            "n_experiences": self.n_experiences, "n_avoided": self.n_avoided,
            "h": _need_dict(self.h), "mood": self.mood,
            "person_top": top(self.value.person), "situation_top": top(self.value.sit),
            "belief": self.belief.summary(),
        })
