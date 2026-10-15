"""経験とプローブの検証（LLM 抽出の出力に相当する構造化 JSON を受ける）。

検証を通ったものは、正規化した新しい dict（元の入力は変更しない）として返す。

v3.1 の変更:
  - 未知の語（類・文脈・概念が lexicon にない）はエラーにしない。ゼロベクトルとして状態に登録し、警告を記録する（SPEC-v3.1 1.6）。
    結果の型（効果表 M の語彙）と actor は、これまでどおり未知ならエラー。
  - 結果は単一の型の代わりに解釈の候補を持てる（1.4）:
      {"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "friendA", "magnitude": 0.8}
  - 経験に任意の `session`（文脈の連続。条件づけの TD をつなぐ範囲、1.2）。

v3.2 の変更:
  - 経験と状況プローブに任意の `announced`（申し出に前もって付いている結果の型のリスト。例: 褒め言葉を伴う申し出なら ["praise"]）。
    予期 Q の手がかり（Q = U_val + λ_CQ ΣC）に `type:<型>` として入る。結果そのものではない（結果は outcomes に書く）。
    選択の前に分かる手がかりは、概念・参加者（個体と類）・announced の型だけ。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .innate import OUTCOME_TYPES

ACTOR_SELF = "self"
ACTOR_WORLD = "world"
RESERVED_IDS = (ACTOR_SELF, ACTOR_WORLD)

_EXP_KEYS = {"id", "context", "concepts", "participants", "outcomes", "optional", "session", "announced"}
_SIT_KEYS = {"id", "context", "concepts", "participants", "announced"}
_ENT_KEYS = {"id", "entity", "class", "context"}
_CON_KEYS = {"id", "concept"}
_CUE_KEYS = {"id", "cue"}


class SchemaError(ValueError):
    """経験・プローブの形式エラー。"""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _check_keys(d: Dict[str, Any], allowed: set, what: str) -> None:
    unknown = set(d) - allowed
    _need(not unknown, f"{what} に未知のキー: {sorted(unknown)}")


def _word(x: Any, what: str) -> str:
    _need(isinstance(x, str) and x != "", f"{what} は空でない文字列です: {x!r}")
    return x


def _participants(raw: Any) -> List[Dict[str, str]]:
    _need(isinstance(raw, list), "participants はリストでなければなりません")
    out: List[Dict[str, str]] = []
    seen = set()
    for p in raw:
        _need(isinstance(p, dict) and set(p) == {"id", "class"}, f"participants の要素は {{id, class}} です: {p!r}")
        pid, pcls = _word(p["id"], "participant の id"), _word(p["class"], "participant の class")
        _need(pid not in RESERVED_IDS, f"participant の id に予約語は使えません: {pid!r}")
        _need(pid not in seen, f"participant の id が重複しています: {pid!r}")
        seen.add(pid)
        out.append({"id": pid, "class": pcls})
    return out


def _concepts(raw: Any) -> List[str]:
    _need(isinstance(raw, list), "concepts はリストでなければなりません")
    out: List[str] = []
    for c in raw:
        _word(c, "concept")
        _need(c not in out, f"concepts が重複しています: {c!r}")
        out.append(c)
    return out


def _announced(raw: Any) -> List[str]:
    _need(isinstance(raw, list), "announced はリストでなければなりません")
    out: List[str] = []
    for t in raw:
        _need(t in OUTCOME_TYPES, f"未知の結果の型です（announced）: {t!r}")
        _need(t not in out, f"announced が重複しています: {t!r}")
        out.append(t)
    return out


def _outcome(o: Any, pids: set) -> Dict[str, Any]:
    _need(isinstance(o, dict), f"outcomes の要素は dict です: {o!r}")
    keys = set(o)
    clear = keys == {"type", "actor", "magnitude"}
    ambiguous = keys == {"interpretations", "actor", "magnitude"}
    _need(clear or ambiguous, f"outcomes の要素は {{type, actor, magnitude}} か {{interpretations, actor, magnitude}} です: {o!r}")
    a, m = o["actor"], o["magnitude"]
    _need(isinstance(a, str) and (a in RESERVED_IDS or a in pids),
          f"actor は self / world / 参加者の id のいずれかです: {a!r}")
    _need(isinstance(m, (int, float)) and not isinstance(m, bool) and 0.0 <= float(m) <= 1.0,
          f"magnitude は 0〜1 の数です: {m!r}")
    if clear:
        t = o["type"]
        _need(t in OUTCOME_TYPES, f"未知の結果の型です: {t!r}（語彙: {', '.join(OUTCOME_TYPES)}）")
        return {"type": t, "actor": a, "magnitude": float(m)}
    interp = o["interpretations"]
    _need(isinstance(interp, dict) and len(interp) >= 2, "interpretations は 2 つ以上の候補を持つ dict です")
    probs: Dict[str, float] = {}
    for t, q in interp.items():
        _need(t in OUTCOME_TYPES, f"未知の結果の型です（interpretations）: {t!r}")
        _need(isinstance(q, (int, float)) and not isinstance(q, bool) and float(q) > 0.0,
              f"解釈の確率は正の数です: {t!r}={q!r}")
        probs[t] = float(q)
    tot = sum(probs.values())
    return {"interpretations": {t: q / tot for t, q in probs.items()}, "actor": a, "magnitude": float(m)}


def validate_experience(exp: Any) -> Dict[str, Any]:
    _need(isinstance(exp, dict), "経験は dict でなければなりません")
    _check_keys(exp, _EXP_KEYS, "経験")
    for k in ("id", "context", "concepts", "participants", "outcomes"):
        _need(k in exp, f"経験に必須のキーがありません: {k}")
    _word(exp["id"], "id")
    context = _word(exp["context"], "context")
    concepts = _concepts(exp["concepts"])
    participants = _participants(exp["participants"])
    pids = {p["id"] for p in participants}
    _need(isinstance(exp["outcomes"], list), "outcomes はリストでなければなりません")
    outcomes = [_outcome(o, pids) for o in exp["outcomes"]]
    optional = exp.get("optional", False)
    _need(isinstance(optional, bool), "optional は真偽値です")
    session: Optional[str] = exp.get("session")
    _need(session is None or (isinstance(session, str) and session != ""), "session は空でない文字列です")
    announced = _announced(exp.get("announced", []))
    return {"id": exp["id"], "context": context, "concepts": concepts, "participants": participants,
            "outcomes": outcomes, "optional": optional, "session": session, "announced": announced}


def validate_situation(p: Dict[str, Any]) -> Dict[str, Any]:
    _check_keys(p, _SIT_KEYS, "状況プローブ")
    _need("context" in p, "状況プローブには context が必要です")
    return {
        "kind": "situation",
        "id": p.get("id", "probe"),
        "context": _word(p["context"], "context"),
        "concepts": _concepts(p.get("concepts", [])),
        "participants": _participants(p.get("participants", [])),
        "announced": _announced(p.get("announced", [])),
    }


def validate_probe(p: Any) -> Dict[str, Any]:
    """プローブ（学習しない評価）: 状況 / エンティティ / 概念 / 手がかり（条件づけられた価値）。"""
    _need(isinstance(p, dict), "プローブは dict でなければなりません")
    if "entity" in p:
        _check_keys(p, _ENT_KEYS, "エンティティプローブ")
        _word(p["entity"], "entity")
        _need("class" in p, "entity には class が必要です")
        out: Dict[str, Any] = {"kind": "entity", "id": p.get("id", "probe"), "entity": p["entity"],
                               "class": _word(p["class"], "class"), "context": None}
        if p.get("context") is not None:
            out["context"] = _word(p["context"], "context")
        return out
    if "concept" in p:
        _check_keys(p, _CON_KEYS, "概念プローブ")
        return {"kind": "concept", "id": p.get("id", "probe"), "concept": _word(p["concept"], "concept")}
    if "cue" in p:
        _check_keys(p, _CUE_KEYS, "手がかりプローブ")
        return {"kind": "cue", "id": p.get("id", "probe"), "cue": _word(p["cue"], "cue")}
    return validate_situation(p)
