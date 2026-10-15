"""経験とプローブの検証（LLM 抽出の出力に相当する構造化 JSON を受ける）。

検証を通ったものは、正規化した新しい dict（元の入力は変更しない）として返す。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .innate import OUTCOME_TYPES
from .lexicon import Lexicon

ACTOR_SELF = "self"
ACTOR_WORLD = "world"
RESERVED_IDS = (ACTOR_SELF, ACTOR_WORLD)

_EXP_KEYS = {"id", "context", "concepts", "participants", "outcomes", "optional"}
_SIT_KEYS = {"id", "context", "concepts", "participants"}
_ENT_KEYS = {"id", "entity", "class", "context"}
_CON_KEYS = {"id", "concept"}


class SchemaError(ValueError):
    """経験・プローブの形式エラー。"""


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _check_keys(d: Dict[str, Any], allowed: set, what: str) -> None:
    unknown = set(d) - allowed
    _need(not unknown, f"{what} に未知のキー: {sorted(unknown)}")


def _participants(raw: Any, lex: Lexicon) -> List[Dict[str, str]]:
    _need(isinstance(raw, list), "participants はリストでなければなりません")
    out: List[Dict[str, str]] = []
    seen = set()
    for p in raw:
        _need(isinstance(p, dict) and set(p) == {"id", "class"}, f"participants の要素は {{id, class}} です: {p!r}")
        pid, pcls = p["id"], p["class"]
        _need(isinstance(pid, str) and pid != "", f"participant の id が不正です: {pid!r}")
        _need(pid not in RESERVED_IDS, f"participant の id に予約語は使えません: {pid!r}")
        _need(pid not in seen, f"participant の id が重複しています: {pid!r}")
        _need(isinstance(pcls, str) and lex.has_class(pcls), f"未知の類です（語彙にありません）: {pcls!r}")
        seen.add(pid)
        out.append({"id": pid, "class": pcls})
    return out


def _context(raw: Any, lex: Lexicon) -> str:
    _need(isinstance(raw, str) and lex.has_context(raw), f"未知の文脈です（語彙にありません）: {raw!r}")
    return raw


def _concepts(raw: Any, lex: Lexicon) -> List[str]:
    _need(isinstance(raw, list), "concepts はリストでなければなりません")
    out: List[str] = []
    for c in raw:
        _need(isinstance(c, str) and lex.has_concept(c), f"未知の概念です（語彙にありません）: {c!r}")
        _need(c not in out, f"concepts が重複しています: {c!r}")
        out.append(c)
    return out


def validate_experience(exp: Any, lex: Lexicon) -> Dict[str, Any]:
    _need(isinstance(exp, dict), "経験は dict でなければなりません")
    _check_keys(exp, _EXP_KEYS, "経験")
    for k in ("id", "context", "concepts", "participants", "outcomes"):
        _need(k in exp, f"経験に必須のキーがありません: {k}")
    _need(isinstance(exp["id"], str) and exp["id"] != "", "id は空でない文字列です")
    context = _context(exp["context"], lex)
    concepts = _concepts(exp["concepts"], lex)
    participants = _participants(exp["participants"], lex)
    pids = {p["id"] for p in participants}
    raw_out = exp["outcomes"]
    _need(isinstance(raw_out, list), "outcomes はリストでなければなりません")
    outcomes: List[Dict[str, Any]] = []
    for o in raw_out:
        _need(isinstance(o, dict) and set(o) == {"type", "actor", "magnitude"},
              f"outcomes の要素は {{type, actor, magnitude}} です: {o!r}")
        t, a, m = o["type"], o["actor"], o["magnitude"]
        _need(t in OUTCOME_TYPES, f"未知の結果の型です: {t!r}（語彙: {', '.join(OUTCOME_TYPES)}）")
        _need(isinstance(a, str) and (a in RESERVED_IDS or a in pids),
              f"actor は self / world / 参加者の id のいずれかです: {a!r}")
        _need(isinstance(m, (int, float)) and not isinstance(m, bool) and 0.0 <= float(m) <= 1.0,
              f"magnitude は 0〜1 の数です: {m!r}")
        outcomes.append({"type": t, "actor": a, "magnitude": float(m)})
    optional = exp.get("optional", False)
    _need(isinstance(optional, bool), "optional は真偽値です")
    return {"id": exp["id"], "context": context, "concepts": concepts, "participants": participants,
            "outcomes": outcomes, "optional": optional}


def validate_situation(p: Dict[str, Any], lex: Lexicon) -> Dict[str, Any]:
    _check_keys(p, _SIT_KEYS, "状況プローブ")
    _need("context" in p, "状況プローブには context が必要です")
    return {
        "kind": "situation",
        "id": p.get("id", "probe"),
        "context": _context(p["context"], lex),
        "concepts": _concepts(p.get("concepts", []), lex),
        "participants": _participants(p.get("participants", []), lex),
    }


def validate_probe(p: Any, lex: Lexicon) -> Dict[str, Any]:
    """プローブ（学習しない評価）: 状況 / エンティティ / 概念。"""
    _need(isinstance(p, dict), "プローブは dict でなければなりません")
    if "entity" in p:
        _check_keys(p, _ENT_KEYS, "エンティティプローブ")
        _need(isinstance(p["entity"], str) and p["entity"] != "", "entity は空でない文字列です")
        _need("class" in p and lex.has_class(p["class"]), f"entity には既知の class が必要です: {p.get('class')!r}")
        ctx = p.get("context")
        out: Dict[str, Any] = {"kind": "entity", "id": p.get("id", "probe"), "entity": p["entity"],
                               "class": p["class"], "context": None}
        if ctx is not None:
            out["context"] = _context(ctx, lex)
        return out
    if "concept" in p:
        _check_keys(p, _CON_KEYS, "概念プローブ")
        _need(isinstance(p["concept"], str) and lex.has_concept(p["concept"]),
              f"未知の概念です（語彙にありません）: {p['concept']!r}")
        return {"kind": "concept", "id": p.get("id", "probe"), "concept": p["concept"]}
    return validate_situation(p, lex)
