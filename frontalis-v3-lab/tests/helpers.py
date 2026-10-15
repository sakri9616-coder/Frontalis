"""テスト共通の補助。判定基準は SPEC 第8節のまま（ここでは変えない）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npnv3 import NPN  # noqa: E402

HIST = ROOT / "data" / "histories"
RESULTS = ROOT / "tests" / "results"
SEEDS = list(range(20))     # SPEC 第8節: 乱数種 0〜19 の 20 本


def load_jsonl(name: str) -> List[Dict[str, Any]]:
    with open(HIST / name, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(name: str) -> Any:
    with open(HIST / name, "r", encoding="utf-8") as f:
        return json.load(f)


def make(seed: int, **kw: Any) -> NPN:
    """全人格は同じ初期状態・同じ係数。違うのは乱数の種とアブレーション指定だけ。"""
    return NPN(seed=seed, **kw)


def feed(npn: NPN, exps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [npn.experience(e) for e in exps]


def persona(name: str, seed: int, **kw: Any) -> NPN:
    n = make(seed, **kw)
    feed(n, load_jsonl(f"{name}.jsonl"))
    return n


def dump(name: str, obj: Any) -> None:
    """REPORT.md 用に、測った数値を tests/results/ に書く。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)


def deep_equal(a: Any, b: Any) -> bool:
    """辞書・リスト・数値・numpy 配列の完全一致（許容誤差なし）。"""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a.keys()) == list(b.keys()) and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def flat_update(entry: Dict[str, Any], keys: Sequence[Any]) -> np.ndarray:
    """台帳の 1 エントリの ΔW を、(モデル, 特徴) × 欲求 のベクトルに並べる。キーにない特徴は 0。"""
    out = []
    for model, name in keys:
        out.append(np.array(entry["dW"][model].get(name, [0.0] * 5)))
    return np.concatenate(out)


def update_keys(entries: Sequence[Dict[str, Any]]) -> List[Any]:
    keys: List[Any] = []
    seen = set()
    for e in entries:
        for model in ("person", "situation"):
            for name in e["dW"][model]:
                if (model, name) not in seen:
                    seen.add((model, name))
                    keys.append((model, name))
    return keys


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")
