"""概念・類・文脈の埋め込み（8 次元）。

実運用では LLM 側の凍結した文埋め込みに置き換える想定。この実装では data/lexicon.json の
手で意味を持たせた属性ベクトルを使う。NPN の状態は、使った語彙の写しを自分で持つ（保存時）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

EMB_DIM = 8
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "lexicon.json"


class Lexicon:
    def __init__(self, data: Dict[str, Any]):
        self._data = {
            "dims": list(data.get("dims", [])),
            "classes": {k: [float(x) for x in v] for k, v in data["classes"].items()},
            "contexts": {k: [float(x) for x in v] for k, v in data["contexts"].items()},
            "concepts": {k: [float(x) for x in v] for k, v in data["concepts"].items()},
        }
        for kind in ("classes", "contexts", "concepts"):
            for name, v in self._data[kind].items():
                if len(v) != EMB_DIM:
                    raise ValueError(f"語彙 {kind}/{name} の次元が {EMB_DIM} ではありません: {len(v)}")
        self._cls = {k: np.array(v) for k, v in self._data["classes"].items()}
        self._ctx = {k: np.array(v) for k, v in self._data["contexts"].items()}
        self._con = {k: np.array(v) for k, v in self._data["concepts"].items()}
        self._ctx_mean = np.mean([self._ctx[k] for k in sorted(self._ctx)], axis=0)

    # --- 読み込み・保存 ---
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Lexicon":
        p = Path(path) if path is not None else DEFAULT_PATH
        with open(p, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data))

    # --- 問い合わせ ---
    def has_class(self, name: str) -> bool:
        return name in self._cls

    def has_context(self, name: str) -> bool:
        return name in self._ctx

    def has_concept(self, name: str) -> bool:
        return name in self._con

    def class_names(self) -> List[str]:
        return sorted(self._cls)

    def context_names(self) -> List[str]:
        return sorted(self._ctx)

    def concept_names(self) -> List[str]:
        return sorted(self._con)

    def cls_emb(self, name: str) -> np.ndarray:
        return self._cls[name]

    def ctx_emb(self, name: Optional[str]) -> np.ndarray:
        """文脈の埋め込み。文脈が与えられないとき（エンティティのプローブ）は、全文脈の平均を使う。"""
        if name is None:
            return self._ctx_mean
        return self._ctx[name]

    def concept_emb(self, name: str) -> np.ndarray:
        return self._con[name]

    def concept_mean(self, names: Iterable[str]) -> np.ndarray:
        names = list(names)
        if not names:
            return np.zeros(EMB_DIM)
        return np.mean([self._con[n] for n in names], axis=0)
