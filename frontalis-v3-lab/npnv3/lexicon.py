"""概念・類・文脈の埋め込み（8 次元）。

v3.1（SPEC-v3.1 1.6）: **埋め込みは NPN が所有する。** 語が初めて学習に現れたとき、lexicon（または外部の埋め込み関数）から
値を**写して状態に保存**し、以後は状態の値だけを使う。lexicon や埋め込みモデルを交換しても、既存の人格の座標は変わらない。
lexicon にない語は、ゼロベクトルとして状態に登録し、警告を記録する。

appraise / probe は状態を変えないので、未登録の語は読むだけで登録しない（lexicon の値、なければゼロ）。
実運用では lexicon を LLM 側の凍結した文埋め込みに置き換える想定。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EMB_DIM = 8
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "lexicon.json"
KINDS = ("cls", "ctx", "concept")
_KEY = {"cls": "classes", "ctx": "contexts", "concept": "concepts"}


class Lexicon:
    """語彙の埋め込みの供給源（凍結した外部の写し）。NPN の状態そのものではない。"""

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
        self._vec = {k: {n: np.array(v) for n, v in self._data[_KEY[k]].items()} for k in KINDS}

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Lexicon":
        p = Path(path) if path is not None else DEFAULT_PATH
        with open(p, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data))

    def lookup(self, kind: str, name: str) -> Optional[np.ndarray]:
        return self._vec[kind].get(name)

    def names(self, kind: str) -> List[str]:
        return sorted(self._vec[kind])

    def rotated(self, rot: np.ndarray) -> "Lexicon":
        """全ベクトルに直交行列 rot を掛けた別の座標の lexicon（U9: LLM の交換を模擬）。"""
        d = self.to_dict()
        for key in ("classes", "contexts", "concepts"):
            d[key] = {n: (rot @ np.array(v)).tolist() for n, v in d[key].items()}
        return Lexicon(d)


class EmbeddingStore:
    """NPN が所有する埋め込み。供給源（Lexicon）は交換できるが、登録済みの値は変わらない。"""

    def __init__(self, source: Lexicon):
        self.source = source
        self.own: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in KINDS}

    # --- 読み出し（状態を変えない）---
    def known(self, kind: str, name: str) -> bool:
        return name in self.own[kind] or self.source.lookup(kind, name) is not None

    def get(self, kind: str, name: str) -> np.ndarray:
        v = self.own[kind].get(name)
        if v is not None:
            return v
        v = self.source.lookup(kind, name)
        return v if v is not None else np.zeros(EMB_DIM)

    def cls(self, name: str) -> np.ndarray:
        return self.get("cls", name)

    def ctx(self, name: Optional[str]) -> np.ndarray:
        """文脈の埋め込み。文脈が与えられないとき（エンティティのプローブ）は、登録済みの文脈の平均（なければゼロ）。"""
        if name is None:
            vs = list(self.own["ctx"].values())
            return np.mean(vs, axis=0) if vs else np.zeros(EMB_DIM)
        return self.get("ctx", name)

    def concept(self, name: str) -> np.ndarray:
        return self.get("concept", name)

    def concept_mean(self, names: Iterable[str]) -> np.ndarray:
        names = list(names)
        if not names:
            return np.zeros(EMB_DIM)
        return np.mean([self.get("concept", n) for n in names], axis=0)

    def unknown_words(self, items: Sequence[Tuple[str, str]]) -> List[str]:
        """(kind, name) のうち、登録も lexicon にもない語（ゼロベクトルになる）の警告文。"""
        out = []
        for kind, name in items:
            if not self.known(kind, name):
                out.append(f"未知の語 {kind}:{name}（lexicon にない）はゼロベクトルとして登録する")
        return out

    # --- 登録（学習時のみ）---
    def register(self, kind: str, name: str) -> None:
        if name in self.own[kind]:
            return
        v = self.source.lookup(kind, name)
        self.own[kind][name] = np.array(v if v is not None else np.zeros(EMB_DIM))

    # --- 保存・復元 ---
    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        meta = {k: list(self.own[k]) for k in KINDS}
        arrays = {f"emb_{k}": (np.array([self.own[k][n] for n in self.own[k]]) if self.own[k]
                               else np.zeros((0, EMB_DIM))) for k in KINDS}
        return meta, arrays

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        for k in KINDS:
            arr = np.array(arrays[f"emb_{k}"], dtype=np.float64)
            self.own[k] = {n: arr[i].copy() for i, n in enumerate(meta[k])}
