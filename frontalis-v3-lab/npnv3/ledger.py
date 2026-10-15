"""更新台帳: 全ての更新を「経験 id・教師・責任・変化量」として記録する（docs/04 第7節）。

不変量: 各特徴について、台帳の ΔW の総和は現在の W に等しい（W は 0 から始まり、更新はすべて台帳を通るため）。
回避した（optional で接近しなかった）経験は kind="avoid" として記録するだけで、価値や信念は変えない。
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import numpy as np


class Ledger:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, rec: Dict[str, Any]) -> None:
        self.entries.append(rec)

    # --- 問い合わせ ---
    def totals(self, model: str) -> Dict[str, np.ndarray]:
        """model = "person" | "situation"。特徴ごとの ΔW の総和（W と一致するはず）。"""
        out: Dict[str, np.ndarray] = {}
        for e in self.entries:
            if e["kind"] != "experience":
                continue
            for name, row in e["dW"][model].items():
                r = np.array(row)
                out[name] = out[name] + r if name in out else r
        return out

    def history_of(self, model: str, feature: str) -> List[Dict[str, Any]]:
        """その特徴を動かした経験の一覧（経験 id、ΔW、そのときの Δu）。"""
        out = []
        for e in self.entries:
            if e["kind"] == "experience" and feature in e["dW"][model]:
                out.append({"id": e["id"], "dW": e["dW"][model][feature], "du": e["du"]})
        return out

    def avoided(self) -> List[str]:
        return [e["id"] for e in self.entries if e["kind"] == "avoid"]

    def count_phi_mass(self) -> float:
        """不変量の確認用: 各 (参加者, 型) の φ_person + φ_ctx = 1 なので、台帳の φ から 1 から引いて足せば ctx の質量。"""
        return float(sum(1.0 - p[2] for e in self.entries if e["kind"] == "experience" for p in e["phi"]))

    # --- 保存・復元 ---
    def to_list(self) -> List[Dict[str, Any]]:
        return self.entries

    @classmethod
    def from_list(cls, items: List[Dict[str, Any]]) -> "Ledger":
        led = cls()
        led.entries = items
        return led
