"""設定。保存・復元の対象（state.json に入る）。全人格で同じ既定値を使う。

SPEC の第4〜6節で固定された係数をここに集約する。アブレーション用の切り替えは既定ですべて OFF。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict


def _person_var() -> Dict[str, float]:
    # 人モデルの事前分散（SPEC 5.1）
    return {"people": 0.01, "cls": 0.04, "ind": 0.25, "emb": 0.02}


def _person_q() -> Dict[str, float]:
    # 人モデルの過程雑音 q（層ごとに固定）
    return {"people": 5e-6, "cls": 2e-5, "ind": 1e-4, "emb": 5e-6}


def _sit_var() -> Dict[str, float]:
    # 状況モデルの事前分散
    return {"world": 0.01, "ctx": 0.1, "concept": 0.1, "cemb": 0.02}


def _sit_q() -> Dict[str, float]:
    return {"world": 5e-6, "ctx": 2e-5, "concept": 2e-5, "cemb": 5e-6}


@dataclass
class Config:
    seed: int = 0

    # --- アブレーション（テスト用。既定はすべて OFF）---
    flat_value: bool = False      # 価値の階層を外す（ind 相当の特徴だけ、事前分散 0.25）
    fixed_teacher: bool = False   # 3-F 型: W[ind] += 0.1 * du（期待に依存しない固定教師）
    no_head: bool = False         # ニューラル事前ヘッドを使わない（prior_cls = m_gen）
    no_scope: bool = False        # スコープ帰属を切る（phi_person = 1）

    # --- 信念ネット（SPEC 4）---
    kappa_e: float = 2.0          # 個体（と文脈）の事前の強さ
    kappa_c: float = 4.0          # 類の事前の強さ
    kappa_g: float = 4.0          # 一般（と世界）の事前の強さ
    prior_a: float = 1.0          # 一般の事前 Beta(1, 4)
    prior_b: float = 4.0
    mag_rate: float = 0.1         # 型ごとの平均の大きさの移動平均
    mag_init: float = 0.6

    # --- ニューラル事前ヘッド（SPEC 4.4）---
    head_hidden: int = 16
    head_lr: float = 0.05
    head_batches: int = 4
    head_batch_size: int = 16
    replay_size: int = 500
    head_min_samples: int = 50

    # --- 価値ネット（SPEC 5.1）---
    obs_noise: float = 0.01       # 観測雑音 R
    person_var: Dict[str, float] = field(default_factory=_person_var)
    person_q: Dict[str, float] = field(default_factory=_person_q)
    sit_var: Dict[str, float] = field(default_factory=_sit_var)
    sit_q: Dict[str, float] = field(default_factory=_sit_q)
    flat_var: float = 0.25        # flat_value のときの事前分散
    fixed_teacher_rate: float = 0.1

    # --- MB と調停（SPEC 5.2）---
    mc_samples: int = 64

    # --- 読み出し（SPEC 5.3, 5.4）---
    s_f: float = 0.5
    beta_mood: float = 0.3
    mood_keep: float = 0.8
    beta_choice: float = 4.0
    explore_bonus: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"未知の設定キー: {sorted(unknown)}")
        return cls(**d)

    def replace(self, **kw: Any) -> "Config":
        return dataclasses.replace(self, **kw)
