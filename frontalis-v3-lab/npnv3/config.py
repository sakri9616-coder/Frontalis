"""設定。保存・復元の対象（state.json に入る）。全人格で同じ既定値を使う。

SPEC の第4〜6節（v3.0）と SPEC-v3.1 で固定された係数をここに集約する。アブレーション用の切り替えは既定ですべて OFF。
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
    no_head: bool = False         # ニューラル事前ヘッドを使わない（head_kind = "none" と同じ）
    no_scope: bool = False        # スコープ帰属を切る（phi_person = 1）
    no_concern: bool = False      # v3.1: 欲求の関心の重み a を 1 に固定
    no_conditioning: bool = False # v3.1: 条件づけられた価値 C を使わない（学習もしない）
    no_reference: bool = False    # v3.1: 参照点項 η_ref を 0 にする
    no_interpretation: bool = False  # v3.1: 曖昧な結果を LLM の確率のまま混合し、信念を使わない
    interpretation_counts: bool = False  # v3.1 実験条件: 曖昧な結果からも（解釈の重みで）信念を計数する
    estimate_kappa: bool = True   # v3.1 1.7: κ_e を周辺尤度で推定（False なら固定 κ_e）
    kappa_per_type: bool = True   # v3.1 改訂: κ_e を（類 × 結果型）ごとに推定（False なら類ごとに 1 つ）。改訂前の挙動の再現用
    kappa_prior_sigma: float = 1.0  # 型ごとの log κ の事前（類全体の log κ の周りの正規分布）の標準偏差
    record_ledger: bool = True    # False のとき台帳に記録しない（長い履歴の検査用。学習には影響しない）

    # --- v3.1: 欲求の関心の重み a（1.1）---
    eta_a: float = 0.02
    a_run_rate: float = 0.05      # |Δu| の移動平均の速さ（SPEC に値の指定なし）

    # --- v3.1: 条件づけられた価値 C（1.2）---
    alpha_c: float = 0.1
    gamma_c: float = 0.6

    # --- v3.1: 受け止め方 f（1.3）---
    lambda_c: float = 1.0
    eta_ref: float = 0.5

    # --- v3.1: ニューラル事前ヘッド（1.5）---
    head_kind: str = "mlp"        # "mlp" | "linear" | "knn1" | "none"
    rho_scale: float = 30.0       # ρ_h = n_head / (n_head + 30)

    # --- v3.1: 類の集中度 κ_e の推定（1.7）---
    kappa_grid_lo: float = 0.3
    kappa_grid_hi: float = 50.0
    kappa_grid_n: int = 12

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

    def __post_init__(self) -> None:
        if self.no_head:
            self.head_kind = "none"
        if self.head_kind not in ("mlp", "linear", "knn1", "none"):
            raise ValueError(f"head_kind が不正です: {self.head_kind!r}")

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
