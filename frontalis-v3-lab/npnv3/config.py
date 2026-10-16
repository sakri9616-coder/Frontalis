"""設定。保存・復元の対象（state.json に入る）。全人格で同じ既定値を使う。

SPEC の第4〜6節（v3.0）と SPEC-v3.1 で固定された係数をここに集約する。アブレーション用の切り替えは既定ですべて OFF。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


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
    no_reattribution: bool = False  # v3.1 改訂 2: 想起（帰属のやり直し）を行わない（責任 φ を経験の時点で凍結する）
    reattribute_every: int = 10   # v3.1 改訂 2: K_r 経験ごとに、全観測について 1 回の掃引（retract → reassign）
    estimate_kappa: bool = True   # v3.1 1.7: κ_e を周辺尤度で推定（False なら固定 κ_e）
    kappa_per_type: bool = True   # v3.1 改訂: κ_e を（類 × 結果型）ごとに推定（False なら類ごとに 1 つ）。改訂前の挙動の再現用
    kappa_prior_sigma: float = 1.0  # 型ごとの log κ の事前（類全体の log κ の周りの正規分布）の標準偏差
    record_ledger: bool = True    # False のとき台帳に記録しない（長い履歴の検査用。学習には影響しない）

    # --- v3.1: 欲求の関心の重み a（1.1）---
    eta_a: float = 0.02
    a_run_rate: float = 0.05      # |Δu| の移動平均の速さ（SPEC に値の指定なし）

    # --- v3.2: 関心の重みの方向（1.3）と、評価基準が学習・行動を変える経路（1.1, 1.2）---
    concern_direction: str = "sensitize"   # "sensitize": 大きく揺さぶられた欲求ほど重みが上がる / "habituate": 下がる（実験条件）
    habituate_floor: float = 5.0           # habituate の目標 = run / max(|Δu_k|, run / 床)。床 = N_NEEDS で感作の上限と対称
    lambda_cq: float = 1.0                 # Q = U_val + λ_CQ ΣC(手がかり)（1.2）
    no_precision_weighting: bool = False   # アブレーション: 観測雑音を R_k = R / a_k にしない（v3.1 の R に戻す）
    no_cq: bool = False                    # アブレーション: C を Q に入れない（v3.1 の Q）

    # --- v3.1: 条件づけられた価値 C（1.2）---
    alpha_c: float = 0.1
    gamma_c: float = 0.6

    # --- v3.3 4.2: 潜在原因による条件づけ価値（Gershman, Blei & Niv 2010 の簡約）---
    single_cause: bool = False     # アブレーション: 原因を 1 つだけにする（K = 1。v3.2 の条件づけと同じ算術）
    lc_max_causes: int = 5         # K: 手がかりごとの原因の最大数
    lc_alpha: float = 0.5          # 新しい原因の事前の質量 α（SPEC-v3.3 4.2）
    lc_tau: float = 30.0           # 原因の事前の時間減衰 τ（経験。SPEC-v3.3 4.2）
    lc_sigma_r: float = 0.15       # 事後の尤度 N(目標; C_{x,k}, σ_r²) の σ_r（SPEC-v3.3 4.2）
    lc_beta: float = 1.0           # 文脈の Dirichlet 予測の疑似計数 β（SPEC に値の指定なし。対称・一様な事前として 1 を、結果を見る前に決めた）
    lc_neg_threshold: float = 0.02      # 復位: 「負の価値を持つ原因」の閾値（C < −0.02。SPEC に指定なし。結果を見る前に決めた）
    lc_aversive_threshold: float = 0.05 # 復位: 「嫌悪の結果」の閾値（事象の Σ a_k Δu_k ≤ −0.05。同上）

    # --- v3.1: 受け止め方 f（1.3）---
    lambda_c: float = 1.0
    eta_ref: float = 0.5

    # --- v3.1: ニューラル事前ヘッド（1.5）---
    head_kind: str = "mixture"    # v3.2 の既定は "mixture"。単一ヘッドは "mlp" | "linear" | "knn1" | "knnk" | "none"
    rho_scale: float = 30.0       # 単一ヘッドの ρ_h = n_head / (n_head + 30)（mixture では使わない）
    head_ll_rate: float = 0.05    # v3.2 1.4: 前向きの対数尤度の移動平均の速さ（max(1/(n+1), 率)）
    head_temperature: float = 1.0 # softmax の温度（SPEC-v3.2: 1）

    # --- v3.3 1.2: 類を除いた事前（leave-class-out）と、転移性能での混合 ---
    # mixture に入れるヘッド（"none" は必須）。X3 の結論（benchmarks/nn_transfer.py: どの条件でも mlp は最良でない）で、既定から mlp を外した。
    # mlp を入れたいときは mixture_heads=("mlp", "linear", "knn1", "none")（設定は残してある）。knnk も選べる。
    mixture_heads: Tuple[str, ...] = ("linear", "knn1", "none")
    head_retrain_every: int = 10  # 類を除いたモデルの再学習の周期（経験。想起 reattribute_every と同じ）
    head_fit_steps: int = 300     # 再学習の全バッチ勾配降下の回数（ベンチマークと共通）
    head_fit_lr: float = 0.5      # 再学習の学習率（同上）
    head_knn_smooth: float = 2.0  # knn の近傍の類の率の平滑（疑似計数。他の類を除いた全体の率へ寄せる）
    head_knn_k: int = 3           # knnk の k（距離重み）

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
    replay_size: int = 500        # v3.3: 再学習に使う直近の観測の数（v3.2 の head_lr / head_batches / head_batch_size / head_min_samples は廃止）

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
        if self.head_kind not in ("mixture", "mlp", "linear", "knn1", "knnk", "none"):
            raise ValueError(f"head_kind が不正です: {self.head_kind!r}")
        self.mixture_heads = tuple(self.mixture_heads)
        known = ("mlp", "linear", "knn1", "knnk", "none")
        if not set(self.mixture_heads) <= set(known) or len(set(self.mixture_heads)) != len(self.mixture_heads):
            raise ValueError(f"mixture_heads が不正です: {self.mixture_heads!r}")
        if "none" not in self.mixture_heads:
            raise ValueError("mixture_heads には none（一般の率）が必要です")
        if self.lc_max_causes < 1:
            raise ValueError("lc_max_causes は 1 以上です")
        if self.concern_direction not in ("sensitize", "habituate"):
            raise ValueError(f"concern_direction が不正です: {self.concern_direction!r}")

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
