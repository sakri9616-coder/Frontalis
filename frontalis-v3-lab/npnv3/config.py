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
    # --- v3.3 第1部の設計の改訂（オーケストレーター承認。結果を見た後の改訂。値は固定）。既定は第2部で spec_literal に戻した（CHANGELOG の M 節）---
    # 既定の選択は**現時点の設計上の選択**で、revised が数値的に壊れているという判定ではない（B1〜B3 = 状態の有界性・学習規則の整合性・分岐の説明可能性は、両方の設計で合格している）。
    # 戻した理由は二つだけ: (1) 事前に「故障」の基準と定めた E1（嫌悪の維持）が revised で不合格になった、(2) 条件づけ価値が行動に効く範囲（W1 の C だけの寄与・W2 の差・U2 の C(praise)）が狭まった。
    # "spec_literal": SPEC-v3.3 4.2 の式のとおり（lc_tau, lc_beta, lc_neg_threshold を使う。柔らかい事後で更新する。**既定**。E1・E2・E8 が合格、E3〜E6・E9 が不合格）。
    # "revised": 冪乗則の時間の重み・原因ごとの試行の履歴・文脈の集中度（最初の原因は広く、後の原因は狭く）・局所 MAP の学習・原因を推論する復位（実験設定。E6 が合格する）。
    lc_design: str = "spec_literal"
    lc_decay_d: float = 1.0        # 原因の事前 Σ_{i∈k} (t − t_i + 1)^(−d) の d（Gershman, Blei & Niv 2010 の距離依存 CRP）
    lc_beta_first: float = 5.0     # 最初に作られた原因（獲得）の文脈の Dirichlet の集中度（広い。未見の文脈にも確率を残す）
    lc_beta_later: float = 0.2     # 2 番目以降に作られた原因（意外さで作られる。消去など）の集中度（狭い。作られた文脈に結びつく。Bouton 2004 の簡約）

    # --- v3.3 第2部 2: 価値の再帰属（訂正 kind = "correction"。SPEC-v3.3 2.3）---
    w_reapp: float = 0.5           # 再評価の強さ。観測雑音 R_corr = R / w_reapp（全人格共通。R は欲求ごとに R/a_k なので、実際は R/(a_k w_reapp)）
    retroactive_value: bool = False  # アブレーション: 価値の更新を、訂正後の事実で最初から再生する（自動消去の対照）
    no_reappraisal: bool = False   # アブレーション: 訂正で価値を一切変えない（信念は訂正する）
    recall_after_correction: bool = False  # 実験設定（SPEC にない。既定 OFF）: 訂正の後に想起（帰属のやり直し）の掃引を 1 回行う。訂正が信念を直した直後は、責任 φ が逐次の推定のまま（CHANGELOG の Y1）

    # --- v3.4 第4節（SPEC-v3.4 4.2〜4.3, 4.6）: 訂正の再評価の規則 ---
    # "v33"（既定）: v3.3 の規則（w_reapp のカルマン更新 1 回）。v3.3 の挙動をビット単位で保つ。
    # "v34": M1（含意された変化 D_i を、情報源の信用 κ̄_s ともっともらしさ LLR_i で重み付けた g_i D_i ずつ、事後共分散に沿って W に吸収する。g_i ∈ (0, 1)）と
    #        M2（κ̄_s は、後の直接の経験での検証の出来事だけで形成される）。M3（罪悪感・後悔・τ・訂正を時計の出来事にする）は Stage 3 で、ここには無い。
    # "v34_or": C-OR（吸収率に上限がない: g^OR = σ(·)(1 + λ_g G_e)。過剰な反省を、経験史に依存する性質として扱う。故障の検査は F1・F2・F3 だけ）。
    reappraisal_mode: str = "v33"
    lambda_g: float = 0.25         # C-OR の λ_g（SPEC-v3.4 4.6）
    kappa_alpha0: float = 1.0      # 情報源の信用 Beta(α, β) の事前（SPEC-v3.4 4.2 手順 3: Beta(1, 1)）
    kappa_beta0: float = 1.0
    g_logit_clip: float = 30.0     # σ の引数の数値の安全のための制限（|z| > 30 では σ が 1 または 0 に丸まり (0, 1) を保てない）。台帳には制限前の z も残す

    # --- v3.3 第2部 3: 選択の正当化 J（二択 kind = "choice"。SPEC-v3.3 3.3, 3.4）---
    eta_j: float = 0.05            # 迷った選択での J の動き η_J
    sigma_conflict: float = 0.1    # 葛藤の強さ c = exp(−(ΔQ/σ_c)²) の σ_c
    j_max: float = 0.2             # J の上限 J_max（J ← J_max tanh(J/J_max)）
    lambda_j: float = 0.02         # 毎経験の J の減衰 λ_J
    kappa_ag: float = 1.0          # 主体性の重み κ_ag: a の更新の速さ η_a (1 + κ_ag · chosen)。chosen = 自分で選んだ経験（optional を受けた、または二択で選んだ）
    no_justification: bool = False # アブレーション: J を作らない・Q に入れない

    # --- v3.3 第2部 3（SPEC-v3.3 第8節）: 正当化 J の実験条件 ---
    # "orig"（既定。Z-orig = 第3節の設計そのまま: 毎経験の減衰 λ_J、上限 J_max、葛藤による自己制限。Z1〜Z7 を事前登録どおりに判定する）|
    # "persist"（Z-persist: 強制的な減衰なし。J は記録された出来事 spread・reaffirm・counter・overwhelm でだけ変わる。上限 J_num は数値の安全のためだけ）|
    # "persist_with_decay"（persist に毎経験の減衰 λ_J を足した変種）。
    # 正当化が持続する・反証に抵抗する・選択で自己強化されることは、それ自体では故障ではない。故障は数値的な故障（有界でない・非数・規則違反・説明できない変化）だけ（B1〜B3、P1〜P7）。
    justification_mode: str = "orig"
    eta_r: float = 0.03            # persist: reaffirm（再び選ぶ）の +η_r (1 − |J|/J_num) と counter（反する選択）の −η_r |J|
    j_num: float = 2.0             # persist: 数値の安全のための上限（心理的な上限ではない）
    rho_dis: float = 0.5           # persist: 反証への抵抗。J > 0 の選択肢を選んで害を受けたとき、人モデルの負の教師の ρ_dis tanh(J) を場面（状況モデル）へ回す
    eta_ov: float = 1.0            # persist: overwhelm（圧倒的な反証）で正の J を減らす割合 = min(η_ov × 超えた分, 1)（乗算型。SPEC に比例の係数の指定なし。試走で加算型だと 1 回の害で J が全部消えたので乗算型・η_ov = 1 に決めた。P 系の実行前）
    record_decisions: bool = False # True のとき、経験ごとの離散的な決定（接近・回避、二択の選択、原因の割当、復位）を npn.decisions に残す（B1 の分岐点の特定）。数値に影響しない

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
        if self.lc_design not in ("revised", "spec_literal"):
            raise ValueError(f"lc_design が不正です: {self.lc_design!r}")
        if self.justification_mode not in ("orig", "persist", "persist_with_decay"):
            raise ValueError(f"justification_mode が不正です: {self.justification_mode!r}")
        if self.concern_direction not in ("sensitize", "habituate"):
            raise ValueError(f"concern_direction が不正です: {self.concern_direction!r}")
        if self.reappraisal_mode not in ("v33", "v34", "v34_or"):
            raise ValueError(f"reappraisal_mode が不正です: {self.reappraisal_mode!r}")

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
