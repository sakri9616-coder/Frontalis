"""信念ネット: スコープ帰属つき階層ベイズ（Beta–Bernoulli、経験ベイズ近似）＋ニューラル事前ヘッド。

予測対象（SPEC 4.1）
  - 参加者 e が型 k の結果を起こす確率  P(y_{e,k}=1 | e, class(e), context)
  - 文脈 c で自分・世界から型 k の結果が起きる確率  P(y_{c,k}=1 | c)

階層（SPEC 4.2）
  人の層: 個体 ind ← 類 cls ← 一般 gen          文脈の層: 文脈 ctx ← 世界 world
  個体  p_ind = (κ_e m_cls + n1) / (κ_e + n)
  類    m_cls = (κ_c prior_cls + Σ w_i r_i) / (κ_c + Σ w_i),  w_i = n_i/(n_i+κ_e),  r_i = n1_i/n_i
  一般  m_gen = (κ_g E[prior] + Σ_cls w_cls r_cls) / (κ_g + Σ_cls w_cls),  w_cls = W_cls/(W_cls+κ_c),  r_cls = Σ w_i r_i / W_cls
  類の有効標本数は個体の数で上限が決まる（W_cls = Σ w_i < 個体数）。

v3.1 の変更
  1.5 事前ヘッドの利用: prior_cls = ρ_h · head + (1 − ρ_h) · m_gen,  ρ_h = n_head/(n_head + 30)（50 件の解禁を廃止）。
      head_kind ∈ {mlp, linear, knn1, none}（基線との比較用）。
  1.7 κ_e を、格子（0.3〜50 の対数 12 点）上の Beta–Binomial 周辺尤度の事後平均で推定（個体が 2 人未満の類は κ_e = 2）。
      設計の改訂（オーケストレーター承認。Kemp, Perfors & Tenenbaum 2007）: 類全体で 1 つではなく、（類 × 結果型）ごとに推定する。
      どの型が個体間でどれだけばらつくかは型ごとに違う（誰でも同じ「つながり」と、人で大きく違う「手柄を取る」）。
      データの少ない型は、類全体の κ（全型をまとめた周辺尤度）へ縮約する: 型ごとの log κ の事後 ∝ 型の周辺尤度 × N(log κ_類, σ²)。
  1.4 曖昧な結果の型は観測なし（マスク）: 解釈が信念を自己訓練するループを作らない。

v3.2 の変更（SPEC-v3.2 1.4）: 事前ヘッドの重みは予測性能で決める
  head_kind = "mixture"（既定）: ヘッド h ∈ {mlp, linear, knn1, none（= 一般の率 m_gen）} ごとに、新しい観測を見る前の予測の対数尤度
  （参加者の観測 1 件 = 観測されたセルの対数尤度の和）を移動平均で記録する（前向き評価）。類の事前は、各ヘッドの予測を
  softmax(移動平均した対数尤度 / 温度 1) で重み付けて混ぜる。ベイズのモデル平均の簡約。NN は、実際に予測が良いときにだけ寄与する。
  観測がまだ 0 件のとき（どのヘッドにも実績がない）は m_gen。単一ヘッド（mlp / linear / knn1 / none）は v3.1 の ρ_h の混合のまま（基線）。

v3.3 の変更（SPEC-v3.3 1.2）: 類を除いた事前（leave-class-out）と、転移性能での混合
  類 c の事前平均を出すヘッドは、c 自身の観測を使わない: knn は c を除いた近傍、全体の率（none）は c を除いたもの、mlp / linear は
  c を除いたデータで学び直したモデル。未見の類（データがない）には全データのモデルを使う。モデルは head_retrain_every（= 10）経験ごとに、
  初期値から全バッチ勾配降下で学び直す（類は高々数十。乱数を使わない）。knn・none は観測の生の計数（y とマスクだけ。責任 φ は入らない）を使うので、
  類 c の観測を書き換えても c の事前（ヘッドの出力）は厳密に変わらない（X1）。
  混合の重みは転移性能で決める: 各ヘッドが、類を除いたモデルで、その類の観測を予測した対数尤度（類単位の交差検証）の移動平均の softmax。
  v3.2 の前向きの対数尤度は階層全体（自分の類の計数を含む）で測っていたので、自分の類を使うヘッドが有利だった。既定のヘッドは X3 の結論で mlp を外した。

設計の改訂 2（オーケストレーター承認）: 想起・再帰属（docs/04 第5節 手順 7）
  参加者が行為者の観測（個体, 文脈, 型ごとの y・マスク・割り当てた φ）を記録として保持する。K_r = 10 経験ごとに、全記録を 1 回掃引する:
  各観測の現在の寄与を計数から引き（retract）、今の信念（π と単位の率）で φ を計算し直し、足し直す（reassign）。オンライン EM の 1 ステップ。
  引く対象は自分の寄与だけ（leave-one-out）なので、自分の割り当てが自分の説明を強める自己強化はない。1 観測の寄与の和は常に 1、計数の総和は不変。
  掃引内の更新は同時（全観測の φ を掃引開始時の計数から求め、計数は記録から作り直す）。価値ネットの教師の分割は再帰属しない（非対称）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from . import heads as H
from .config import Config
from .innate import N_TYPES, OUTCOME_TYPES
from .lexicon import EMB_DIM, EmbeddingStore
from .mlp import MLP, LogReg

_P_MIN = 1e-4
_P_MAX = 1.0 - 1e-4
HEAD_NAMES = ("mlp", "linear", "knn1", "knnk", "none")     # ヘッドの名前の一覧（mixture に入れるのは cfg.mixture_heads。none = 他の類をまとめた率）
TRAINED_HEADS = ("mlp", "linear")                            # 学習するヘッド（他は計数から計算）


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def lgamma_np(x: np.ndarray) -> np.ndarray:
    """ベクトル化した log Γ(x)（x ≥ 0）。8 回の漸化式 ＋ スターリング級数。math.lgamma と 1e-10 以内で一致。"""
    x = np.asarray(x, dtype=np.float64)
    prod = np.ones_like(x)
    y = x
    for _ in range(8):
        prod = prod * y
        y = y + 1.0
    inv = 1.0 / y
    inv2 = inv * inv
    series = inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0)))
    return (y - 0.5) * np.log(y) - y + 0.9189385332046727 + series - np.log(prod)


def _loglik_grid(N1: np.ndarray, N0: np.ndarray, prior_a: float, prior_b: float, grid: np.ndarray):
    """格子の各 κ での Beta–Binomial 周辺尤度の対数（個体について和、型ごと）。形は (格子, 正の計数のある型)。
    正の計数が 1 つもない型は個体差の情報がほぼないので除く（active でない）。"""
    active = N1.sum(axis=0) > 1e-9
    if not active.any():
        return None, active
    a1, a0 = N1[:, active], N0[:, active]
    t1, t0 = a1.sum(axis=0), a0.sum(axis=0)
    m = (t1 + prior_a) / (t1 + t0 + prior_a + prior_b)           # 類の率（型ごと）
    kap = grid[:, None, None]
    al = kap * m[None, None, :]
    be = kap * (1.0 - m)[None, None, :]
    ll = (lgamma_np(a1[None] + al) + lgamma_np(a0[None] + be) - lgamma_np(a1[None] + a0[None] + kap)
          - (lgamma_np(al) + lgamma_np(be) - lgamma_np(kap + 0.0 * al))).sum(axis=1)
    return ll, active


def estimate_kappa(N1: np.ndarray, N0: np.ndarray, prior_a: float, prior_b: float, grid: np.ndarray) -> Optional[float]:
    """類の個体差の集中度 κ_e を、格子上の Beta–Binomial 周辺尤度（弱い対数一様事前）の事後平均で推定する（全型をまとめた、類全体の値）。
    N1, N0: (個体, 型) の（責任で重み付けた）計数。情報がなければ None。"""
    ll, _ = _loglik_grid(N1, N0, prior_a, prior_b, grid)
    if ll is None:
        return None
    tot = ll.sum(axis=1)
    w = np.exp(tot - tot.max())
    return float((w * grid).sum() / w.sum())


def estimate_kappa_by_type(N1: np.ndarray, N0: np.ndarray, prior_a: float, prior_b: float, grid: np.ndarray,
                           sigma: float) -> Optional[Tuple[float, np.ndarray]]:
    """（類 × 結果型）ごとの κ_e（設計の改訂。Kemp, Perfors & Tenenbaum 2007 の階層の超仮説）。
    戻り値は (類全体の κ, 型ごとの κ（型数,）)。
      類全体の κ: 全型をまとめた周辺尤度の事後平均（estimate_kappa と同じ）。
      型 k の κ_k: log κ の事後 ∝ 型 k だけの周辺尤度 × N(log κ_類, σ²)（格子上で正規化し、κ の事後平均）。
        データが少ない型は事前（類全体の値）に縮約し、個体間で大きくばらつく型は小さく、誰でも同じ型は大きくなる。
      正の計数が 1 つもない型は類全体の κ。"""
    ll, active = _loglik_grid(N1, N0, prior_a, prior_b, grid)
    if ll is None:
        return None
    tot = ll.sum(axis=1)
    wc = np.exp(tot - tot.max())
    kc = float((wc * grid).sum() / wc.sum())
    kvec = np.full(N1.shape[1], kc)
    logprior = -0.5 * ((np.log(grid) - np.log(kc)) / sigma) ** 2
    lp = ll + logprior[:, None]
    w = np.exp(lp - lp.max(axis=0, keepdims=True))
    kvec[active] = (w * grid[:, None]).sum(axis=0) / w.sum(axis=0)
    return kc, kvec


class Prediction:
    """ある（文脈, 参加者）に対する信念の予測。状態は変えない。"""
    __slots__ = ("p_person", "p_ind", "p_ctxp", "pi_p", "pi_c", "p_sw", "var_person", "var_sw",
                 "known", "head_used", "n_ind", "m_cls", "rho")

    def __init__(self, **kw: Any):
        for k, v in kw.items():
            setattr(self, k, v)


class BeliefNet:
    def __init__(self, cfg: Config, emb: EmbeddingStore, rng: Optional[np.random.Generator]):
        self.cfg = cfg
        self.emb = emb
        K = N_TYPES
        self.ind_ids: List[str] = []
        self.ind_cls: List[str] = []
        self.ind_idx: Dict[str, int] = {}
        self.N1 = np.zeros((0, K))
        self.N0 = np.zeros((0, K))
        self.ctx_names: List[str] = []
        self.ctx_idx: Dict[str, int] = {}
        self.CP1 = np.zeros((0, K))   # 文脈で「人が k を起こした」計数（責任 φ_ctx で重み付け）
        self.CP0 = np.zeros((0, K))
        self.CS1 = np.zeros((0, K))   # 文脈で「自分・世界が k を起こした」計数
        self.CS0 = np.zeros((0, K))
        self.pi = np.ones((K, 2))     # π_k の計数: 列 0 = 人のせい、列 1 = 場面のせい（Dirichlet(1,1)）
        self.mag = np.full(K, cfg.mag_init)
        self.mag_n = np.zeros(K)
        self.n_obs = 0                # ヘッドの学習標本数 n_head（参加者の観測の総数）
        self.prior_mean = cfg.prior_a / (cfg.prior_a + cfg.prior_b)
        self.head: Optional[Union[MLP, LogReg]] = None       # 主たる学習ヘッド（全データのモデル。mlp があれば mlp、なければ linear）
        self.heads: Dict[str, Union[MLP, LogReg]] = {}       # 学習するヘッド（名前 → 全データのモデル。未見の類に使う）
        self.heads_init: Dict[str, Union[MLP, LogReg]] = {}  # 学び直しの出発点（初期値。mlp の重みの初期値だけが乱数）
        self.loo: Dict[str, List[Union[MLP, LogReg]]] = {}   # 類 c を除いたデータで学び直したモデル（名前 → 生の計数の類の添字順のリスト）
        self.ll_ema = np.zeros(len(HEAD_NAMES))              # ヘッドごとの転移性能（類を除いた予測の対数尤度）の移動平均（HEAD_NAMES の順。mixture に入らないヘッドは 0）
        self.ll_cnt = 0                                      # 移動平均に入れた経験の数（参加者のいる経験）
        self.rb_x = np.zeros((cfg.replay_size, 2 * EMB_DIM))
        self.rb_y = np.zeros((cfg.replay_size, K))
        self.rb_m = np.ones((cfg.replay_size, K))
        self.rb_cls = np.zeros(cfg.replay_size, dtype=np.int64)    # v3.3: 各行の類（rawcls の添字）と文脈（ctx_names の添字）
        self.rb_ctxi = np.zeros(cfg.replay_size, dtype=np.int64)
        self.rb_len = 0
        self.rb_ptr = 0
        self.rawcls: List[str] = []                          # v3.3: 観測のある類。類ごとの観測の生の計数（y とマスクだけ。φ は入らない）
        self.rawcls_idx: Dict[str, int] = {}
        self.raw_n1 = np.zeros((0, K))
        self.raw_n0 = np.zeros((0, K))
        self.exp_count = 0            # 想起の掃引・ヘッドの学び直しの周期のための、学習した経験の数
        self.rec_n = 0                # 想起のための観測の記録（参加者が行為者の観測）
        self.rec_ind = np.zeros(0, dtype=np.int64)
        self.rec_ctx = np.zeros(0, dtype=np.int64)
        self.rec_y = np.zeros((0, K))
        self.rec_m = np.zeros((0, K))
        self.rec_phi = np.zeros((0, K))
        self.rec_exp = np.zeros(0, dtype=np.int64)          # v3.3 第2部: 記録の行が属する経験の通し番号（学習した経験の順。訂正が参照先の観測を見つける）
        self.rec_ser = np.zeros(0, dtype=np.int64)          # 同: 観測の通し番号（n_obs。リプレイ緩衝の行を見つける）
        self.rb_ser = np.full(cfg.replay_size, -1, dtype=np.int64)    # リプレイ緩衝の各行の観測の通し番号
        self.sw_n = 0                                       # 自分・世界が起こす結果の観測の記録（経験ごとに 1 行。訂正で取り消す）
        self.sw_exp = np.zeros(0, dtype=np.int64)
        self.sw_ctx = np.zeros(0, dtype=np.int64)
        self.sw_y = np.zeros((0, K))
        self.sw_m = np.zeros((0, K))
        self._kcache: Dict[str, Any] = {}          # κ_e の記憶（純粋な関数の記憶。状態ではない）
        self._agg_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None   # 類の集計の記憶（同上。計数が変わるたびに捨てる）
        self._grid = np.exp(np.linspace(np.log(cfg.kappa_grid_lo), np.log(cfg.kappa_grid_hi), cfg.kappa_grid_n))
        bias = np.full(K, _logit(self.prior_mean))
        trained = self.trained_names()
        if "mlp" in trained:
            if rng is None:
                raise ValueError("ヘッドの初期化には乱数が必要です")
            self.heads["mlp"] = MLP(2 * EMB_DIM, cfg.head_hidden, K, rng, out_bias=bias)
        if "linear" in trained:
            self.heads["linear"] = LogReg(2 * EMB_DIM, K, out_bias=bias)
        for nm, hd in self.heads.items():
            self.heads_init[nm] = hd.copy()
            self.loo[nm] = []
        self.head = self.heads.get("mlp", self.heads.get("linear"))

    # ------------------------------------------------------------------ ヘッドの構成
    def active_names(self) -> Tuple[str, ...]:
        """事前に使うヘッド。mixture は cfg.mixture_heads（HEAD_NAMES の順）、単一ヘッドはそのヘッドだけ。"""
        kind = self.cfg.head_kind
        if kind == "mixture":
            return tuple(h for h in HEAD_NAMES if h in self.cfg.mixture_heads)
        return (kind,)

    def trained_names(self) -> Tuple[str, ...]:
        return tuple(h for h in self.active_names() if h in TRAINED_HEADS)

    # ------------------------------------------------------------------ 登録
    def _add_individual(self, pid: str, cls: str) -> int:
        i = len(self.ind_ids)
        self.ind_ids.append(pid)
        self.ind_cls.append(cls)
        self.ind_idx[pid] = i
        z = np.zeros((1, N_TYPES))
        self.N1 = np.vstack([self.N1, z])
        self.N0 = np.vstack([self.N0, z])
        return i

    def _add_context(self, name: str) -> int:
        i = len(self.ctx_names)
        self.ctx_names.append(name)
        self.ctx_idx[name] = i
        z = np.zeros((1, N_TYPES))
        self.CP1 = np.vstack([self.CP1, z]); self.CP0 = np.vstack([self.CP0, z])
        self.CS1 = np.vstack([self.CS1, z]); self.CS0 = np.vstack([self.CS0, z])
        return i

    def check_class(self, pid: str, cls: str) -> None:
        i = self.ind_idx.get(pid)
        if i is not None and self.ind_cls[i] != cls:
            raise ValueError(f"個体 {pid!r} の類が以前（{self.ind_cls[i]!r}）と異なります: {cls!r}")

    # ------------------------------------------------------------------ ヘッド
    def head_weights(self) -> np.ndarray:
        """mixture のヘッドの重み（HEAD_NAMES の順。mixture に入らないヘッドは 0）= softmax(転移性能の移動平均 / 温度)。
        観測が 0 件なら none に全部。単一ヘッドの種類では、そのヘッドへの one-hot（none は none へ。ρ_h による一般の率との混合は rho() 側）。"""
        names = HEAD_NAMES
        if self.cfg.head_kind != "mixture":
            return np.array([1.0 if nm == self.cfg.head_kind else 0.0 for nm in names])
        if self.n_obs == 0:
            return np.array([1.0 if nm == "none" else 0.0 for nm in names])
        act = np.array([nm in self.cfg.mixture_heads for nm in names])
        z = self.ll_ema / self.cfg.head_temperature
        e = np.where(act, np.exp(z - z[act].max()), 0.0)
        return e / e.sum()

    def head_weights_dict(self) -> Dict[str, float]:
        return {nm: float(w) for nm, w in zip(HEAD_NAMES, self.head_weights())}

    def rho(self) -> float:
        """事前に対するヘッドの寄与。単一ヘッド: ρ_h = n_head / (n_head + 30)（none なら 0）。
        mixture: 学習するヘッド・近傍ヘッドの重みの和（none の重みを除く）。観測が 0 件なら 0。"""
        kind = self.cfg.head_kind
        if kind == "none" or self.n_obs == 0:
            return 0.0
        if kind == "mixture":
            return float(1.0 - self.head_weights()[HEAD_NAMES.index("none")])
        return self.n_obs / (self.n_obs + self.cfg.rho_scale)

    def head_trained(self) -> bool:
        """互換: ヘッドが事前に寄与している（ρ_h > 0）か。v3.1 では 50 件の解禁はない。"""
        return self.rho() > 0.0

    # ------------------------------------------------------------------ 集計
    def _kappas(self, cls: str, rows: List[int]) -> Tuple[float, np.ndarray]:
        """類の κ_e: (類全体の値, 型ごとの値)。推定しない・個体が 2 人未満・情報がないときは固定の κ_e。"""
        cfg = self.cfg
        default = (cfg.kappa_e, np.full(N_TYPES, cfg.kappa_e))
        if not cfg.estimate_kappa or len(rows) < 2:
            return default
        v = self._kcache.get(cls)
        if v is None:
            if cfg.kappa_per_type:
                est = estimate_kappa_by_type(self.N1[rows], self.N0[rows], cfg.prior_a, cfg.prior_b, self._grid,
                                             cfg.kappa_prior_sigma)
            else:
                e = estimate_kappa(self.N1[rows], self.N0[rows], cfg.prior_a, cfg.prior_b, self._grid)
                est = None if e is None else (e, np.full(N_TYPES, e))
            v = default if est is None else est
            self._kcache[cls] = v
        return v

    def kappa_of(self, cls: str) -> float:
        """類全体の κ_e（全型をまとめた推定）。"""
        return self._kappas(cls, [i for i, c in enumerate(self.ind_cls) if c == cls])[0]

    def kappa_by_type(self, cls: str) -> np.ndarray:
        """（類 × 結果型）ごとの κ_e。"""
        return self._kappas(cls, [i for i, c in enumerate(self.ind_cls) if c == cls])[1]

    def _class_agg(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """類ごとの (W = Σ w_i, WR = Σ w_i r_i, その類の型ごとの κ_e)。計数が変わるまで同じ値なので記憶する（純粋な関数の記憶。状態ではない）。"""
        if self._agg_cache is None:
            self._agg_cache = self._class_agg_compute()
        return self._agg_cache

    def _class_agg_compute(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        groups: Dict[str, List[int]] = {}
        for i, c in enumerate(self.ind_cls):
            groups.setdefault(c, []).append(i)
        for c in sorted(groups):
            rows = groups[c]
            ke = self._kappas(c, rows)[1]
            N1, N0 = self.N1[rows], self.N0[rows]
            n = N1 + N0
            w = n / (n + ke[None, :])
            r = np.divide(N1, n, out=np.zeros_like(n), where=n > 0)
            out[c] = (w.sum(axis=0), (w * r).sum(axis=0), ke)
        return out

    def _gen_mean(self, agg: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        num = np.full(N_TYPES, cfg.kappa_g * self.prior_mean)
        den = np.full(N_TYPES, cfg.kappa_g)
        for c in sorted(agg):
            W, WR, _ = agg[c]
            rc = np.divide(WR, W, out=np.zeros_like(W), where=W > 1e-12)
            wc = W / (W + cfg.kappa_c)
            num = num + wc * rc
            den = den + wc
        return num / den, den - cfg.kappa_g      # 平均、有効標本数 Σ w_cls

    def _world_mean(self, C1: np.ndarray, C0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        n = C1 + C0
        w = n / (n + cfg.kappa_e)
        r = np.divide(C1, n, out=np.zeros_like(n), where=n > 0)
        num = cfg.kappa_g * self.prior_mean + (w * r).sum(axis=0)
        den = cfg.kappa_g + w.sum(axis=0)
        return num / den, w.sum(axis=0)

    # ------------------------------------------------------------------ 類を除いた事前（v3.3 1.2）
    def _raw_class(self, cls: str) -> int:
        """観測のある類の添字（なければ登録）。"""
        i = self.rawcls_idx.get(cls)
        if i is None:
            i = len(self.rawcls)
            self.rawcls.append(cls)
            self.rawcls_idx[cls] = i
            z = np.zeros((1, N_TYPES))
            self.raw_n1 = np.vstack([self.raw_n1, z])
            self.raw_n0 = np.vstack([self.raw_n0, z])
        return i

    def _model_for(self, nm: str, cls: str) -> Union[MLP, LogReg]:
        """類 cls の事前を出すときの学習ヘッド。既知の類（観測がある）には、その類を除いて学び直したモデル、
        未見の類（と、まだ学び直していない類）には全データのモデル。"""
        ci = self.rawcls_idx.get(cls)
        models = self.loo.get(nm, [])
        if ci is not None and ci < len(models):
            return models[ci]
        return self.heads[nm]

    def head_outputs(self, classes: List[str], ctx: Optional[str],
                     names: Optional[Tuple[str, ...]] = None) -> Dict[str, np.ndarray]:
        """各ヘッドが出す類の事前平均（行 = 類、形 (n, K)）。どのヘッドも、予測する類自身の観測を使わない（leave-class-out）。
        mlp / linear: その類を除いて学び直したモデル（未見の類は全データのモデル）。knn1 / knnk: その類を除いた近傍の類の率（観測の生の計数）。
        none: その類を除いた全体の率。状態は変えない。"""
        cfg = self.cfg
        names = self.active_names() if names is None else names
        n = len(classes)
        out: Dict[str, np.ndarray] = {}
        if any(nm in TRAINED_HEADS for nm in names):
            ce = self.emb.ctx(ctx)
        emb_tab = None
        for nm in names:
            if nm in TRAINED_HEADS:
                rows = [np.clip(self._model_for(nm, c).predict_proba(np.concatenate([self.emb.cls(c), ce]))[0], _P_MIN, _P_MAX)
                        for c in classes]
                out[nm] = np.array(rows).reshape(n, N_TYPES)
            elif nm in ("knn1", "knnk"):
                if emb_tab is None:
                    emb_tab = np.array([self.emb.cls(c) for c in self.rawcls]).reshape(len(self.rawcls), EMB_DIM)
                k = 1 if nm == "knn1" else cfg.head_knn_k
                out[nm] = np.array([H.knn_rate(emb_tab, self.raw_n1, self.raw_n0, self.emb.cls(c), self.rawcls_idx.get(c), k,
                                               cfg.kappa_g, self.prior_mean, cfg.head_knn_smooth) for c in classes]).reshape(n, N_TYPES)
            else:
                out[nm] = np.array([H.pooled_rate(self.raw_n1, self.raw_n0, self.rawcls_idx.get(c), cfg.kappa_g, self.prior_mean)
                                    for c in classes]).reshape(n, N_TYPES)
        return out

    def _priors(self, classes: List[str], ctx: Optional[str], m_gen: np.ndarray,
                agg: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        """類の事前平均 prior_cls（行 = 参加者）。
        mixture: Σ_h w_h prior_h（w は転移性能の softmax。prior_h は類を除いたヘッドの出力）。観測が 0 件なら m_gen。
        単一ヘッド: ρ_h · head + (1 − ρ_h) · m_gen（head は類を除いたヘッドの出力。none は m_gen）。"""
        kind = self.cfg.head_kind
        n = len(classes)
        if kind == "none" or n == 0 or self.n_obs == 0:
            return np.tile(m_gen, (n, 1)), False
        if kind == "mixture":
            hp = self.head_outputs(classes, ctx)
            w = self.head_weights()
            return sum(float(w[HEAD_NAMES.index(nm)]) * hp[nm] for nm in hp), True
        rho = self.rho()
        if rho <= 0.0:
            return np.tile(m_gen, (n, 1)), False
        head = self.head_outputs(classes, ctx, (kind,))[kind]
        return rho * head + (1.0 - rho) * m_gen[None, :], True

    # ------------------------------------------------------------------ 予測
    def predict(self, ctx: Optional[str], participants: List[Dict[str, str]],
                prior_override: Optional[np.ndarray] = None) -> Prediction:
        """prior_override: 類の事前平均（行 = 参加者）を差し替える（ヘッドごとの前向き評価用）。"""
        cfg = self.cfg
        K = N_TYPES
        ke0, kc, kg = cfg.kappa_e, cfg.kappa_c, cfg.kappa_g
        agg = self._class_agg()
        m_gen, W_gen = self._gen_mean(agg)
        v_gen = m_gen * (1 - m_gen) / (kg + W_gen + 1.0)
        # --- 文脈の層（人が起こす k / 自分・世界が起こす k）---
        mw_p, Ww_p = self._world_mean(self.CP1, self.CP0)
        mw_s, Ww_s = self._world_mean(self.CS1, self.CS0)
        vw_p = mw_p * (1 - mw_p) / (kg + Ww_p + 1.0)
        vw_s = mw_s * (1 - mw_s) / (kg + Ww_s + 1.0)
        ci = self.ctx_idx.get(ctx) if ctx is not None else None
        if ci is None:
            p_ctxp, n_c = mw_p.copy(), np.zeros(K)
            p_sw, n_s = mw_s.copy(), np.zeros(K)
        else:
            n_c = self.CP1[ci] + self.CP0[ci]
            p_ctxp = (ke0 * mw_p + self.CP1[ci]) / (ke0 + n_c)
            n_s = self.CS1[ci] + self.CS0[ci]
            p_sw = (ke0 * mw_s + self.CS1[ci]) / (ke0 + n_s)
        v_ctxp = p_ctxp * (1 - p_ctxp) / (ke0 + n_c + 1.0) + vw_p
        v_sw = p_sw * (1 - p_sw) / (ke0 + n_s + 1.0) + vw_s
        # --- スコープの事前 ---
        if cfg.no_scope:
            pi_p = np.ones(K); pi_c = np.zeros(K)
        else:
            tot = self.pi.sum(axis=1)
            pi_p = self.pi[:, 0] / tot
            pi_c = self.pi[:, 1] / tot
        # --- 人の層 ---
        nP = len(participants)
        p_person = np.zeros((nP, K)); p_ind = np.zeros((nP, K)); p_ctxp_all = np.zeros((nP, K))
        var_person = np.zeros((nP, K)); n_ind_all = np.zeros((nP, K)); m_cls_all = np.zeros((nP, K))
        known = []
        if prior_override is not None:
            priors, head_used = prior_override, True
        else:
            priors, head_used = self._priors([p["class"] for p in participants], ctx, m_gen, agg)
        for j, p in enumerate(participants):
            c = p["class"]
            W, WR, ke = agg.get(c, (np.zeros(K), np.zeros(K), np.full(K, ke0)))
            m_cls = (kc * priors[j] + WR) / (kc + W)
            v_cls = m_cls * (1 - m_cls) / (kc + W + 1.0)
            i = self.ind_idx.get(p["id"])
            if i is None:
                n_i = np.zeros(K)
                pi_ind = m_cls
                known.append(False)
            else:
                n_i = self.N1[i] + self.N0[i]
                pi_ind = (ke * m_cls + self.N1[i]) / (ke + n_i)
                known.append(True)
            v_ind = pi_ind * (1 - pi_ind) / (ke + n_i + 1.0)
            p_ind[j] = pi_ind
            p_ctxp_all[j] = p_ctxp
            p_person[j] = pi_p * pi_ind + pi_c * p_ctxp
            var_person[j] = v_ind + v_cls + v_gen + v_ctxp
            n_ind_all[j] = n_i
            m_cls_all[j] = m_cls
        return Prediction(p_person=np.clip(p_person, _P_MIN, _P_MAX), p_ind=p_ind, p_ctxp=p_ctxp_all,
                          pi_p=pi_p, pi_c=pi_c, p_sw=np.clip(p_sw, _P_MIN, _P_MAX),
                          var_person=var_person, var_sw=v_sw, known=known, head_used=head_used,
                          n_ind=n_ind_all, m_cls=m_cls_all, rho=self.rho())

    # ------------------------------------------------------------------ 責任 φ
    def responsibility(self, pred: Prediction, y: np.ndarray) -> np.ndarray:
        """観測 y_{e,k} の責任 φ_person（行 = 参加者、列 = 型）。φ_ctx = 1 - φ_person。"""
        if self.cfg.no_scope or y.shape[0] == 0:
            return np.ones_like(y)
        pi_ind = np.clip(pred.p_ind, _P_MIN, _P_MAX)
        pi_ctx = np.clip(pred.p_ctxp, _P_MIN, _P_MAX)
        bp = np.where(y > 0.5, pi_ind, 1.0 - pi_ind)
        bc = np.where(y > 0.5, pi_ctx, 1.0 - pi_ctx)
        a = pred.pi_p[None, :] * bp
        b = pred.pi_c[None, :] * bc
        return a / (a + b)

    # ------------------------------------------------------------------ 転移性能（v3.3 1.2。v3.2 1.4 の前向き評価を、類を除く形に直した）
    def _update_head_ll(self, ctx: str, participants: List[Dict[str, str]], y: np.ndarray, mask: np.ndarray) -> None:
        """各ヘッドの転移性能を移動平均に入れる。ヘッド h の予測 = その参加者の類を除いたヘッドの出力 prior_h(類)
        （その類の観測を使わずに、類の埋め込みだけから出した、型ごとの率。階層の個体・類の計数は通さない）。
        観測 1 件の対数尤度 = 観測されたセル（mask = 1）の y log p + (1 − y) log(1 − p) の和。参加者が複数いる経験では参加者で平均する。
        新しい観測を見る前の状態で計算する（前向き ＋ 類を除く = 類単位の交差検証）。未見の類には全データのモデルが使われる。
        この関数は計数にもヘッドにも触れない。"""
        if self.cfg.head_kind != "mixture" or not participants:
            return
        classes = [p["class"] for p in participants]
        hp = self.head_outputs(classes, ctx)
        ll = np.zeros(len(HEAD_NAMES))
        for nm, pr in hp.items():
            pr = np.clip(pr, _P_MIN, _P_MAX)
            cell = y * np.log(pr) + (1.0 - y) * np.log(1.0 - pr)
            ll[HEAD_NAMES.index(nm)] = float((mask * cell).sum(axis=1).mean())
        rate = max(1.0 / (self.ll_cnt + 1.0), self.cfg.head_ll_rate)
        self.ll_ema = self.ll_ema + rate * (ll - self.ll_ema)
        self.ll_cnt += 1

    # ------------------------------------------------------------------ ヘッドの学び直し（v3.3 1.2）
    def retrain_heads(self) -> None:
        """全データのモデルと、類ごとにその類を除いたモデルを、初期値から学び直す（全バッチ勾配降下。乱数を使わない）。
        データ = リプレイ緩衝（直近 replay_size 件の観測）。同じ（類, 文脈）の行を集計して重み付きで学ぶ。
        出力バイアスは、そのモデルが見るデータの全体の率（その類を除く）のロジットから始める。"""
        cfg = self.cfg
        names = self.trained_names()
        L = self.rb_len
        if not names or L == 0:
            return
        pairs = np.stack([self.rb_cls[:L], self.rb_ctxi[:L]], axis=1)
        uniq, first, inv = np.unique(pairs, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        G = uniq.shape[0]
        n1_g = np.zeros((G, N_TYPES)); n_g = np.zeros((G, N_TYPES))
        np.add.at(n1_g, inv, self.rb_m[:L] * self.rb_y[:L])
        np.add.at(n_g, inv, self.rb_m[:L])
        rows_g = np.bincount(inv, minlength=G).astype(np.float64)
        X_g = self.rb_x[:L][first]
        n_cls = len(self.rawcls)
        for nm in names:
            self.loo[nm] = []
        for e in [None] + list(range(n_cls)):
            sel = np.ones(G, dtype=bool) if e is None else (uniq[:, 0] != e)
            Xs, n1s, ns, total = X_g[sel], n1_g[sel], n_g[sel], float(rows_g[sel].sum())
            t = ns.sum(axis=0)
            base = (n1s.sum(axis=0) + cfg.kappa_g * self.prior_mean) / (t + cfg.kappa_g)
            for nm in names:
                if e is None:
                    model = self.heads[nm]
                else:
                    model = self.heads_init[nm].copy()
                H.fit_weighted(model, self.heads_init[nm], Xs, n1s, ns, total, base, cfg.head_fit_lr, cfg.head_fit_steps)
                if e is not None:
                    self.loo[nm].append(model)

    def loo_cv_loglik(self) -> Dict[str, Any]:
        """（X4 の報告用）リプレイ緩衝の観測について、類を除いて学び直したヘッドの、類単位の交差検証の対数尤度。
        ここで学び直す（self を変える）ので、呼ぶ側は deepcopy してから使う。戻り値は ヘッドごとの {観測あたりの平均, 類の平均, 類ごと}。"""
        self.retrain_heads()
        L = self.rb_len
        out: Dict[str, Any] = {}
        if L == 0:
            return out
        names = [nm for nm in HEAD_NAMES if nm in self.cfg.mixture_heads or nm == self.cfg.head_kind]
        per_cls: Dict[str, Dict[str, List[float]]] = {nm: {} for nm in names}
        for r in range(L):
            cls = self.rawcls[int(self.rb_cls[r])]
            ctx = self.ctx_names[int(self.rb_ctxi[r])]
            hp = self.head_outputs([cls], ctx, tuple(names))
            y, m = self.rb_y[r], self.rb_m[r]
            for nm in names:
                p = np.clip(hp[nm][0], _P_MIN, _P_MAX)
                per_cls[nm].setdefault(cls, []).append(float((m * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).sum()))
        for nm in names:
            allv = [v for vs in per_cls[nm].values() for v in vs]
            out[nm] = {"per_observation": float(np.mean(allv)),
                       "per_class_mean": float(np.mean([np.mean(vs) for vs in per_cls[nm].values()])),
                       "per_class": {c: float(np.mean(vs)) for c, vs in sorted(per_cls[nm].items())}}
        return out

    # ------------------------------------------------------------------ 更新
    def update(self, ctx: str, participants: List[Dict[str, str]], y_person: np.ndarray, y_mask: np.ndarray,
               y_sw: np.ndarray, sw_mask: np.ndarray, phi: np.ndarray, outcomes: List[Dict[str, Any]],
               rng: np.random.Generator, exp_serial: int = -1) -> Dict[str, Any]:
        """計数 → π → 型の大きさ → リプレイ緩衝・類ごとの生の計数 → （周期ごとに）類を除いたヘッドの学び直し。
        y_mask = 0 のセル（曖昧な結果の型）は観測なしとして、計数にも π にもヘッドの学習にも入れない。"""
        cfg = self.cfg
        n_obs0 = self.n_obs                                            # この経験の観測の通し番号は n_obs0 + j（参加者 j）
        self._update_head_ll(ctx, participants, y_person, y_mask)      # 前向き評価: 更新の前の状態で
        self._agg_cache = None
        ci = self.ctx_idx.get(ctx)
        if ci is None:
            ci = self._add_context(ctx)
        for j, p in enumerate(participants):
            i = self.ind_idx.get(p["id"])
            if i is None:
                i = self._add_individual(p["id"], p["class"])
            m = y_mask[j]
            if not (cfg.no_scope or cfg.no_reattribution):
                self._rec_append(i, ci, y_person[j], m, phi[j], exp_serial, n_obs0 + j)
            self.N1[i] += m * phi[j] * y_person[j]
            self.N0[i] += m * phi[j] * (1.0 - y_person[j])
            self.CP1[ci] += m * (1.0 - phi[j]) * y_person[j]
            self.CP0[ci] += m * (1.0 - phi[j]) * (1.0 - y_person[j])
            if not cfg.no_scope:
                self.pi[:, 0] += m * phi[j]
                self.pi[:, 1] += m * (1.0 - phi[j])
            self._kcache.pop(p["class"], None)
        self.CS1[ci] += sw_mask * y_sw
        self.CS0[ci] += sw_mask * (1.0 - y_sw)
        self._sw_append(exp_serial, ci, y_sw, sw_mask)
        for o in outcomes:                       # 型の大きさ: 明確な結果だけ
            if "type" not in o:
                continue
            k = OUTCOME_TYPES.index(o["type"])
            rate = max(1.0 / (self.mag_n[k] + 1.0), cfg.mag_rate)
            self.mag[k] += rate * (o["magnitude"] - self.mag[k])
            self.mag_n[k] += 1.0
        ce = self.emb.ctx(ctx)
        for j, p in enumerate(participants):
            rc = self._raw_class(p["class"])
            self.raw_n1[rc] += y_mask[j] * y_person[j]
            self.raw_n0[rc] += y_mask[j] * (1.0 - y_person[j])
            self.rb_x[self.rb_ptr] = np.concatenate([self.emb.cls(p["class"]), ce])
            self.rb_y[self.rb_ptr] = y_person[j]
            self.rb_m[self.rb_ptr] = y_mask[j]
            self.rb_cls[self.rb_ptr] = rc
            self.rb_ctxi[self.rb_ptr] = ci
            self.rb_ser[self.rb_ptr] = self.n_obs
            self.rb_ptr = (self.rb_ptr + 1) % cfg.replay_size
            self.rb_len = min(self.rb_len + 1, cfg.replay_size)
            self.n_obs += 1
        self.exp_count += 1
        retrained = False
        if cfg.head_retrain_every > 0 and self.exp_count % cfg.head_retrain_every == 0 and self.trained_names():
            self.retrain_heads()          # 類を除いたモデルの学び直し（想起と同じ周期）
            retrained = True
        # 想起: K_r 経験ごとに、全記録を 1 回掃引する
        recall = None
        if cfg.reattribute_every > 0 and self.exp_count % cfg.reattribute_every == 0:
            recall = self.reattribute()
        return {"head_loss": None, "recall": recall, "retrained": retrained}

    # ------------------------------------------------------------------ 想起・再帰属（設計の改訂 2）
    def _rec_append(self, i: int, c: int, y: np.ndarray, m: np.ndarray, phi: np.ndarray, exp_serial: int = -1, ser: int = -1) -> None:
        n = self.rec_n
        if n == self.rec_ind.shape[0]:
            cap = max(64, 2 * n)
            K = N_TYPES
            self.rec_ind = np.concatenate([self.rec_ind, np.zeros(cap - n, dtype=np.int64)])
            self.rec_ctx = np.concatenate([self.rec_ctx, np.zeros(cap - n, dtype=np.int64)])
            self.rec_exp = np.concatenate([self.rec_exp, np.zeros(cap - n, dtype=np.int64)])
            self.rec_ser = np.concatenate([self.rec_ser, np.zeros(cap - n, dtype=np.int64)])
            for name in ("rec_y", "rec_m", "rec_phi"):
                setattr(self, name, np.vstack([getattr(self, name), np.zeros((cap - n, K))]))
        self.rec_ind[n], self.rec_ctx[n] = i, c
        self.rec_exp[n], self.rec_ser[n] = exp_serial, ser
        self.rec_y[n], self.rec_m[n], self.rec_phi[n] = y, m, phi
        self.rec_n = n + 1

    def _sw_append(self, exp_serial: int, c: int, y: np.ndarray, m: np.ndarray) -> None:
        n = self.sw_n
        if n == self.sw_exp.shape[0]:
            cap = max(64, 2 * n)
            self.sw_exp = np.concatenate([self.sw_exp, np.zeros(cap - n, dtype=np.int64)])
            self.sw_ctx = np.concatenate([self.sw_ctx, np.zeros(cap - n, dtype=np.int64)])
            for name in ("sw_y", "sw_m"):
                setattr(self, name, np.vstack([getattr(self, name), np.zeros((cap - n, N_TYPES))]))
        self.sw_exp[n], self.sw_ctx[n] = exp_serial, c
        self.sw_y[n], self.sw_m[n] = y, m
        self.sw_n = n + 1

    # ------------------------------------------------------------------ 訂正（v3.3 第2部 2.3 手順 1）: 参照先の経験の観測を取り消し、訂正後の観測を加える
    def can_correct(self) -> bool:
        """訂正は、観測の記録（想起の記録）を使う。スコープ帰属なし・想起なしの設定では記録がない。"""
        return not (self.cfg.no_scope or self.cfg.no_reattribution)

    def withdraw_experience(self, serial: int) -> Dict[str, Any]:
        """経験 serial の観測（参加者が行為者の観測、自分・世界が起こす結果の観測、類ごとの生の計数）を取り消す。
        取り消した行の古い値を返す（install_experience に渡す）。信念は「その経験がなかった」状態になる（計数は記録から作り直す）。"""
        R = self.rec_n
        rows = np.nonzero(self.rec_exp[:R] == serial)[0]
        old = {"rows": rows, "ind": self.rec_ind[rows].copy(), "y": self.rec_y[rows].copy(), "m": self.rec_m[rows].copy(), "phi": self.rec_phi[rows].copy()}
        for r in range(rows.size):
            rc = self._raw_class(self.ind_cls[int(old["ind"][r])])
            self.raw_n1[rc] -= old["m"][r] * old["y"][r]
            self.raw_n0[rc] -= old["m"][r] * (1.0 - old["y"][r])
        sw = np.nonzero(self.sw_exp[: self.sw_n] == serial)[0]
        old["sw_rows"] = sw
        old["sw_y"], old["sw_m"], old["sw_ctx"] = self.sw_y[sw].copy(), self.sw_m[sw].copy(), self.sw_ctx[sw].copy()
        for q in range(sw.size):
            ci = int(old["sw_ctx"][q])
            self.CS1[ci] -= old["sw_m"][q] * old["sw_y"][q]
            self.CS0[ci] -= old["sw_m"][q] * (1.0 - old["sw_y"][q])
        self.sw_m[sw] = 0.0
        self.rec_m[rows] = 0.0
        self._rebuild_counts()
        return old

    def install_experience(self, old: Dict[str, Any], y_person: np.ndarray, y_mask: np.ndarray, phi: np.ndarray,
                           y_sw: np.ndarray, sw_mask: np.ndarray) -> None:
        """withdraw_experience で取り消した行に、訂正後の観測 y・マスク・責任 φ を入れる（記録・生の計数・リプレイ緩衝・計数）。
        φ は、取り消した後の信念で呼び出し側が求める。mag（型ごとの大きさの移動平均）は取り消せないので触れない。"""
        rows = old["rows"]
        for r in range(rows.size):
            row = int(rows[r])
            self.rec_y[row], self.rec_m[row], self.rec_phi[row] = y_person[r], y_mask[r], phi[r]
            rc = self._raw_class(self.ind_cls[int(old["ind"][r])])
            self.raw_n1[rc] += y_mask[r] * y_person[r]
            self.raw_n0[rc] += y_mask[r] * (1.0 - y_person[r])
            slot = np.nonzero(self.rb_ser == self.rec_ser[row])[0]
            if slot.size:
                self.rb_y[slot[0]] = y_person[r]
                self.rb_m[slot[0]] = y_mask[r]
        for q in range(old["sw_rows"].size):
            row = int(old["sw_rows"][q])
            ci = int(old["sw_ctx"][q])
            self.sw_y[row], self.sw_m[row] = y_sw, sw_mask
            self.CS1[ci] += sw_mask * y_sw
            self.CS0[ci] += sw_mask * (1.0 - y_sw)
        self._rebuild_counts()
        if self.trained_names():
            self.retrain_heads()          # 以後の予測は訂正後の信念（リプレイ緩衝・生の計数を直したので、類を除いたモデルを学び直す）

    def _rebuild_counts(self) -> None:
        """計数と π を、記録（観測 y・マスク・φ）から作り直す。引いて足す更新の浮動小数点の誤差を残さない。"""
        R = self.rec_n
        ind, ctx = self.rec_ind[:R], self.rec_ctx[:R]
        y, m, phi = self.rec_y[:R], self.rec_m[:R], self.rec_phi[:R]
        wp, wc = m * phi, m * (1.0 - phi)
        ni, nc = len(self.ind_ids), len(self.ctx_names)
        for k in range(N_TYPES):
            self.N1[:, k] = np.bincount(ind, weights=wp[:, k] * y[:, k], minlength=ni)
            self.N0[:, k] = np.bincount(ind, weights=wp[:, k] * (1.0 - y[:, k]), minlength=ni)
            self.CP1[:, k] = np.bincount(ctx, weights=wc[:, k] * y[:, k], minlength=nc)
            self.CP0[:, k] = np.bincount(ctx, weights=wc[:, k] * (1.0 - y[:, k]), minlength=nc)
        self.pi[:, 0] = 1.0 + wp.sum(axis=0)
        self.pi[:, 1] = 1.0 + wc.sum(axis=0)
        self._kcache = {}
        self._agg_cache = None

    def reattribute(self) -> Optional[Dict[str, float]]:
        """想起: 全記録について 1 回の掃引（retract → reassign）。オンライン EM の 1 ステップ（leave-one-out）。
        各観測の現在の寄与（人 φ、文脈 1−φ）を自分の単位の計数と π から引いた状態で、今の信念（個体の率・文脈の率・π）から
        φ を計算し直し、足し直す。1 観測の寄与の和は 1、計数の総和は変わらない。"""
        cfg = self.cfg
        R = self.rec_n
        if R == 0 or cfg.no_scope or cfg.no_reattribution:
            return None
        ind, ctx = self.rec_ind[:R], self.rec_ctx[:R]
        y, m, phi = self.rec_y[:R], self.rec_m[:R], self.rec_phi[:R]
        ke0 = cfg.kappa_e
        # 掃引の開始時点の類・世界の水準（掃引の間は固定）
        agg = self._class_agg()
        m_gen, _ = self._gen_mean(agg)
        mw_p, _ = self._world_mean(self.CP1, self.CP0)
        kc = cfg.kappa_c
        cls_names = [self.ind_cls[i] for i in ind]
        m_cls = np.zeros((R, N_TYPES))
        ke = np.zeros((R, N_TYPES))
        by_ctx: Dict[int, List[int]] = {}
        for r in range(R):
            by_ctx.setdefault(int(ctx[r]), []).append(r)
        classes = sorted(agg)
        for c_idx, rows in by_ctx.items():
            priors, _ = self._priors(classes, self.ctx_names[c_idx], m_gen, agg)
            table = {c: (kc * priors[j] + agg[c][1]) / (kc + agg[c][0]) for j, c in enumerate(classes)}
            for r in rows:
                m_cls[r] = table[cls_names[r]]
        for c in classes:
            sel = [r for r in range(R) if cls_names[r] == c]
            ke[sel] = agg[c][2][None, :]
        # retract: 自分の寄与だけを引く
        own_p, own_c = m * phi, m * (1.0 - phi)
        N1 = self.N1[ind] - own_p * y
        N0 = self.N0[ind] - own_p * (1.0 - y)
        C1 = self.CP1[ctx] - own_c * y
        C0 = self.CP0[ctx] - own_c * (1.0 - y)
        p_ind = np.clip((ke * m_cls + N1) / (ke + N1 + N0), _P_MIN, _P_MAX)
        p_ctx = np.clip((ke0 * mw_p[None, :] + C1) / (ke0 + C1 + C0), _P_MIN, _P_MAX)
        pi_p = (self.pi[:, 0][None, :] - own_p) / (self.pi.sum(axis=1)[None, :] - m)
        pi_p = np.clip(pi_p, 1e-6, 1.0 - 1e-6)
        # reassign
        bp = np.where(y > 0.5, p_ind, 1.0 - p_ind)
        bc = np.where(y > 0.5, p_ctx, 1.0 - p_ctx)
        a, b = pi_p * bp, (1.0 - pi_p) * bc
        new = np.where(m > 0.5, a / (a + b), phi)
        d = np.abs(new - phi)[m > 0.5]
        self.rec_phi[:R] = new
        self._rebuild_counts()
        piv = self.pi[:, 0] / self.pi.sum(axis=1)
        return {"n_records": float(R), "mean_abs_dphi": float(d.mean()) if d.size else 0.0,
                "max_abs_dphi": float(d.max()) if d.size else 0.0,
                "pi_person_min": float(piv.min()), "pi_person_max": float(piv.max())}

    # ------------------------------------------------------------------ 要約（CLI show 用）
    def summary(self) -> Dict[str, Any]:
        agg = self._class_agg()
        m_gen, _ = self._gen_mean(agg)
        out: Dict[str, Any] = {"n_individuals": len(self.ind_ids), "n_contexts": len(self.ctx_names),
                               "n_obs": self.n_obs, "head_kind": self.cfg.head_kind, "rho_h": self.rho(),
                               "head_trained": self.head_trained(), "head_weights": self.head_weights_dict(),
                               "mixture_heads": list(self.cfg.mixture_heads) if self.cfg.head_kind == "mixture" else [self.cfg.head_kind],
                               "pi_person": {t: float(self.pi[k, 0] / self.pi[k].sum()) for k, t in enumerate(OUTCOME_TYPES)},
                               "m_gen": {t: float(m_gen[k]) for k, t in enumerate(OUTCOME_TYPES)},
                               "class_rate": {}, "kappa_e": {}}
        kc = self.cfg.kappa_c
        for c, (W, WR, ke) in agg.items():
            m_cls = (kc * m_gen + WR) / (kc + W)
            out["class_rate"][c] = {t: float(m_cls[k]) for k, t in enumerate(OUTCOME_TYPES)}
            out["kappa_e"][c] = float(self.kappa_of(c))
            out.setdefault("kappa_e_by_type", {})[c] = {t: float(ke[k]) for k, t in enumerate(OUTCOME_TYPES)}
        return out

    # ------------------------------------------------------------------ 保存・復元
    def get_state(self) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        meta = {"ind_ids": list(self.ind_ids), "ind_cls": list(self.ind_cls), "ctx_names": list(self.ctx_names),
                "n_obs": int(self.n_obs), "rb_len": int(self.rb_len), "rb_ptr": int(self.rb_ptr),
                "head_kind": self.cfg.head_kind, "exp_count": int(self.exp_count), "rec_n": int(self.rec_n),
                "ll_cnt": int(self.ll_cnt), "rawcls": list(self.rawcls), "sw_n": int(self.sw_n),
                "loo_n": {nm: len(ms) for nm, ms in self.loo.items()}}
        arrays = {"b_N1": self.N1, "b_N0": self.N0, "b_CP1": self.CP1, "b_CP0": self.CP0, "b_CS1": self.CS1,
                  "b_CS0": self.CS0, "b_pi": self.pi, "b_mag": self.mag, "b_mag_n": self.mag_n,
                  "b_rb_x": self.rb_x, "b_rb_y": self.rb_y, "b_rb_m": self.rb_m,
                  "b_rec_ind": self.rec_ind[: self.rec_n], "b_rec_ctx": self.rec_ctx[: self.rec_n],
                  "b_rec_y": self.rec_y[: self.rec_n], "b_rec_m": self.rec_m[: self.rec_n],
                  "b_rec_phi": self.rec_phi[: self.rec_n], "b_ll_ema": self.ll_ema,
                  "b_rec_exp": self.rec_exp[: self.rec_n], "b_rec_ser": self.rec_ser[: self.rec_n], "b_rb_ser": self.rb_ser,
                  "b_sw_exp": self.sw_exp[: self.sw_n], "b_sw_ctx": self.sw_ctx[: self.sw_n],
                  "b_sw_y": self.sw_y[: self.sw_n], "b_sw_m": self.sw_m[: self.sw_n],
                  "b_rb_cls": self.rb_cls, "b_rb_ctxi": self.rb_ctxi, "b_raw_n1": self.raw_n1, "b_raw_n0": self.raw_n0}
        for nm, hd in self.heads.items():
            arrays.update(hd.get_arrays(f"b_head_{nm}_"))
            arrays.update(self.heads_init[nm].get_arrays(f"b_hinit_{nm}_"))
            for i, m in enumerate(self.loo[nm]):
                arrays.update(m.get_arrays(f"b_loo_{nm}_{i}_"))
        return meta, arrays

    def set_state(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
        self.ind_ids = list(meta["ind_ids"]); self.ind_cls = list(meta["ind_cls"])
        self.ind_idx = {p: i for i, p in enumerate(self.ind_ids)}
        self.ctx_names = list(meta["ctx_names"]); self.ctx_idx = {c: i for i, c in enumerate(self.ctx_names)}
        self.n_obs = int(meta["n_obs"]); self.rb_len = int(meta["rb_len"]); self.rb_ptr = int(meta["rb_ptr"])
        g = lambda k: np.array(arrays[k], dtype=np.float64)
        self.N1, self.N0 = g("b_N1"), g("b_N0")
        self.CP1, self.CP0, self.CS1, self.CS0 = g("b_CP1"), g("b_CP0"), g("b_CS1"), g("b_CS0")
        self.pi, self.mag, self.mag_n = g("b_pi"), g("b_mag"), g("b_mag_n")
        self.rb_x, self.rb_y, self.rb_m = g("b_rb_x"), g("b_rb_y"), g("b_rb_m")
        self.exp_count = int(meta["exp_count"]); self.rec_n = int(meta["rec_n"])
        self.rec_ind = np.array(arrays["b_rec_ind"], dtype=np.int64); self.rec_ctx = np.array(arrays["b_rec_ctx"], dtype=np.int64)
        self.rec_y, self.rec_m, self.rec_phi = g("b_rec_y"), g("b_rec_m"), g("b_rec_phi")
        self.rec_exp = np.array(arrays["b_rec_exp"], dtype=np.int64); self.rec_ser = np.array(arrays["b_rec_ser"], dtype=np.int64)
        self.rb_ser = np.array(arrays["b_rb_ser"], dtype=np.int64)
        self.sw_n = int(meta["sw_n"])
        self.sw_exp = np.array(arrays["b_sw_exp"], dtype=np.int64); self.sw_ctx = np.array(arrays["b_sw_ctx"], dtype=np.int64)
        self.sw_y, self.sw_m = g("b_sw_y"), g("b_sw_m")
        self.rb_cls = np.array(arrays["b_rb_cls"], dtype=np.int64); self.rb_ctxi = np.array(arrays["b_rb_ctxi"], dtype=np.int64)
        self.rawcls = list(meta["rawcls"]); self.rawcls_idx = {c: i for i, c in enumerate(self.rawcls)}
        self.raw_n1, self.raw_n0 = g("b_raw_n1"), g("b_raw_n0")
        for nm, hd in self.heads.items():
            hd.set_arrays(arrays, f"b_head_{nm}_")
            self.heads_init[nm].set_arrays(arrays, f"b_hinit_{nm}_")
            self.loo[nm] = []
            for i in range(int(meta["loo_n"].get(nm, 0))):
                m = self.heads_init[nm].copy()
                m.set_arrays(arrays, f"b_loo_{nm}_{i}_")
                self.loo[nm].append(m)
        self.ll_ema = g("b_ll_ema"); self.ll_cnt = int(meta["ll_cnt"])
        self._kcache = {}
        self._agg_cache = None
