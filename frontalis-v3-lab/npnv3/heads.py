"""事前ヘッドの共通部品（v3.3 SPEC 1.2、1.3）: 類を除いた（leave-class-out）予測と、その学習。

人格の信念ネット（belief.py）と、人格から独立した合成ベンチマーク（benchmarks/nn_transfer.py）が、同じ部品を使う。
ヘッドは「類の埋め込み（と文脈）から、その類の個体が結果の型 k を起こす率の事前平均」を出す。

  mlp    隠れ層 16 の小さな MLP（tanh、ロジット出力）。全バッチ勾配降下で、初期値から学習し直す。
  linear ロジスティック回帰（MLP と同じ入出力・同じ学習）。
  knn1   最も近い（類の埋め込みのユークリッド距離）他の類の率。
  knnk   最も近い k 個（既定 3）の他の類の率の、距離の逆数での重み付き平均。
  none   他の類をまとめた全体の率（類の情報を使わない）。

「他の類」: 予測の対象の類 c 自身の観測は使わない（exclude = c の添字）。未見の類（データがない）は何も除かない。
knn・none が使うのは、類ごとの観測の生の計数 (n1, n0)（観測 y とマスクだけ。責任 φ や信念の事後は入らない）。
したがって、類 c の観測を書き換えても、c を除いたヘッドの出力は厳密に変わらない（X1）。

率の平滑: 全体の率 g = (Σ n1 + κ_g m0) / (Σ n + κ_g)（Beta 事前。κ_g = 4、m0 = 0.2 は信念ネットと同じ）。
近傍の類の率 r̃ = (n1 + s g) / (n + s)（s = 2。計数が少ない類の率を全体の率へ寄せる）。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

P_MIN = 1e-4
P_MAX = 1.0 - 1e-4


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_MIN, P_MAX)
    return np.log(p / (1.0 - p))


def pooled_rate(n1: np.ndarray, n0: np.ndarray, exclude: Optional[int], kappa_g: float, prior_mean: float) -> np.ndarray:
    """類 exclude を除いた全体の率（型ごと、形 (K,)）。n1, n0: (類の数, K) の生の計数。"""
    keep = np.ones(n1.shape[0], dtype=bool)
    if exclude is not None:
        keep[exclude] = False
    t1 = n1[keep].sum(axis=0)
    t = (n1[keep] + n0[keep]).sum(axis=0)
    return (t1 + kappa_g * prior_mean) / (t + kappa_g)


def knn_rate(emb: np.ndarray, n1: np.ndarray, n0: np.ndarray, query: np.ndarray, exclude: Optional[int], k: int,
             kappa_g: float, prior_mean: float, smooth: float) -> np.ndarray:
    """最も近い k 個の（データのある）他の類の、平滑した率の距離重み平均。k = 1 なら最近傍の類の率。
    emb: (類の数, 次元)。データのある他の類がなければ全体の率。距離が同じ類は添字の小さい方が先。"""
    g = pooled_rate(n1, n0, exclude, kappa_g, prior_mean)
    n = n1 + n0
    has = n.sum(axis=1) > 0
    if exclude is not None:
        has[exclude] = False
    idx = np.nonzero(has)[0]
    if idx.size == 0:
        return g.copy()
    d = np.linalg.norm(emb[idx] - query[None, :], axis=1)
    order = np.argsort(d, kind="stable")[: max(1, int(k))]
    sel = idx[order]
    r = (n1[sel] + smooth * g[None, :]) / (n[sel] + smooth)
    if sel.size == 1:
        return r[0]
    w = 1.0 / (d[order] + 1e-9)
    return (w[:, None] * r).sum(axis=0) / w.sum()


def fit_weighted(model: Any, init: Any, X: np.ndarray, n1: np.ndarray, n: np.ndarray, total: float,
                 base_rate: np.ndarray, lr: float, steps: int) -> None:
    """model を init の重みから学習し直す（全バッチ勾配降下）。出力バイアスは base_rate のロジットから始める。

    X: (U, 次元) の異なる入力（類×文脈）。n1, n: (U, K) の正の観測の計数と観測の計数（マスクで観測されたセルだけ）。
    total: 観測の行数（参加者の観測 1 件 = 1 行）。損失は MLP.sgd_step と同じ（行で平均、型で和、二値交差エントロピー）で、
    同じ入力の行を集計した重み付きの形（勾配は行ごとの SGD の期待値と同じ）。乱数を使わない。"""
    model.load_from(init)
    model.set_output_bias(logit(base_rate))
    U = X.shape[0]
    if U == 0 or total <= 0 or steps <= 0:
        return
    rate = np.divide(n1, n, out=np.zeros_like(n1), where=n > 0)
    mask = n * (U / float(total))
    for _ in range(steps):
        model.sgd_step(X, rate, lr, mask=mask)
