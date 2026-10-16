"""SPEC-v3.3 1.3 合成ベンチマーク: 未見の類への転移で、ヘッド（mlp / linear / knn1 / knnk / none）を公平に比べる。

人格の実行とは独立（NPN を使わない）。ただしヘッドの実装は人格と共通（npnv3/mlp.py の MLP・LogReg と npnv3/heads.py）。
実行: python3 benchmarks/nn_transfer.py [--sensitivity] [--json OUT]

【設計（SPEC-v3.3 1.3 のとおり。細部は結果を見る前にここで固定した）】
  類 32 個（学習 24、未見 8）。埋め込みは 8 次元の正規乱数を正規化したもの（(c) 群の世界だけ群の構造から作る）。結果型は 4 つ。
  学習類ごとの観測数 n ∈ {5, 20}。1 観測 = 4 つの型それぞれを真の率で独立に 0/1 に抽出したもの。
  真の率の 4 つの世界（logit は型 j ごと。e は単位ベクトル、z = √8 e は各成分が標準偏差 1 になるよう尺度を直したもの）:
    (a) 線形    logit_j = W_j · e，W の各成分 ~ N(0, 1.5²)（logit の標準偏差は 1.5）
    (b) 非線形  logit_j = z_a z_b + 1.5 sin(z_c)，(a, b, c) = (0,1,2), (3,4,5), (6,7,0), (1,3,5)
                （SPEC の例 2 e1 e2 + 1.5 sin(3 e3) は単位ベクトルでは積の項が小さすぎる（標準偏差 0.25）ので、z に直して標準偏差を 1.5 前後にそろえた）
    (c) 群      類 i は群 i mod 4 に属する（学習 24 は各群 6、未見 8 は各群 2）。群 g の率は型ごとに U(0.1, 0.9)。
                埋め込み = 正規化(群の中心(単位ベクトル) + 0.3 × 単位ベクトルの雑音)
    (d) 無関係  率 = sigmoid(1.5 ξ)，ξ ~ N(0, 1) を類・型ごとに独立に（埋め込みと無関係。(a) と同じ周辺分布）
  評価: 未見の類 8 個×型 4 つについて、真の率 p に対する交差エントロピー −[p log p̂ + (1−p) log(1−p̂)] の平均（期待対数損失）。
        予測 p̂ は [1e-4, 1−1e-4] に切る（人格と同じ）。乱数種 20 本（種ごとに世界・埋め込み・観測を引き直す）。ヘッドは同じデータで対にして比べる。
        参考として、真の率のエントロピーを引いた「超過損失（KL）」も報告する（判定には使わない）。
  ヘッド（人格と共通の部品）:
    none   学習類をまとめた全体の率 (Σ n1 + 4×0.2) / (Σ n + 4)
    knn1   最も近い学習類の率 (n1 + 2 g) / (n + 2)（g = 全体の率）
    knnk   最も近い 3 つの学習類の同じ率の、距離の逆数での重み付き平均
    linear ロジスティック回帰（入力 8 次元 → 4 ロジット）。出力バイアスを全体の率のロジットから始め、全バッチ勾配降下 300 回、学習率 0.5
    mlp    隠れ層 16（tanh）の MLP。linear と同じ初期バイアス・同じ学習。重みの初期値は種ごとの乱数
  学習の予算（300 回・0.5）は、凸な linear の損失がほぼ収束する大きさとして、結果を見る前に決めた（学習損失だけで確認。テストの成績は見ていない）。
  mlp は正則化も早期停止もない（収束させないことが暗黙の正則化になる）。予算への感度は --sensitivity で別に報告する（判定には使わない）。

【X3 の決定則（事前登録。SPEC-v3.3 1.4）】
  「mlp が最良」= ある（世界, n）で、(i) 平均の交差エントロピーが全ヘッドで最小、(ii) 2 番目に良いヘッドより 5% 以上小さい、
  (iii) 20 本中 16 本以上で 5 ヘッドの中で最小。一つでも成り立つ（世界, n）があれば (i) NN 固有の役割は「その条件の構造の転移」。
  一つもなければ (ii) NN 固有の役割は確認されなかった。既定のヘッドから mlp を外す（設定は残す）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npnv3.heads import P_MAX, P_MIN, fit_weighted, knn_rate, pooled_rate  # noqa: E402
from npnv3.mlp import MLP, LogReg  # noqa: E402

N_CLASSES, N_TRAIN, DIM, N_TYPES = 32, 24, 8, 4
WORLDS = ("linear", "nonlinear", "group", "unrelated")
NS = (5, 20)
HEADS = ("mlp", "linear", "knn1", "knnk", "none")
SEEDS = 20
HIDDEN = 16
FIT_STEPS = 300
FIT_LR = 0.5
KAPPA_G, PRIOR_MEAN, SMOOTH, KNN_K = 4.0, 0.2, 2.0, 3
TRIPLES = ((0, 1, 2), (3, 4, 5), (6, 7, 0), (1, 3, 5))
# X3 の決定則
MARGIN = 0.05
MIN_SEEDS_BEST = 16


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_world(world: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """世界を引く。戻り値は 埋め込み E (32, 8) と 真の率 P (32, 4)。学習類は先頭 24、未見は後ろ 8。"""
    rng = np.random.default_rng([seed, WORLDS.index(world), 1])
    if world == "group":
        centers = _unit(rng.normal(size=(4, DIM)))
        group = np.arange(N_CLASSES) % 4
        E = _unit(centers[group] + 0.3 * _unit(rng.normal(size=(N_CLASSES, DIM))))
        R = rng.uniform(0.1, 0.9, size=(4, N_TYPES))
        return E, R[group]
    E = _unit(rng.normal(size=(N_CLASSES, DIM)))
    if world == "linear":
        W = rng.normal(0.0, 1.5, size=(N_TYPES, DIM))
        return E, _sigmoid(E @ W.T)
    if world == "nonlinear":
        Z = np.sqrt(DIM) * E
        logit = np.stack([Z[:, a] * Z[:, b] + 1.5 * np.sin(Z[:, c]) for a, b, c in TRIPLES], axis=1)
        return E, _sigmoid(logit)
    if world == "unrelated":
        return E, _sigmoid(1.5 * rng.normal(size=(N_CLASSES, N_TYPES)))
    raise ValueError(world)


def observe(P_train: np.ndarray, n: int, seed: int, world: str) -> Tuple[np.ndarray, np.ndarray]:
    """学習類ごとに n 観測（各観測 = 4 型の 0/1）を抽出。戻り値は正の計数 n1 と負の計数 n0（形 (24, 4)）。"""
    rng = np.random.default_rng([seed, WORLDS.index(world), 2, n])
    Y = (rng.random((P_train.shape[0], n, N_TYPES)) < P_train[:, None, :]).astype(np.float64)
    n1 = Y.sum(axis=1)
    return n1, float(n) - n1


def predict_heads(E_train: np.ndarray, n1: np.ndarray, n0: np.ndarray, E_test: np.ndarray, seed: int, world: str,
                  steps: int = FIT_STEPS, lr: float = FIT_LR, heads: Tuple[str, ...] = HEADS) -> Dict[str, np.ndarray]:
    """各ヘッドの未見の類への予測（形 (8, 4)）。"""
    out: Dict[str, np.ndarray] = {}
    n = n1 + n0
    g = pooled_rate(n1, n0, None, KAPPA_G, PRIOR_MEAN)
    if "none" in heads:
        out["none"] = np.tile(g, (E_test.shape[0], 1))
    for nm, k in (("knn1", 1), ("knnk", KNN_K)):
        if nm in heads:
            out[nm] = np.array([knn_rate(E_train, n1, n0, e, None, k, KAPPA_G, PRIOR_MEAN, SMOOTH) for e in E_test])
    total = float(n.sum() / N_TYPES)                      # 観測の行数（全型がそろった観測）
    if "linear" in heads:
        m = LogReg(DIM, N_TYPES)
        fit_weighted(m, LogReg(DIM, N_TYPES), E_train, n1, n, total, g, lr, steps)
        out["linear"] = m.predict_proba(E_test)
    if "mlp" in heads:
        init = MLP(DIM, HIDDEN, N_TYPES, np.random.default_rng([seed, WORLDS.index(world), 3]))
        m = init.copy()
        fit_weighted(m, init, E_train, n1, n, total, g, lr, steps)
        out["mlp"] = m.predict_proba(E_test)
    return {k: np.clip(v, P_MIN, P_MAX) for k, v in out.items()}


def cross_entropy(P: np.ndarray, p_hat: np.ndarray) -> float:
    """真の率 P に対する交差エントロピー（未見の類×型の平均）。"""
    return float(np.mean(-(P * np.log(p_hat) + (1.0 - P) * np.log(1.0 - p_hat))))


def entropy(P: np.ndarray) -> float:
    return float(np.mean(-(P * np.log(P) + (1.0 - P) * np.log(1.0 - P))))


def run_benchmark(seeds: int = SEEDS, steps: int = FIT_STEPS, lr: float = FIT_LR, heads: Tuple[str, ...] = HEADS) -> Dict[str, Any]:
    """全条件（4 世界 × 2 観測数）× 種を実行する。戻り値は条件ごとの種別の誤差と集計。"""
    cells: Dict[str, Any] = {}
    for world in WORLDS:
        for n in NS:
            ce = {h: [] for h in heads}
            ent: List[float] = []
            for seed in range(seeds):
                E, P = make_world(world, seed)
                n1, n0 = observe(P[:N_TRAIN], n, seed, world)
                pred = predict_heads(E[:N_TRAIN], n1, n0, E[N_TRAIN:], seed, world, steps, lr, heads)
                for h in heads:
                    ce[h].append(cross_entropy(P[N_TRAIN:], pred[h]))
                ent.append(entropy(P[N_TRAIN:]))
            cells[f"{world}|{n}"] = summarize(ce, ent, heads)
    return {"steps": steps, "lr": lr, "seeds": seeds, "heads": list(heads), "cells": cells,
            "x3": decide_x3(cells, heads)}


def summarize(ce: Dict[str, List[float]], ent: List[float], heads: Tuple[str, ...]) -> Dict[str, Any]:
    arr = np.array([ce[h] for h in heads])                    # (heads, seeds)
    mean = arr.mean(axis=1)
    order = np.argsort(mean, kind="stable")
    best_per_seed = np.argmin(arr, axis=0)
    e = float(np.mean(ent))
    res: Dict[str, Any] = {
        "ce_mean": {h: float(m) for h, m in zip(heads, mean)},
        "ce_sem": {h: float(arr[i].std(ddof=1) / np.sqrt(arr.shape[1])) for i, h in enumerate(heads)},
        "excess_kl_mean": {h: float(m - e) for h, m in zip(heads, mean)},
        "entropy_mean": e,
        "seeds_best": {h: int(np.sum(best_per_seed == i)) for i, h in enumerate(heads)},
        "rank": [heads[i] for i in order],
    }
    if "mlp" in heads:
        others = [i for i in order if heads[i] != "mlp"]
        second = float(mean[others[0]]) if others else float("nan")
        m = float(mean[heads.index("mlp")])
        res["mlp"] = {"ce": m, "second_best_head": heads[others[0]] if others else None, "second_best_ce": second,
                      "relative_gap_to_second": (second - m) / second if others else float("nan"),
                      "seeds_mlp_best": int(np.sum(best_per_seed == heads.index("mlp"))),
                      "mlp_is_best": bool(heads[order[0]] == "mlp")}
        # 超過損失（KL）での同じ指標（参考。判定には使わない）
        kl = mean - e
        so = float(kl[others[0]]) if others else float("nan")
        res["mlp"]["relative_gap_kl_to_second"] = (so - float(kl[heads.index("mlp")])) / so if (others and so > 0) else float("nan")
    return res


def decide_x3(cells: Dict[str, Any], heads: Tuple[str, ...]) -> Dict[str, Any]:
    """事前登録した決定則。"""
    if "mlp" not in heads:
        return {"conclusion": "mlp を含まない実行", "mlp_best_conditions": []}
    winners = []
    for key, c in cells.items():
        m = c["mlp"]
        if m["mlp_is_best"] and m["relative_gap_to_second"] >= MARGIN and m["seeds_mlp_best"] >= MIN_SEEDS_BEST:
            winners.append(key)
    if winners:
        concl = ("(i) 少なくとも 1 つの条件で mlp が最良: NN 固有の役割は「その条件の構造の転移」（" + ", ".join(winners) + "）")
    else:
        concl = "(ii) どの条件でも mlp は最良でない: NN 固有の役割は確認されなかった。既定のヘッドから mlp を外す（設定は残す）"
    return {"conclusion": concl, "mlp_best_conditions": winners, "margin": MARGIN, "min_seeds_best": MIN_SEEDS_BEST}


def markdown_table(res: Dict[str, Any], key: str = "ce_mean") -> str:
    heads = res["heads"]
    lines = ["| 世界 | n | " + " | ".join(heads) + " | 最良 | mlp と 2 位の差 | mlp が最良の種 |",
             "| --- | ---: | " + " | ".join(["---:"] * len(heads)) + " | --- | ---: | ---: |"]
    for k, c in res["cells"].items():
        w, n = k.split("|")
        vals = c[key]
        best = min(vals, key=vals.get)
        cells = []
        for h in heads:
            s = f"{vals[h]:.4f}"
            cells.append(f"**{s}**" if h == best else s)
        gap = f"{100 * c['mlp']['relative_gap_to_second']:+.1f}%" if "mlp" in c else "-"
        sb = f"{c['mlp']['seeds_mlp_best']}/{res['seeds']}" if "mlp" in c else "-"
        lines.append(f"| {w} | {n} | " + " | ".join(cells) + f" | {best} | {gap} | {sb} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="NN の転移ベンチマーク（SPEC-v3.3 1.3）")
    ap.add_argument("--sensitivity", action="store_true", help="学習の予算（回数）への感度を別に報告する（判定には使わない）")
    ap.add_argument("--json", default=None, help="結果を JSON に書く")
    args = ap.parse_args()
    res = run_benchmark()
    print("## 交差エントロピー（未見の類 8 個×型 4 つの平均、種 20 本の平均。小さいほど良い。太字が最良）")
    print(markdown_table(res, "ce_mean"))
    print("\n## 超過損失 KL（交差エントロピー − 真の率のエントロピー。参考）")
    print(markdown_table(res, "excess_kl_mean"))
    print("\n## X3:", res["x3"]["conclusion"])
    if args.sensitivity:
        res["sensitivity"] = {}
        for steps in (100, 1000):
            r = run_benchmark(steps=steps)
            res["sensitivity"][str(steps)] = r
            print(f"\n## 感度: 学習 {steps} 回（参考。判定には使わない）")
            print(markdown_table(r, "ce_mean"))
            print("X3 の決定則を当てると:", r["x3"]["conclusion"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
