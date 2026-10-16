"""診断（テストではない）: X3 の結論（mlp は最良でない）が、ベンチマークの細部の選び方に左右されないかを確かめる。判定には使わない。結果は tests/results/diag_x3.json。

X3 の判定は事前登録した決定則（benchmarks/nn_transfer.py 冒頭）で、SPEC-v3.3 1.3 の世界・ヘッド・種の数のまま（`tests/test_x_nn_transfer.py` の X2・X3）。
ここは、その判定を見た後に足した、感度の確認である（判定の結論は変えない）:
  literal      (b) 非線形の世界を、SPEC の例のとおり logit = 2·e1·e2 + 1.5·sin(3·e3)（単位ベクトルのまま。4 つの型は添字の組 (0,1,2), (3,4,5), (6,7,0), (1,3,5)）にしたもの。
               ベンチマーク本体は、単位ベクトルでは積の項が小さすぎる（標準偏差 0.25）ので z = √8 e に直した版を使っている。その選び方で X3 が変わらないか。
  many_classes 学習類を 24 → 96 に増やしたもの（未見の類は 8 のまま）。非線形の世界と線形の世界で、n = 5, 20。mlp の不利が学習類の少なさ（24）に由来するかを見る。
               Frontalis の実データでは類は数個から数十個なので、これは「NN 固有の役割が現れる条件」の探索であって、人格の既定には使わない。
同じ決定則（mlp が 2 位より 5% 以上小さく、20 本中 16 本以上で最良）を当てて、結論を報告する。
実行: python3 tests/diagnostics_x3.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import dump  # noqa: E402  (helpers が benchmarks/ を sys.path に入れる)
import nn_transfer as bench  # noqa: E402

DIM, N_TYPES = bench.DIM, bench.N_TYPES


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def world_literal(seed, n_total=32):
    rng = np.random.default_rng([seed, 90, 1])
    E = unit(rng.normal(size=(n_total, DIM)))
    logit = np.stack([2.0 * E[:, a] * E[:, b] + 1.5 * np.sin(3.0 * E[:, c]) for a, b, c in bench.TRIPLES], axis=1)
    return E, sigmoid(logit)


def world_scaled(kind, seed, n_total):
    """ベンチマーク本体と同じ式（z = √8 e の非線形、線形）で、類の数だけを変えたもの。"""
    rng = np.random.default_rng([seed, 91, 1])
    E = unit(rng.normal(size=(n_total, DIM)))
    if kind == "linear":
        W = rng.normal(0.0, 1.5, size=(N_TYPES, DIM))
        return E, sigmoid(E @ W.T)
    Z = np.sqrt(DIM) * E
    logit = np.stack([Z[:, a] * Z[:, b] + 1.5 * np.sin(Z[:, c]) for a, b, c in bench.TRIPLES], axis=1)
    return E, sigmoid(logit)


def run_cell(world_fn, n_train, n_obs, seeds=bench.SEEDS):
    ce = {h: [] for h in bench.HEADS}
    ent = []
    for seed in range(seeds):
        E, P = world_fn(seed)
        n1, n0 = bench.observe(P[:n_train], n_obs, seed, "nonlinear")
        pred = bench.predict_heads(E[:n_train], n1, n0, E[n_train:], seed, "nonlinear")
        for h in bench.HEADS:
            ce[h].append(bench.cross_entropy(P[n_train:], pred[h]))
        ent.append(bench.entropy(P[n_train:]))
    return bench.summarize(ce, ent, bench.HEADS)


def x3_rule(cells):
    return [k for k, c in cells.items() if c["mlp"]["mlp_is_best"] and c["mlp"]["relative_gap_to_second"] >= bench.MARGIN
            and c["mlp"]["seeds_mlp_best"] >= bench.MIN_SEEDS_BEST]


def show(title, cells):
    print(f"\n## {title}")
    print("| 条件 | " + " | ".join(bench.HEADS) + " | 最良 | mlp と 2 位の差 | mlp が最良の種 |")
    print("| --- | " + " | ".join(["---:"] * len(bench.HEADS)) + " | --- | ---: | ---: |")
    for k, c in cells.items():
        v = c["ce_mean"]
        best = min(v, key=v.get)
        print(f"| {k} | " + " | ".join((f"**{v[h]:.4f}**" if h == best else f"{v[h]:.4f}") for h in bench.HEADS)
              + f" | {best} | {100 * c['mlp']['relative_gap_to_second']:+.1f}% | {c['mlp']['seeds_mlp_best']}/{bench.SEEDS} |")
    w = x3_rule(cells)
    print("決定則を当てると:", ("(i) mlp が最良: " + ", ".join(w)) if w else "(ii) どの条件でも mlp は最良でない")


def main():
    res = {}
    cells = {f"literal|{n}": run_cell(world_literal, bench.N_TRAIN, n) for n in bench.NS}
    res["literal"] = {"cells": cells, "mlp_best_conditions": x3_rule(cells)}
    show("literal: SPEC の例のとおりの非線形（単位ベクトル。学習類 24）", cells)
    for n_train in (96,):
        for kind in ("linear", "nonlinear"):
            cells = {f"{kind}|n_train={n_train}|{n}": run_cell(lambda s, k=kind, nt=n_train: world_scaled(k, s, nt + 8), n_train, n) for n in bench.NS}
            res[f"many_classes_{kind}_{n_train}"] = {"cells": cells, "mlp_best_conditions": x3_rule(cells)}
            show(f"many_classes: {kind}、学習類 {n_train}", cells)
    dump("diag_x3", res)


if __name__ == "__main__":
    main()
