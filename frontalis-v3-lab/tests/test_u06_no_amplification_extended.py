"""U6 自己増幅なし（拡張）: 全機構を有効にした閉ループで、状態の摂動が増幅しない。

SPEC-v3.1 第 3 節: 全機構を有効にして、同じ経験史を 2000 経験まで延長し、状態に 1e-3 の摂動を入れて共通乱数で再生する。
合格: 全ての値が有界（|C| < 10、a の各成分 < 5）。最終状態の摂動の比 ‖Δ最終‖/‖Δ初期‖ が 20 本の中央値で 1 以下。

測り方（実装前に固定した細部）:
  全機構 = 既定の設定（関心の重み a、条件づけ C、参照点、曖昧な結果の解釈、事前ヘッド、κ_e の推定、スコープ帰属、optional の接近・回避）。
  経験史 = generate_extended_history(型, 種, 2000)。型は種に応じて R, K, X を巡回（種 0 = R, 1 = K, 2 = X, 3 = R, ...）。
    相手は 12 人の個体が繰り返し現れ、8% の協働に曖昧な結果、20% の協働は optional な申し出。
  摂動の入れ方: 最初の 200 経験の後の状態に、価値ネットの重み W（人・状況）、条件づけ C、関心の重み a、気分 m、欲求の状態 h へ
    独立なガウス雑音を入れ、全体のノルムを 1e-3 に正規化する（a は摂動の後に和 5 へ再正規化）。残りの 1800 経験を、
    基準と摂動の 2 つの NPN に同じ順序で与える。optional の接近・回避の乱数は同じ状態から始める（共通乱数）。
  ‖Δ‖ = 上の 5 種の状態（特徴名で対応づけ、片方にしかない特徴は 0）の差の L2 ノルム。Σ・信念の計数は摂動しない。
  有界の検査: 基準の再生の 100 経験ごとに max|C| と max a_k を記録し、最大値が上の上限未満。
"""
import copy
import unittest

import numpy as np

from helpers import KINDS, SEEDS, dump, make, mh, pmap

WARMUP = 200
TOTAL = 2000
EPS = 1e-3


def state_vec(n):
    d = {}
    for tag, kal in (("pW", n.value.person), ("sW", n.value.sit)):
        for i, name in enumerate(kal.names):
            for k in range(5):
                d[(tag, name, k)] = float(kal.W[i, k])
    for name, c in zip(n.cond.names, n.cond.C):
        d[("C", name)] = float(c)
    for k in range(5):
        d[("a", k)] = float(n.concern.a[k]); d[("h", k)] = float(n.h[k])
    d[("m",)] = float(n.mood)
    return d


def distance(n1, n2):
    a, b = state_vec(n1), state_vec(n2)
    return float(np.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in set(a) | set(b))))


def perturb(n, rng):
    """価値ネット W・C・a・m・h に、全体のノルムが EPS のガウス雑音を入れる。"""
    shapes = [n.value.person.W.shape, n.value.sit.W.shape, n.cond.C.shape, (5,), (), (5,)]
    noise = [rng.normal(size=s) for s in shapes]
    norm = np.sqrt(sum(float(np.sum(x ** 2)) for x in noise))
    noise = [x * EPS / norm for x in noise]
    n.value.person.W = n.value.person.W + noise[0]
    n.value.sit.W = n.value.sit.W + noise[1]
    n.cond.C = n.cond.C + noise[2]
    a = n.concern.a + noise[3]
    n.concern.a = 5.0 * a / a.sum()
    n.mood = n.mood + float(noise[4])
    n.h = n.h + noise[5]


def worker(seed):
    kind = KINDS[seed % 3]
    hist = mh.generate_extended_history(kind, seed, TOTAL)
    base = make(seed, record_ledger=False)
    for e in hist[:WARMUP]:
        base.experience(e)
    pert = copy.deepcopy(base)
    perturb(pert, np.random.default_rng([seed, 77]))
    d0 = distance(base, pert)
    pert.rng.bit_generator.state = base.rng.bit_generator.state          # 共通乱数
    max_c, max_a, flips = 0.0, 0.0, 0
    for i, e in enumerate(hist[WARMUP:]):
        r1 = base.experience(e)
        r2 = pert.experience(e)
        if e["optional"] and r1["approached"] != r2["approached"]:
            flips += 1
        if i % 100 == 0 or i == TOTAL - WARMUP - 1:
            max_c = max(max_c, float(np.abs(base.cond.C).max()), float(np.abs(pert.cond.C).max()))
            max_a = max(max_a, float(base.concern.a.max()), float(pert.concern.a.max()))
    return {"seed": seed, "kind": kind, "ratio": distance(base, pert) / d0, "d0": d0, "max_C": max_c, "max_a": max_a,
            "choice_flips": flips, "n_avoided": base.n_avoided, "n_exp": base.n_experiences,
            "n_features": [len(base.value.person.names), len(base.value.sit.names), len(base.cond.names)]}


class U6NoAmplificationExtended(unittest.TestCase):
    def test_bounded_and_perturbation_does_not_grow(self):
        rows = pmap(worker, SEEDS)
        ratios = [r["ratio"] for r in rows]
        dump("U06", {"median_ratio": float(np.median(ratios)), "max_ratio": float(np.max(ratios)),
                     "ratios": ratios, "max_abs_C": max(r["max_C"] for r in rows), "max_a": max(r["max_a"] for r in rows),
                     "choice_flips_total": int(sum(r["choice_flips"] for r in rows)),
                     "kinds": [r["kind"] for r in rows], "n_experiences": rows[0]["n_exp"],
                     "avoided_mean": float(np.mean([r["n_avoided"] for r in rows])),
                     "features_seed0": rows[0]["n_features"]})
        self.assertLess(max(r["max_C"] for r in rows), 10.0)
        self.assertLess(max(r["max_a"] for r in rows), 5.0)
        self.assertLessEqual(float(np.median(ratios)), 1.0)


if __name__ == "__main__":
    unittest.main()
