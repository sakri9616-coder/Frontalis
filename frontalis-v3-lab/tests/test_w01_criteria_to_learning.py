"""W1 評価基準 → 学習: 関心の重み a と条件づけ価値 C が、次の学習（価値の重み W）を変えるか。

SPEC-v3.2 第 2 節: 人格 P と Q を育てた後、P の a と C だけを Q のものに入れ替え、同じ 20 経験を与える。入れ替えない場合と比べる。
合格: W（価値の重み）の最終値の差の相対ノルムが 0.05 以上（20 本の平均）。`no_concern` かつ `no_conditioning` では 0。

測り方（実装前に固定した細部）:
  P = R 型、Q = K 型の人格（generate_history、36 件、乱数種ごとに別の経験史）。
  同じ 20 経験 = ランダムな世界（generate_random_history(seed + 500, 20, offer_rate = 0.25)）から抽出。協働の約 4 分の 1 が申し出（optional）。
    P の複製 2 つ（乱数の状態も同じ）に与える: 基準（入れ替えなし）と、a と C だけを Q のものに入れ替えたもの。
  相対ノルム = ‖W_入替 − W_基準‖ / ‖W_基準‖。W は人モデルと状況モデルの全特徴×5 欲求（特徴名で対応づけ、片方にしかない特徴は 0）。
  a → W の経路（欲求ごとの観測雑音 R/a_k）と、C → Q → 選択 → 経験 → W の経路（申し出の接近・回避が変わる）の両方を含む。
  内訳（判定しない）: a だけ、C だけを入れ替えたとき。申し出のない 20 経験（a → W の経路だけ）。
  no_concern かつ no_conditioning の人格では a = 1、C = 0 なので、入れ替えは何も変えない（差は厳密に 0）。
"""
import copy
import unittest

import numpy as np

from helpers import SEEDS, dump, feed, mh, persona_gen, pmap


def w_vec(n):
    d = {}
    for tag, kal in (("p", n.value.person), ("s", n.value.sit)):
        for i, name in enumerate(kal.names):
            for k in range(5):
                d[(tag, name, k)] = float(kal.W[i, k])
    return d


def rel_diff(n1, n2):
    a, b = w_vec(n1), w_vec(n2)
    keys = set(a) | set(b)
    num = np.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))
    den = np.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys))
    return float(num / den)


def swapped(p, q, parts):
    s = copy.deepcopy(p)
    if "a" in parts:
        s.concern = copy.deepcopy(q.concern)
    if "C" in parts:
        s.cond = copy.deepcopy(q.cond)
    return s


def run(p, q, exps, parts):
    base, sw = copy.deepcopy(p), swapped(p, q, parts)
    feed(base, exps); feed(sw, exps)
    return rel_diff(sw, base)


def worker(seed):
    exps = mh.generate_random_history(seed + 500, 20, offer_rate=0.25)
    exps_free = mh.generate_random_history(seed + 500, 20, offer_rate=0.0)
    out = {"seed": seed, "n_offers": sum(e["optional"] for e in exps)}
    p, q = persona_gen("R", seed), persona_gen("K", seed)
    out["a_P"] = [float(x) for x in p.concern.a]
    out["a_Q"] = [float(x) for x in q.concern.a]
    out["both"] = run(p, q, exps, {"a", "C"})
    out["a_only"] = run(p, q, exps, {"a"})
    out["C_only"] = run(p, q, exps, {"C"})
    out["both_no_offers"] = run(p, q, exps_free, {"a", "C"})
    out["a_only_no_offers"] = run(p, q, exps_free, {"a"})
    p0 = persona_gen("R", seed, no_concern=True, no_conditioning=True)
    q0 = persona_gen("K", seed, no_concern=True, no_conditioning=True)
    out["ablated_both"] = run(p0, q0, exps, {"a", "C"})
    # アブレーション: 精度の重み付けを外す（a は動くが R_k = R）と、a の入れ替えは申し出の選択を通る経路だけになる
    pn = persona_gen("R", seed, no_precision_weighting=True)
    qn = persona_gen("K", seed, no_precision_weighting=True)
    out["a_only_no_precision_weighting_no_offers"] = run(pn, qn, exps_free, {"a"})
    return out


class W1CriteriaToLearning(unittest.TestCase):
    def test_swapping_a_and_c_changes_what_is_learned(self):
        rows = pmap(worker, SEEDS)
        m = lambda k: float(np.mean([r[k] for r in rows]))
        res = {"mean_relative_norm_both": m("both"), "min_relative_norm_both": float(np.min([r["both"] for r in rows])),
               "mean_a_only": m("a_only"), "mean_C_only": m("C_only"),
               "mean_both_no_offers (a -> W only)": m("both_no_offers"),
               "mean_a_only_no_offers": m("a_only_no_offers"),
               "mean_a_only_no_precision_weighting_no_offers": m("a_only_no_precision_weighting_no_offers"),
               "ablated_no_concern_no_conditioning_max": float(np.max([r["ablated_both"] for r in rows])),
               "mean_offers_in_20": m("n_offers"),
               "mean_abs_a_difference_P_vs_Q": float(np.mean([np.abs(np.array(r["a_P"]) - np.array(r["a_Q"])).mean() for r in rows])),
               "seeds_ge_0.05": int(sum(r["both"] >= 0.05 for r in rows)), "both": [r["both"] for r in rows]}
        dump("W01", res)
        self.assertGreaterEqual(res["mean_relative_norm_both"], 0.05, res)
        self.assertEqual(res["ablated_no_concern_no_conditioning_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
