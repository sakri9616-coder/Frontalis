"""W7 ヘッドの重み: 予測性能で決まる混合の重みを報告する（合否にはしない）。

SPEC-v3.2 第 2 節: T8 の課題と R/K/X の履歴で、各ヘッドの混合の重みを報告する。どのヘッドが選ばれたかを数値で報告する。
研究思想 §10「NN が必ず優れている前提にしない」: NN が勝たない課題があれば、そう書く。

測り方（実装前に固定した細部）:
  課題 = (1) T8（data/histories/T8_train.jsonl、120 経験: 共同作業者は手柄を取る率 0.7、師・見知らぬ人は 0.05）、
         (2) R/K/X の経験史（generate_history、36 件）、(3) ランダムな世界（generate_random_history、種ごと、60 件）。種 0〜19。
  重み = 学習の後の softmax(前向き対数尤度の移動平均) の 4 つ（mlp, linear, knn1, none）。種の平均と、各種で最大の重みのヘッドの数。
  検査するのは重みが確率分布であることだけ（合否にしない数値は dump に出す）。
"""
import unittest

import numpy as np

from helpers import KINDS, SEEDS, dump, feed, load_jsonl, make, mh, persona_gen, pmap
from npnv3.belief import HEAD_NAMES


def worker(seed):
    out = {"seed": seed}
    n = make(seed)
    feed(n, load_jsonl("T8_train.jsonl"))
    out["T8"] = n.belief.head_weights_dict()
    out["T8_ll"] = [float(x) for x in n.belief.ll_ema]
    for k in KINDS:
        out[k] = persona_gen(k, seed).belief.head_weights_dict()
    m = make(seed)
    feed(m, mh.generate_random_history(seed, 60, offer_rate=0.0))
    out["random"] = m.belief.head_weights_dict()
    return out


def summarize(rows, key):
    mean = {h: float(np.mean([r[key][h] for r in rows])) for h in HEAD_NAMES}
    best = {h: int(sum(max(r[key], key=r[key].get) == h for r in rows)) for h in HEAD_NAMES}
    return {"mean_weight": mean, "seeds_where_head_has_max_weight": best,
            "mean_nn_weight (1 - none)": 1.0 - mean["none"]}


class W7HeadWeights(unittest.TestCase):
    def test_report_head_weights(self):
        rows = pmap(worker, SEEDS)
        res = {key: summarize(rows, key) for key in ("T8", "R", "K", "X", "random")}
        res["T8_forward_loglik_ema_mean"] = {h: float(np.mean([r["T8_ll"][i] for r in rows])) for i, h in enumerate(HEAD_NAMES)}
        dump("W07", res)
        for r in rows:
            for key in ("T8", "R", "K", "X", "random"):
                self.assertAlmostEqual(sum(r[key].values()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
