"""T8 NN の転移。
共同作業者（手柄を取る率が高い）と、師・見知らぬ人（低い）を学習した後、共同作業者に埋め込みが似た
未見の類「バンド仲間」への予測が、共同作業者の傾向へ寄る。
合格（SPEC 第 8 節）: ヘッドあり: 未見の類の予測と共同作業者の予測の差が、no_head より 30% 以上小さい。
予測 = 未見の個体（どちらも未見）の P(credit_taken | music_session)。種 0〜19 の平均。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, feed, load_jsonl, make, pmap

COLLAB = {"entity": "t8_new_collab", "class": "collaborator", "context": "music_session"}
BAND = {"entity": "t8_new_band", "class": "band_mate", "context": "music_session"}


def gap(npn):
    pc, pb = npn.probe(COLLAB), npn.probe(BAND)
    return abs(pc["p_person"]["credit_taken"] - pb["p_person"]["credit_taken"]), pc, pb


def worker(seed):
    train = load_jsonl("T8_train.jsonl")
    h, nh = make(seed), make(seed, no_head=True)
    feed(h, train); feed(nh, train)
    gh, pc_h, pb_h = gap(h)
    gn, pc_n, pb_n = gap(nh)
    return {"seed": seed, "gap_head": gh, "gap_nohead": gn,
            "collab_head": pc_h["p_person"]["credit_taken"], "band_head": pb_h["p_person"]["credit_taken"],
            "collab_nohead": pc_n["p_person"]["credit_taken"], "band_nohead": pb_n["p_person"]["credit_taken"],
            "head_trained": h.belief.head_trained(), "rho_h": h.belief.rho(), "head_weights": h.belief.head_weights_dict()}


class T8Transfer(unittest.TestCase):
    def test_head_transfers_to_unseen_class(self):
        rows = pmap(worker, SEEDS)       # v3.2: 種を並列化（計算は同じ）
        gh = float(np.mean([r["gap_head"] for r in rows])); gn = float(np.mean([r["gap_nohead"] for r in rows]))
        reduction = 1.0 - gh / gn
        dump("T08", {"gap_with_head": gh, "gap_no_head": gn, "reduction": reduction,
                     "seeds_reduction_ge_30pct": int(sum(r["gap_head"] <= 0.7 * r["gap_nohead"] for r in rows)),
                     "seed0": rows[0]})
        self.assertTrue(all(r["head_trained"] for r in rows))
        self.assertGreaterEqual(reduction, 0.30)


if __name__ == "__main__":
    unittest.main()
