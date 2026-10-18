"""T7 信念の一般化（H4）。
「手柄を取る」を同じ相手から 8 回 vs 8 人から 1 回ずつ学んだ後の、未見の共同作業者への
P(credit_taken) の上昇（学習前との差）。合格: 8 人の方が大きい（20 本の平均と、全本）。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, feed, load_jsonl, make

PROBE = {"entity": "t7_unseen", "class": "collaborator", "context": "music_session"}


def p_credit(npn):
    return npn.probe(PROBE)["p_person"]["credit_taken"]


class T7Generalization(unittest.TestCase):
    def test_eight_people_generalize_more_than_one_person_eight_times(self):
        rows = []
        for seed in SEEDS:
            same, diff = make(seed), make(seed)
            before = p_credit(same)
            feed(same, load_jsonl("T7_same.jsonl")); feed(diff, load_jsonl("T7_diff.jsonl"))
            rows.append({"seed": seed, "before": before, "rise_same": p_credit(same) - before,
                         "rise_diff": p_credit(diff) - before})
        rs = float(np.mean([r["rise_same"] for r in rows])); rd = float(np.mean([r["rise_diff"] for r in rows]))
        wins = int(sum(r["rise_diff"] > r["rise_same"] for r in rows))
        dump("T07", {"before": rows[0]["before"], "rise_same_partner_x8": rs, "rise_eight_partners_x1": rd,
                     "seeds_where_eight_partners_larger": wins, "n_seeds": len(rows)})
        self.assertGreater(rd, rs)
        self.assertEqual(wins, len(rows))


if __name__ == "__main__":
    unittest.main()
