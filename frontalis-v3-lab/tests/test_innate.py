"""生得の層: 語彙、効用、寄与の分解（SPEC 第3節）。"""
import unittest

import numpy as np

from helpers import make  # noqa: F401  (パス設定のため)
from npnv3 import innate


class InnateTests(unittest.TestCase):
    def test_vocabulary_covers_spec_minimum(self):
        """SPEC の最低限の語彙（18 型）に、v3.1 で credit_shared を加えた。"""
        need = {"connection", "rejection", "support", "betrayal", "harm", "praise", "criticism", "credit_taken",
                "credit_given", "credit_shared", "taught", "achievement", "failure", "autonomy_gain", "autonomy_loss", "novelty",
                "monotony", "safety_threat", "comfort"}
        self.assertEqual(set(innate.OUTCOME_TYPES), need)
        self.assertEqual(innate.M.shape, (19, 5))       # v3.0 の 18 型 + credit_shared
        self.assertTrue(np.all(innate.M[innate.TYPE_INDEX['praise']] == 0))   # v3.1: praise は生得の効果を持たない

    def test_utility_shapes(self):
        h = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertTrue(np.allclose(innate.utility(h)[:4], 3 * (1 - np.exp(-1 / 3))))
        self.assertEqual(innate.utility(h)[4], 0.0)                       # h = h* で罰なし
        self.assertAlmostEqual(innate.utility(np.array([1, 1, 1, 1, 0.4]))[4], -0.5 * 0.36)
        self.assertEqual(innate.utility(np.array([1, 1, 1, 1, 2.0]))[4], 0.0)  # 超過側に効用はない
        a = innate.utility(np.array([[1.0] * 5, [2.0] * 5, [3.0] * 5, [4.0] * 5]))[:, 0]   # 等間隔の h
        self.assertTrue(np.all(np.diff(a) > 0) and np.all(np.diff(np.diff(a)) < 0))   # 単調増加・逓減

    def test_contributions_sum_to_event_utility_change(self):
        types = ["connection", "credit_taken", "taught", "achievement", "betrayal", "harm"]
        mags = [1.0, 0.8, 0.6, 0.8, 0.5, 0.4]
        rows = innate.outcome_rows(types, mags)
        for h in (np.ones(5), np.array([3.0, 0.5, 2.0, 1.0, 0.6]), np.array([5.9, 0.0, 1.0, 1.0, 0.2])):
            dh, du, contrib = innate.contributions(h, rows)
            nz = np.abs(dh) > 1e-12
            self.assertTrue(np.allclose(contrib.sum(axis=0)[nz], du[nz], atol=1e-12))
            self.assertTrue(np.all(contrib[:, ~nz] == 0))

    def test_zero_net_change_gives_zero_contribution(self):
        rows = innate.outcome_rows(["connection", "rejection"], [1.0, 1.0])      # R: +0.30 と -0.30 で相殺
        dh, du, contrib = innate.contributions(np.ones(5), rows)
        self.assertEqual(dh[0], 0.0)
        self.assertEqual(du[0], 0.0)
        self.assertTrue(np.all(contrib[:, 0] == 0.0))

    def test_h_dynamics(self):
        h = np.ones(5)
        h2 = innate.next_h(h, np.array([0.3, 0, 0, 0, 0]))
        self.assertAlmostEqual(h2[0], 1.3 + 0.1 * (1 - 1.3))
        self.assertTrue(np.all(innate.next_h(np.full(5, 5.9), np.full(5, 1.0)) <= 6.0))
        self.assertTrue(np.all(innate.next_h(np.full(5, 0.05), np.full(5, -1.0)) >= 0.0))

    def test_spec_example_delta_h(self):
        """SPEC 第2節の例の Δh は、docs/04 第6節の値（R +0.30, C +0.24, A -0.24, S +0.18）と一致する。"""
        rows = innate.outcome_rows(["connection", "achievement", "credit_taken", "taught"], [1.0, 0.8, 0.8, 0.6])
        dh = rows.sum(axis=0)
        self.assertTrue(np.allclose(dh, [0.30, 0.24, -0.24, 0.18, 0.0]))


if __name__ == "__main__":
    unittest.main()
