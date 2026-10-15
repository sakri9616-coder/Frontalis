"""U5 閉ループ: 人格が、経験の中身を変える（接近・回避）。

SPEC-v3.1 第 3 節: 共同作業の申し出（optional）を 40 回、各人格に与える。
合格: K 型の接近率が R 型より 0.15 以上低い（20 本平均）。回避した申し出からは信念が更新されない（計数が不変）。

測り方（実装前に固定した細部）:
  K 型・R 型は乱数種ごとの生成器の経験史（36 件）で育てる。申し出は generate_offers(kind, seed, 40): 型の世界から抽出した結果
  （接近した場合に起きること）を持つ optional: true の共同作業。各人格が自分の型の世界の申し出を受ける（その人生が続く閉ループ）。
  相手は 6 人の個体が繰り返し現れる。接近率 = 40 回のうち接近した割合。
  回避の検査: 回避した申し出の前後で、信念の計数（個体・文脈・自分/世界・π・型の大きさ）が完全に同じ。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, mh, persona_gen, pmap


def belief_snapshot(b):
    return [b.N1.copy(), b.N0.copy(), b.CP1.copy(), b.CP0.copy(), b.CS1.copy(), b.CS0.copy(), b.pi.copy(),
            b.mag.copy(), b.mag_n.copy(), np.array([b.n_obs])]


def worker(seed):
    out = {"seed": seed, "violations": 0, "avoided": 0}
    for kind in ("K", "R"):
        n = persona_gen(kind, seed)
        approached = []
        for e in mh.generate_offers(kind, seed, 40):
            before = belief_snapshot(n.belief)
            ap = n.experience(e)
            approached.append(bool(ap["approached"]))
            if not ap["approached"]:
                out["avoided"] += 1
                after = belief_snapshot(n.belief)
                if not all(np.array_equal(x, y) for x, y in zip(before, after)):
                    out["violations"] += 1
        out[f"rate_{kind}"] = float(np.mean(approached))
        out[f"rate_first20_{kind}"] = float(np.mean(approached[:20]))
        out[f"rate_last20_{kind}"] = float(np.mean(approached[20:]))
    return out


class U5ClosedLoop(unittest.TestCase):
    def test_pessimistic_persona_approaches_less_and_avoidance_leaves_beliefs(self):
        rows = pmap(worker, SEEDS)
        rk = float(np.mean([r["rate_K"] for r in rows])); rr = float(np.mean([r["rate_R"] for r in rows]))
        dump("U05", {"approach_rate_K": rk, "approach_rate_R": rr, "difference_R_minus_K": rr - rk,
                     "first20": [float(np.mean([r["rate_first20_K"] for r in rows])), float(np.mean([r["rate_first20_R"] for r in rows]))],
                     "last20": [float(np.mean([r["rate_last20_K"] for r in rows])), float(np.mean([r["rate_last20_R"] for r in rows]))],
                     "avoided_offers_total": int(sum(r["avoided"] for r in rows)),
                     "belief_changes_after_avoidance": int(sum(r["violations"] for r in rows)),
                     "seeds_R_minus_K_ge_0.15": int(sum(r["rate_R"] - r["rate_K"] >= 0.15 for r in rows))})
        self.assertGreaterEqual(rr - rk, 0.15)
        self.assertEqual(sum(r["violations"] for r in rows), 0)
        self.assertGreater(sum(r["avoided"] for r in rows), 0)


if __name__ == "__main__":
    unittest.main()
