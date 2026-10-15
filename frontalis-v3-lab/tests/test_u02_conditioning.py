"""U2 随伴による価値の形成: 生得の効果を持たない手がかりが、経験の随伴関係から価値を得る。

SPEC-v3.1 第 3 節: 生得の効果を持たない手がかり（praise）が、経験史 H1 では後の害の前に、H2 では後の支えの前に現れる。
それ以外の統計は同じ。合格: praise の条件づけられた価値 C の符号が、H1 で負、H2 で正（20 本中 18 本以上）。
no_conditioning では差が 0。

測り方（実装前に固定した細部）:
  H1/H2 は data/make_histories.py の generate_praise_history。24 回の「褒められる → 同じ相手から害（H1）/ 支え（H2）」の対
  （同じ文脈の連続）に、約半数で独りの出来事が挟まる。場面・大きさ・相手・合間の出来事は種が同じなら H1 と H2 で同一。
  praise の価値 = 手がかり "type:praise" の C。praise は効果表 M で全欲求 0（生得の効果なし）。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, make, mh, pmap
from npnv3 import innate


def worker(seed):
    out = {"seed": seed}
    for name, kw in (("full", {}), ("no_conditioning", {"no_conditioning": True})):
        for var in ("H1", "H2"):
            n = make(seed, **kw)
            for e in mh.generate_praise_history(var, seed):
                n.experience(e)
            out[f"{name}_{var}"] = n.probe({"cue": "type:praise"})["C"]
    return out


class U2Conditioning(unittest.TestCase):
    def test_praise_value_follows_contingency(self):
        self.assertTrue(np.all(innate.M[innate.TYPE_INDEX["praise"]] == 0))        # 生得の効果なし
        rows = pmap(worker, SEEDS)
        ok = sum(r["full_H1"] < 0 < r["full_H2"] for r in rows)
        dump("U02", {"seeds_with_H1_negative_and_H2_positive": ok, "n_seeds": len(rows),
                     "C_praise_H1_mean": float(np.mean([r["full_H1"] for r in rows])),
                     "C_praise_H2_mean": float(np.mean([r["full_H2"] for r in rows])),
                     "C_praise_H1": [r["full_H1"] for r in rows], "C_praise_H2": [r["full_H2"] for r in rows],
                     "no_conditioning": [[r["no_conditioning_H1"], r["no_conditioning_H2"]] for r in rows]})
        self.assertGreaterEqual(ok, 18)
        for r in rows:                                                              # no_conditioning では差が 0
            self.assertEqual(r["no_conditioning_H1"], 0.0)
            self.assertEqual(r["no_conditioning_H2"], 0.0)


if __name__ == "__main__":
    unittest.main()
