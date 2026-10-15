"""U3 効果表 M への頑健性: 価値が M の写しにならず、経験の関数として残るか。

SPEC-v3.1 第 3 節: 効果表 M の各要素を ±30% の乱数で揺らす（種ごとに別）、または結果型の名前を一貫して置換する。
合格: U1(a) の f の人格間の順位と、U2 の符号が、元の M のときと 80% 以上の本で一致。

測り方（実装前に固定した細部）:
  「±30% の乱数で揺らす」を採る（名前の置換は、実装が名前に依存しないことしか見ないので、ここでは行わない）。
  M の 0 でない要素に、種ごとの一様乱数 U(−0.3, +0.3) を掛ける（0 の要素は 0 のまま。praise の行は 0 のまま）。
  同じ種の同じ経験史を、元の M と揺らした M の両方で通す。
  順位 = h と m を揃えた f の、R・K・X の昇順の並び。符号 = (H1 の C_praise の符号, H2 の C_praise の符号)。
  順位の一致率と符号の一致率が、どちらも 0.8 以上。

W5（SPEC-v3.2 第 2 節）: U3 の順位一致を、順位をランダムに入れ替えた場合の一致率（偶然水準）と比べる。合格: 一致率 − 偶然水準 ≥ 0.15。
  測り方（実装前に固定した細部）: 偶然水準 = 元の M の順位の列と、揺らした M の順位の列を、種の間でランダムに組み替えたとき
  （20 本の順位の並びを 10000 回ランダムに並べ替えて元の列と突き合わせる）の一致率の平均。順位の分布が偏っている（大半が同じ順位）と
  偶然水準が高くなるので、単純な 1/6 より厳しい基準になる。U3 と同じ計算結果（rows）を使う。
"""
import unittest

import numpy as np

from helpers import KINDS, SEEDS, align_shell, dump, load_json, make, mh, persona_gen, pmap
from npnv3 import innate

NEW = load_json("new_event.json")


def ranking(seed, table):
    f = {}
    for k in KINDS:
        n = make(seed, effect_table=table)
        for e in mh.generate_history(k, seed, 36):
            n.experience(e)
        f[k] = align_shell(n).appraise(NEW)["f"]
    return tuple(sorted(KINDS, key=lambda k: f[k])), f


def praise_signs(seed, table):
    out = []
    for var in ("H1", "H2"):
        n = make(seed, effect_table=table)
        for e in mh.generate_praise_history(var, seed):
            n.experience(e)
        out.append(int(np.sign(n.probe({"cue": "type:praise"})["C"])))
    return tuple(out)


def worker(seed):
    pert = innate.perturb_table(np.array(innate.M), np.random.default_rng([seed, 31]), 0.3)
    r0, f0 = ranking(seed, None)
    r1, f1 = ranking(seed, pert)
    s0, s1 = praise_signs(seed, None), praise_signs(seed, pert)
    return {"seed": seed, "rank_orig": r0, "rank_pert": r1, "sign_orig": s0, "sign_pert": s1, "f_orig": f0, "f_pert": f1}


_ROWS = None


def get_rows():
    """U3 と W5 で同じ計算を共有する（モジュール内で 1 回だけ）。"""
    global _ROWS
    if _ROWS is None:
        _ROWS = pmap(worker, SEEDS)
    return _ROWS


class U3EffectTableRobustness(unittest.TestCase):
    def test_rank_and_sign_survive_perturbing_M(self):
        rows = get_rows()
        rank_agree = float(np.mean([r["rank_orig"] == r["rank_pert"] for r in rows]))
        sign_agree = float(np.mean([r["sign_orig"] == r["sign_pert"] for r in rows]))
        dump("U03", {"rank_agreement": rank_agree, "sign_agreement": sign_agree,
                     "original_ranking_counts": {" < ".join(k): sum(r["rank_orig"] == k for r in rows)
                                                 for k in {r["rank_orig"] for r in rows}},
                     "perturbed_ranking_counts": {" < ".join(k): sum(r["rank_pert"] == k for r in rows)
                                                  for k in {r["rank_pert"] for r in rows}},
                     "sign_patterns_orig": {str(k): sum(r["sign_orig"] == k for r in rows) for k in {r["sign_orig"] for r in rows}},
                     "sign_patterns_pert": {str(k): sum(r["sign_pert"] == k for r in rows) for k in {r["sign_pert"] for r in rows}}})
        self.assertGreaterEqual(rank_agree, 0.8)
        self.assertGreaterEqual(sign_agree, 0.8)


class W5ChanceLevel(unittest.TestCase):
    def test_rank_agreement_beats_chance_level(self):
        rows = get_rows()
        orig = [r["rank_orig"] for r in rows]
        pert = [r["rank_pert"] for r in rows]
        agree = float(np.mean([a == b for a, b in zip(orig, pert)]))
        rng = np.random.default_rng(2024)
        chance = float(np.mean([np.mean([a == pert[j] for a, j in zip(orig, rng.permutation(len(pert)))]) for _ in range(10000)]))
        # 参考: 順位が 3! = 6 通りの一様な並びなら 1/6
        dump("W05", {"rank_agreement": agree, "chance_level_shuffled_pairing": chance, "difference": agree - chance,
                     "uniform_chance_1_over_6": 1.0 / 6.0, "n_distinct_rankings_orig": len(set(orig)),
                     "ranking_counts_orig": {" < ".join(k): orig.count(k) for k in set(orig)},
                     "ranking_counts_pert": {" < ".join(k): pert.count(k) for k in set(pert)}})
        self.assertGreaterEqual(agree - chance, 0.15)


if __name__ == "__main__":
    unittest.main()
