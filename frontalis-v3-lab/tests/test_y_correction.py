"""Y1〜Y7 価値の再帰属: 事実の訂正と、当時の主観的経験を分ける（SPEC-v3.3 第 2 節）。判定基準は実装前に固定されたまま（閾値は一度も変えていない）。

プロトコルの細部（SPEC に細部がなかった点）は tests/yz_protocol.py の冒頭に、数値を見る前に書いた。乱数種 0〜19（種ごとに手柄の大きさが変わる。NPN の種は MB の標本だけに効く）。
判定は 20 本の平均どうしの比（E 系と同じ。Y5 は全種で完全一致）。

  Y1 事実の訂正       A の P(credit_taken) が、対照の値まで 80% 以上近づく: (P_mis − P_訂正後)/(P_mis − P_ctrl) ≥ 0.8
  Y2 自動消去しない   訂正の直後の、A への価値の対照との差が、訂正前の差の 30% 以上、90% 以下で残る
  Y3 再学習できる     訂正の後、A と良い経験を 10 回: 残った差が、訂正直後の 30% 以下
  Y4 対照             retroactive_value では、訂正直後に差の 90% 以上が消える
  Y5 当時の記録の保存 訂正の前後で、台帳の当時の f、φ、Δu、感情（経験の記録の全項目）が完全一致
  Y6 訂正後の解釈     訂正後に A が関わる曖昧な出来事を解釈: 「取られた」の重みの、対照との差が 50% 以上縮む
  Y7 訂正がない場合   訂正を与えない誤解: 信念も価値も、対照との差が 70% 以上残る
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, pmap
import yz_protocol as Y

_ROWS = None


def rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = pmap(Y.y_worker, SEEDS)
    return _ROWS


def mean(vs):
    return float(np.mean(list(vs)))


def m(f):
    return mean(f(r) for r in rows())


def stats():
    """各判定の数値（20 本の平均どうしの比）。"""
    rs = rows()
    P_mis, P_ctrl, P_post = m(lambda r: r["pre"]["P_mis"]), m(lambda r: r["pre"]["P_ctrl"]), m(lambda r: r["post"]["P"])
    l_mis, l_ctrl, l_post, l_retro = m(lambda r: r["pre"]["like_mis"]), m(lambda r: r["pre"]["like_ctrl"]), m(lambda r: r["post"]["like"]), m(lambda r: r["retro"]["like"])
    d_pre, d_post, d_retro = l_mis - l_ctrl, l_post - l_ctrl, l_retro - l_ctrl
    l10_c, l10_k = m(lambda r: r["Y3"]["like_corrected"]), m(lambda r: r["Y3"]["like_ctrl"])
    w_mis, w_ctrl, w_post = m(lambda r: r["pre"]["w_mis"]), m(lambda r: r["pre"]["w_ctrl"]), m(lambda r: r["post"]["w"])
    P7m, P7c = m(lambda r: r["Y7"]["P_mis"]), m(lambda r: r["Y7"]["P_ctrl"])
    l7m, l7c = m(lambda r: r["Y7"]["like_mis"]), m(lambda r: r["Y7"]["like_ctrl"])
    return {
        "Y1": {"P_mis": P_mis, "P_ctrl": P_ctrl, "P_after_corrections": P_post, "closeness": (P_mis - P_post) / (P_mis - P_ctrl),
               "closeness_per_seed": [(r["pre"]["P_mis"] - r["post"]["P"]) / (r["pre"]["P_mis"] - r["pre"]["P_ctrl"]) for r in rs],
               "reference_individual_part_p_ind": {"mis": m(lambda r: r["pre"]["Pind_mis"]), "ctrl": m(lambda r: r["pre"]["Pind_ctrl"]), "after": m(lambda r: r["post"]["Pind"]),
                                                   "closeness": (m(lambda r: r["pre"]["Pind_mis"]) - m(lambda r: r["post"]["Pind"])) / (m(lambda r: r["pre"]["Pind_mis"]) - m(lambda r: r["pre"]["Pind_ctrl"]))},
               "reference_recall_after_correction_option (not the default; P compared with the same ctrl)": {
                   "P": m(lambda r: r["post_recall_option"]["P"]), "closeness": (P_mis - m(lambda r: r["post_recall_option"]["P"])) / (P_mis - P_ctrl)}},
        "Y2": {"like_mis": l_mis, "like_ctrl": l_ctrl, "like_after_corrections": l_post, "d_pre": d_pre, "d_post": d_post, "remaining_fraction": d_post / d_pre,
               "remaining_per_seed": [(r["post"]["like"] - r["pre"]["like_ctrl"]) / (r["pre"]["like_mis"] - r["pre"]["like_ctrl"]) for r in rs],
               "reference_conditioned_C_of_A (not changed by corrections)": {"mis": m(lambda r: r["pre"]["C_mis"]), "ctrl": m(lambda r: r["pre"]["C_ctrl"]), "after": m(lambda r: r["post"]["C"])}},
        "Y3": {"like_corrected_after_10_good": l10_c, "like_ctrl_after_10_good": l10_k, "d_10": l10_c - l10_k, "d_post": d_post, "ratio": abs(l10_c - l10_k) / abs(d_post)},
        "Y4": {"like_retroactive": l_retro, "d_retro": d_retro, "d_pre": d_pre, "ratio_to_pre": abs(d_retro) / abs(d_pre), "fraction_removed": 1.0 - d_retro / d_pre,
               "reference_after_10_good": m(lambda r: r["Y3_retro"]["like"]) - l10_k},
        "Y5": {"all_seeds_ledger_identical": all(r["Y5_ledger_unchanged"] for r in rs)},
        "Y6": {"w_mis": w_mis, "w_ctrl": w_ctrl, "w_after_corrections": w_post, "shrink": 1.0 - (w_post - w_ctrl) / (w_mis - w_ctrl)},
        "Y7": {"belief_remaining": (P7m - P7c) / (P_mis - P_ctrl), "value_remaining": (l7m - l7c) / d_pre,
               "P_mis_after": P7m, "P_ctrl_after": P7c, "like_mis_after": l7m, "like_ctrl_after": l7c},
    }


class YCorrection(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        dump("Y01_Y07", stats())

    def test_Y1_the_fact_is_corrected(self):
        self.assertGreaterEqual(stats()["Y1"]["closeness"], 0.8)

    def test_Y2_the_value_is_not_erased_automatically(self):
        r = stats()["Y2"]["remaining_fraction"]
        self.assertGreaterEqual(r, 0.3)
        self.assertLessEqual(r, 0.9)

    def test_Y3_the_remaining_difference_can_be_relearned(self):
        self.assertLessEqual(stats()["Y3"]["ratio"], 0.3)

    def test_Y4_retroactive_value_erases_the_difference(self):
        self.assertLessEqual(stats()["Y4"]["ratio_to_pre"], 0.1)

    def test_Y5_the_record_at_the_time_is_unchanged(self):
        for r in rows():
            self.assertTrue(r["Y5_ledger_unchanged"], r["seed"])

    def test_Y6_interpretation_after_the_correction(self):
        self.assertGreaterEqual(stats()["Y6"]["shrink"], 0.5)

    def test_Y7_an_uncorrected_misunderstanding_stays(self):
        s = stats()["Y7"]
        self.assertGreaterEqual(s["belief_remaining"], 0.7)
        self.assertGreaterEqual(s["value_remaining"], 0.7)


if __name__ == "__main__":
    unittest.main()
