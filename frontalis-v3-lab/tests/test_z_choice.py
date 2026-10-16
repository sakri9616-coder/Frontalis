"""Z1〜Z7 選択による関心学習: 正当化と、経験に根拠のある価値を分ける（SPEC-v3.3 第 3 節）。判定基準は実装前に固定されたまま（閾値は一度も変えていない）。

プロトコルの細部（SPEC に細部がなかった点）は tests/yz_protocol.py の冒頭に、数値を見る前に書いた。乱数種 0〜19。

  Z1 迷った選択でだけ生じる   易しい選択（|ΔQ| > 0.3）と迷う選択（|ΔQ| < 0.05）を 20 回ずつ: 易しい選択での J の総変化 < 0.01、迷う選択での選んだ側と選ばなかった側の J の差 ≥ 0.02
  Z2 有界・正帰還なし         同じ迷う二択を 1000 回: max|J| ≤ J_max、1e-3 の摂動の比 ≤ 1（中央値。U6/W6 と同じ。最大も報告）、後半 500 回の J の増分が前半の 10% 以下（20 本の平均）
  Z3 根拠のない正当化は薄れる J を作った後、その選択肢が現れない経験を 100 回: J の残存 ≤ 20%、同じ期間の経験に根拠のある W の残存 ≥ 70%
  Z4 経験が正当化に勝つ       J で好むようになった選択肢に反する結果（害）を与える: 5 回以内に選好が反転する（全種で）
  Z5 主体性の重み             同じ |Δu| の経験を「選んだ」場合と「強制された」場合: a の変化の比 ≥ 1.5
  Z6 自己強化なし             効用の変化がない（Δu = 0）選択を 100 回: a の変化 < 1e-6
  Z7 J は経験の教師にならない  同じ経験列を強制で与え、J だけを入れ替える: W と C の学習がビット単位で同一
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, pmap
import yz_protocol as Y
from npnv3 import Config

J_MAX = Config().j_max
_CACHE = {}


def rows(name):
    if name not in _CACHE:
        fn = {"z1": Y.z1_worker, "z2": Y.z2_worker, "z3": Y.z3_worker, "z4": Y.z4_worker, "z5": Y.z5_worker, "z6": Y.z6_worker, "z7": Y.z7_worker}[name]
        _CACHE[name] = pmap(fn, SEEDS)
    return _CACHE[name]


def mean(vs):
    return float(np.mean(list(vs)))


class ZChoice(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        out = {}
        if "z1" in _CACHE:
            r = _CACHE["z1"]
            out["Z1"] = {"easy_min_abs_dQ (precondition > 0.3)": min(x["easy_min_abs_dQ"] for x in r),
                         "easy_total_update_max": max(x["easy_total_update"] for x in r), "easy_net_change_max": max(x["easy_net_change"] for x in r),
                         "hesitant_max_abs_dQ (precondition < 0.05)": max(x["hes_max_abs_dQ"] for x in r),
                         "hesitant_min_gap": min(x["hes_min_gap"] for x in r), "hesitant_mean_gap": mean(x["hes_mean_gap"] for x in r)}
        if "z2" in _CACHE:
            r = _CACHE["z2"]
            d1, d2 = mean(x["delta1"] for x in r), mean(x["delta2"] for x in r)
            out["Z2"] = {"max_abs_J_over_seeds": max(x["max_abs_J"] for x in r), "J_max": J_MAX,
                         "perturbation_ratio_median": float(np.median([x["ratio"] for x in r])), "perturbation_ratio_max": max(x["ratio"] for x in r),
                         "seeds_with_choice_flips_between_base_and_perturbed": int(sum(x["flips"] > 0 for x in r)),
                         "delta1_mean (norm J_500 - norm J_0)": d1, "delta2_mean (norm J_1000 - norm J_500)": d2, "delta2_over_delta1": d2 / d1,
                         "delta2_over_delta1_per_seed": [x["delta2"] / x["delta1"] for x in r], "norm_J_500_mean": mean(x["norm_500"] for x in r),
                         "norm_J_1000_mean": mean(x["norm_1000"] for x in r), "J_end_by_seed": [x["J_end"] for x in r]}
        if "z3" in _CACHE:
            r = _CACHE["z3"]
            out["Z3"] = {"J_remaining_mean": mean(x["J_remaining"] for x in r), "J_remaining_max": max(x["J_remaining"] for x in r),
                         "W_remaining_min": min(x["W_remaining"] for x in r), "W_remaining_mean": mean(x["W_remaining"] for x in r),
                         "J_before_mean": mean(x["J_before"] for x in r), "W_before_mean": mean(x["W_before"] for x in r)}
        if "z4" in _CACHE:
            r = _CACHE["z4"]
            out["Z4"] = {"reversed_at_by_seed (None = not within 10)": [x["reversed_at"] for x in r],
                         "seeds_reversed_within_5": int(sum(x["reversed_at"] is not None and x["reversed_at"] <= 5 for x in r)),
                         "seeds_never_reversed_within_10": int(sum(x["reversed_at"] is None for x in r)),
                         "gap_before_mean": mean(x["gap_before"] for x in r), "J_pref_mean": mean(x["J_pref"] for x in r),
                         "gap_after_k_harms_mean": [mean(x["gaps"][k] for x in r) for k in range(10)]}
        if "z5" in _CACHE:
            r = _CACHE["z5"]
            out["Z5"] = {"ratio_mean": mean(x["ratio"] for x in r), "ratio_min": min(x["ratio"] for x in r), "attempts_max": max(x["attempts"] for x in r),
                         "d_forced_mean": mean(x["d_forced"] for x in r), "d_chosen_mean": mean(x["d_chosen"] for x in r)}
        if "z6" in _CACHE:
            r = _CACHE["z6"]
            out["Z6"] = {"a_change_max": max(x["a_change"] for x in r), "max_abs_du_total": max(x["max_abs_du_total"] for x in r), "n_learned": r[0]["n_learned"]}
        if "z7" in _CACHE:
            r = _CACHE["z7"]
            out["Z7"] = {"all_identical": all(all(x["identical"].values()) for x in r), "checked": sorted(r[0]["identical"]),
                         "min_over_seeds_of_max_abs_Q_difference (non-vacuous if > 0)": min(x["max_abs_Q_difference"] for x in r), "n_experiences": r[0]["n"]}
        dump("Z01_Z07", out)

    def test_Z1_justification_arises_only_in_hesitant_choices(self):
        r = rows("z1")
        for x in r:
            self.assertGreater(x["easy_min_abs_dQ"], 0.3, x["seed"])                # 前提: 易しい選択
            self.assertLess(x["hes_max_abs_dQ"], 0.05, x["seed"])                   # 前提: 迷う選択
            self.assertLess(x["easy_total_update"], 0.01, x["seed"])
            self.assertGreaterEqual(x["hes_min_gap"], 0.02, x["seed"])

    def test_Z2_bounded_and_no_positive_feedback(self):
        r = rows("z2")
        for x in r:
            self.assertLessEqual(x["max_abs_J"], J_MAX, x["seed"])
        self.assertLessEqual(float(np.median([x["ratio"] for x in r])), 1.0)
        d1, d2 = mean(x["delta1"] for x in r), mean(x["delta2"] for x in r)
        self.assertGreater(d1, 0.0)                                                 # 前半では J ができている（検査が空でない）
        self.assertLessEqual(d2, 0.1 * d1)

    def test_Z3_unfounded_justification_fades_and_founded_value_stays(self):
        r = rows("z3")
        self.assertLessEqual(mean(x["J_remaining"] for x in r), 0.2)
        self.assertGreaterEqual(min(x["W_remaining"] for x in r), 0.7)
        self.assertGreater(min(x["J_before"] for x in r), 0.01)                     # 前提: J ができている（残存の比は尺度に依らない。検査が空でないことの確認だけ）

    def test_Z4_experience_beats_justification(self):
        for x in rows("z4"):
            self.assertIsNotNone(x["reversed_at"], (x["seed"], x["gaps"]))
            self.assertLessEqual(x["reversed_at"], 5, (x["seed"], x["gaps"]))

    def test_Z5_agency_weight(self):
        r = rows("z5")
        self.assertEqual(sum(x["ratio"] is None for x in r), 0)
        self.assertGreaterEqual(mean(x["ratio"] for x in r), 1.5)

    def test_Z6_no_self_reinforcement_without_utility_change(self):
        for x in rows("z6"):
            self.assertEqual(x["max_abs_du_total"], 0.0, x["seed"])                 # 前提: Δu = 0
            self.assertLess(x["a_change"], 1e-6, x["seed"])

    def test_Z7_justification_is_not_a_teacher_of_experience(self):
        for x in rows("z7"):
            self.assertTrue(all(x["identical"].values()), (x["seed"], x["identical"]))
            self.assertGreater(x["max_abs_Q_difference"], 1e-3, x["seed"])          # 検査が空でない: Q は J で違う


if __name__ == "__main__":
    unittest.main()
