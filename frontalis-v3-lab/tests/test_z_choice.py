"""Z1〜Z8 選択による関心学習: 正当化と、経験に根拠のある価値を分ける（SPEC-v3.3 第 3 節、第 8 節）。判定基準は実装前に固定されたまま（閾値は一度も変えていない）。

プロトコルの細部（SPEC に細部がなかった点）は tests/yz_protocol.py の冒頭に、数値を見る前に書いた。乱数種 0〜19。

**読み方（SPEC-v3.3 8.1〜8.2）: ここは実験条件 Z-orig（第 3 節の J: 毎経験の減衰 λ_J = 0.02、上限 J_max = 0.2、葛藤による自己制限）の Z1〜Z7 を、事前登録どおりの定義で判定する。
Z2〜Z4 が不合格でも、それは「この条件の J が持続する（Z3）・反証に抵抗する（Z4）・摂動で分岐する（Z2）」という性質の記録であり、数値的な故障を意味しない。
故障の判定は B1〜B3（test_b_health.py）と、Z-persist の P1〜P7（test_p_persist.py）。** 第 7.3 節の Z 系の再定義は第 8.2 節で撤回されている（定義は変えていない）。

  Z1 迷った選択でだけ生じる   易しい選択（|ΔQ| > 0.3）と迷う選択（|ΔQ| < 0.05）を 20 回ずつ: 易しい選択での J の総変化 < 0.01、迷う選択での選んだ側と選ばなかった側の J の差 ≥ 0.02
  Z2 有界・正帰還なし         同じ迷う二択を 1000 回: max|J| ≤ J_max、1e-3 の摂動の比 ≤ 1（中央値。U6/W6 と同じ。最大も報告）、後半 500 回の J の増分が前半の 10% 以下（20 本の平均）
  Z3 根拠のない正当化は薄れる J を作った後、その選択肢が現れない経験を 100 回: J の残存 ≤ 20%、同じ期間の経験に根拠のある W の残存 ≥ 70%
  Z4 経験が正当化に勝つ       J で好むようになった選択肢に反する結果（害）を与える: 5 回以内に選好が反転する（全種で）
  Z5 主体性の重み             同じ |Δu| の経験を「選んだ」場合と「強制された」場合: a の変化の比 ≥ 1.5
  Z6 自己強化なし             効用の変化がない（Δu = 0）選択を 100 回: a の変化 < 1e-6
  Z7 J は経験の教師にならない  同じ経験列を強制で与え、J だけを入れ替える: W と C の学習がビット単位で同一（Z-orig の性質。Z-persist には適用しない: 反証への抵抗は教師を変える。P5）
  Z8 正当化の追跡（追加）      二択ごとに、葛藤 c・ΔQ・J の変化を台帳に記録する。explain_preference が予期 Q を「経験に根拠のある部分」と「正当化 J の部分」に分けて返す。保存・復元で完全一致する
                              （Z-orig と Z-persist の両方で測る。判定は恒等式（≤ 1e-12）と完全一致）
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
        if name.startswith("z8_"):
            _CACHE[name] = pmap(Y.z8_worker, [(s, name[3:]) for s in SEEDS])
            return _CACHE[name]
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
        z8 = {}
        for mode in ("orig", "persist"):
            if f"z8_{mode}" in _CACHE:
                r = _CACHE[f"z8_{mode}"]
                z8[mode] = {"max_errors": {k: max(x["errors"][k] for x in r) for k in r[0]["errors"]}, "explain_is_pure_all": all(x["explain_is_pure"] for x in r),
                            "restore_same_state_all": all(x["restore_same_state"] for x in r), "restore_same_explain_all": all(x["restore_same_explain"] for x in r),
                            "restore_same_future_all": all(x["restore_same_future"] for x in r), "choices_per_seed": r[0]["n_choices"],
                            "seeds_where_J_decides_the_preference (Q and grounded-only disagree)": int(sum(x["example_J_decides"]["justification_decides"] for x in r)),
                            "example_seed0_explain_preference": r[0]["example_J_decides"]}
        if z8:
            dump("Z08", z8)

    def test_Z1_justification_arises_only_in_hesitant_choices(self):
        r = rows("z1")
        for x in r:
            self.assertGreater(x["easy_min_abs_dQ"], 0.3, x["seed"])                # 前提: 易しい選択
            self.assertLess(x["hes_max_abs_dQ"], 0.05, x["seed"])                   # 前提: 迷う選択
            self.assertLess(x["easy_total_update"], 0.01, x["seed"])
            self.assertGreaterEqual(x["hes_min_gap"], 0.02, x["seed"])

    def test_Z2_bounded_and_no_positive_feedback(self):
        """事前登録どおりの判定。不合格の項目（摂動の比、後半の増分）があれば、それは Z-orig の J の性質の記録（SPEC 8.2）。"""
        r = rows("z2")
        for x in r:
            self.assertLessEqual(x["max_abs_J"], J_MAX, x["seed"])
        self.assertLessEqual(float(np.median([x["ratio"] for x in r])), 1.0)
        d1, d2 = mean(x["delta1"] for x in r), mean(x["delta2"] for x in r)
        self.assertGreater(d1, 0.0)                                                 # 前半では J ができている（検査が空でない）
        self.assertLessEqual(d2, 0.1 * d1)

    def test_Z3_unfounded_justification_fades_and_founded_value_stays(self):
        """事前登録どおりの判定（J の残存 ≤ 20%、W の残存 ≥ 70%）。不合格なら、それは Z-orig の J の性質の記録（SPEC 8.2）。"""
        r = rows("z3")
        self.assertLessEqual(mean(x["J_remaining"] for x in r), 0.2)
        self.assertGreaterEqual(min(x["W_remaining"] for x in r), 0.7)
        self.assertGreater(min(x["J_before"] for x in r), 0.01)                     # 前提: J ができている（残存の比は尺度に依らない。検査が空でないことの確認だけ）

    def test_Z4_experience_beats_justification(self):
        """事前登録どおりの判定（5 回以内に選好が反転する。全種で）。不合格の場合、それは「この条件（Z-orig）の J が反証に抵抗する」という性質の記録で、数値的な故障ではない
        （SPEC 8.2。故障の判定は B1〜B3）。何回で反転したかは Z01_Z07.json に残る。"""
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

    def _z8(self, mode):
        for x in rows(f"z8_{mode}"):
            for k, v in x["errors"].items():
                self.assertLessEqual(v, 1e-12, (x["seed"], k))                        # 台帳の c・ΔQ・ΔJ と explain_preference の恒等式
            self.assertEqual(x["ledger_choice_entries"], x["n_choices"], x["seed"])    # 二択ごとに台帳の記録がある
            self.assertTrue(x["explain_is_pure"], x["seed"])                          # 説明は状態も乱数も変えない
            self.assertTrue(x["restore_same_state"], x["seed"])                       # 保存・復元で完全一致
            self.assertTrue(x["restore_same_explain"], x["seed"])
            self.assertTrue(x["restore_same_future"], x["seed"])                      # 続きの選択も同一
            e = x["example_J_decides"]
            self.assertNotEqual(e["delta"]["justification"], 0.0, x["seed"])          # 検査が空でない: J の部分がある
            self.assertAlmostEqual(e["delta"]["Q"], e["delta"]["grounded"] + e["delta"]["justification"], places=12)

    def test_Z8_choice_ledger_and_explain_preference_Z_orig(self):
        self._z8("orig")

    def test_Z8_choice_ledger_and_explain_preference_Z_persist(self):
        self._z8("persist")


if __name__ == "__main__":
    unittest.main()
