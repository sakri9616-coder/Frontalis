"""P1〜P7 Z-persist（強制的な減衰を持たない正当化。SPEC-v3.3 第 8.3〜8.4 節）。判定基準は実装前に固定されたまま（閾値は一度も変えていない）。
プロトコルの細部は tests/p_protocol.py の冒頭に、数値を見る前に書いた（試走で変えた点は CHANGELOG の「v3.3 第2部」に記録）。乱数種 0〜19。

**この条件の合格は「J が持続・増幅・抵抗すること」を問わない。問うのは、数値的な健全性と追跡可能性だけ**（SPEC 8.1。心理的な非合理性と数値的な故障は区別する）。
J の持続の大きさ・反証への抵抗の強さ・何回で選好が反転するかは、合否にせず報告する。

  P1 源の追跡           J のすべての変化に源の種類（spread・reaffirm・counter・overwhelm・cap）が記録され、台帳の源ごとの和が J に完全に一致する
  P2 無条件の正帰還なし 二択の経験や、その選択肢の結果がない区間で、J が一切変化しない（厳密に 0）
  P3 数値的な健全性     二択 2000 回で有限・|J| ≤ J_num・B2・B3（W6 の経験史 + 二択 + 訂正の 2000 経験の B 系の実行（test_b_health.py の Z_persist）も B2・B3 に合格）
  P4 持続の由来         再び選んだ回数だけが違う 2 つの履歴（0 回と 10 回）の後に反証を同じ回数与える。J の残存の差の 95% 以上が、台帳の reaffirm の和と overwhelm の和で説明される
  P5 抵抗の由来         正当化された選択肢への反証で、W の学習の減少分が、台帳の割り引きの記録と一致する（相対誤差 1% 以内）。何回で選好が反転するかは報告のみ
  P6 増幅の由来         再び選ぶことの繰り返しで、J の増加が reaffirm の記録の和に完全に一致する。他の源から増えない
  P7 経験史による個人差 J の持続・抵抗・増幅の違いを、源の記録で分解して報告する（合否なし）

読み方: P4 の「残存の差」は、反証でも反証を受ける選択そのものが再び選ぶこと（reaffirm）になる（コミットメント）ので、反証を与えても reaffirm が起きる。
overwhelm は、予測誤差 δ = f − tanh(Q/s_f) が正当化で割り引ける量 tanh(J) を超えた分で起きる。J が Q に入るので、J が作った期待に結果が届かなければ（繰り返しの満足による飽和を含む）
反証でなくても overwhelm は起きる（P6 の報告項目 "erased_by_other_sources"）。これは数値的な故障ではなく、この機構の性質（CHANGELOG の解釈の節）。
"""
import unittest

import numpy as np

from helpers import KINDS, SEEDS, dump, pmap
import b_protocol as B
import p_protocol as P
import test_b_health as TB
from npnv3 import Config

_CACHE = {}


def rows(name):
    if name not in _CACHE:
        _CACHE[name] = pmap(getattr(P, f"{name}_worker"), SEEDS)
    return _CACHE[name]


def mean(vs):
    return float(np.mean(list(vs)))


class ZPersist(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        out = {"J_num": Config().j_num, "eta_r": Config().eta_r, "rho_dis": Config().rho_dis, "eta_ov": Config().eta_ov}
        if "p1" in _CACHE:
            r = _CACHE["p1"]
            out["P1"] = {"err_source_sums_max": max(x["err_source_sums"] for x in r), "err_cue_total_vs_J_max": max(x["err_cue_total_vs_J"] for x in r),
                         "reaffirm_formula_error_max": max(x["reaffirm_formula_error"] for x in r), "unknown_sources": sorted({s for x in r for s in x["unknown_sources"]}),
                         "event_counts_min_over_seeds": {s: min(x["event_counts"][s] for x in r) for s in P.SOURCES},
                         "event_counts_mean": {s: mean(x["event_counts"][s] for x in r) for s in P.SOURCES}}
        if "p2" in _CACHE:
            r = _CACHE["p2"]
            out["P2"] = {"persist": {"all_seeds_J_unchanged": all(x["persist"]["J_unchanged"] for x in r), "all_sources_unchanged": all(x["persist"]["sources_unchanged"] for x in r),
                                     "new_justification_entries_total": sum(x["persist"]["new_justification_entries"] for x in r),
                                     "J_before_max_min": min(x["persist"]["J_before_max"] for x in r), "learned_min": min(x["persist"]["learned"] for x in r),
                                     "avoided_min": min(x["persist"]["avoided"] for x in r)},
                         "contrast_report_only": {m: {"J_ratio_after_before_mean": mean(x[m]["J_ratio_after_before"] for x in r),
                                                      "new_justification_entries_mean": mean(x[m]["new_justification_entries"] for x in r)} for m in ("persist_with_decay", "orig")}}
        if "p3" in _CACHE:
            r = _CACHE["p3"]
            out["P3"] = {"stress_2000_choices": {"max_abs_J_over_seeds": max(x["max_abs_J"] for x in r), "J_num": r[0]["J_num"], "all_finite": all(x["finite"] for x in r),
                                                 "B2_violations": [(x["seed"], x["B2"]["violations"]) for x in r if x["B2"]["violations"]],
                                                 "B3_violations": [(x["seed"], x["B3_violations"]) for x in r if x["B3_violations"]],
                                                 "event_counts_mean": {s: mean(x["event_counts"][s] for x in r) for s in P.SOURCES}},
                         "W6_history_with_choices_and_corrections": "B01_B03.json の Z_persist"}
        if "p4" in _CACHE:
            r = _CACHE["p4"]
            out["P4"] = {"max_unexplained_fraction": max(x["unexplained_fraction"] for x in r), "min_abs_delta": min(abs(x["delta"]) for x in r),
                         "reaffirm_count_difference": sorted({x["reaffirm_count_difference"] for x in r}),
                         "J_residual_H0_mean": mean(x["H0"]["J_residual"] for x in r), "J_residual_H10_mean": mean(x["H10"]["J_residual"] for x in r),
                         "J_peak_H0_mean": mean(x["H0"]["J_peak_before_refutation"] for x in r), "J_peak_H10_mean": mean(x["H10"]["J_peak_before_refutation"] for x in r),
                         "delta_mean": mean(x["delta"] for x in r), "explained_mean": mean(x["explained"] for x in r),
                         "reaffirm_sum_diff_mean": mean(x["H10"]["sums"]["reaffirm"] - x["H0"]["sums"]["reaffirm"] for x in r),
                         "overwhelm_sum_diff_mean": mean(x["H10"]["sums"]["overwhelm"] - x["H0"]["sums"]["overwhelm"] for x in r),
                         "remaining_fraction_H0_mean": mean(x["H0"]["remaining_fraction"] for x in r), "remaining_fraction_H10_mean": mean(x["H10"]["remaining_fraction"] for x in r),
                         "refutation_chosen_ok_all": all(x["refutation_chosen_ok"] for x in r)}
        if "p5" in _CACHE:
            r = _CACHE["p5"]
            out["P5"] = {"person_rel_error_max": max(x["person"]["rel_error"] for x in r), "R_meas_norm_min": min(x["person"]["R_meas_norm"] for x in r),
                         "R_meas_norm_mean": mean(x["person"]["R_meas_norm"] for x in r), "dW_control_norm_mean": mean(x["person"]["dW_control_norm"] for x in r),
                         "reduction_fraction_of_person_learning_mean": mean(x["person"]["R_meas_norm"] / x["person"]["dW_control_norm"] for x in r),
                         "discount_q_mean": mean(x["discount_q"] for x in r), "teachers_total_equal_all": all(x["teachers_total_equal"] for x in r),
                         "reversal_report_only": {tag: {"seeds_reversed_Q_within_10": int(sum(x["reversal"][tag]["reversed_at_Q"] is not None for x in r)),
                                                        "reversed_at_Q": [x["reversal"][tag]["reversed_at_Q"] for x in r],
                                                        "seeds_reversed_grounded_within_10": int(sum(x["reversal"][tag]["reversed_at_grounded"] is not None for x in r)),
                                                        "gap_Q_after_10_mean": mean(x["reversal"][tag]["gap_Q"][-1] for x in r),
                                                        "J_fav_end_mean": mean(x["reversal"][tag]["J_fav_end"] for x in r)} for tag in ("with_justification", "control")}}
        if "p6" in _CACHE:
            r = _CACHE["p6"]
            out["P6"] = {"err_gross_increase_vs_reaffirm_max": max(x["err_gross_increase_vs_reaffirm"] for x in r), "other_sources_max_positive": max(x["other_sources_max_positive"] for x in r),
                         "err_net_vs_all_events_max": max(x["err_net_vs_all_events"] for x in r), "reaffirm_formula_error_max": max(x["reaffirm_formula_error"] for x in r),
                         "max_conflict_over_seeds": max(x["max_conflict"] for x in r),
                         "gross_increase_fav_mean": mean(x["gross_increase_fav"] for x in r), "erased_by_overwhelm_fav_mean": mean(x["erased_by_other_sources_fav"].get("overwhelm", 0.0) for x in r),
                         "J_fav_before_mean": mean(x["J_fav_before"] for x in r), "J_fav_after_mean": mean(x["J_fav_after"] for x in r),
                         "orig_J_fav_after_mean": mean(x["orig_J_fav_after"] for x in r), "orig_J_max": r[0]["orig_J_max"]}
        if "p7" in _CACHE:
            r = _CACHE["p7"]
            by_kind = {}
            for x in r:
                by_kind.setdefault(KINDS[x["seed"] % 3], []).append(x)
            keys = ("sum_abs_J", "n_cues", "n_discounted_choices", "q_mean_when_discounted", "moved_abs_total")
            out["P7_report_only"] = {"by_kind": {k: {"n": len(v), **{m: mean(t[m] for t in v) for m in keys},
                                                     "source_sums_signed_mean": {s: mean(t["source_sums_signed"][s] for t in v) for s in P.SOURCES},
                                                     "source_sums_abs_over_cues_mean": {s: mean(t["source_sums_abs_over_cues"][s] for t in v) for s in P.SOURCES}} for k, v in by_kind.items()},
                                     "range_over_seeds": {m: [min(t[m] for t in r), max(t[m] for t in r)] for m in keys},
                                     "example_seed0": {"J": r[0]["J"], "by_source": r[0]["by_source"]}, "per_persona_sum_abs_J": {x["seed"]: x["sum_abs_J"] for x in r}}
        dump("P01_P07", out)

    def test_P1_every_change_of_J_is_recorded_with_its_source(self):
        for x in rows("p1"):
            self.assertLessEqual(x["err_source_sums"], 1e-12, x["seed"])              # 台帳の源ごとの和 = 人格の源ごとの和
            self.assertLessEqual(x["err_cue_total_vs_J"], 1e-12, x["seed"])           # 台帳の全源の和 = J
            self.assertEqual(x["unknown_sources"], [], x["seed"])
            self.assertLessEqual(x["reaffirm_formula_error"], 1e-12, x["seed"])
            for s in ("spread", "reaffirm", "counter", "overwhelm"):
                self.assertGreaterEqual(x["event_counts"][s], 1, (x["seed"], s))      # 検査が空でない

    def test_P2_without_choices_J_does_not_change_at_all(self):
        for x in rows("p2"):
            p = x["persist"]
            self.assertGreater(p["J_before_max"], 0.01, x["seed"])                    # 前提: J が立っている
            self.assertGreater(p["learned"], 100, x["seed"])
            self.assertGreater(p["avoided"], 10, x["seed"])                           # 前提: 回避した申し出もある
            self.assertTrue(p["J_unchanged"], x["seed"])                              # 厳密に 0
            self.assertTrue(p["sources_unchanged"], x["seed"])
            self.assertTrue(p["chosen_before_unchanged"], x["seed"])
            self.assertEqual(p["new_justification_entries"], 0, x["seed"])
            self.assertFalse(x["persist_with_decay"]["J_unchanged"], x["seed"])        # 対照: 減衰の変種と orig は時間で変わる（検査が空でない）
            self.assertFalse(x["orig"]["J_unchanged"], x["seed"])

    def test_P3_numerically_sound_over_2000_choices(self):
        for x in rows("p3"):
            self.assertTrue(x["finite"], x["seed"])
            self.assertLessEqual(x["max_abs_J"], x["J_num"], x["seed"])
            self.assertEqual(x["B2"]["violations"], {}, x["seed"])
            self.assertEqual(x["B3_violations"], [], (x["seed"], x["B3"]))
            self.assertGreater(x["B3"]["teacher_checked"], 1900, x["seed"])
            self.assertGreater(x["event_counts"]["reaffirm"], 100, x["seed"])
        for x in TB.rows("Z_persist"):                                                # W6 の経験史 + 二択 + 訂正の 2000 経験（B 系の実行）
            self.assertEqual(x["B2"]["base"]["violations"], {}, x["seed"])
            self.assertEqual(x["B2"]["perturbed"]["violations"], {}, x["seed"])
            self.assertEqual(x["B3"]["violations"], [], x["seed"])
            self.assertLessEqual(x["B2"]["base"]["max_abs_J"], x["B2"]["base"]["J_bound"], x["seed"])

    def test_P4_where_persistence_comes_from_is_explained_by_reaffirm_and_overwhelm(self):
        for x in rows("p4"):
            self.assertEqual(x["reaffirm_count_difference"], 10, x["seed"])           # 前提: 再び選んだ回数の差は 10
            self.assertTrue(x["refutation_chosen_ok"], x["seed"])
            self.assertGreater(abs(x["delta"]), 1e-3, x["seed"])                      # 前提: 残存の差がある
            self.assertLessEqual(abs(x["delta"] - x["explained"]), 0.05 * abs(x["delta"]), (x["seed"], x["delta"], x["explained"], x["other_sources_difference"]))

    def test_P5_resistance_to_refutation_is_explained_by_the_ledger_discount(self):
        for x in rows("p5"):
            self.assertIsNotNone(x["discount_q"], x["seed"])
            self.assertGreater(x["discount_q"], 0.0, x["seed"])                       # 前提: 正当化された選択肢への反証
            self.assertGreater(x["person"]["R_meas_norm"], 1e-6, x["seed"])
            self.assertLessEqual(x["person"]["rel_error"], 0.01, (x["seed"], x["person"]))
            self.assertTrue(x["teachers_total_equal"], x["seed"])                     # 割り引きは教師の合計（人 + 状況）を変えない

    def test_P6_amplification_equals_the_recorded_reaffirm_sum(self):
        for x in rows("p6"):
            self.assertGreater(x["reaffirm_sum_fav"], 0.2, x["seed"])                 # 前提: 増幅している
            self.assertLessEqual(x["err_gross_increase_vs_reaffirm"], 1e-9, x["seed"])
            self.assertLessEqual(x["other_sources_max_positive"], 1e-9, x["seed"])    # 他の源から増えない
            self.assertLessEqual(x["err_net_vs_all_events"], 1e-12, x["seed"])
            self.assertLessEqual(x["reaffirm_formula_error"], 1e-12, x["seed"])
            self.assertLess(x["max_conflict"], 1e-7, x["seed"])                       # 前提: 葛藤のない再選択（spread が混ざらない。混ざる量は上の 1e-9 の検査で見る）

    def test_P7_individual_differences_by_choice_history_report_only(self):
        r = rows("p7")
        by_kind = {}
        for x in r:
            by_kind.setdefault(KINDS[x["seed"] % 3], []).append(x)
        self.assertEqual(len(by_kind), 3)
        for x in r:
            self.assertGreater(x["n_choices"], 200, x["seed"])
            for cue, row in x["by_source"].items():                                   # 分解は閉じている: 源ごとの和 = J
                self.assertAlmostEqual(sum(row.values()), x["J"].get(cue, 0.0), places=9)


if __name__ == "__main__":
    unittest.main()
