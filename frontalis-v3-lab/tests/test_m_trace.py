"""M1〜M5 復活の機構の追跡（SPEC-v3.3 6.3。結果を見る前に固定）。E3〜E6 が数値として合格しても、機構として戻ったのか、係数で嫌悪を強めただけなのかを区別する。

測り方の細部は tests/m_trace.py の冒頭に、M の結果を計算する前に書いた（SPEC 6.3 の表の閾値は変えていない）。20 本の乱数種（E 系と同じプロトコル・同じ種）。

  M1  E3・E4・E5・E6 のうち**合格したもの**で、重み付けの移り変わりの項が Δresp の 80% 以上（20 本の合計どうしの比）。不合格のものは報告のみ
  M2  全種・全プロトコル（消去直後、E1、E5、E6、E7 で 70% に達した時点）で、テスト時点の |C_獲得| ≤ 獲得直後の |C_獲得| + 0.02
  M3  合格したもので、E3・E4 は文脈、E5 は時間、E6 は加わった試行の段が、獲得の原因の事前の対数比の変化 Δρ の 50% 以上（Δρ > 0 のとき）。不合格のものは報告のみ
  M4  全種で、消去 20 試行のうち獲得の原因が MAP に選ばれた試行 ≤ 5、かつ消去中の獲得の原因の |ΔC| の和 ≤ 0.05
  M5  割当の記録が決定的（同じ種で 2 回走らせて一致）で、保存 → 復元で assignment_trace と explain_response が完全一致

E3〜E6 が不合格のとき、割当の記録から「なぜ戻らなかったか」を説明する（結果は tests/results/M01_M05.json の narrative と、CHANGELOG）。

v3.3 第2部: 既定の設計を spec_literal（SPEC 4.2 のとおり。柔らかい事後で更新）に戻した。判定（MTrace）は既定の設計で行い、M01_M05.json に書く。
revised（局所 MAP。実験設定）の同じ測定は報告用（MTraceRevisedReport。M01_M05_revised.json。合否に使わない）。spec_literal の M2・M4 は、割当の記録が示すとおり
消去の学習が獲得の原因を壊す（柔らかい事後で古い原因が更新される）ので破れる。これは機構の事実であり、閾値は変えていない。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, pmap
from m_trace import m_worker

_CACHE = {}


def _worker(args):
    seed, design = args
    return m_worker(seed, design)


def rows(design=None):
    if design not in _CACHE:
        _CACHE[design] = pmap(_worker, [(s, design) for s in SEEDS])
    return _CACHE[design]


def mean(vs):
    return float(np.mean(vs))


def e_passed(design=None):
    """E3・E4・E5・E6 の合否（SPEC-v3.3 4.4 の閾値。tests/test_e_extinction.py と同じ式）。"""
    rs = rows(design)
    m = lambda k: mean([r["E"][k] for r in rs])
    return {"E3": m("ABA") >= 0.5, "E4": m("ABC") >= 0.4,
            "E5": mean([r["E"]["E5_after"] - r["E"]["E2_B"] for r in rs]) >= 0.15,
            "E6": mean([r["E"]["E6_reinstated"] - r["E"]["E6_control"] for r in rs]) >= 0.15}


def summarize(design=None):
    rs = rows(design)
    passed = e_passed(design)
    res = {"design": rs[0]["design"], "E_passed (SPEC 4.4 thresholds)": passed, "M1": {}, "M3": {}}
    for nm in ("E3", "E4", "E5", "E6"):
        dl = sum(r["M1"][nm]["delta"] for r in rs)
        w = sum(r["M1"][nm]["weighting"] for r in rs)
        v = sum(r["M1"][nm]["value"] for r in rs)
        res["M1"][nm] = {"judged": passed[nm], "sum_delta_resp": dl, "sum_weighting_term": w, "sum_value_term": v,
                         "weighting_share": (w / dl) if abs(dl) > 1e-9 else None, "max_identity_error": max(abs(r["M1"][nm]["identity_error"]) for r in rs),
                         "mean_resp_before": mean([r["M1"][nm]["resp0"] for r in rs]), "mean_resp_after": mean([r["M1"][nm]["resp1"] for r in rs])}
        d = [r["M3"][nm] for r in rs]
        ok = [x for x in d if x["defined"]]
        key = {"E3": "context", "E4": "context", "E5": "time", "E6": "trials"}[nm]
        tot = sum(x["delta"] for x in ok)
        res["M3"][nm] = {"judged": passed[nm], "factor": key, "n_defined": len(ok), "sum_delta_rho": tot,
                         "sum_step_context": sum(x["step_context"] for x in ok), "sum_step_time": sum(x["step_time"] for x in ok),
                         "sum_step_trials": sum(x["step_trials"] for x in ok), "expected_factor_share": (sum(x[f"step_{key}"] for x in ok) / tot) if tot > 1e-9 else None,
                         "mean_rho_before": mean([x["rho0"] for x in ok]) if ok else None, "mean_rho_after": mean([x["rho1"] for x in ok]) if ok else None}
    res["M2"] = {"max_increase_over_seeds_and_protocols": max(r["M2"]["max_increase"] for r in rs),
                 "max_increase_without_E7 (protocols that do not learn x)": max(max(v - abs(r["C_acq_post"]) for k, v in r["M2"]["abs_C_at_test"].items() if not k.startswith("E7")) for r in rs),
                 "max_increase_E7 (re-acquisition)": max(r["M2"]["abs_C_at_test"]["E7 (at the trial reaching 70%)"] - abs(r["C_acq_post"]) for r in rs),
                 "seeds_violating_M2": int(sum(r["M2"]["max_increase"] > 0.02 for r in rs)),
                 "mean_C_acq_post_abs": mean([abs(r["C_acq_post"]) for r in rs]),
                 "mean_abs_C_at_test": {k: mean([r["M2"]["abs_C_at_test"][k] for r in rs]) for k in rs[0]["M2"]["abs_C_at_test"]},
                 "seeds_acq_cause_alive_after_extinction": int(sum(r["M2"]["acq_cause_alive_after_extinction"] for r in rs))}
    res["M4"] = {"seeds_violating_selected_le_5": int(sum(r["M4"]["acq_cause_selected"] > 5 for r in rs)),
                 "seeds_violating_dC_sum_le_0.05": int(sum(r["M4"]["acq_cause_abs_dC_sum"] > 0.05 for r in rs)),
                 "seeds_violating_either": int(sum(r["M4"]["acq_cause_selected"] > 5 or r["M4"]["acq_cause_abs_dC_sum"] > 0.05 for r in rs)),
                 "acq_cause_selected_by_seed": [r["M4"]["acq_cause_selected"] for r in rs],
                 "acq_cause_abs_dC_sum_by_seed": [round(r["M4"]["acq_cause_abs_dC_sum"], 4) for r in rs],
                 "max_acq_cause_selected_in_20_extinction_trials": max(r["M4"]["acq_cause_selected"] for r in rs),
                 "mean_acq_cause_selected": mean([r["M4"]["acq_cause_selected"] for r in rs]),
                 "max_acq_cause_abs_dC_sum": max(r["M4"]["acq_cause_abs_dC_sum"] for r in rs),
                 "mean_acq_cause_abs_dC_sum": mean([r["M4"]["acq_cause_abs_dC_sum"] for r in rs]),
                 "mean_acq_post_mass_in_20_extinction_updates (soft posterior mass the acquisition cause received)": mean([r["M4"]["acq_post_mass"] for r in rs]),
                 "mean_extinction_updates_assigned_to_new_cause": mean([r["M4"]["assigned_to_new_cause"] for r in rs]),
                 "mean_reuse_events": mean([r["M4"]["reuse_events"] for r in rs]),
                 "seeds_where_acq_cause_was_replaced_by_reuse (reported)": int(sum(r["M4"]["replaced_acq_cause"] for r in rs)),
                 "n_extinction_updates_per_seed": sorted(set(r["M4"]["n_extinction_updates"] for r in rs))}
    res["M5"] = {"all_deterministic": all(r["M5"]["deterministic"] for r in rs), "all_restored_equal": all(r["M5"]["restored_equal"] for r in rs),
                 "mean_records": mean([r["M5"]["n_records"] for r in rs])}
    res["narrative_seed0"] = rs[0]["narrative"]
    res["per_seed_acq_uid"] = [r["acq_uid"] for r in rs]
    return res


class MTrace(unittest.TestCase):
    """判定。既定の設計（spec_literal）。"""

    @classmethod
    def tearDownClass(cls):
        dump("M01_M05", summarize())

    def test_M1_recovery_is_reweighting_of_causes(self):
        rs = rows()
        passed = e_passed()
        for nm in ("E3", "E4", "E5", "E6"):
            if not passed[nm]:
                continue                                                            # 不合格のものは報告のみ
            dl = sum(r["M1"][nm]["delta"] for r in rs)
            w = sum(r["M1"][nm]["weighting"] for r in rs)
            self.assertGreaterEqual(w / dl, 0.8, nm)

    def test_M2_the_acquisition_value_is_not_strengthened(self):
        for r in rows():
            self.assertLessEqual(r["M2"]["max_increase"], 0.02, (r["seed"], r["M2"]["abs_C_at_test"], r["M2"]["C_acq_post_abs"]))

    def test_M3_the_expected_factor_carries_the_shift(self):
        rs = rows()
        passed = e_passed()
        for nm, key in (("E3", "context"), ("E4", "context"), ("E5", "time"), ("E6", "trials")):
            if not passed[nm]:
                continue
            ok = [r["M3"][nm] for r in rs if r["M3"][nm]["defined"]]
            tot = sum(x["delta"] for x in ok)
            self.assertGreater(tot, 0.0, nm)
            self.assertGreaterEqual(sum(x[f"step_{key}"] for x in ok) / tot, 0.5, nm)

    def test_M4_extinction_does_not_destroy_the_acquisition_cause(self):
        for r in rows():
            self.assertEqual(r["M4"]["n_extinction_updates"], 20, r["seed"])
            self.assertLessEqual(r["M4"]["acq_cause_selected"], 5, r["seed"])
            self.assertLessEqual(r["M4"]["acq_cause_abs_dC_sum"], 0.05, r["seed"])

    def test_M5_assignment_records_are_deterministic_and_restorable(self):
        for r in rows():
            self.assertTrue(r["M5"]["deterministic"], r["seed"])
            self.assertTrue(r["M5"]["restored_equal"], r["seed"])
            self.assertGreater(r["M5"]["n_records"], 0, r["seed"])


class MTraceRevisedReport(unittest.TestCase):
    """報告用: revised（局所 MAP。実験設定）の同じ測定。合否には使わない（M01_M05_revised.json）。検査するのは記録の決定性と恒等式だけ。"""

    @classmethod
    def tearDownClass(cls):
        dump("M01_M05_revised", summarize("revised"))

    def test_report_revised_records_are_deterministic_and_the_decomposition_is_an_identity(self):
        rs = rows("revised")
        self.assertEqual(rs[0]["design"], "revised")
        for r in rs:
            self.assertTrue(r["M5"]["deterministic"], r["seed"])
            self.assertTrue(r["M5"]["restored_equal"], r["seed"])
            for nm in ("E3", "E4", "E5", "E6"):
                self.assertLess(abs(r["M1"][nm]["identity_error"]), 1e-12, (r["seed"], nm))


if __name__ == "__main__":
    unittest.main()
