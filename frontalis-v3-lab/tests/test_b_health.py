"""B1〜B3 数値的な健全性（SPEC-v3.3 第 7.2 節）: 人格形成の機構が、(1) 状態が有界で、(2) 学習規則が自分の定義どおりに動き、(3) 分岐や変化の原因が記録から説明できること。
判定基準は実装前に固定されたもの（閾値は一度も変えていない）。プロトコルの細部は tests/b_protocol.py の冒頭（数値を見る前に書いた）。

**W6 の微小摂動の収縮性（比 ≤ 1）は数学的な安定性の指標であって、人格形成の健全性そのものではない（SPEC 7.1）。** 離散的な選択（接近・回避、二択、潜在原因の割当、復位）が一つ分かれれば、
その後の軌道が離れて比が 1 を超えるのは、機構の故障ではなく人格の個別性でありうる。ここでは、故障とみなすものを次の三つに限る。W6 の元のテストと記録は残してあり、ここでは判定に使わない。

  B1 分岐の因果的説明可能性   比 > 1 の実行について、最初の離散的な決定の分岐点より前の比 ≤ 1 + 1e-6、かつ離散的な決定をすべて基準の実行に合わせて強制した再生の最終の比 ≤ 1。
                              この二つがともに成り立つ実行が、比 > 1 の実行の 95% 以上（比 > 1 の実行がなければ対象なしで合格）。
                              → 「離れたのは、記録された離散的な決定が分かれたから。それ以外の経路では離れていない」ことの確認。
  B2 状態の有界性             全イベントで: 有限、|W| ≤ 10、|C| ≤ 2、0 ≤ a ≤ 5、|J| ≤ J の上限、計数 ≥ 0、原因数 ≤ K。Σ は対称・半正定値。
  B3 学習規則の整合性         台帳の ΔW・ΔC_k・J の出来事の和が状態に一致、φ の和（質量）が 1、W の教師 = Δu を φ で分けたもの（Z-persist は割り引きを含めて）、a の目標 = |Δu|、
                              J は二択の経験の出来事（と減衰）でだけ変わる、価値ネット自身の出力は教師に入らない。

条件（どれも全機構を有効にし、W6 と同じ 2000 経験、乱数種 0〜19 の 20 本）:
  spec_literal / revised   既定の設計と revised の両方（W6 と同じ経験史。二択・訂正なし）。「revised は数値的に壊れているか」への答えはここ（B1〜B3）で見る。
  Z-orig / Z-persist       spec_literal の上で、W6 の経験史に二択（8 経験ごと）と訂正（100 経験ごと）を挟んだもの。正当化 J の二つの条件（Z-orig = SPEC 3.3 そのまま、Z-persist = 8.3）。
"""
import unittest
from typing import Any, Dict, List

import numpy as np

from helpers import SEEDS, dump, pmap
import b_protocol as B

CONDITIONS = {
    "spec_literal": ("spec_literal", "orig", False),
    "revised": ("revised", "orig", False),
    "Z_orig": ("spec_literal", "orig", True),
    "Z_persist": ("spec_literal", "persist", True),
}
_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def rows(cond: str) -> List[Dict[str, Any]]:
    if cond not in _CACHE:
        d, m, wc = CONDITIONS[cond]
        _CACHE[cond] = pmap(B.b_worker, [(s, d, m, wc) for s in SEEDS])
    return _CACHE[cond]


def b1_summary(r: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt1 = [x for x in r if x["ratio"] > 1.0]
    ok = [x for x in gt1 if x["B1_a_ok"] and x["B1_b_ok"]]
    kinds: Dict[str, int] = {}
    for x in gt1:
        bi = x["branch_info"]
        key = "no branch found" if bi is None else "+".join(bi["differs"])
        kinds[key] = kinds.get(key, 0) + 1
    return {"ratios_median": float(np.median([x["ratio"] for x in r])), "ratios_max": max(x["ratio"] for x in r), "n_runs": len(r), "n_ratio_gt_1": len(gt1),
            "n_explained": len(ok), "fraction_explained": (len(ok) / len(gt1)) if gt1 else None,
            "runs_with_ratio_gt_1": [{"seed": x["seed"], "ratio": x["ratio"], "branch_event": (x["branch_info"] or {}).get("event_id"), "branch_index_after_warmup": x["branch_index"],
                                      "branch_differs": (x["branch_info"] or {}).get("differs"), "pre_branch_max_ratio": x["pre_branch_max_ratio"], "forced_replay_ratio": x.get("forced_ratio"),
                                      "a_ok": x["B1_a_ok"], "b_ok": x["B1_b_ok"]} for x in gt1],
            "first_branch_kinds_among_ratio_gt_1": kinds,
            "n_runs_with_a_branch": sum(x["branch_index"] is not None for x in r),
            "pre_branch_max_ratio_over_all_runs": max(x["pre_branch_max_ratio"] for x in r),
            "ratio_by_seed": [x["ratio"] for x in r]}


def b2_summary(r: List[Dict[str, Any]]) -> Dict[str, Any]:
    def agg(which: str) -> Dict[str, Any]:
        s = [x["B2"][which] for x in r]
        viol: Dict[str, int] = {}
        for t in s:
            for k, v in t["violations"].items():
                viol[k] = viol.get(k, 0) + v
        return {"max_abs_W": max(t["max_abs_W"] for t in s), "max_abs_C": max(t["max_abs_C"] for t in s), "a_min": min(t["a_range"][0] for t in s), "a_max": max(t["a_range"][1] for t in s),
                "max_abs_J": max(t["max_abs_J"] for t in s), "J_bound": s[0]["J_bound"], "min_count": min(t["min_count"] for t in s), "max_causes": max(t["max_causes"] for t in s), "K": s[0]["K"],
                "max_S_asymmetry": max(t["max_S_asymmetry"] for t in s), "min_S_eigenvalue": min(t["min_S_eigenvalue"] for t in s), "violations": viol,
                "checks_total": sum(t["n_checks"] for t in s)}
    return {"base": agg("base"), "perturbed": agg("perturbed")}


def b3_summary(r: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["ledger_W_max_error", "ledger_C_max_error", "phi_mass_max_error", "pi_mass_max_error", "teacher_max_error", "teacher_sum_vs_du_max_error", "a_replay_max_error",
            "J_ledger_sum_max_error", "J_sources_sum_max_error"]
    out: Dict[str, Any] = {k: max(x["B3"][k] for x in r) for k in keys}
    out["J_events_outside_choice"] = sum(x["B3"]["J_events_outside_choice"] for x in r)
    out["teacher_checked_total"] = sum(x["B3"]["teacher_checked"] for x in r)
    out["teacher_skipped_corrected_total"] = sum(x["B3"]["teacher_skipped_corrected"] for x in r)
    out["value_independence_checks"] = sum(x["B3"]["value_independence_checks"] for x in r)
    out["value_independence_ok"] = sum(x["B3"]["value_independence_ok"] for x in r)
    out["n_choice_entries_total"] = sum(x["B3"]["n_choice_entries"] for x in r)
    out["runs_with_violations"] = [(x["seed"], x["B3"]["violations"]) for x in r if x["B3"]["violations"]]
    return out


class BHealth(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        out = {"protocol": "tests/b_protocol.py（冒頭に判定の定義）", "WARMUP": B.WARMUP, "perturbation_norm": B.EPS, "events": B.TOTAL}
        for c, r in _CACHE.items():
            out[c] = {"design": CONDITIONS[c][0], "justification_mode": CONDITIONS[c][1], "with_choices_and_corrections": CONDITIONS[c][2],
                      "n_events": r[0]["n_events"], "n_choices": r[0]["n_choices"], "n_corrections": r[0]["n_corrections"],
                      "B1": b1_summary(r), "B2": b2_summary(r), "B3": b3_summary(r)}
        if "Z_persist" in _CACHE:
            r = _CACHE["Z_persist"]
            out["Z_persist_J_sources_seed0"] = {"J": r[0]["J_final"], "by_source": r[0]["J_sources_final"]}
        if "Z_orig" in _CACHE:
            r = _CACHE["Z_orig"]
            out["Z_orig_J_sources_seed0"] = {"J": r[0]["J_final"], "by_source": r[0]["J_sources_final"]}
        dump("B01_B03", out)

    # ------------------------------------------------------------------ 検査が空でないことの前提
    def _nonvacuous(self, cond: str) -> None:
        r = rows(cond)
        for x in r:
            self.assertEqual(x["n_events"], B.TOTAL + (x["n_choices"] + x["n_corrections"]), (cond, x["seed"]))
            self.assertGreater(x["B3"]["teacher_checked"], 1000, (cond, x["seed"]))
            self.assertGreater(x["B2"]["base"]["n_checks"], 1000, (cond, x["seed"]))
            self.assertGreaterEqual(x["B3"]["value_independence_checks"], 9, (cond, x["seed"]))
        if CONDITIONS[cond][2]:
            self.assertTrue(all(x["n_choices"] > 200 and x["n_corrections"] >= 15 for x in r))
            self.assertTrue(all(x["B3"]["n_choice_entries"] > 200 for x in r))
            self.assertGreater(max(x["B2"]["base"]["max_abs_J"] for x in r), 0.01)         # J が実際に動いている

    def _b1(self, cond: str) -> None:
        self._nonvacuous(cond)
        s = b1_summary(rows(cond))
        if s["n_ratio_gt_1"] == 0:
            return                                                                  # 比 > 1 の実行がない: 対象なしで合格
        self.assertGreaterEqual(s["fraction_explained"], 0.95, s["runs_with_ratio_gt_1"])

    def _b2(self, cond: str) -> None:
        for x in rows(cond):
            self.assertEqual(x["B2"]["base"]["violations"], {}, (cond, x["seed"], "base"))
            self.assertEqual(x["B2"]["perturbed"]["violations"], {}, (cond, x["seed"], "perturbed"))

    def _b3(self, cond: str) -> None:
        for x in rows(cond):
            self.assertEqual(x["B3"]["violations"], [], (cond, x["seed"], x["B3"]))

    # ------------------------------------------------------------------ 既定の設計 spec_literal と revised（W6 の経験史）
    def test_B1_spec_literal(self):
        self._b1("spec_literal")

    def test_B2_spec_literal(self):
        self._b2("spec_literal")

    def test_B3_spec_literal(self):
        self._b3("spec_literal")

    def test_B1_revised(self):
        self._b1("revised")

    def test_B2_revised(self):
        self._b2("revised")

    def test_B3_revised(self):
        self._b3("revised")

    # ------------------------------------------------------------------ 正当化の二つの条件（二択と訂正を含む経験史）
    def test_B1_Z_orig(self):
        self._b1("Z_orig")

    def test_B2_Z_orig(self):
        self._b2("Z_orig")

    def test_B3_Z_orig(self):
        self._b3("Z_orig")

    def test_B1_Z_persist(self):
        self._b1("Z_persist")

    def test_B2_Z_persist(self):
        self._b2("Z_persist")

    def test_B3_Z_persist(self):
        self._b3("Z_persist")


if __name__ == "__main__":
    unittest.main()
