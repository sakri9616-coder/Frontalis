"""E1〜E9 嫌悪の維持・消去・再学習・再発（SPEC-v3.3 第 4 節）。判定基準は実装前に固定されたまま（閾値は一度も変えていない）。

プロトコルは tests/e_protocol.py の冒頭に書いた（手がかり praise_from_X、文脈 A = studio・B = cafe_chat・未見の C = garden、獲得 12 試行 →
消去 20 試行 → 各テスト。各テストは消去直後の状態の写しから分岐。20 本の乱数種で、害の大きさ・無関係な経験の中身が変わる）。
比 = C / C_獲得直後（獲得直後の文脈 A の C を 1 とする。負の C どうしの比）。判定は 20 本の平均。`single_cause`（K = 1）は同じプロトコルを v3.2 の条件づけで走らせた対照。

判定（SPEC-v3.3 4.4 の表のとおり）:
  E1 維持     消去なしで無関係な経験 50 回 → 比 ≥ 0.7
  E2 消去     文脈 B で 20 試行の消去の後、B でテスト → 比 ≤ 0.3。`single_cause` も ≤ 0.3
  E3 復元 ABA 消去の後 A でテスト → ≥ 0.5。`single_cause` では ≤ 0.3
  E4 復元 ABC 消去の後 未見の C でテスト → ≥ 0.4。`single_cause` では ≤ 0.3
  E5 自発的回復 消去の後 無関係な経験 50 回、その後 B でテスト → 消去直後より 0.15 以上高い。`single_cause` では増加 0.05 未満
  E6 復位     消去の後 手がかりなしの害 2 回、その後 B でテスト → 復位なしの対照より 0.15 以上高い。`single_cause` では 0.05 未満
  E7 節約     消去の後 B で再獲得 → 獲得直後の 70% に達するまでの試行数が、初めての獲得の 0.6 倍以下。`single_cause` では 0.9 倍以上
  E8 故障でない (a) |C| ≤ 2、原因数 ≤ K。(b) 別の手がかりの C の変化 ≤ 0.05。(c) 状態の保存・復元で完全一致
  E9 経験史による個人差 消去を 1 つの文脈で行った人格と 3 つの文脈で行った人格の ABC 復元 → 3 文脈の方が 0.1 以上小さい

測り方の細部（実装前に決めた）:
  E5 の「消去直後」= 同じ人格の B でのテスト（E2 の値）。E6 の対照 = 同じ時間（2 経験）だけ無関係な経験を与えた人格。
  E7 の試行数 = 比が 0.7 に初めて達する試行の番号（達しなければ打ち切りの値 41）。初めての獲得は文脈 A の 12 試行の間（0.7 に達しなければ 13）、
  再獲得は文脈 B で最大 40 試行。判定は 20 本の試行数の平均どうしの比。
  E8(b) の別の手がかりは「一度も x と対にならない中立の手がかり tea_time」（e_protocol.py 冒頭）。
  （v3.3 の再開後に足した E8(b) の 2 つ目の検査: tea_time は C = 0 で、変化が自明に 0 になる（最大の変化 0.0 ちょうど）。前任が参考にしていた evening_walk（comfort と対）も、
  C が 0.009 しかなく、閾値 0.05 より小さいので空に近い。そこで、値を持つ別の手がかり「別の嫌悪の手がかり concept:other_fear」（文脈 A で害と対に 12 試行、獲得直後の C ≈ −0.2）を、
  x のプロトコル全体（獲得 12 試行 + 消去 20 試行、64 経験）の前後で比べる別の実行にした（`e_protocol.valued_cue_leak`）。判定は閾値 0.05 のまま（変化の絶対値）。
  参考: 同じ経験数の無関係な経験だけを与えた対照も報告する（時間の重みだけによる変化）。この検査を足したのは、E 系の結果を見た後。主プロトコルの数値は変えていない。）
  すべて 20 本の平均。single_cause の対照も同じ種で走らせる。
"""
import unittest

import numpy as np

from e_protocol import EXT_CONTEXTS_3, CTX_B, run_variant, valued_cue_leak
from helpers import SEEDS, dump, pmap


def worker(seed):
    return {"seed": seed,
            "latent": run_variant(seed),
            "single": run_variant(seed, single_cause=True),
            "e9_one": run_variant(seed, ext_contexts=(CTX_B,), full=False),
            "e9_three": run_variant(seed, ext_contexts=EXT_CONTEXTS_3, full=False)}


def valued_worker(args):
    seed, single = args
    return valued_cue_leak(seed, single)


_ROWS = None


def rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = pmap(worker, SEEDS)
    return _ROWS


def mean(vs):
    return float(np.mean(vs))


def col(which, key):
    return [r[which][key] for r in rows()]


class EExtinction(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        rs = rows()
        res = {}
        for which in ("latent", "single"):
            res[which] = {
                "E1_maintenance": mean(col(which, "E1")),
                "E2_extinction_B": mean(col(which, "E2_B")),
                "E3_ABA": mean(col(which, "ABA")),
                "E4_ABC": mean(col(which, "ABC")),
                "E5_after_unrelated_B": mean(col(which, "E5_after")),
                "E5_increase": mean(np.array(col(which, "E5_after")) - np.array(col(which, "E2_B"))),
                "E6_reinstated": mean(col(which, "E6_reinstated")),
                "E6_control": mean(col(which, "E6_control")),
                "E6_difference": mean(np.array(col(which, "E6_reinstated")) - np.array(col(which, "E6_control"))),
                "E7_first_trials": mean(col(which, "E7_first")),
                "E7_re_trials": mean(col(which, "E7_re")),
                "E7_ratio": mean(col(which, "E7_re")) / mean(col(which, "E7_first")),
                "n_causes_after_acq": mean(col(which, "n_causes_after_acq")),
                "n_causes_after_ext": mean(col(which, "n_causes_after_ext")),
                "C_acq_mean": mean(col(which, "C_acq")),
                "acq_curve_ratio_mean": [mean([r[which]["acq_curve_ratio"][i] for r in rs]) for i in range(12)],
                "re_curve_ratio_mean": [mean([r[which]["re_curve_ratio"][i] for r in rs]) for i in range(12)],
                "seeds_E1_ge_0.7": int(sum(v >= 0.7 for v in col(which, "E1"))),
                "seeds_ABA_ge_0.5": int(sum(v >= 0.5 for v in col(which, "ABA"))),
                "seeds_ABC_ge_0.4": int(sum(v >= 0.4 for v in col(which, "ABC"))),
                "E8a_max_abs_cause_value": max(r[which]["bounds"]["max_abs_cause_value"] for r in rs),
                "E8a_max_abs_response": max(r[which]["bounds"]["max_abs_response"] for r in rs),
                "E8a_max_causes": max(r[which]["bounds"]["max_causes"] for r in rs),
                "E8b_tea_time_max_change": max(abs(r[which]["ctl_after"]["tea_time"] - r[which]["ctl_before"]["tea_time"]) for r in rs),
                "E8b_reference_evening_walk_max_change (not judged; C is only ~0.01)": max(
                    abs(r[which]["ctl_after"]["evening_walk"] - r[which]["ctl_before"]["evening_walk"]) for r in rs),
                "E8b_reference_evening_walk_after_50_unrelated (not judged)": max(
                    abs(r[which]["ctl_after_unrelated"]["evening_walk"] - r[which]["ctl_before"]["evening_walk"]) for r in rs),
                "E8b_reference_evening_walk_value_before": float(np.mean([r[which]["ctl_before"]["evening_walk"] for r in rs])),
                "E8c_all_equal": all(r[which]["E8c_restore_state_equal_at_save"] and r[which]["E8c_restore_continue_equal"] for r in rs),
            }
        res["E9"] = {"ABC_one_context": mean([r["e9_one"]["ABC"] for r in rs]),
                     "ABC_three_contexts": mean([r["e9_three"]["ABC"] for r in rs]),
                     "difference_one_minus_three": mean([r["e9_one"]["ABC"] - r["e9_three"]["ABC"] for r in rs]),
                     "ABC_one_context_single_cause_reference": mean([run_variant(s, single_cause=True, ext_contexts=(CTX_B,), full=False)["ABC"] for s in SEEDS[:5]]),
                     "n_causes_after_ext_one": mean([r["e9_one"]["n_causes_after_ext"] for r in rs]),
                     "n_causes_after_ext_three": mean([r["e9_three"]["n_causes_after_ext"] for r in rs])}
        dump("E01_E09", res)

    # --- E1 -------------------------------------------------------------
    def test_E1_maintenance(self):
        self.assertGreaterEqual(mean(col("latent", "E1")), 0.7)

    # --- E2 -------------------------------------------------------------
    def test_E2_extinction(self):
        self.assertLessEqual(mean(col("latent", "E2_B")), 0.3)
        self.assertLessEqual(mean(col("single", "E2_B")), 0.3)

    # --- E3 -------------------------------------------------------------
    def test_E3_renewal_ABA(self):
        self.assertGreaterEqual(mean(col("latent", "ABA")), 0.5)
        self.assertLessEqual(mean(col("single", "ABA")), 0.3)

    # --- E4 -------------------------------------------------------------
    def test_E4_renewal_ABC(self):
        self.assertGreaterEqual(mean(col("latent", "ABC")), 0.4)
        self.assertLessEqual(mean(col("single", "ABC")), 0.3)

    # --- E5 -------------------------------------------------------------
    def test_E5_spontaneous_recovery(self):
        inc = lambda w: mean(np.array(col(w, "E5_after")) - np.array(col(w, "E2_B")))
        self.assertGreaterEqual(inc("latent"), 0.15)
        self.assertLess(inc("single"), 0.05)

    # --- E6 -------------------------------------------------------------
    def test_E6_reinstatement(self):
        d = lambda w: mean(np.array(col(w, "E6_reinstated")) - np.array(col(w, "E6_control")))
        self.assertGreaterEqual(d("latent"), 0.15)
        self.assertLess(d("single"), 0.05)

    # --- E7 -------------------------------------------------------------
    def test_E7_savings(self):
        ratio = lambda w: mean(col(w, "E7_re")) / mean(col(w, "E7_first"))
        self.assertLessEqual(ratio("latent"), 0.6)
        self.assertGreaterEqual(ratio("single"), 0.9)

    # --- E8 -------------------------------------------------------------
    def test_E8a_bounded(self):
        for w in ("latent", "single"):
            for r in rows():
                b = r[w]["bounds"]
                self.assertLessEqual(b["max_abs_cause_value"], 2.0, (w, r["seed"]))
                self.assertLessEqual(b["max_abs_response"], 2.0, (w, r["seed"]))
        self.assertLessEqual(max(r["latent"]["bounds"]["max_causes"] for r in rows()), 5)
        self.assertLessEqual(max(r["single"]["bounds"]["max_causes"] for r in rows()), 1)

    def test_E8b_other_cue_unchanged(self):
        for w in ("latent", "single"):
            for r in rows():
                self.assertLessEqual(abs(r[w]["ctl_after"]["tea_time"] - r[w]["ctl_before"]["tea_time"]), 0.05, (w, r["seed"]))

    def test_E8b_valued_other_cue_unchanged(self):
        """値を持つ別の手がかり（別の嫌悪の手がかり other_fear。C ≈ −0.2）が、x のプロトコル全体の前後で変わる量 ≤ 0.05（v3.3 の再開後に足した。冒頭の注）。"""
        out = {}
        for w, single in (("latent", False), ("single", True)):
            rs = pmap(valued_worker, [(s, single) for s in SEEDS])
            out[w] = {"value_before_mean": mean([r["before"] for r in rs]),
                      "max_abs_change_protocol": max(abs(r["after_protocol"] - r["before"]) for r in rs),
                      "max_abs_change_time_matched_unrelated_control (reported, not judged)": max(abs(r["after_control"] - r["before"]) for r in rs),
                      "max_abs_protocol_minus_control (reported)": max(abs(r["after_protocol"] - r["after_control"]) for r in rs)}
        dump("E08b_valued_cue", out)
        for w in out:
            self.assertGreater(abs(out[w]["value_before_mean"]), 0.1)                          # 検査が空でない
            self.assertLessEqual(out[w]["max_abs_change_protocol"], 0.05, w)

    def test_E8c_save_restore_exact(self):
        for w in ("latent", "single"):
            for r in rows():
                self.assertTrue(r[w]["E8c_restore_state_equal_at_save"], (w, r["seed"]))
                self.assertTrue(r[w]["E8c_restore_continue_equal"], (w, r["seed"]))

    # --- E9 -------------------------------------------------------------
    def test_E9_extinction_in_three_contexts_reduces_renewal(self):
        d = mean([r["e9_one"]["ABC"] - r["e9_three"]["ABC"] for r in rows()])
        self.assertGreaterEqual(d, 0.1)


if __name__ == "__main__":
    unittest.main()
