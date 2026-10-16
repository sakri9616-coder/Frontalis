"""v3.4 Stage 4 の単体テスト: 残りの条件（A-g-const・A-κ-fixed・local-reattr）、Shapley 分解、並列ランナー（直列とビット単位で同じ）、変異体 MU1〜MU8、検査の基準線、
乱数種の守り、凍結ファイル（freeze-protocol-v2.json）。

乱数種: 開発用の 3490〜3499 だけ（SPEC 第 5.1 節）。変異の種 3450〜3459 は、このテストでは実行しない（実行するのは stage4_mutation.py だけ）。
評価 3400〜3439・予約 3500〜3539 はどこでも実行しない（モジュールの先頭で確認する）。
"""
import copy
import json
import math
import os
import pickle
import subprocess
import sys
import unittest

import numpy as np

from helpers import ROOT, cos  # noqa: F401

sys.path.insert(0, str(ROOT / "data"))
import make_histories_v34 as H  # noqa: E402
import mutants_v34 as MU  # noqa: E402
import ra_checks_v34 as CK  # noqa: E402
import ra_parallel as PAR  # noqa: E402
import ra_protocol as RP  # noqa: E402
from npnv3 import NPN  # noqa: E402
from npnv3.config import Config  # noqa: E402
from npnv3.innate import NEEDS  # noqa: E402

DEV_SEEDS = tuple(range(3490, 3500))
FORBIDDEN = set(range(3400, 3440)) | set(range(3450, 3460)) | set(range(3500, 3540))
USED_SEEDS = (3490, 3491)
assert set(USED_SEEDS) <= set(DEV_SEEDS) and not (set(USED_SEEDS) & FORBIDDEN)

V34 = RP._V34


def led(npn, kind):
    return [e for e in npn.ledger.entries if e["kind"] == kind]


def canon(x):
    """ビット単位の比較用の正規形（浮動小数点は 16 進表記、配列はバイト列のハッシュ）。pickle のバイト列は共有オブジェクトの有無で変わるので使わない。"""
    import hashlib
    if isinstance(x, np.ndarray):
        return ["nd", str(x.dtype), list(x.shape), hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()]
    if isinstance(x, (float, np.floating)):
        return ["f", float(x).hex()]
    if isinstance(x, dict):
        return ["d", [[str(k), canon(v)] for k, v in x.items()]]
    if isinstance(x, (list, tuple)):
        return ["l", [canon(v) for v in x]]
    return ["o", repr(x)]


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, errors="replace").stdout
    return str(pid) in out


def run_hist(seed, bg, src, err, n, cond, upto="all"):
    """条件 cond（ra_protocol の Condition）で、COR の直前から cor_overrides を効かせて実行した人格。"""
    c = RP.get_condition(cond)
    h = H.make_history(seed, bg, src, err, n, 0)
    npn = RP._prefix(seed, h, c.prefix_overrides, "mis")
    for k, v in c.cor_overrides:
        setattr(npn.cfg, k, v)
    hook = RP.local_reattribute if c.local_reattr else None
    RP.feed_items(npn, h["cor"], hook)
    if upto == "all":
        RP.feed_items(npn, h["fol"])
    return npn, h


class RemainingConditions(unittest.TestCase):
    def test_defaults_are_inactive_and_validated(self):
        cfg = Config()
        self.assertIsNone(cfg.g_const)
        self.assertFalse(cfg.kappa_fixed)
        for bad in (0.0, 1.0, -0.1, float("nan"), 1.5):
            with self.assertRaises(ValueError):
                Config(g_const=bad)
        self.assertEqual(Config(g_const=0.25).g_const, 0.25)

    def test_all_protocol_conditions_are_implemented(self):
        for name in ("A-g-const", "A-κ-fixed", "A-kappa-fixed", "local-reattr", "local-reattr-v33", "C-none-v34", "C-v34", "C-OR"):
            self.assertTrue(RP.get_condition(name).implemented, name)
        self.assertEqual(RP._STUBS, [])

    def test_A_g_const_fixes_g_and_still_records_z_and_llr(self):
        a, _ = run_hist(3490, "betray", "unreliable", "interp", 4, "A-g-const", upto="cor")
        b, _ = run_hist(3490, "betray", "unreliable", "interp", 4, "C-v34", upto="cor")
        ca, cb = led(a, "correction"), led(b, "correction")
        self.assertEqual(len(ca), len(cb))
        for x, y in zip(ca[:4], cb[:4]):                          # SRC の訂正（COR の前）は同じ
            self.assertEqual(x["m1"]["g"], y["m1"]["g"])
        for x, y in zip(ca[4:], cb[4:]):                          # COR の訂正は g ≡ 0.5。z・LLR・κ̄ は記録される
            self.assertEqual(x["m1"]["g"], 0.5)
            self.assertTrue(math.isfinite(x["m1"]["z"]) and math.isfinite(x["m1"]["llr"]))
            self.assertNotEqual(y["m1"]["g"], 0.5)
        self.assertFalse(np.array_equal(a.value.person.W, b.value.person.W))
        self.assertEqual(Config().g_const, None)                  # 既定の設定には触れない

    def test_A_kappa_fixed_ignores_learned_credibility_and_skips_verification(self):
        a, _ = run_hist(3490, "trust", "reliable", "report", 4, "A-κ-fixed", upto="cor")
        b, _ = run_hist(3490, "trust", "reliable", "report", 4, "C-v34", upto="cor")
        n_ver_pre = len(led(b, "verification"))
        self.assertGreater(n_ver_pre, 0)                          # SRC-reliable: COR の前に検証がある
        for x in led(a, "correction")[4:]:
            self.assertEqual(x["m1"]["kappa_bar"], 0.5)
            self.assertEqual(x["m1"]["kappa_alpha_beta"], [1.0, 1.0])
        self.assertTrue(all(y["m1"]["kappa_bar"] != 0.5 for y in led(b, "correction")[4:]))
        self.assertEqual(a.ver_pending[-4:] if a.ver_pending else [], [])                 # COR の訂正は検証待ちを登録しない
        h = H.make_history(3490, "trust", "reliable", "report", 4, 0)
        RP.feed_items(a, h["fol"])
        RP.feed_items(b, h["fol"])
        self.assertEqual(len(led(a, "verification")), n_ver_pre)                          # COR の後の検証もない
        self.assertGreater(len(led(b, "verification")), n_ver_pre)

    def test_variants_do_not_touch_the_state_before_COR(self):
        out = RP.run_individual(3490, "betray", "unreliable", "interp", 3, 0, ["C-v34", "A-g-const", "A-κ-fixed", "local-reattr"], "dev")
        pre = out["cond"]["C-v34"]["R_pre"]
        for c in ("A-g-const", "A-κ-fixed", "local-reattr"):
            self.assertTrue(np.array_equal(out["cond"][c]["R_pre"], pre), c)
            self.assertTrue(np.isfinite(out["cond"][c]["R_cor"]).all() and np.isfinite(out["cond"][c]["R_fol"]).all())
        self.assertFalse(np.array_equal(out["cond"]["A-g-const"]["R_cor"], out["cond"]["C-v34"]["R_cor"]))

    def test_local_reattr_changes_only_rows_of_the_referred_participants_and_treats_the_control_alike(self):
        h = H.make_history(3490, "betray", "none", "report", 3, 0)
        base = RP._prefix(3490, h, V34, "mis")
        a = copy.deepcopy(base)
        RP.feed_items(a, h["cor"][:1])
        b = copy.deepcopy(a)
        phi0 = b.belief.rec_phi[: b.belief.rec_n].copy()
        RP.local_reattribute(b, h["cor"][0])
        phi1 = b.belief.rec_phi[: b.belief.rec_n]
        pids = {H.A, H.O}
        sel = np.array([b.belief.ind_ids[int(i)] in pids for i in b.belief.rec_ind[: b.belief.rec_n]])
        self.assertTrue(np.array_equal(phi0[~sel], phi1[~sel]))                               # 選ばなかった行は変わらない
        self.assertFalse(np.array_equal(phi0[sel], phi1[sel]))                                # 選んだ行は変わる
        # 選んだ行の値は、全体の想起の掃引（ヤコビ型）と同じ
        full = copy.deepcopy(a)
        full.belief.reattribute()
        self.assertTrue(np.array_equal(full.belief.rec_phi[: full.belief.rec_n][sel], phi1[sel]))
        # 価値・条件づけ・関心の重みは変えない
        self.assertTrue(np.array_equal(a.value.person.W, b.value.person.W) and np.array_equal(a.cond.C, b.cond.C) and np.array_equal(a.concern.a, b.concern.a))
        # 計数は記録から作り直した値と一致
        b2 = copy.deepcopy(b.belief, {id(b.emb): b.emb})
        b2._rebuild_counts()
        for k in ("N1", "N0", "CP1", "CP0", "pi"):
            self.assertLessEqual(np.abs(getattr(b.belief, k) - getattr(b2, k)).max(), 1e-12)
        # 条件: 対照にも同じ処理（R_ctrl_cor が返る）
        out = RP.run_individual(3490, "betray", "none", "report", 3, 0, ["local-reattr", "local-reattr-v33", "C-v34"], "dev")
        for c in ("local-reattr", "local-reattr-v33"):
            self.assertIn("R_ctrl_cor", out["cond"][c])
            self.assertIn("R_ctrl_fol", out["cond"][c])
            self.assertFalse(np.array_equal(out["cond"][c]["R_ctrl_cor"], out["cond"][c]["R_ctrl_pre"]))
        self.assertNotIn("R_ctrl_cor", out["cond"]["C-v34"])

    def test_C_none_v34_is_a_control_that_does_not_absorb(self):
        n, _ = run_hist(3490, "trust", "reliable", "report", 3, "C-none-v34", upto="cor")
        for c in led(n, "correction"):
            self.assertNotIn("m1", c)


class ShapleyDecomposition(unittest.TestCase):
    def test_shapley_values_on_known_games(self):
        pl = ["a", "b", "c"]
        def game(f):
            return {frozenset(s for i, s in enumerate(pl) if m >> i & 1): f(sum(m >> i & 1 for i in range(3))) for m in range(8)}
        sym = RP.shapley_values(game(lambda k: float(k * k)), pl)                           # 対称: 全員同じ
        self.assertAlmostEqual(sym["a"], 3.0)
        self.assertAlmostEqual(sym["b"], 3.0)
        v = {frozenset(s): 0.0 for s in ([], ["a"], ["b"], ["c"], ["a", "b"], ["a", "c"], ["b", "c"], ["a", "b", "c"])}
        v[frozenset(["a", "b"])] = v[frozenset(["a", "b", "c"])] = 1.0                     # a と b が揃って初めて価値。c は何もしない（ダミー）
        phi = RP.shapley_values(v, pl)
        self.assertAlmostEqual(phi["a"], 0.5)
        self.assertAlmostEqual(phi["b"], 0.5)
        self.assertAlmostEqual(phi["c"], 0.0)
        self.assertAlmostEqual(sum(phi.values()), v[frozenset(pl)] - v[frozenset()])        # 効率性

    def test_run_shapley_decomposes_the_gap_between_two_personas_with_efficiency(self):
        players = ("G1", "G2", "G3")
        r = RP.run_shapley(3490, "none", "interp", "betray", "trust", "dev", n=3, players=players)
        self.assertIsNotNone(r)
        self.assertEqual(len(r["CE_subsets"]), 8)
        self.assertEqual(r["CE_subsets"][""], r["CE_target"])                                # 何も入れ替えなければ元の CE
        R_pre = np.array([RP.run_individual(3490, bg, "none", "interp", 3, 0, ["C-v34"], "dev")["cond"]["C-v34"]["R_pre"] for bg in H.BGS])
        scale, mask = RP.scale_of(R_pre)
        res = RP.shapley_from_run(r, scale, mask)
        self.assertGreater(res["gap"], 0.0)
        self.assertAlmostEqual(res["v_empty"], 0.0, places=12)
        self.assertAlmostEqual(sum(res["phi"].values()), res["v_all"] - res["v_empty"], places=10)        # 効率性
        self.assertEqual(set(res["phi"]), set(players))

    def test_split_G5_groups_swap_kappa_and_tau_separately(self):
        t, _ = run_hist(3490, "betray", "unreliable", "interp", 3, "C-v34", upto="cor")
        s, _ = run_hist(3490, "trust", "unreliable", "interp", 3, "C-v34", upto="cor")
        k = RP.swap_groups(t, s, ["G5k"])
        self.assertEqual(k.src_beta, s.src_beta)
        self.assertEqual(k.log_tau, t.log_tau)
        u = RP.swap_groups(t, s, ["G5t"])
        self.assertEqual(u.log_tau, s.log_tau)
        self.assertEqual(u.src_beta, t.src_beta)
        self.assertEqual(RP.SHAPLEY_PLAYERS, ("G1", "G2", "G3", "G4", "G5", "G6"))
        self.assertEqual(len(RP.SHAPLEY_PLAYERS_SPLIT_G5), 7)

    def test_swap_with_nocorr_returns_the_swapped_persons_no_correction_branch(self):
        r = RP.run_swap(3490, "none", "interp", "betray", "trust", RP.VAL_SWAP, "C-v34", "dev", n=3, nocorr=True)
        self.assertIsNotNone(r)
        for k in ("R_cor", "R_fol", "R_cor_nc", "R_fol_nc"):
            self.assertTrue(np.isfinite(r[k]).all(), k)
        self.assertFalse(np.array_equal(r["R_fol"], r["R_fol_nc"]))
        plain = RP.run_swap(3490, "none", "interp", "betray", "trust", RP.VAL_SWAP, "C-v34", "dev", n=3)
        self.assertNotIn("R_cor_nc", plain)
        self.assertTrue(np.array_equal(plain["R_fol"], r["R_fol"]))                       # nocorr を足しても訂正ありの枝は変わらない


class ParallelRunner(unittest.TestCase):
    def tasks(self):
        t = []
        for seed in (3490, 3491):
            for bg in ("betray", "trust"):
                t.append(("ra_protocol:run_individual", (seed, bg, "none", "interp", 3, 0, ["C-v34", "A-g-const", "clamp-τ"], "dev")))
        t.append(("ra_checks_v34:swap_task", (None, 3490, "report", "dev", 3)))
        t.append(("ra_checks_v34:seed_task", (None, 3491, "dev", 0)))
        return t

    def test_parallel_results_are_bit_identical_to_serial_and_only_our_workers_are_used(self):
        tasks = self.tasks()
        serial, info_s = PAR.run_tasks(tasks, serial=True)
        par, info_p = PAR.run_tasks(tasks, workers=3)
        self.assertEqual(canon(serial), canon(par))                                     # ビット単位で同じ（浮動小数点は 16 進、配列はバイト列のハッシュ）
        self.assertTrue(info_s["serial"] and info_s["pids"] == [os.getpid()])
        self.assertEqual(info_p["workers"], 3)
        self.assertEqual(len(info_p["pids"]), 3)
        self.assertNotIn(os.getpid(), info_p["pids"])
        self.assertEqual(info_p["exit_codes"], [0, 0, 0])
        self.assertTrue(set(info_p["worker_pids_of_tasks"]) <= set(info_p["pids"]))
        for pid in info_p["pids"]:                                                       # プールを閉じた後にワーカーは残らない
            self.assertFalse(pid_alive(pid), pid)
        self.assertEqual(len(info_p["seconds"]), len(tasks))

    def test_order_is_preserved_and_worker_errors_surface(self):
        tasks = [("ra_protocol:run_individual", (3490, "betray", "none", "interp", 2, 0, ["C-v34"], "dev")),
                 ("ra_protocol:run_individual", (3400, "betray", "none", "interp", 2, 0, ["C-v34"], "dev"))]      # 評価の種: 拒否される
        with self.assertRaises(PermissionError):
            PAR.run_tasks(tasks, serial=True)
        with self.assertRaises(PermissionError):
            PAR.run_tasks(tasks, workers=2)
        ok = [("ra_protocol:run_individual", (s, "betray", "none", "interp", 2, 0, ["C-v34"], "dev")) for s in (3492, 3490, 3491)]
        res, _ = PAR.run_tasks(ok, workers=2)
        self.assertEqual([r["meta"]["seed"] for r in res], [3492, 3490, 3491])

    def test_wall_clock_estimate_is_a_greedy_makespan(self):
        self.assertAlmostEqual(PAR.estimate_wall([3, 3, 3, 3], 2), 6.0)
        self.assertAlmostEqual(PAR.estimate_wall([10, 1, 1, 1, 1], 2), 10.0)
        self.assertAlmostEqual(PAR.estimate_wall([1.0] * 100, 10), 10.0)
        self.assertEqual(PAR.MAX_WORKERS, 10)


class Mutants(unittest.TestCase):
    def test_every_mutant_is_documented_applies_and_restores_the_classes(self):
        self.assertEqual(MU.MUTANT_IDS, tuple(f"MU{i}" for i in range(1, 9)))
        from npnv3.value import ValueNet
        names = [(NPN, "_m1_absorb"), (NPN, "_tau_step"), (NPN, "_correct"), (NPN, "_m3_evaluate"), (NPN, "_interp_weights"), (ValueNet, "absorb")]
        before = {(c, n): c.__dict__[n] for c, n in names}
        for m in MU.MUTANT_IDS:
            self.assertTrue(MU.MUTANTS[m]["description"])
            with MU.apply_mutant(m):
                changed = [n for (c, n) in names if c.__dict__[n] is not before[(c, n)]]
                self.assertEqual(len(changed), 1, (m, changed))
            self.assertTrue(all(c.__dict__[n] is before[(c, n)] for c, n in names), m)
        with MU.apply_mutant(None):
            pass
        with MU.apply_mutant("baseline"):
            self.assertTrue(all(c.__dict__[n] is before[(c, n)] for c, n in names))
        self.assertEqual(set(MU.MUST_DETECT), {"MU2", "MU3", "MU4", "MU5", "MU7"})
        self.assertEqual(MU.MIN_DETECTED, 7)

    def test_the_mutations_do_what_their_documentation_says(self):
        h = H.make_history(3490, "betray", "none", "interp", 3, 0)

        def go(m):
            with MU.apply_mutant(m):
                n = RP._prefix(3490, h, V34, "mis")
                w_pre = n.value.person.W.copy()
                RP.feed_items(n, h["cor"])
                RP.feed_items(n, h["fol"])
                return n, w_pre
        base, _ = go(None)
        n1, _ = go("MU1")
        self.assertTrue(all(c["m1"]["g"] == 0.5 for c in led(n1, "correction")))
        n2, _ = go("MU2")
        self.assertEqual(n2.log_tau, {})
        self.assertTrue(any(ce["tau_raw"] != 0.0 for ce in led(n2, "correction_experience")))          # 計算・記録される
        self.assertTrue(all(ce["tau_step"] == 0.0 for ce in led(n2, "correction_experience")))
        self.assertNotEqual(base.log_tau, {})
        n3, _ = go("MU3")
        self.assertFalse(np.array_equal(n3.value.person.S, base.value.person.S))                      # Σ が縮む（v3.3 型）
        n4, _w4 = go("MU4")
        c4 = led(n4, "correction")[0]
        self.assertTrue(np.all(np.isclose(np.array([c4["m1"]["pred_change"]["A"][k] for k in NEEDS]), 3.0 * np.array([c4["m1"]["delta"]["A"][k] for k in NEEDS]), atol=1e-9)))
        n5, _ = go("MU5")
        f_led = [e["f"] for e in led(n5, "experience") if e["id"] == "j0"][0]
        f_base = [e["f"] for e in led(base, "experience") if e["id"] == "j0"][0]
        self.assertNotEqual(f_led, f_base)
        n7, _ = go("MU7")
        self.assertFalse(np.array_equal(RP.read_R(n7, "collaborator"), RP.read_R(base, "collaborator")))
        self.assertEqual(NPN._tau_step(1.0, 0.0, 0.0, 0.5), 0.5)                                       # 変異の外では元に戻っている
        with MU.apply_mutant("MU8"):
            self.assertEqual(NPN._tau_step(1.0, 0.0, 0.0, 0.5), 0.0)
            self.assertEqual(NPN._tau_step(0.0, 1.0, 0.0, 0.5), -0.5)
        with MU.apply_mutant("MU6"):
            n6 = NPN(seed=3490, reappraisal_mode="v34")
            self.assertEqual(n6.cfg.tau_max, 3.0)                                                      # 設定そのものは変えない。_m3_evaluate の中だけ

    def test_the_hundred_guilt_corrections_stay_bounded_without_the_mutation_and_escape_with_it(self):
        r0 = CK.seed_task(None, 3490, "dev", stress_n=12)
        self.assertTrue(r0["inv"]["H1"]["ok"] and r0["inv"]["RA8c"]["ok"] and r0["inv"]["H9"]["ok"])
        self.assertAlmostEqual(r0["ev"]["stress"]["log_tau_end"], -math.log(3.0), places=12)
        r6 = CK.seed_task("MU6", 3490, "dev", stress_n=12)
        self.assertFalse(r6["inv"]["H1"]["ok"])
        self.assertFalse(r6["inv"]["RA8c"]["ok"])
        self.assertLess(r6["ev"]["stress"]["tau_end"], 1.0 / 3.0)


class ChecksBaselineAndDetection(unittest.TestCase):
    CELL = (3490, "trust", "reliable", "report")

    def run_cell(self, mutant, cross=False):
        seed, bg, src, err = self.CELL
        return CK.cell_task(mutant, seed, bg, src, err, 3, "dev", cross, False)

    def test_unmutated_run_passes_every_invariant_check(self):
        r = self.run_cell(None, cross=True)
        bad = {k: v["fails"] for k, v in r["inv"].items() if not v["ok"]}
        self.assertEqual(bad, {})
        for k in ("RA3_3", "RA3_4", "RA3_5", "RA5_formula", "RA5_pred", "RA5_sigma", "RA6", "RA8a", "RA8b", "H2", "H4", "H7", "H8", "RA12"):
            self.assertIn(k, r["inv"], k)

    def test_checks_detect_the_mutants_they_are_meant_for(self):
        want = {"MU1": "RA5_formula", "MU4": "RA5_pred", "MU5": "RA6", "MU3": "RA5_sigma"}
        for m, k in want.items():
            r = self.run_cell(m)
            self.assertFalse(r["inv"][k]["ok"], (m, k))

    def test_guard_allows_only_the_phase_seeds_and_never_evaluation_or_reserved(self):
        RP.guard_seeds([3450, 3459], "mut")                            # Stage 4 で許可
        RP.guard_seeds([3490], "dev")
        RP.guard_seeds([3460], "diag")
        for phase in ("dev", "diag", "mut"):
            for s in (3400, 3439, 3500, 3539):
                with self.assertRaises(PermissionError):
                    RP.guard_seeds([s], phase)
        with self.assertRaises(PermissionError):
            RP.guard_seeds([3450], "dev")
        with self.assertRaises(ValueError):
            RP.guard_seeds([3400], "eval")
        self.assertEqual(set(RP.PHASES), {"dev", "diag", "mut"})
        self.assertFalse(set(RP.STAGE_ALLOWED) & (set(RP.SEEDS_EVAL) | set(RP.SEEDS_RESERVED)))

    def test_ra12_and_candidate_helpers(self):
        self.assertEqual(CK.ra12_ok([0.0, 0.01], [0.0, -1.0])[0], True)                # 直前の |成分| ≤ 0.02 は数えない
        self.assertFalse(CK.ra12_ok([1.0, 1.0], [0.1, 0.1])[0])                         # 50% 未満に縮む
        self.assertFalse(CK.ra12_ok([1.0, 1.0, 1.0], [1.0, -1.0, -1.0])[0])             # 符号が 2 つ反転
        self.assertTrue(CK.ra12_ok([1.0, 1.0], [2.0, 1.5])[0])                           # 上昇は崩壊ではない
        self.assertEqual(CK.CANDIDATE_CHECKS, ["C_tau_rule", "C_D_recompute"])


class FreezeFile(unittest.TestCase):
    def test_freeze_protocol_v2_matches_the_files_and_v1_is_untouched(self):
        d = ROOT / "tests" / "results" / "v34"
        v1 = json.loads((d / "freeze-protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(v1["version"], 1)
        self.assertEqual(set(v1["files"]), {"frontalis-v3-lab/tests/ra_protocol.py", "frontalis-v3-lab/data/make_histories_v34.py"})
        p2 = d / "freeze-protocol-v2.json"
        if not p2.exists():
            self.skipTest("freeze-protocol-v2.json は Stage 4 の最後に書く")
        v2 = json.loads(p2.read_text(encoding="utf-8"))
        self.assertEqual(v2["version"], 2)
        for rel, h in v2["files"].items():
            self.assertEqual(RP.file_sha256(ROOT.parent / rel), h, rel)
        for need_ in ("tests/ra_protocol.py", "data/make_histories_v34.py", "tests/mutants_v34.py", "tests/ra_parallel.py", "tests/ra_checks_v34.py", "tests/stage4_mutation.py"):
            self.assertIn("frontalis-v3-lab/" + need_, v2["files"])


if __name__ == "__main__":
    unittest.main()
