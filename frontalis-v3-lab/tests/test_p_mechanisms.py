"""Z-persist（justification_mode = "persist" / "persist_with_decay"。SPEC-v3.3 第 8.3 節）の機構の単体検査。判定基準 P1〜P7 は test_p_persist.py、B1〜B3 は test_b_health.py。
ここは「式どおりに動くか」「源が記録されるか」「割り引きの記録」「決定の記録と強制再生（B1 の仕組み）」「保存・復元」。
"""
import copy
import tempfile
import unittest

import numpy as np

from helpers import deep_equal, make
import p_protocol as P
import yz_protocol as Y
from npnv3 import NPN, Config


def states_equal(x, y):
    mx, ax = x.get_state()
    my, ay = y.get_state()
    return deep_equal(mx, my) and ax.keys() == ay.keys() and all(np.array_equal(ax[k], ay[k]) for k in ax)


def last_just(n, after=None):
    es = [e for e in n.ledger.entries if e["kind"] == "justification" and (after is None or e["after"] == after)]
    return es[-1]


def ev_of(entry, source=None, cue=None):
    return [e for e in entry["events"] if (source is None or e["source"] == source) and (cue is None or e["cue"] == cue)]


class Config_(unittest.TestCase):
    def test_default_is_orig_and_unknown_modes_are_refused(self):
        self.assertEqual(Config().justification_mode, "orig")
        for m in ("orig", "persist", "persist_with_decay"):
            self.assertEqual(Config(justification_mode=m).justification_mode, m)
        with self.assertRaises(ValueError):
            Config(justification_mode="forever")
        c = Config(justification_mode="persist")
        self.assertEqual((c.eta_r, c.j_num, c.rho_dis), (0.03, 2.0, 0.5))


class Events(unittest.TestCase):
    def test_first_choice_is_spread_only_and_nothing_else_moves_J_in_persist(self):
        n = make(0, justification_mode="persist")
        r = n.experience(Y.hesitant_choice("e0", "ja", "jb"))
        c = r["choice"]
        won, lost = ("concept:ja", "concept:jb") if c["chosen"] == 0 else ("concept:jb", "concept:ja")
        ent = last_just(n)
        self.assertEqual(sorted({e["source"] for e in ent["events"]}), ["spread"])
        self.assertEqual(n.just[won], n.cfg.eta_j * c["conflict"])                   # 有界化も減衰もない（orig では tanh と (1 − λ_J)）
        self.assertEqual(n.just[lost], -n.cfg.eta_j * c["conflict"])
        before = dict(n.just)
        for i in range(30):                                                           # 二択でない経験では、時間が経っても何も変わらない
            n.experience({"id": f"z{i}", "context": "workshop", "concepts": ["x"], "participants": [], "outcomes": [{"type": "novelty", "actor": "world", "magnitude": 0.3}]})
        self.assertEqual(n.just, before)
        self.assertEqual(sum(1 for e in n.ledger.entries if e["kind"] == "justification"), 1)

    def test_orig_is_the_same_case_but_bounded_and_decayed(self):
        n = make(0)
        n.experience(Y.hesitant_choice("e0", "ja", "jb"))
        self.assertLess(max(abs(v) for v in n.just.values()), n.cfg.eta_j)             # tanh と減衰で 0.05 より小さい
        self.assertEqual(sorted({e["source"] for e in last_just(n)["events"]}), ["bound", "decay", "spread"])

    def test_reaffirm_follows_eta_r_times_one_minus_J_over_J_num_and_needs_a_previous_choice(self):
        n, fav, oth = P.base_persona(0)
        x = f"concept:{fav}"
        self.assertIn(x, n.chosen_before)
        j0 = n.just[x]
        r = P.run_forced(n, P.rechoice(fav, oth, "r0"), 0)
        ent = last_just(n, "r0")
        re_ = ev_of(ent, "reaffirm", x)
        self.assertEqual(len(re_), 1)
        self.assertEqual(re_[0]["delta"], n.cfg.eta_r * (1.0 - abs(j0) / n.cfg.j_num))
        # 以前に選んだことのない手がかりは reaffirm にならない（最初の選択は spread だけ）
        m = make(0, justification_mode="persist")
        m.experience(P.rechoice("never_a", "never_b", "r1"))
        self.assertEqual(ev_of(last_just(m), "reaffirm"), [])
        # 葛藤がなくても起きる（易しい二択でも reaffirm）
        self.assertLess(r["choice"]["conflict"], 1e-6)

    def test_counter_when_the_unchosen_side_has_positive_J(self):
        n, fav, oth = P.base_persona(1)
        x = f"concept:{fav}"
        self.assertGreater(n.just[x], 0.0)
        j0 = n.just[x]
        P.run_forced(n, P.rechoice(fav, oth, "c0"), 1)                                # 反対側（oth）を選ぶ
        ent = last_just(n, "c0")
        co = ev_of(ent, "counter", x)
        self.assertEqual(len(co), 1)
        self.assertEqual(co[0]["delta"], -n.cfg.eta_r * j0)
        self.assertEqual(ev_of(ent, "reaffirm", x), [])

    def test_cap_is_for_numerical_safety_and_is_recorded_as_an_event(self):
        n = make(0, justification_mode="persist")
        n.just = {"concept:ja": 1.9, "concept:jb": -1.95}
        new, ev = n._j_step_persist([("concept:ja", "reaffirm", 0.3), ("concept:jb", "spread", -0.2), ("concept:jc", "spread", 0.1)])
        self.assertEqual(new["concept:ja"], n.cfg.j_num)
        self.assertEqual(new["concept:jb"], -n.cfg.j_num)
        self.assertEqual(new["concept:jc"], 0.1)
        caps = {c: d for c, s_, d in ev if s_ == "cap"}
        self.assertAlmostEqual(caps["concept:ja"], 2.0 - 2.2, places=14)
        self.assertAlmostEqual(caps["concept:jb"], -2.0 - (-2.15), places=14)
        self.assertNotIn("concept:jc", caps)
        # 出来事がなければ何も変えない（時間の減衰もない）
        same, ev0 = n._j_step_persist([])
        self.assertEqual(same, n.just)
        self.assertEqual(ev0, [])
        # 心理的な上限ではない: ふつうの履歴では上限に届かない（B2 の P3）
        self.assertGreaterEqual(n.cfg.j_num, 10 * n.cfg.j_max)

    def test_overwhelm_reduces_positive_J_by_the_excess_in_proportion(self):
        n, fav, oth = P.base_persona(2)
        for i in range(8):
            P.run_forced(n, P.rechoice(fav, oth, f"o{i}"), 0)
        x = f"concept:{fav}"
        pos = {c: v for c, v in n.just.items() if c in (x, "ind:ally", "cls:friend") and v > 0.0}
        jp = sum(pos.values())
        r = P.run_forced(n, P.rechoice(fav, oth, "ov", harm=True), 0)
        ent = last_just(n, "ov")
        ov = ev_of(ent, "overwhelm")
        excess = max(max(0.0, -r["delta"]) - float(np.tanh(jp)), 0.0)
        self.assertGreater(excess, 0.0)
        frac = min(n.cfg.eta_ov * excess, 1.0)
        self.assertEqual({e["cue"] for e in ov}, set(pos))                            # 正の J の手がかりだけ
        for e in ov:
            self.assertAlmostEqual(e["delta"], -frac * pos[e["cue"]], places=14)
        # 負の J と、選ばなかった側は動かさない
        self.assertEqual(ev_of(ent, "overwhelm", f"concept:{oth}"), [])

    def test_no_overwhelm_without_positive_J_on_the_chosen_side(self):
        n = Y.warm_persona(0, justification_mode="persist")
        P.run_forced(n, P.rechoice("fresh_a", "fresh_b", "n0", harm=True), 0)
        self.assertEqual(ev_of(last_just(n, "n0"), "overwhelm"), [])

    def test_persist_with_decay_decays_every_experience_and_records_it(self):
        n = make(0, justification_mode="persist_with_decay")
        n.experience(Y.hesitant_choice("d0", "ja", "jb"))
        j = dict(n.just)
        n.experience({"id": "d1", "context": "workshop", "concepts": ["x"], "participants": [], "outcomes": []})
        for c, v in j.items():
            self.assertAlmostEqual(n.just[c], (1.0 - n.cfg.lambda_j) * v, places=14)
        ent = last_just(n, "d1")
        self.assertEqual(sorted({e["source"] for e in ent["events"]}), ["decay"])
        self.assertAlmostEqual(sum(n.just_src["concept:ja"].values()), n.just["concept:ja"], places=12)

    def test_no_justification_ablation_builds_no_J_in_persist_either(self):
        n = make(0, justification_mode="persist", no_justification=True)
        for i in range(10):
            n.experience(Y.hesitant_choice(f"nj{i}", "a", "b"))
        self.assertEqual(n.just, {})
        self.assertEqual(sum(1 for e in n.ledger.entries if e["kind"] == "justification"), 0)

    def test_without_choices_persist_orig_and_decay_variants_learn_identically(self):
        """二択がなければ、どの条件でも W・C・a は同一（J は学習の教師にならない。persist の抵抗は二択で J > 0 のときだけ）。"""
        hist = Y.forced_history("R")[:40]
        ns = [make(0, justification_mode=m) for m in ("orig", "persist", "persist_with_decay")]
        for e in hist:
            for n in ns:
                n.experience(copy.deepcopy(e))
        for n in ns[1:]:
            self.assertTrue(np.array_equal(ns[0].value.person.W, n.value.person.W) and np.array_equal(ns[0].value.sit.W, n.value.sit.W))
            self.assertTrue(np.array_equal(ns[0].cond.CK, n.cond.CK) and np.array_equal(ns[0].concern.a, n.concern.a))


class Discount(unittest.TestCase):
    def setUp(self):
        base, self.fav, self.oth = P.base_persona(0)
        for i in range(10):
            P.run_forced(base, P.rechoice(self.fav, self.oth, f"s{i}"), 0)
        self.jst = copy.deepcopy(base)
        self.ctl = copy.deepcopy(base)
        self.ctl.cfg = self.ctl.cfg.replace(rho_dis=0.0)                              # 対照: 同じ J・同じ状態で、抵抗（割り引き）だけを切る
        self.ref = P.rechoice(self.fav, self.oth, "refute", harm=True)

    def test_negative_part_of_person_teacher_moves_to_the_situation_by_rho_dis_tanh_J(self):
        ap_c = P.run_forced(self.ctl, self.ref, 0)
        ap_j = P.run_forced(self.jst, self.ref, 0)
        d = ap_j["discount"]
        # 割り引きの割合: ρ_dis tanh(選んだ側だけの手がかりの正の J の和)。台帳の記録から再計算する
        q = d["q"]
        self.assertAlmostEqual(q, self.jst.cfg.rho_dis * float(np.tanh(d["J_chosen"])), places=14)
        self.assertGreater(q, 0.0)
        tc, tj = ap_c["teachers"], ap_j["teachers"]
        needs = list(tc["situation"].keys())
        moved = np.array(d["moved"]["ally"])
        t_c = np.array([tc["person"]["ally"][k] for k in needs])
        t_j = np.array([tj["person"]["ally"][k] for k in needs])
        s_c = np.array([tc["situation"][k] for k in needs])
        s_j = np.array([tj["situation"][k] for k in needs])
        self.assertTrue(np.allclose(moved, q * np.minimum(t_c, 0.0), atol=1e-14))
        self.assertTrue(np.allclose(t_j, t_c - moved, atol=1e-14))                    # 人の教師は負の部分が (1 − q) 倍、正の部分は変わらない
        self.assertTrue(np.allclose(s_j, s_c + moved, atol=1e-14))                    # 場面の教師は moved だけ負へ
        self.assertTrue(np.allclose(t_j + s_j, t_c + s_c, atol=1e-14))                # 合計は変わらない
        self.assertTrue(np.all(t_j[t_c > 0] == t_c[t_c > 0]))
        self.assertLess(float(np.abs(moved).sum()), float(np.abs(np.minimum(t_c, 0.0)).sum()))   # q < 1
        self.assertTrue(float(np.abs(moved).sum()) > 0.0)

    def test_the_discount_is_in_the_ledger_and_the_experience_log(self):
        P.run_forced(self.jst, self.ref, 0)
        ent = last_just(self.jst, "refute")
        self.assertIn("discount", ent)
        self.assertEqual(ent["discount"]["q"], self.jst.exp_log[-1]["discount_q"])
        P.run_forced(self.ctl, self.ref, 0)
        self.assertEqual(self.ctl.exp_log[-1]["discount_q"], 0.0)                     # 抵抗なし（ρ_dis = 0）なら割り引きは 0
        self.assertTrue(all(float(np.abs(m).sum()) == 0.0 for m in last_just(self.ctl, "refute")["discount"]["moved"].values()))

    def test_learning_difference_is_the_kalman_response_to_the_recorded_discount(self):
        r = P.p5_worker(0)
        self.assertLess(r["person"]["rel_error"], 1e-9)

    def test_no_discount_in_orig_and_J_is_not_a_teacher_there(self):
        base = Y.warm_persona(0, justification_mode="orig")
        fav, oth = P.phase1(base)
        for i in range(10):
            P.run_forced(base, P.rechoice(fav, oth, f"s{i}"), 0)
        a, b = copy.deepcopy(base), copy.deepcopy(base)
        b.just = {}
        ra = P.run_forced(a, self.ref, 0)
        rb = P.run_forced(b, self.ref, 0)
        self.assertNotIn("discount", ra)
        self.assertEqual(ra["teachers"], rb["teachers"])
        self.assertTrue(np.array_equal(a.value.person.W, b.value.person.W) and np.array_equal(a.value.sit.W, b.value.sit.W))

    def test_no_discount_without_positive_J_on_the_chosen_side(self):
        n = Y.warm_persona(0, justification_mode="persist")                           # J が一つもない
        ap = P.run_forced(n, P.rechoice("fresh_a", "fresh_b", "fr", harm=True), 0)
        self.assertNotIn("discount", ap)
        self.assertEqual(ev_of(last_just(n, "fr"), "overwhelm"), [])

    def test_replay_values_reproduces_the_discounted_learning(self):
        for i in range(4):
            P.run_forced(self.jst, P.rechoice(self.fav, self.oth, f"rp{i}", harm=True), 0)
        self.assertTrue(any(r["discount_q"] > 0 for r in self.jst.exp_log))
        nv = self.jst._replay_values()
        for kal, ref in ((nv.person, self.jst.value.person), (nv.sit, self.jst.value.sit)):
            self.assertEqual(kal.names, ref.names)
            self.assertTrue(np.allclose(kal.W, ref.W, atol=1e-10))

    def test_ledger_W_totals_equal_the_models_with_discount(self):
        for i in range(4):
            P.run_forced(self.jst, P.rechoice(self.fav, self.oth, f"lt{i}", harm=True), 0)
        for model, kal in (("person", self.jst.value.person), ("situation", self.jst.value.sit)):
            tot = self.jst.ledger.totals(model)
            for nm, row in tot.items():
                self.assertTrue(np.allclose(kal.W[kal.idx[nm]], row, atol=1e-10))


class DecisionsAndForcedReplay(unittest.TestCase):
    def events(self):
        out = [dict(e) for e in Y.forced_history("R")[:12]]
        out.append({"id": "opt1", "context": "studio", "concepts": ["music"], "participants": [], "outcomes": [{"type": "praise", "actor": "world", "magnitude": 0.5}],
                    "optional": True, "announced": ["praise"], "session": "o1"})
        out += [Y.hesitant_choice(f"dh{i}", "da", "db") for i in range(6)]
        return out

    def test_decisions_are_recorded_per_non_correction_experience_and_do_not_change_numbers(self):
        a = make(0, record_decisions=True, justification_mode="persist")
        b = make(0, justification_mode="persist")
        for e in self.events():
            a.experience(copy.deepcopy(e))
            b.experience(copy.deepcopy(e))
        self.assertEqual(len(a.decisions), len(self.events()))
        ma, aa = a.get_state()
        mb, ab = b.get_state()
        self.assertTrue(all(np.array_equal(aa[k], ab[k]) for k in aa) and aa.keys() == ab.keys())       # 数値は同一（設定の record_decisions だけが違う）
        for k in ("justification", "justification_sources", "chosen_before"):
            self.assertEqual(ma[k], mb[k])
        self.assertEqual(b.decisions, [])
        ch = [d for d in a.decisions if d["id"].startswith("dh")]
        self.assertTrue(all(d["choice"] in (0, 1) for d in ch))
        self.assertTrue(all(isinstance(d["assign"], dict) for d in a.decisions))

    def test_forced_replay_of_own_decisions_reproduces_the_run_bit_for_bit(self):
        for design in ("spec_literal", "revised"):
            ev = self.events()
            base = make(3, record_decisions=True, justification_mode="persist", lc_design=design)
            start = copy.deepcopy(base)
            for e in ev:
                base.experience(copy.deepcopy(e))
            rep = copy.deepcopy(start)
            rep.force_decisions = base.decisions
            rep._force_i = 0
            for e in ev:
                rep.experience(copy.deepcopy(e))
            self.assertTrue(states_equal(base, rep), design)

    def test_forcing_a_different_choice_changes_only_what_that_decision_causes_and_keeps_the_random_stream(self):
        base = make(3, record_decisions=True, justification_mode="persist")
        start = copy.deepcopy(base)
        ch = Y.hesitant_choice("fc", "fa", "fb")
        base.experience(copy.deepcopy(ch))
        side = base.decisions[0]["choice"]
        other = copy.deepcopy(start)
        other.force_decisions = [{"choice": 1 - side}]
        other._force_i = 0
        r = other.experience(copy.deepcopy(ch))
        self.assertEqual(r["choice"]["chosen"], 1 - side)
        self.assertEqual(other.rng.bit_generator.state, base.rng.bit_generator.state)   # 乱数は同じだけ引く
        won = "concept:fa" if r["choice"]["chosen"] == 0 else "concept:fb"
        self.assertGreater(other.just[won], 0.0)

    def test_forcing_the_approach_decision_of_an_optional_offer(self):
        n = make(0, record_decisions=True)
        e = {"id": "off", "context": "studio", "concepts": ["music"], "participants": [], "outcomes": [{"type": "praise", "actor": "world", "magnitude": 0.5}],
             "optional": True, "announced": ["praise"], "session": "of"}
        for approach in (True, False):
            k = copy.deepcopy(n)
            k.force_decisions = [{"approach": approach}]
            k._force_i = 0
            r = k.experience(copy.deepcopy(e))
            self.assertEqual(r["approached"], approach)
            self.assertEqual(k.decisions[-1]["approach"], approach)


class PersistState(unittest.TestCase):
    def test_roundtrip_keeps_sources_and_previous_choices_and_continues_identically(self):
        n, fav, oth = P.base_persona(0)
        for i in range(5):
            P.run_forced(n, P.rechoice(fav, oth, f"st{i}"), 0)
        P.run_forced(n, P.rechoice(fav, oth, "stf", harm=True), 0)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        self.assertEqual(n.just, m.just)
        self.assertEqual(n.just_src, m.just_src)
        self.assertEqual(n.chosen_before, m.chosen_before)
        self.assertEqual(m.cfg.justification_mode, "persist")
        self.assertTrue(states_equal(n, m))
        self.assertEqual([r["discount_q"] for r in n.exp_log], [r["discount_q"] for r in m.exp_log])
        for i in range(4):
            ch = P.rechoice(fav, oth, f"stg{i}", harm=(i % 2 == 0))
            self.assertTrue(deep_equal(P.run_forced(n, ch, 0), P.run_forced(m, ch, 0)))
        self.assertTrue(states_equal(n, m))

    def test_schema_refuses_a_state_saved_with_another_mode_only_by_config_not_silently(self):
        n = make(0, justification_mode="persist")
        n.experience(Y.hesitant_choice("sm", "a", "b"))
        meta, arrs = n.get_state()
        self.assertEqual(meta["config"]["justification_mode"], "persist")
        self.assertEqual(sorted(meta["justification_sources"]), sorted(n.just_src))


if __name__ == "__main__":
    unittest.main()
