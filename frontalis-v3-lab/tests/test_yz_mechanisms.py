"""訂正（kind = "correction"。SPEC-v3.3 第 2 節）と二択（kind = "choice"。第 3 節）の機構の単体検査。判定基準 Y1〜Y7・Z1〜Z7 は test_y_correction.py・test_z_choice.py。
ここは「式どおりに動くか」「不変量が保たれるか」「保存・復元・決定性」。
"""
import copy
import tempfile
import unittest

import numpy as np

from helpers import deep_equal, feed, load_jsonl, make, mh, persona
import yz_protocol as Y
from npnv3 import NPN, Config, SchemaError
from npnv3.innate import TYPE_INDEX
from npnv3.schema import validate_choice, validate_correction, validate_experience
from npnv3.value import Concern

CT = TYPE_INDEX["credit_taken"]


def mis_persona(seed=0, **kw):
    n = make(seed, **kw)
    Y.feed(n, Y.y_history(seed, True))
    return n


def states_equal(x, y):
    mx, ax = x.get_state()
    my, ay = y.get_state()
    return deep_equal(mx, my) and ax.keys() == ay.keys() and all(np.array_equal(ax[k], ay[k]) for k in ax)


def correction(i=0, seed=0):
    return Y.y_corrections(seed)[i]


class SchemaOfNewKinds(unittest.TestCase):
    def test_kind_experience_is_optional_and_other_kinds_are_not_experiences(self):
        e = Y.y_history(0, True)[0]
        validate_experience({**e, "kind": "experience"})
        with self.assertRaises(SchemaError):
            validate_experience({**e, "kind": "correction"})
        with self.assertRaises(SchemaError):
            make(0).experience({**e, "kind": "teleport"})

    def test_correction_schema(self):
        c = correction()
        validate_correction(c)
        for mutate in (lambda x: x.pop("refers_to"), lambda x: x.update(corrected_outcomes="no"), lambda x: x.update(extra=1),
                       lambda x: x["corrected_outcomes"][0].update(type="teleport"), lambda x: x["corrected_outcomes"][0].update(magnitude=1.5),
                       lambda x: x.update(source=""), lambda x: x.update(kind="experience")):
            bad = copy.deepcopy(c); mutate(bad)
            with self.assertRaises(SchemaError):
                validate_correction(bad)

    def test_choice_schema(self):
        ch = Y.hesitant_choice("c", "a", "b")
        validate_choice(ch)
        one = copy.deepcopy(ch); one["options"] = one["options"][:1]
        three = copy.deepcopy(ch); three["options"].append(three["options"][0])
        miss = copy.deepcopy(ch); miss["outcomes_by_option"].pop("1")
        actor = copy.deepcopy(ch); actor["outcomes_by_option"]["0"] = [{"type": "harm", "actor": "nobody", "magnitude": 0.5}]
        extra = copy.deepcopy(ch); extra["x"] = 1
        noctx = copy.deepcopy(ch); noctx["options"][0].pop("context")
        for bad in (one, three, miss, actor, extra, noctx):
            with self.assertRaises(SchemaError):
                validate_choice(bad)

    def test_appraise_refuses_correction_and_previews_choice_without_changing_state(self):
        n = mis_persona()
        with self.assertRaises(SchemaError):
            n.appraise(correction())
        before = copy.deepcopy(n)
        r = n.appraise(Y.hesitant_choice("pc", "a", "b"))
        self.assertIsNone(r["chosen"])
        self.assertAlmostEqual(r["p_choose"][0] + r["p_choose"][1], 1.0, places=14)
        self.assertTrue(states_equal(n, before))                              # 乱数の状態も含めて変わらない


class CorrectionBelief(unittest.TestCase):
    def test_withdraw_and_install_keep_all_counts_consistent_with_the_records(self):
        n = mis_persona()
        for i in range(8):
            n.experience(correction(i))
        b = n.belief
        R = b.rec_n
        ind, ctx, y, mk, phi = b.rec_ind[:R], b.rec_ctx[:R], b.rec_y[:R], b.rec_m[:R], b.rec_phi[:R]
        wp, wc = mk * phi, mk * (1 - phi)
        for k in range(y.shape[1]):
            self.assertTrue(np.allclose(b.N1[:, k], np.bincount(ind, weights=wp[:, k] * y[:, k], minlength=len(b.ind_ids)), atol=1e-12))
            self.assertTrue(np.allclose(b.CP1[:, k], np.bincount(ctx, weights=wc[:, k] * y[:, k], minlength=len(b.ctx_names)), atol=1e-12))
        self.assertTrue(np.allclose(b.pi[:, 0], 1.0 + wp.sum(axis=0), atol=1e-12))
        # 類ごとの生の計数 = 記録の観測の和（この履歴の全観測は記録にある）
        for c in b.rawcls:
            rows = [r for r in range(R) if b.ind_cls[int(ind[r])] == c]
            self.assertTrue(np.allclose(b.raw_n1[b.rawcls_idx[c]], (mk[rows] * y[rows]).sum(axis=0), atol=1e-12), c)
            self.assertTrue(np.allclose(b.raw_n0[b.rawcls_idx[c]], (mk[rows] * (1 - y[rows])).sum(axis=0), atol=1e-12), c)
        # 自分・世界の観測: 文脈ごとの和
        for ci in range(len(b.ctx_names)):
            sel = b.sw_ctx[: b.sw_n] == ci
            self.assertTrue(np.allclose(b.CS1[ci], (b.sw_m[: b.sw_n][sel] * b.sw_y[: b.sw_n][sel]).sum(axis=0), atol=1e-12))
        # リプレイ緩衝の行も訂正後の観測
        for r in range(R):
            slot = np.nonzero(b.rb_ser == b.rec_ser[r])[0]
            if slot.size:
                self.assertTrue(np.array_equal(b.rb_y[slot[0]], y[r]) and np.array_equal(b.rb_m[slot[0]], mk[r]))

    def test_the_corrected_observation_replaces_the_original_one(self):
        n = mis_persona()
        b = n.belief
        iA, iO = b.ind_idx["A"], b.ind_idx["organizer"]
        rows = np.nonzero(b.rec_exp[: b.rec_n] == 0)[0]                       # 経験 0（j0）の観測の行
        self.assertEqual(b.rec_y[rows[0], CT], 1.0)                           # 当時: A が手柄を取った
        n.experience(correction(0))
        ra = [r for r in np.nonzero(b.rec_exp[: b.rec_n] == 0)[0] if b.rec_ind[r] == iA][0]
        ro = [r for r in np.nonzero(b.rec_exp[: b.rec_n] == 0)[0] if b.rec_ind[r] == iO][0]
        self.assertEqual(b.rec_y[ra, CT], 0.0)
        self.assertEqual(b.rec_y[ro, CT], 1.0)

    def test_restating_the_same_outcomes_leaves_the_observations_and_value_teachers_equal(self):
        n = mis_persona()
        b0 = copy.deepcopy(n.belief)
        e0 = n.exp_log[0]
        same = {"id": "same", "kind": "correction", "refers_to": "j0", "corrected_outcomes": copy.deepcopy(e0["outcomes"])}
        w0 = (n.value.person.W.copy(), n.value.person.S.copy())
        r = n.experience(same)
        self.assertTrue(np.array_equal(n.belief.rec_y[: b0.rec_n], b0.rec_y[: b0.rec_n]))
        self.assertTrue(np.array_equal(n.belief.raw_n1, b0.raw_n1) and np.array_equal(n.belief.raw_n0, b0.raw_n0))
        for pid, d in r["shifts"].items():                                    # 教師の差は、φ が逐次の推定で少し変わる分だけ（結果は同じ）
            self.assertLess(max(abs(v) for v in d.values()), 0.1)

    def test_second_correction_is_relative_to_the_current_facts_and_keeps_the_original(self):
        n = mis_persona()
        n.experience(correction(0))
        back = {"id": "back", "kind": "correction", "refers_to": "j0", "corrected_outcomes": copy.deepcopy(n.exp_log[0]["original_outcomes"])}
        n.experience(back)
        rec = n.exp_log[0]
        self.assertEqual(rec["corrections"], ["c0", "back"])
        self.assertEqual(rec["outcomes"], rec["original_outcomes"])
        self.assertEqual(rec["original_outcomes"][0]["actor"], "A")           # 当時の結果は変わらない

    def test_ambiguous_corrected_outcomes_are_masked_not_counted(self):
        n = mis_persona()
        amb = {"id": "amb-c", "kind": "correction", "refers_to": "j0",
               "corrected_outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "A", "magnitude": 0.8}]}
        n.experience(amb)
        b = n.belief
        iA = b.ind_idx["A"]
        r = [r for r in np.nonzero(b.rec_exp[: b.rec_n] == 0)[0] if b.rec_ind[r] == iA][0]
        self.assertEqual(b.rec_m[r, CT], 0.0)                                 # 曖昧な型は観測なし

    def test_errors(self):
        n = mis_persona()
        with self.assertRaises(ValueError):
            n.experience({"id": "x", "kind": "correction", "refers_to": "nope", "corrected_outcomes": []})
        bad = correction(0); bad["corrected_outcomes"][0]["actor"] = "stranger_x"
        with self.assertRaises(ValueError):
            n.experience(bad)
        for kw in ({"no_scope": True}, {"no_reattribution": True}):
            m = make(0, **kw)
            Y.feed(m, Y.y_history(0, True)[:2])
            with self.assertRaises(ValueError):
                m.experience(correction(0))

    def test_an_avoided_experience_cannot_be_corrected(self):
        n = make(0)
        n.cond.set_single("type:praise", -2.0)
        e = {"id": "o", "context": "studio", "concepts": ["music"], "participants": [], "outcomes": [{"type": "praise", "actor": "world", "magnitude": 0.5}],
             "optional": True, "announced": ["praise"]}
        for i in range(60):
            m = copy.deepcopy(n)
            m.rng = np.random.default_rng(i)
            r = m.experience({**e, "id": f"o{i}"})
            if not r["approached"]:
                with self.assertRaises(ValueError):
                    m.experience({"id": "c", "kind": "correction", "refers_to": f"o{i}", "corrected_outcomes": []})
                return
        self.fail("回避する場合が見つからなかった")


class CorrectionValue(unittest.TestCase):
    def test_reappraisal_is_one_kalman_update_with_observation_noise_R_over_a_w_reapp(self):
        n = mis_persona()
        a = n.concern.a.copy()
        ref = copy.deepcopy(n.value)
        c = correction(0)
        r = n.experience(c)
        # 独立に再計算: 参加者ごとに shift つきの更新（観測雑音 R/(a_k w_reapp)）
        for p in n.exp_log[0]["participants"]:
            sh = np.array([r["shifts"][p["id"]][k] for k in ("R", "C", "A", "S", "Sa")])
            if np.any(np.abs(sh) > 1e-12):
                ref.person.update(ref.person_feats(p["id"], p["class"]), np.zeros(5), a * n.cfg.w_reapp, shift=sh)
        self.assertTrue(np.array_equal(n.value.person.W, ref.person.W))
        self.assertTrue(np.array_equal(n.value.person.S, ref.person.S))

    def test_confident_values_move_less(self):
        """確信の高い（分散の小さい）古い価値ほど、動きにくい: 同じ訂正でも、経験が多いほど動きが小さい。"""
        moves = []
        for n_prior in (1, 8):
            n = make(0)
            hist = Y.y_history(0, True)
            Y.feed(n, hist[: 2 * n_prior])                                    # j0..j{n_prior-1} と f
            before = n.value.person.W[n.value.person.idx["ind:A"]].copy()
            n.experience(correction(0))
            moves.append(float(np.abs(n.value.person.W[n.value.person.idx["ind:A"]] - before).sum()))
        self.assertGreater(moves[0], moves[1])

    def test_conditioned_value_concern_clock_mood_pending_are_not_touched(self):
        n = mis_persona()
        snap = copy.deepcopy(n)
        n.experience(correction(0))
        for key in ("CK", "TK", "AL", "CNT", "UID"):
            self.assertTrue(np.array_equal(getattr(n.cond, key), getattr(snap.cond, key)), key)
        self.assertTrue(np.array_equal(n.concern.a, snap.concern.a) and n.concern.run == snap.concern.run)
        self.assertEqual((n.n_experiences, n.n_avoided, n._clock(), n.mood, n.pending), (snap.n_experiences, snap.n_avoided, snap._clock(), snap.mood, snap.pending))
        self.assertTrue(np.array_equal(n.h, snap.h))
        self.assertTrue(np.array_equal(n.value.sit.W, snap.value.sit.W))      # 状況モデルは訂正で動かない
        self.assertEqual(n.n_corrections, 1)

    def test_no_reappraisal_changes_the_belief_but_not_the_value_at_all(self):
        n = mis_persona(no_reappraisal=True)
        w = (n.value.person.W.copy(), n.value.person.S.copy(), n.value.sit.W.copy())
        pa = Y.belief_A(n)
        Y.feed(n, Y.y_corrections(0))
        self.assertTrue(np.array_equal(n.value.person.W, w[0]) and np.array_equal(n.value.person.S, w[1]) and np.array_equal(n.value.sit.W, w[2]))
        self.assertLess(Y.belief_A(n), pa)                                    # 信念は訂正される

    def test_retroactive_value_replays_from_the_start_and_matches_the_uncorrected_replay_when_nothing_changes(self):
        n = mis_persona()
        nv = n._replay_values()                                               # 訂正がなければ、再生は当時の学習と一致する
        self.assertTrue(np.array_equal(nv.person.W, n.value.person.W) and np.array_equal(nv.person.S, n.value.person.S))
        self.assertTrue(np.array_equal(nv.sit.W, n.value.sit.W) and np.array_equal(nv.sit.S, n.value.sit.S))
        self.assertEqual(nv.person.names, n.value.person.names)

    def test_replay_matches_the_live_value_for_a_long_history_with_ambiguity_and_optional_offers(self):
        n = make(3)
        for e in mh.generate_history("K", 3, 36):
            n.experience(e)
        nv = n._replay_values()
        self.assertTrue(np.array_equal(nv.person.W, n.value.person.W) and np.array_equal(nv.sit.W, n.value.sit.W))

    def test_ledger_totals_equal_W_after_corrections_in_every_mode(self):
        for kw in ({}, {"retroactive_value": True}, {"no_reappraisal": True}):
            n = mis_persona(**kw)
            Y.feed(n, Y.y_corrections(0))
            for model, kal in (("person", n.value.person), ("situation", n.value.sit)):
                tot = n.ledger.totals(model)
                for nm, row in tot.items():
                    self.assertTrue(np.allclose(kal.W[kal.idx[nm]], row, atol=1e-12), (kw, model, nm))
                for nm, w in zip(kal.names, kal.W):
                    if nm not in tot:
                        self.assertTrue(np.all(w == 0.0), (kw, nm))

    def test_the_record_at_the_time_is_never_modified(self):
        n = mis_persona()
        before = copy.deepcopy(n.ledger.entries)
        Y.feed(n, Y.y_corrections(0))
        self.assertEqual(n.ledger.entries[: len(before)], before)
        self.assertEqual([e["kind"] for e in n.ledger.entries[len(before):]], ["correction"] * 8)
        ent = n.ledger.entries[-1]
        for key in ("refers_to", "original_outcomes", "corrected_outcomes", "shifts", "like", "Q", "f", "emotions", "belief_change", "dW", "reappraisal"):
            self.assertIn(key, ent)
        self.assertEqual(ent["original_outcomes"][0]["actor"], "A")
        self.assertEqual(ent["corrected_outcomes"][0]["actor"], "organizer")

    def test_correction_readout_relief_and_guilt_are_bounded_and_do_not_teach(self):
        n = mis_persona()
        r = n.experience(correction(0))
        for v in r["emotions"].values():
            self.assertTrue(0.0 <= v <= 1.0)
        self.assertTrue(-1.0 <= r["f"] <= 1.0)

    def test_correction_is_deterministic_and_survives_save_and_restore(self):
        a, b = mis_persona(), mis_persona()
        Y.feed(a, Y.y_corrections(0)[:4]); Y.feed(b, Y.y_corrections(0)[:4])
        self.assertTrue(states_equal(a, b))
        with tempfile.TemporaryDirectory() as d:
            a.save(d)
            m = NPN.load(d)
        self.assertTrue(states_equal(a, m))
        rest = Y.y_corrections(0)[4:] + Y.good_with_A(3)
        for e in rest:
            ra, rm = a.experience(copy.deepcopy(e)), m.experience(copy.deepcopy(e))
            self.assertTrue(deep_equal(ra, rm))
        self.assertTrue(states_equal(a, m))
        # 保存後の訂正でも、経験記録の位置が復元される
        self.assertEqual(m._exp_pos, a._exp_pos)

    def test_retrain_of_heads_follows_the_corrected_buffer(self):
        n = mis_persona(mixture_heads=("mlp", "linear", "knn1", "none"))
        Y.feed(n, Y.y_corrections(0))
        for nm in ("mlp", "linear"):
            self.assertEqual(len(n.belief.loo[nm]), len(n.belief.rawcls))


class ChoiceMechanics(unittest.TestCase):
    def test_probabilities_are_the_softmax_of_beta_times_q(self):
        n = Y.warm_persona(0)
        r = n.appraise(Y.easy_choice(0))
        q = [o["Q"] for o in r["options"]]
        z = n.cfg.beta_choice * np.array(q)
        p = np.exp(z - z.max()); p /= p.sum()
        self.assertTrue(np.allclose(r["p_choose"], p, atol=1e-14))
        self.assertAlmostEqual(r["dQ"], q[0] - q[1], places=14)
        self.assertAlmostEqual(r["conflict"], float(np.exp(-((q[0] - q[1]) / n.cfg.sigma_conflict) ** 2)), places=14)

    def test_one_random_draw_per_choice_and_the_chosen_side_is_what_is_learned(self):
        n = Y.warm_persona(0)
        ref = np.random.default_rng(); ref.bit_generator.state = n.rng.bit_generator.state
        u = ref.random()
        ch = Y.easy_choice(0)
        p1 = n.appraise(ch)["p_choose"][1]
        r = n.experience(ch)
        self.assertEqual(r["choice"]["chosen"], int(u < p1))
        self.assertEqual(n.rng.bit_generator.state, ref.bit_generator.state)         # 1 回だけ引いた
        self.assertEqual(r["kind"], "choice")
        # 選んだ側の結果だけを観測する
        won = r["choice"]["chosen"]
        types = {o["type"] for o in r["contributions"]}
        self.assertEqual(types, {"connection"} if won == 0 else {"harm"})

    def test_unchosen_side_leaves_no_trace_in_belief_value_or_conditioning(self):
        n = make(0)
        ch = {"id": "u", "kind": "choice",
              "options": [{"context": "studio", "concepts": ["only_a"], "participants": [{"id": "pa", "class": "friend"}]},
                          {"context": "workshop", "concepts": ["only_b"], "participants": [{"id": "pb", "class": "rival"}]}],
              "outcomes_by_option": {"0": [{"type": "connection", "actor": "pa", "magnitude": 0.5}], "1": [{"type": "harm", "actor": "pb", "magnitude": 0.5}]}}
        r = n.experience(ch)
        lost = "only_b" if r["choice"]["chosen"] == 0 else "only_a"
        lost_p = "pb" if r["choice"]["chosen"] == 0 else "pa"
        self.assertNotIn(f"concept:{lost}", n.value.sit.idx)
        self.assertNotIn(f"ind:{lost_p}", n.value.person.idx)
        self.assertNotIn(lost_p, n.belief.ind_idx)
        self.assertEqual(n.n_experiences, 1)

    def test_conflict_justification_update_bound_and_decay_follow_the_formulas(self):
        cfg = Config()
        n = make(0)
        r = n.experience(Y.hesitant_choice("h", "ja", "jb"))
        c = r["choice"]
        self.assertAlmostEqual(c["conflict"], 1.0, places=14)                       # ΔQ = 0
        won, lost = ("concept:ja", "concept:jb") if c["chosen"] == 0 else ("concept:jb", "concept:ja")
        self.assertAlmostEqual(c["dJ"][won], cfg.eta_j * c["conflict"], places=14)
        self.assertAlmostEqual(c["dJ"][lost], -cfg.eta_j * c["conflict"], places=14)
        self.assertNotIn("ctx:studio", c["dJ"])                                     # 両方に共通の手がかりは相殺
        expect = (1 - cfg.lambda_j) * cfg.j_max * np.tanh(cfg.eta_j / cfg.j_max)
        self.assertAlmostEqual(n.just[won], expect, places=14)
        self.assertAlmostEqual(n.just[lost], -expect, places=14)
        # 次の経験（関係のない）では減衰だけ
        n.experience({"id": "z", "context": "workshop", "concepts": [], "participants": [], "outcomes": []})
        self.assertAlmostEqual(n.just[won], (1 - cfg.lambda_j) * expect, places=14)

    def test_J_enters_Q_additively_in_appraise_and_probe(self):
        n = make(0)
        n.just = {"concept:music": 0.1, "cls:collaborator": -0.04, "ctx:studio": 0.02}
        e = {"id": "q", "context": "studio", "concepts": ["music"], "participants": [{"id": "p", "class": "collaborator"}], "outcomes": []}
        ap = n.appraise(e)
        self.assertAlmostEqual(ap["J_Q"], 0.1 - 0.04 + 0.02, places=14)
        self.assertAlmostEqual(ap["Q"], ap["Q_val"] + ap["C_Q"] + ap["J_Q"], places=14)
        pr = n.probe({"context": "studio", "concepts": ["music"], "participants": [{"id": "p", "class": "collaborator"}]})
        self.assertAlmostEqual(pr["J_Q"], ap["J_Q"], places=14)
        self.assertAlmostEqual(pr["Q"], pr["Q_val"] + pr["C_Q"] + pr["J_Q"], places=14)       # （MB の標本はプローブと経験で別なので、Q どうしは比べない）
        self.assertEqual(n.probe({"concept": "music"})["J"], 0.1)
        self.assertEqual(n.probe({"cue": "cls:collaborator"})["J"], -0.04)
        off = make(0, no_justification=True)
        off.just = dict(n.just)
        self.assertEqual(off.appraise(e)["J_Q"], 0.0)
        self.assertEqual(off.probe({"concept": "music"})["J"], 0.0)

    def test_bounded_decays_while_avoiding_and_never_exceeds_J_max(self):
        n = make(0, record_ledger=False)
        for i in range(300):
            n.experience({"id": f"b{i}", "kind": "choice", "options": [{"context": "studio", "concepts": ["a"]}, {"context": "studio", "concepts": ["b"]}],
                          "outcomes_by_option": {"0": [], "1": []}})
            self.assertLessEqual(max(abs(v) for v in n.just.values()), n.cfg.j_max)
        # 回避した申し出でも時間は経つ（減衰）
        m = make(0)
        m.just = {"concept:music": 0.1}
        m.cond.set_single("type:praise", -3.0)
        e = {"id": "o", "context": "studio", "concepts": ["zzz"], "participants": [], "outcomes": [{"type": "praise", "actor": "world", "magnitude": 0.5}],
             "optional": True, "announced": ["praise"]}
        for i in range(80):
            k = copy.deepcopy(m)
            k.rng = np.random.default_rng(i)
            r = k.experience({**e, "id": f"o{i}"})
            if not r["approached"]:
                self.assertAlmostEqual(k.just["concept:music"], 0.1 * (1 - m.cfg.lambda_j), places=14)
                return
        self.fail("回避する場合が見つからなかった")

    def test_no_justification_ablation_never_builds_J_and_matches_Q_without_it(self):
        n = make(0, no_justification=True)
        for i in range(10):
            n.experience(Y.hesitant_choice(f"nj{i}", "a", "b"))
        self.assertEqual(n.just, {})
        ap = n.appraise(Y.hesitant_choice("p", "a", "b"))
        self.assertEqual([o["J_Q"] for o in ap["options"]], [0.0, 0.0])

    def test_J_changes_nothing_that_is_learned_a_forced_sequence_is_bit_identical(self):
        r = Y.z7_worker(0)
        self.assertTrue(all(r["identical"].values()))

    def test_ledger_has_a_choice_entry_before_the_experience_entry(self):
        n = make(0)
        n.experience(Y.hesitant_choice("l", "a", "b"))
        kinds = [e["kind"] for e in n.ledger.entries]
        self.assertEqual(kinds[-2:], ["choice", "experience"])
        ch = n.ledger.entries[-2]
        for key in ("chosen", "Q", "p_choose", "dQ", "conflict", "dJ", "J_before", "J_after"):
            self.assertIn(key, ch)
        self.assertEqual(n.ledger.entries[-1]["id"], "l")

    def test_choice_runs_through_the_same_learning_as_an_experience(self):
        """二択で選んだ側は、同じ内容の通常の経験と、学習が同じ（a の更新の速さだけ、自分で選んだ分が違う）。"""
        n = Y.warm_persona(0)
        m = copy.deepcopy(n)
        ch = Y.easy_choice(0)
        r = n.experience(ch)
        w = r["choice"]["chosen"]
        o = ch["options"][w]
        m.experience({"id": ch["id"], "context": o["context"], "concepts": o["concepts"], "participants": o["participants"],
                      "outcomes": ch["outcomes_by_option"][str(w)], "optional": False})
        self.assertTrue(np.array_equal(n.value.person.W, m.value.person.W) and np.array_equal(n.value.sit.W, m.value.sit.W))
        self.assertTrue(np.array_equal(n.cond.CK, m.cond.CK))
        self.assertFalse(np.array_equal(n.concern.a, m.concern.a))                  # 主体性の重み

    def test_choice_state_roundtrip_with_J(self):
        n = make(0)
        for i in range(12):
            n.experience(Y.hesitant_choice(f"r{i}", "a", "b"))
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        self.assertEqual(n.just, m.just)
        self.assertTrue(states_equal(n, m))
        for i in range(12, 20):
            self.assertTrue(deep_equal(n.experience(Y.hesitant_choice(f"r{i}", "a", "b")), m.experience(Y.hesitant_choice(f"r{i}", "a", "b"))))
        self.assertTrue(states_equal(n, m))


class AgencyWeight(unittest.TestCase):
    def test_update_speed_is_eta_a_times_one_plus_kappa_chosen(self):
        du = np.array([0.3, -0.1, 0.05, 0.0, -0.2])
        cfg0, cfg1 = Config(kappa_ag=0.0), Config(kappa_ag=1.0)
        a0, a1, a2 = Concern(cfg0), Concern(cfg1), Concern(cfg1)
        a0.update(du, True); a1.update(du, True); a2.update(du, False)
        d0, d1, d2 = a0.a - 1.0, a1.a - 1.0, a2.a - 1.0
        # 正規化の前の更新は η(target − a) の (1 + κ chosen) 倍: 同じ向きで、chosen は forced の約 2 倍
        self.assertTrue(np.array_equal(a0.a, a2.a))                                # κ = 0 なら chosen でも forced と同じ
        self.assertGreater(np.linalg.norm(d1) / np.linalg.norm(d2), 1.9)
        self.assertLess(np.linalg.norm(d1) / np.linalg.norm(d2), 2.1)
        self.assertAlmostEqual(float(a1.a.sum()), 5.0, places=12)

    def test_target_is_still_the_absolute_utility_change_so_no_change_means_no_update(self):
        c = Concern(Config())
        c.update(np.array([0.3, 0.0, 0.0, 0.0, 0.0]), False)
        a = c.a.copy()
        for chosen in (False, True):
            c.update(np.zeros(5), chosen)
            self.assertTrue(np.array_equal(c.a, a))

    def test_optional_offer_accepted_counts_as_chosen_and_a_forced_one_does_not(self):
        n = Y.warm_persona(0)
        for attempt in range(50):
            m = copy.deepcopy(n)
            m.rng = np.random.default_rng([0, attempt])
            r = m.experience({**Y.Z5_EXP, "optional": True})
            if r["approached"]:
                break
        ref = Concern(n.cfg); ref.a, ref.run, ref.n = n.concern.a.copy(), n.concern.run, n.concern.n
        du = np.array([r["du"][k] for k in ("R", "C", "A", "S", "Sa")])
        ref.update(du, True)
        self.assertTrue(np.array_equal(m.concern.a, ref.a))
        f = copy.deepcopy(n)
        f.experience({**Y.Z5_EXP, "optional": False})
        ref2 = Concern(n.cfg); ref2.a, ref2.run, ref2.n = n.concern.a.copy(), n.concern.run, n.concern.n
        ref2.update(du, False)
        self.assertTrue(np.array_equal(f.concern.a, ref2.a))


class PersistenceAndCompatibility(unittest.TestCase):
    def test_schema_version_is_7_and_older_states_are_refused(self):
        from npnv3.npn import SCHEMA_VERSION, StateVersionError
        self.assertEqual(SCHEMA_VERSION, 7)
        n = make(0)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            import json, os
            p = os.path.join(d, "state.json")
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            doc["schema_version"] = 6
            with open(p, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            with self.assertRaises(StateVersionError):
                NPN.load(d)

    def test_ordinary_histories_are_unchanged_when_kappa_ag_is_zero_and_nothing_new_is_used(self):
        """第 2 部の追加は、使わなければ（κ_ag = 0、訂正も二択もなし）v3.3 第 1 部と同じ数値。X（optional なし）は κ_ag に依らない。"""
        a, b = persona("X", 1), persona("X", 1, kappa_ag=0.0)
        self.assertTrue(np.array_equal(a.value.person.W, b.value.person.W) and np.array_equal(a.concern.a, b.concern.a))
        self.assertEqual(a.just, {})
        self.assertEqual(a.n_corrections, 0)

    def test_experience_log_has_one_entry_per_learned_experience(self):
        n = persona("K", 0)
        self.assertEqual(len(n.exp_log), n.n_experiences)
        self.assertEqual([r["serial"] for r in n.exp_log], list(range(n.n_experiences)))
        self.assertEqual(n._exp_pos[n.exp_log[3]["id"]], 3)


if __name__ == "__main__":
    unittest.main()
