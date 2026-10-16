"""潜在原因による条件づけ価値（SPEC-v3.3 4.2）の機構の単体検査。判定基準 E1〜E9 は test_e_extinction.py。ここは「式どおりに動くか」。

  事前    p(k) ∝ N_k · (n_k(c) + β)/(N_k + β S) · exp(−(t − t_k)/τ)，新しい原因は ∝ α
  応答    Σ_k p(k) C_k
  事後    事前 × N(y_x; C_k, σ_r²)，新しい原因が最大なら作る（満員なら N_k exp(−(t−t_k)/τ) が最小の原因を再利用）
  学習    ΔC_k = (α_C / 手がかりの数) 事後_k (y_x − C_k)，文脈の計数 += 事後，t_k は事後が最大の原因だけ現在に
  復位    手がかりなしの嫌悪の結果で、負の価値を持つ原因の t_k を現在に
  single_cause（K = 1）は v3.2 の算術と完全に同じ
"""
import copy
import tempfile
import unittest

import numpy as np

from helpers import deep_equal, feed, load_jsonl, make, mh, persona
from npnv3 import NPN, Config
from npnv3.value import Conditioning


def cond(**kw):
    return Conditioning(Config(**kw))


def put_causes(cd, cue, values, counts, times, ctxs=("A", "B")):
    """手がかり cue に、原因を手で置く（white-box）。counts[k][j] = 原因 k の文脈 ctxs[j] の計数。"""
    i = cd._register(cue)
    for c in ctxs:
        cd._ctx_col(c, True)
    for k, (v, cnt, t) in enumerate(zip(values, counts, times)):
        cd.CK[i, k] = v
        cd.AL[i, k] = 1.0
        cd.TK[i, k] = t
        for j, c in enumerate(ctxs):
            cd.CNT[i, k, cd.ctx_idx[c]] = cnt[j]
    return i


class PriorAndResponse(unittest.TestCase):
    def test_prior_weight_formula_and_response(self):
        cd = cond()
        put_causes(cd, "x", [-0.2, 0.0], [[10.0, 0.0], [0.0, 6.0]], [10.0, 40.0])
        cfg = cd.cfg
        for ctx, t in (("A", 50.0), ("B", 50.0), ("A", 100.0)):
            S = 2.0                                                    # 見た文脈は A, B の 2 つ（ctx は既知）
            w = []
            for N, nc, tk in ((10.0, 10.0 if ctx == "A" else 0.0, 10.0), (6.0, 0.0 if ctx == "A" else 6.0, 40.0)):
                w.append(N * (nc + cfg.lc_beta) / (N + cfg.lc_beta * S) * np.exp(-(t - tk) / cfg.lc_tau))
            p = np.array(w) / (sum(w) + cfg.lc_alpha)
            self.assertAlmostEqual(cd.value_of("x", ctx, t), float(p[0] * -0.2 + p[1] * 0.0), places=14)

    def test_unseen_context_counts_as_one_more_category(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[9.0, 1.0]], [10.0])         # A と B を見た（計数 9 と 1）
        cfg = cd.cfg
        t = 12.0
        N = 10.0
        w_seen = N * (1.0 + cfg.lc_beta) / (N + cfg.lc_beta * 2.0) * np.exp(-(t - 10.0) / cfg.lc_tau)      # B は見た文脈
        w_new = N * (0.0 + cfg.lc_beta) / (N + cfg.lc_beta * 3.0) * np.exp(-(t - 10.0) / cfg.lc_tau)       # C は未見: 範疇が 1 つ増える
        self.assertAlmostEqual(cd.value_of("x", "B", t), -0.2 * w_seen / (w_seen + cfg.lc_alpha), places=14)
        self.assertAlmostEqual(cd.value_of("x", "C", t), -0.2 * w_new / (w_new + cfg.lc_alpha), places=14)

    def test_context_free_read_ignores_the_context_term(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[10.0, 0.0]], [10.0])
        t = 12.0
        w = 10.0 * np.exp(-(t - 10.0) / cd.cfg.lc_tau)
        self.assertAlmostEqual(cd.value_of("x", None, t), -0.2 * w / (w + cd.cfg.lc_alpha), places=14)

    def test_time_decay_shrinks_the_response_toward_zero_exponentially_in_the_weight(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[10.0, 0.0]], [0.0])
        v = [cd.value_of("x", "A", t) for t in (0.0, 30.0, 60.0, 120.0)]
        self.assertTrue(all(a < 0.0 for a in v))
        self.assertTrue(abs(v[0]) > abs(v[1]) > abs(v[2]) > abs(v[3]))

    def test_unknown_cue_and_no_cause_give_zero(self):
        cd = cond()
        self.assertEqual(cd.value_of("nothing", "A", 5.0), 0.0)
        self.assertEqual(cd.value(["a", "b"], "A", 5.0), 0.0)

    def test_sum_over_cues(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[10.0, 0.0]], [10.0])
        put_causes(cd, "y", [0.1], [[3.0, 5.0]], [11.0])
        self.assertAlmostEqual(cd.value(["x", "y", "z"], "A", 12.0), cd.value_of("x", "A", 12.0) + cd.value_of("y", "A", 12.0), places=14)


class StructuralLimits(unittest.TestCase):
    """SPEC-v3.3 4.2 の式そのものの性質（判定ではない）。E5・E9 が不合格になる原因の一つを、式の性質として固定する（tests/diagnostics_e_oracle.py）。"""

    def test_without_learning_the_response_magnitude_never_grows_with_time(self):
        """すべての原因の重みは同じ exp(−t/τ) で縮み、新しい原因 α の質量だけが一定なので、学習のない間の |応答| は増えない（価値がすべて ≤ 0 のとき）。
        したがって E5（自発的回復: 消去の後、無関係な経験だけを与えると応答が戻る）は、この式では原理的に起きない。"""
        rng = np.random.default_rng(7)
        for _ in range(300):
            cd = cond()
            K = int(rng.integers(1, 6))
            vals = -rng.uniform(0.0, 0.6, size=K) * (rng.random(K) < 0.8)
            put_causes(cd, "x", list(vals), [[float(rng.uniform(0, 20)), float(rng.uniform(0, 20))] for _ in range(K)],
                       [float(rng.uniform(0, 60)) for _ in range(K)])
            ctx = ["A", "B", "C"][int(rng.integers(3))]
            prev = abs(cd.value_of("x", ctx, 60.0))
            for dt in (1, 5, 20, 50, 200):
                cur = abs(cd.value_of("x", ctx, 60.0 + dt))
                self.assertLessEqual(cur, prev + 1e-15)
                prev = cur

    def test_weight_in_an_unseen_context_saturates_so_it_barely_depends_on_how_counts_are_spread(self):
        """未見の文脈での原因の重み N_k (0 + β)/(N_k + β S) は、N_k が大きいと β に飽和する。消去の計数を 1 つの文脈に集めても 3 つの文脈に散らしても
        （S が 3 → 5）、N_k = 20 で 0.87 → 0.80 の違いしかない。E9（3 つの文脈で消去すると ABC の復元が小さい）がこの式から出にくい理由。"""
        beta = Config().lc_beta
        w1 = 20.0 * beta / (20.0 + beta * 3.0)
        w3 = 20.0 * beta / (20.0 + beta * 5.0)
        self.assertGreater(w3, 0.75)
        self.assertLess(w1 - w3, 0.1)


class UpdateRules(unittest.TestCase):
    def test_first_update_creates_a_cause_and_learns_with_alpha_c(self):
        cd = cond()
        dC, det = cd.update(["x"], -0.3, "A", 5.0)
        i = cd.idx["x"]
        self.assertEqual(int(cd.AL[i].sum()), 1)
        self.assertEqual(det["x"]["created"], 0)
        self.assertAlmostEqual(cd.CK[i, 0], cd.cfg.alpha_c * 1.0 * (-0.3 - 0.0), places=14)    # 事後 1、新しい原因の価値 0 から
        self.assertAlmostEqual(cd.CNT[i, 0, cd.ctx_idx["A"]], 1.0, places=14)                  # 文脈の計数 += 事後
        self.assertEqual(cd.TK[i, 0], 5.0)
        self.assertAlmostEqual(dC["x"], cd.CK[i, 0], places=14)

    def test_posterior_is_prior_times_gaussian_likelihood(self):
        cd = cond()
        put_causes(cd, "x", [-0.2, 0.0], [[10.0, 0.0], [0.0, 6.0]], [10.0, 40.0])
        i = cd.idx["x"]
        cfg = cd.cfg
        t, y = 50.0, 0.0
        w = np.array([10.0 * (0.0 + 1.0) / (10.0 + 2.0) * np.exp(-(t - 10.0) / cfg.lc_tau),
                      6.0 * (6.0 + 1.0) / (6.0 + 2.0) * np.exp(-(t - 40.0) / cfg.lc_tau)])
        lik = np.exp(-0.5 * ((y - np.array([-0.2, 0.0])) / cfg.lc_sigma_r) ** 2)
        new = cfg.lc_alpha * np.exp(-0.5 * (y / cfg.lc_sigma_r) ** 2)
        z = float((w * lik).sum() + new)
        post = w * lik / z
        before = cd.CK[i].copy(); cnt = cd.CNT[i].copy()
        _, det = cd.update(["x"], y, "B", t)
        self.assertTrue(np.allclose(det["x"]["post"][:2], post, atol=1e-14))
        self.assertAlmostEqual(det["x"]["post_new"], new / z, places=14)
        self.assertIsNone(det["x"]["created"])                                                  # 新しい原因は最大でない
        self.assertTrue(np.allclose(cd.CK[i, :2], before[:2] + cfg.alpha_c * post * (y - before[:2]), atol=1e-14))
        self.assertTrue(np.allclose(cd.CNT[i, :2, cd.ctx_idx["B"]], cnt[:2, cd.ctx_idx["B"]] + post, atol=1e-14))
        self.assertEqual(cd.TK[i, int(np.argmax(post))], t)                                      # 事後が最大の原因だけ時刻を更新
        self.assertEqual(cd.TK[i, 1 - int(np.argmax(post))], [10.0, 40.0][1 - int(np.argmax(post))])

    def test_new_cause_is_created_when_old_causes_cannot_explain_the_target(self):
        cd = cond()
        put_causes(cd, "x", [-0.5], [[3.0, 0.0]], [10.0])         # 強く条件づけられた原因（価値 −0.5）。N が小さい
        _, det = cd.update(["x"], 0.0, "B", 11.0)                  # 何も起きない: z = 3.3 → 尤度が小さい
        self.assertEqual(det["x"]["created"], 1)
        self.assertEqual(int(cd.AL[cd.idx["x"]].sum()), 2)
        self.assertAlmostEqual(cd.CK[cd.idx["x"], 0], -0.5, delta=0.01)    # 古い原因はほとんど動かない（事後が小さい）

    def test_full_set_reuses_the_cause_with_smallest_decayed_mass(self):
        cd = cond(lc_max_causes=2)
        put_causes(cd, "x", [-0.5, 0.3], [[20.0, 0.0], [2.0, 0.0]], [10.0, 10.0])
        i = cd.idx["x"]
        _, det = cd.update(["x"], 0.0, "B", 11.0)                  # どちらにも説明できない目標 → 新しい原因が最大 → 満員なので N が最小の原因（添字 1）を再利用
        self.assertTrue(det["x"]["reused"])
        self.assertEqual(det["x"]["created"], 1)
        self.assertEqual(int(cd.AL[i].sum()), 2)
        self.assertAlmostEqual(cd.CK[i, 0], -0.5, delta=0.01)
        self.assertLess(abs(cd.CK[i, 1]), 0.05)                    # 価値 0 から始まった新しい原因

    def test_number_of_causes_never_exceeds_K(self):
        rng = np.random.default_rng(0)
        for K in (2, 3, 5):
            cd = cond(lc_max_causes=K)
            for t in range(200):
                cd.update(["x"], float(rng.choice([-0.6, -0.3, 0.0, 0.3, 0.6])), ["A", "B", "C"][t % 3], float(t))
                self.assertLessEqual(int(cd.AL[cd.idx["x"]].sum()), K)

    def test_multiple_cues_share_one_target_by_explaining_away(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[10.0, 0.0]], [10.0])
        put_causes(cd, "z", [0.0], [[1.0, 0.0]], [10.0])
        rx, rz = cd.value_of("x", "A", 12.0), cd.value_of("z", "A", 12.0)
        _, det = cd.update(["x", "z"], -0.3, "A", 12.0)
        self.assertAlmostEqual(det["x"]["y_x"], -0.3 - rz, places=14)       # 他の手がかりの応答を引いた目標
        self.assertAlmostEqual(det["z"]["y_x"], -0.3 - rx, places=14)
        self.assertAlmostEqual(det["x"]["resp_before"], rx, places=14)

    def test_dC_is_the_sum_of_cause_changes(self):
        cd = cond()
        rng = np.random.default_rng(1)
        for t in range(40):
            before = cd.CK.copy()
            dC, det = cd.update(["x", "y"], float(rng.normal(0, 0.3)), ["A", "B"][t % 2], float(t))
            for c in ("x", "y"):
                i = cd.idx[c]
                b = before[i] if i < before.shape[0] else np.zeros(cd.K)
                self.assertAlmostEqual(dC[c], float((cd.CK[i] - b).sum()), places=13)
                self.assertTrue(np.allclose(det[c]["dCk"], cd.CK[i] - b, atol=1e-13))


class SingleCauseIsV32(unittest.TestCase):
    def test_single_cause_update_is_the_v32_nlms_rule(self):
        cd = cond(single_cause=True)
        rng = np.random.default_rng(2)
        ref = {}                                                  # v3.2 の更新の参照実装
        names = ["a", "b", "c", "d"]
        for t in range(60):
            cues = list(rng.choice(names, size=int(rng.integers(1, 4)), replace=False))
            y = float(rng.normal(0, 0.4))
            err = y - sum(ref.get(c, 0.0) for c in cues)
            d = cd.cfg.alpha_c * err / len(cues)
            for c in cues:
                ref[c] = ref.get(c, 0.0) + d
            cd.update(cues, y, ["A", "B"][t % 2], float(t))
        for c, v in ref.items():
            self.assertEqual(cd.value_of(c, "A", 99.0), v)       # 完全に同じ（== で）
            self.assertEqual(cd.value_of(c, "Z", 0.0), v)         # 文脈・時刻に依存しない
        self.assertEqual(cd.CNT.shape[2], 0)                       # 文脈の計数は使わない

    def test_single_cause_has_exactly_one_alive_cause_per_cue(self):
        n = persona("K", 0, single_cause=True)
        self.assertEqual(n.cond.K, 1)
        self.assertTrue(np.all(n.cond.AL == 1.0))


class Reinstatement(unittest.TestCase):
    def test_only_negative_causes_of_other_cues_are_refreshed(self):
        cd = cond()
        put_causes(cd, "neg", [-0.2, 0.1], [[5.0, 0.0], [5.0, 0.0]], [3.0, 3.0])
        put_causes(cd, "pos", [0.2], [[5.0, 0.0]], [3.0])
        put_causes(cd, "tiny", [-0.01], [[5.0, 0.0]], [3.0])
        put_causes(cd, "present", [-0.3], [[5.0, 0.0]], [3.0])
        refreshed = cd.reinstate(["present"], 40.0)
        self.assertEqual(refreshed, [("neg", 0)])
        self.assertEqual(cd.TK[cd.idx["neg"], 0], 40.0)
        self.assertEqual(cd.TK[cd.idx["neg"], 1], 3.0)
        self.assertEqual(cd.TK[cd.idx["pos"], 0], 3.0)
        self.assertEqual(cd.TK[cd.idx["tiny"], 0], 3.0)             # 閾値（−0.02）より浅い
        self.assertEqual(cd.TK[cd.idx["present"], 0], 3.0)          # 提示された手がかりは対象外

    def test_reinstatement_raises_the_response_through_recency(self):
        cd = cond()
        put_causes(cd, "x", [-0.2, 0.0], [[10.0, 0.0], [0.0, 20.0]], [10.0, 80.0])
        before = cd.value_of("x", "B", 90.0)
        cd.reinstate([], 90.0)
        after = cd.value_of("x", "B", 90.0)
        self.assertLess(after, before)                              # 負の価値の原因の重みが増えたので、応答はより負になる
        self.assertLess(abs(before), 0.2)

    def test_single_cause_never_reinstates(self):
        cd = cond(single_cause=True)
        cd.update(["x"], -0.5, "A", 1.0)
        self.assertEqual(cd.reinstate([], 10.0), [])

    def test_aversive_event_without_the_cue_triggers_it_in_the_npn(self):
        n = make(0)
        # x を条件づける
        for k in range(6):
            s = f"s{k}"
            n.experience({"id": f"c{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": s})
            n.experience({"id": f"u{k}", "context": "studio", "concepts": [], "participants": [],
                          "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": s})
        i = n.cond.idx["concept:praise_from_X"]
        t0 = float(n.cond.TK[i].max())
        for k in range(8):                                           # 時間を進める
            n.experience({"id": f"f{k}", "context": "solo_practice", "concepts": ["music"], "participants": [],
                          "outcomes": [], "session": f"f{k}"})
        self.assertEqual(float(n.cond.TK[i].max()), t0)
        n.experience({"id": "h1", "context": "cafe_chat", "concepts": [], "participants": [],
                      "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": "h"})
        self.assertGreater(float(n.cond.TK[i].max()), t0)
        self.assertEqual(float(n.cond.TK[i].max()), float(n.n_experiences + n.n_avoided - 1))
        entry = n.ledger.entries[-1]
        self.assertIn(["concept:praise_from_X", 0], entry["td"]["reinstated"])


class IntegrationInNPN(unittest.TestCase):
    def test_ledger_cause_totals_equal_cause_values(self):
        n = make(0)
        for e in mh.generate_praise_history("H1", 3, 12):
            n.experience(e)
        tot = {}
        for en in n.ledger.entries:
            if en["kind"] != "experience":
                continue
            for cue, d in en["td"]["lc"].items():
                v = np.array(d["dCk"])
                tot[cue] = tot[cue] + v if cue in tot else v
        self.assertTrue(len(tot) > 3)
        for cue, v in tot.items():
            self.assertTrue(np.allclose(n.cond.CK[n.cond.idx[cue]], v, atol=1e-12), cue)

    def test_appraise_and_probe_do_not_change_conditioning_state(self):
        n = persona("K", 1)
        before = copy.deepcopy(n)
        n.appraise(load_jsonl("R.jsonl")[3])
        n.probe({"cue": "type:harm", "context": "music_session"})
        n.probe({"context": "music_session", "concepts": ["collaboration"], "participants": [{"id": "p", "class": "collaborator"}]})
        for key in ("CK", "TK", "AL", "CNT"):
            self.assertTrue(np.array_equal(getattr(n.cond, key), getattr(before.cond, key)), key)
        self.assertEqual(n.cond.ctx_names, before.cond.ctx_names)

    def test_clock_counts_learned_and_avoided_experiences(self):
        n = make(0)
        self.assertEqual(n._clock(), 0.0)
        e = {"id": "o1", "context": "studio", "concepts": ["music"], "participants": [],
             "outcomes": [{"type": "praise", "actor": "world", "magnitude": 0.5}], "optional": True, "announced": ["praise"]}
        n.cond.set_single("type:praise", -2.0)
        learned = avoided = 0
        for i in range(30):
            ee = copy.deepcopy(e); ee["id"] = f"o{i}"
            r = n.experience(ee)
            learned += int(r["approached"]); avoided += int(not r["approached"])
        self.assertGreater(avoided, 0)
        self.assertEqual(n.n_experiences, learned)
        self.assertEqual(n.n_avoided, avoided)
        self.assertEqual(n._clock(), float(learned + avoided))

    def test_pending_carries_context_and_time(self):
        n = make(0)
        n.experience({"id": "a", "context": "cafe_chat", "concepts": ["conversation"], "participants": [], "outcomes": [], "session": "s"})
        self.assertEqual(n.pending["context"], "cafe_chat")
        self.assertEqual(n.pending["t"], 0.0)
        n.experience({"id": "b", "context": "studio", "concepts": [], "participants": [], "outcomes": [], "session": "s"})
        self.assertEqual(n.pending["t"], 1.0)

    def test_save_restore_keeps_causes_counts_times_and_continues_identically(self):
        n = persona("K", 2)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        for key in ("CK", "TK", "AL", "CNT"):
            self.assertTrue(np.array_equal(getattr(n.cond, key), getattr(m.cond, key)), key)
        self.assertEqual(n.cond.ctx_names, m.cond.ctx_names)
        rest = load_jsonl("R.jsonl")
        self.assertTrue(deep_equal(feed(n, copy.deepcopy(rest)), feed(m, copy.deepcopy(rest))))
        self.assertEqual(n.cond.value_of("type:harm", "music_session"), m.cond.value_of("type:harm", "music_session"))

    def test_loading_with_a_different_K_is_refused(self):
        n = persona("K", 0)
        meta, arrays = n.get_state()
        m = make(0, single_cause=True)
        with self.assertRaises(ValueError):
            m.cond.set_state(meta["conditioning"], arrays)

    def test_no_conditioning_leaves_no_state(self):
        n = make(0, no_conditioning=True)
        feed(n, load_jsonl("K.jsonl"))
        self.assertEqual(len(n.cond.names), 0)
        self.assertEqual(n.cond.CNT.shape[0], 0)

    def test_cue_probe_reports_causes_and_takes_a_context(self):
        n = persona("K", 0)
        r = n.probe({"cue": "concept:collaboration", "context": "music_session"})
        self.assertEqual(r["context"], "music_session")
        self.assertGreaterEqual(r["n_causes"], 1)
        self.assertTrue(r["known"])
        self.assertNotEqual(n.probe({"cue": "concept:collaboration"})["C"], 0.0)
        self.assertFalse(n.probe({"cue": "concept:never_seen"})["known"])


if __name__ == "__main__":
    unittest.main()
