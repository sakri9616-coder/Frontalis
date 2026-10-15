"""v3.1 の各機構の単体検査（SPEC-v3.1 第 1 節）: 関心の重み a、条件づけ C、参照点、解釈、埋め込みの所有、κ_e の推定、生成器。"""
import copy
import math
import tempfile
import unittest

import numpy as np

from helpers import deep_equal, feed, load_json, load_jsonl, make, mh, persona
from npnv3 import NPN, innate
from npnv3.belief import estimate_kappa, lgamma_np


def exp(eid, outcomes, ctx="cafe_chat", concepts=("conversation",), participants=(), session=None, **kw):
    e = {"id": eid, "context": ctx, "concepts": list(concepts), "participants": list(participants),
         "outcomes": outcomes, "optional": False}
    if session is not None:
        e["session"] = session
    e.update(kw)
    return e


def person(pid, cls="friend"):
    return {"id": pid, "class": cls}


class ConcernTests(unittest.TestCase):
    """1.1 欲求の関心の重み a"""

    def test_starts_at_one_sums_to_five_and_is_bounded(self):
        n = make(0)
        self.assertTrue(np.array_equal(n.a, np.ones(5)))
        for e in mh.generate_history("K", 1, 60):
            n.experience(e)
            self.assertAlmostEqual(float(n.a.sum()), 5.0, places=12)
            self.assertTrue(np.all(n.a > 0) and np.all(n.a < 5))

    def test_strongly_shaken_need_gains_weight_regardless_of_sign(self):
        up, down = make(0), make(0)
        for i in range(30):
            up.experience(exp(f"u{i}", [{"type": "novelty", "actor": "world", "magnitude": 0.5},
                                        {"type": "achievement", "actor": "self", "magnitude": 0.5}]))
            down.experience(exp(f"d{i}", [{"type": "monotony", "actor": "world", "magnitude": 0.5},
                                          {"type": "achievement", "actor": "self", "magnitude": 0.5}]))
        # 刺激が大きく揺さぶられた（正でも負でも）人格は、刺激の重みが 1 を超える
        self.assertGreater(up.a[3], 1.0)
        self.assertGreater(down.a[3], 1.0)

    def test_target_is_exogenous_value_state_does_not_enter(self):
        a, b = persona("K", 0), persona("K", 0)
        rng = np.random.default_rng(1)
        b.value.person.W = b.value.person.W + rng.normal(0, 0.5, b.value.person.W.shape)
        b.cond.C = b.cond.C + 0.3
        e = load_json("new_event.json")
        a.experience(e); b.experience(e)
        self.assertTrue(np.array_equal(a.a, b.a))

    def test_ablation_no_concern(self):
        n = make(0, no_concern=True)
        feed(n, load_jsonl("K.jsonl"))
        self.assertTrue(np.array_equal(n.a, np.ones(5)))

    def test_empty_experience_does_not_move_a(self):
        n = persona("K", 0)
        a0 = n.a.copy()
        n.experience(exp("rest", []))
        self.assertTrue(np.array_equal(n.a, a0))


class ConditioningTests(unittest.TestCase):
    """1.2 条件づけられた価値 C（TD(0)）"""

    def test_cue_before_reward_acquires_value(self):
        n = make(0)
        p = person("p1")
        n.experience(exp("a", [{"type": "praise", "actor": "p1", "magnitude": 0.8}], participants=[p], session="s"))
        self.assertEqual(n.cond.value_of("type:praise"), 0.0)                    # まだ次の事象を見ていない
        r = n.experience(exp("b", [{"type": "harm", "actor": "p1", "magnitude": 0.8}], participants=[p], session="s"))
        # y = r_b + γ C_frozen(z_b)（C はまだ 0）。r_b は害の効用（負）
        y = r["value_change"]["conditioned"]["td_target"]
        self.assertAlmostEqual(y, r["du_weighted"], places=12)
        self.assertLess(y, 0.0)
        self.assertLess(n.cond.value_of("type:praise"), 0.0)
        self.assertEqual(n.cond.value_of("type:harm"), 0.0)                      # b の手がかりは次の事象でしか更新されない

    def test_session_break_gives_zero_target(self):
        n = make(0)
        n.experience(exp("a", [{"type": "praise", "actor": "p1", "magnitude": 0.8}], participants=[person("p1")], session="s1"))
        r = n.experience(exp("b", [{"type": "harm", "actor": "p2", "magnitude": 0.8}], participants=[person("p2")], session="s2"))
        self.assertEqual(r["value_change"]["conditioned"]["td_target"], 0.0)     # 途切れたら次の価値は 0
        self.assertEqual(n.cond.value_of("type:praise"), 0.0)                    # 0 に向けて動くだけ（初期値 0）

    def test_unspecified_sessions_chain(self):
        n = make(0)
        n.experience(exp("a", [{"type": "praise", "actor": "p1", "magnitude": 0.8}], participants=[person("p1")]))
        r = n.experience(exp("b", [{"type": "harm", "actor": "p1", "magnitude": 0.8}], participants=[person("p1")]))
        self.assertIsNotNone(r["value_change"]["conditioned"]["td_target"])
        self.assertNotEqual(r["value_change"]["conditioned"]["td_target"], 0.0)

    def test_bootstrap_is_contraction_and_target_ignores_value_net(self):
        a, b = persona("K", 0), persona("K", 0)
        rng = np.random.default_rng(2)
        b.value.sit.W = b.value.sit.W + rng.normal(0, 0.5, b.value.sit.W.shape)       # W を変えても C の更新は同じ
        new = load_json("new_event.json")
        a.experience(new); b.experience(new)
        self.assertTrue(np.array_equal(a.cond.C, b.cond.C))

    def test_no_conditioning_ablation(self):
        n = make(0, no_conditioning=True)
        feed(n, load_jsonl("K.jsonl"))
        self.assertEqual(len(n.cond.names), 0)
        self.assertEqual(n.appraise(load_json("new_event.json"))["f_parts"]["conditioned"], 0.0)

    def test_ledger_records_C_updates_and_totals_match(self):
        n = make(0)
        for e in mh.generate_praise_history("H1", 3, 6):
            n.experience(e)
        tot = {}
        for en in n.ledger.entries:
            for k, v in en["td"]["dC"].items():
                tot[k] = tot.get(k, 0.0) + v
        for k, v in tot.items():
            self.assertAlmostEqual(n.cond.value_of(k), v, places=12)

    def test_pending_survives_save_and_load(self):
        n = make(0)
        n.experience(exp("a", [{"type": "praise", "actor": "p1", "magnitude": 0.8}], participants=[person("p1")], session="s"))
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        nxt = exp("b", [{"type": "harm", "actor": "p1", "magnitude": 0.8}], participants=[person("p1")], session="s")
        self.assertTrue(deep_equal(n.experience(nxt), m.experience(nxt)))
        self.assertEqual(n.cond.C.tolist(), m.cond.C.tolist())


class ReferencePointTests(unittest.TestCase):
    """1.3 受け止め方 f"""

    def test_parts_sum_into_f_and_reference_depends_on_expectation(self):
        n = persona("K", 0)
        ap = n.appraise(load_json("new_event.json"))
        p = ap["f_parts"]
        self.assertAlmostEqual(p["innate"], float(n.a @ np.array([ap["du"][k] for k in innate.NEEDS])), places=12)
        self.assertAlmostEqual(p["reference"], 0.5 * (p["innate"] - ap["Q"]), places=12)
        f = math.tanh((p["innate"] + p["conditioned"] + p["reference"]) / 0.5 + 0.3 * ap["mood"])
        self.assertAlmostEqual(ap["f"], f, places=12)

    def test_no_reference_ablation(self):
        n = persona("K", 0, no_reference=True)
        self.assertEqual(n.appraise(load_json("new_event.json"))["f_parts"]["reference"], 0.0)

    def test_f_is_never_a_teacher(self):
        """f・C・気分を勝手に変えても、価値ネットの教師・信念の更新は変わらない。"""
        a, b = persona("K", 0), persona("K", 0)
        b.mood = 0.9; b.cond.C = b.cond.C + 0.2
        e = load_json("new_event.json")
        ra, rb = a.experience(e), b.experience(e)
        self.assertNotEqual(ra["f"], rb["f"])
        self.assertTrue(deep_equal(ra["teachers"], rb["teachers"]))
        self.assertTrue(np.array_equal(a.value.person.W, b.value.person.W))
        self.assertTrue(np.array_equal(a.belief.N1, b.belief.N1))


class InterpretationTests(unittest.TestCase):
    """1.4 曖昧な結果の解釈"""

    AMB = {"id": "amb", "context": "project_meeting", "concepts": ["collaboration"], "participants": [person("c1", "collaborator")],
           "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "c1", "magnitude": 0.8}],
           "optional": False}

    def test_weights_are_llm_probability_times_belief(self):
        n = persona("K", 0)
        ap = n.appraise(self.AMB)
        w = ap["interpretations"][0]["weights"]
        pr = n.probe({"entity": "c1", "class": "collaborator", "context": "project_meeting"})["p_person"]
        expect = 0.5 * pr["credit_taken"] / (0.5 * pr["credit_taken"] + 0.5 * pr["credit_shared"])
        self.assertAlmostEqual(w["credit_taken"], expect, places=12)
        self.assertAlmostEqual(w["credit_taken"] + w["credit_shared"], 1.0, places=12)

    def test_effect_is_the_mixture(self):
        n = persona("R", 0)
        ap = n.appraise(self.AMB)
        w = ap["interpretations"][0]["weights"]
        row = 0.8 * (w["credit_taken"] * n.M[innate.TYPE_INDEX["credit_taken"]] + w["credit_shared"] * n.M[innate.TYPE_INDEX["credit_shared"]])
        dh = np.array([ap["dh"][k] for k in innate.NEEDS])
        self.assertTrue(np.allclose(dh, row, atol=1e-12))

    def test_ambiguous_types_are_not_counted_by_default(self):
        n = persona("K", 0)
        N1, N0, CS1 = n.belief.N1.copy(), n.belief.N0.copy(), n.belief.CS1.copy()
        n.experience(self.AMB)
        i = n.belief.ind_idx["c1"]
        cols = [innate.TYPE_INDEX["credit_taken"], innate.TYPE_INDEX["credit_shared"]]
        self.assertEqual(float(n.belief.N1[i, cols].sum() + n.belief.N0[i, cols].sum()), 0.0)   # 観測なし
        other = [k for k in range(innate.N_TYPES) if k not in cols]
        self.assertGreater(float(n.belief.N0[i, other].sum()), 0.0)                              # ほかの型は y = 0 と数える

    def test_interpretation_counts_condition_counts_with_weights(self):
        n = persona("K", 0, interpretation_counts=True)
        ap = n.appraise(self.AMB)
        n.experience(self.AMB)
        i = n.belief.ind_idx["c1"]
        k = innate.TYPE_INDEX["credit_taken"]
        self.assertGreater(float(n.belief.N1[i, k]), 0.0)
        self.assertGreater(float(n.belief.N0[i, k]), 0.0)

    def test_no_interpretation_uses_llm_probabilities_only(self):
        n = persona("K", 0, no_interpretation=True)
        w = n.appraise(self.AMB)["interpretations"][0]["weights"]
        self.assertAlmostEqual(w["credit_taken"], 0.5, places=12)

    def test_misreading_is_kept_in_the_value(self):
        """誤解は誤解のまま価値に残る: K は『取られた』寄りの教師で学ぶ。R は『共有された』寄り。"""
        k, r = persona("K", 0), persona("R", 0)
        tk = k.appraise(self.AMB)["teachers"]
        tr = r.appraise(self.AMB)["teachers"]
        self.assertLess(sum(tk["person"]["c1"].values()) + sum(tk["situation"].values()),
                        sum(tr["person"]["c1"].values()) + sum(tr["situation"].values()))


class EmbeddingOwnershipTests(unittest.TestCase):
    """1.6 埋め込みは NPN が所有する"""

    def test_registered_on_first_learning_not_on_reading(self):
        n = make(0)
        n.appraise(load_json("new_event.json")); n.probe({"concept": "music"})
        self.assertEqual({k: len(v) for k, v in n.emb.own.items()}, {"cls": 0, "ctx": 0, "concept": 0})
        n.experience(load_json("new_event.json"))
        self.assertIn("collaborator", n.emb.own["cls"]); self.assertIn("music_session", n.emb.own["ctx"])
        self.assertEqual(set(n.emb.own["concept"]), {"music", "collaboration"})

    def test_owned_values_do_not_follow_the_source(self):
        n = persona("K", 0)
        before = n.emb.cls("collaborator").copy()
        n.emb.source = n.emb.source.rotated(np.eye(8)[::-1])                  # 供給源を別の座標に替える
        self.assertTrue(np.array_equal(n.emb.cls("collaborator"), before))

    def test_unknown_word_is_registered_as_zero_with_warning(self):
        n = make(0)
        e = exp("w", [{"type": "connection", "actor": "p", "magnitude": 0.5}], participants=[person("p", "alien")], ctx="moon")
        ap = n.experience(e)
        self.assertEqual(len(ap["warnings"]), 2)
        self.assertEqual(len(n.warnings), 2)
        self.assertTrue(np.all(n.emb.own["cls"]["alien"] == 0.0))


class KappaTests(unittest.TestCase):
    """1.7 類ごとの κ_e の経験ベイズ推定"""

    def test_lgamma_matches_math(self):
        x = np.concatenate([np.linspace(0.001, 10, 400), np.linspace(10, 500, 200)])
        err = np.max(np.abs(lgamma_np(x) - np.array([math.lgamma(v) for v in x])))
        self.assertLess(err, 1e-8)

    def test_heterogeneous_class_has_small_kappa_homogeneous_has_large(self):
        grid = np.exp(np.linspace(np.log(0.3), np.log(50), 12))
        n = 10
        het1 = np.array([[1.0] * 3 if i % 2 == 0 else [0.0] * 3 for i in range(n)]) * 6    # 個体ごとに 0% か 100%
        het0 = 6 - het1
        hom1 = np.full((n, 3), 3.0); hom0 = np.full((n, 3), 3.0)                           # 全員 50%
        kh = estimate_kappa(het1, het0, 1.0, 4.0, grid)
        ko = estimate_kappa(hom1, hom0, 1.0, 4.0, grid)
        self.assertLess(kh, 2.0)
        self.assertGreater(ko, 5.0)
        self.assertGreater(ko, 3 * kh)

    def test_fewer_than_two_individuals_keeps_the_prior(self):
        n = make(0)
        n.experience(exp("a", [{"type": "connection", "actor": "p", "magnitude": 0.5}], participants=[person("p", "friend")]))
        self.assertEqual(n.belief.kappa_of("friend"), 2.0)
        n.experience(exp("b", [{"type": "connection", "actor": "q", "magnitude": 0.5}], participants=[person("q", "friend")]))
        self.assertNotEqual(n.belief.kappa_of("friend"), 2.0)

    def test_fixed_kappa_ablation(self):
        n = make(0, estimate_kappa=False)
        feed(n, load_jsonl("K.jsonl"))
        self.assertEqual(n.belief.kappa_of("collaborator"), 2.0)


class GeneratorTests(unittest.TestCase):
    """1.8 確率的な経験史の生成"""

    def test_seeds_give_different_histories_and_same_seed_is_deterministic(self):
        a, b, c = mh.generate_history("K", 1), mh.generate_history("K", 2), mh.generate_history("K", 1)
        self.assertEqual(a, c)
        self.assertNotEqual(a, b)

    def test_contains_optional_offers_and_matches_world(self):
        hist = [e for s in range(20) for e in mh.generate_history("K", s, 60)]
        self.assertTrue(any(e["optional"] for e in hist))
        collab = [e for e in hist if e["participants"] and e["participants"][0]["class"] == "collaborator"]
        taken = np.mean([any(o.get("type") == "credit_taken" for o in e["outcomes"]) for e in collab])
        self.assertAlmostEqual(float(taken), 0.75, delta=0.06)               # K の世界の真の確率 0.75

    def test_praise_histories_share_structure(self):
        h1, h2 = mh.generate_praise_history("H1", 4), mh.generate_praise_history("H2", 4)
        self.assertEqual(len(h1), len(h2))
        strip = lambda es: [(e["context"], e["session"], [(o["actor"], o["magnitude"]) for o in e["outcomes"]]) for e in es]
        self.assertEqual(strip(h1), strip(h2))
        self.assertEqual({o["type"] for e in h1 for o in e["outcomes"]} - {"achievement"}, {"praise", "harm"})
        self.assertEqual({o["type"] for e in h2 for o in e["outcomes"]} - {"achievement"}, {"praise", "support"})


class HeadKindTests(unittest.TestCase):
    def test_all_head_kinds_run_and_none_ignores_embeddings(self):
        for kind in ("mlp", "linear", "knn1", "none"):
            n = make(0, head_kind=kind)
            feed(n, load_jsonl("T8_train.jsonl")[:40])
            p = n.probe({"entity": "x", "class": "band_mate", "context": "music_session"})["p_person"]["credit_taken"]
            self.assertTrue(0.0 < p < 1.0, kind)

    def test_knn1_uses_nearest_known_class_rate(self):
        n = make(0, head_kind="knn1")
        feed(n, load_jsonl("T8_train.jsonl"))
        bm = n.probe({"entity": "x", "class": "band_mate", "context": "music_session"})["m_cls"]["credit_taken"]
        col = n.probe({"entity": "y", "class": "collaborator", "context": "music_session"})["m_cls"]["credit_taken"]
        self.assertGreater(bm, 0.25)       # 最も近い既知の類は共同作業者（手柄を取る率が高い）
        self.assertGreater(col, 0.25)


if __name__ == "__main__":
    unittest.main()
