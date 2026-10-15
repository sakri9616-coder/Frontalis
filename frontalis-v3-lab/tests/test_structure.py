"""構造の検査: 継承、帰属、事前ヘッド、アブレーションの切り替え。"""
import unittest

import numpy as np

from helpers import feed, load_json, load_jsonl, make, persona


class InheritanceTests(unittest.TestCase):
    def test_new_individual_inherits_class_value(self):
        n = persona("K", 0)
        like = n.probe({"entity": "someone_new", "class": "collaborator"})
        self.assertFalse(like["known"])
        kal = n.value.person
        emb = n.emb.cls("collaborator")
        by_need = (kal.W[kal.idx["people"]] + kal.W[kal.idx["cls:collaborator"]]
                   + sum(kal.W[kal.idx[f"emb:{d}"]] * v for d, v in enumerate(emb) if v != 0))
        self.assertAlmostEqual(like["like"], float(n.a @ by_need), places=12)     # v3.1: 関心の重み a で合計する
        self.assertLess(like["like"], 0.0)               # K は共同作業者に手柄を取られた

    def test_experience_with_new_individual_starts_deviation_at_zero(self):
        n = persona("K", 0)
        n.experience(load_json("new_event.json"))
        self.assertIn("ind:friendA", n.value.person.idx)
        # 新しい個体の偏差は、事前平均 0 から始まり、類へも一般へも「個体の数に比例して」しか動かない
        ledger = n.ledger.entries[-1]["dW"]["person"]
        self.assertGreater(np.abs(ledger["ind:friendA"]).sum(), np.abs(ledger["cls:collaborator"]).sum())
        self.assertGreater(np.abs(ledger["cls:collaborator"]).sum(), np.abs(ledger["people"]).sum())


class ScopeTests(unittest.TestCase):
    def test_responsibilities_sum_to_one_and_follow_belief(self):
        n = persona("K", 0)
        ap = n.appraise(load_json("new_event.json"))
        for p in ap["phi"]:
            self.assertAlmostEqual(p["phi_person"] + p["phi_ctx"], 1.0, places=12)
            self.assertTrue(0.0 < p["phi_person"] < 1.0)

    def test_no_scope_gives_full_responsibility_to_person(self):
        n = make(0, no_scope=True)
        feed(n, load_jsonl("K.jsonl"))
        ap = n.appraise(load_json("new_event.json"))
        self.assertTrue(all(p["phi_person"] == 1.0 for p in ap["phi"]))
        # 状況モデルの教師は、自分・世界が行為者の結果だけ
        self.assertAlmostEqual(ap["teachers"]["situation"]["A"], 0.0)

    def test_teacher_partition_is_exhaustive(self):
        """人の教師 + 状況の教師 = 事象の効用変化 Δu（φ で分けても総和は変わらない）。"""
        n = persona("R", 0)
        ap = n.appraise(load_json("new_event.json"))
        for k in ("R", "C", "A", "S", "Sa"):
            tot = ap["teachers"]["situation"][k] + sum(t[k] for t in ap["teachers"]["person"].values())
            self.assertAlmostEqual(tot, ap["du"][k], places=12)


class HeadTests(unittest.TestCase):
    def test_head_is_trained_on_observations_not_on_posteriors(self):
        """目標は観測そのもの（0/1）。リプレイ緩衝には y だけが入り、事後平均は入らない。"""
        n = make(0)
        feed(n, load_jsonl("T8_train.jsonl")[:40])
        ys = n.belief.rb_y[: n.belief.rb_len]
        self.assertTrue(set(np.unique(ys)) <= {0.0, 1.0})
        self.assertLessEqual(n.belief.rb_len, 500)

    def test_head_mixture_weight_follows_sample_count(self):
        """v3.1（SPEC-v3.1 1.5）: 50 件の解禁は廃止。prior_cls = ρ_h head + (1 − ρ_h) m_gen、ρ_h = n/(n+30)。"""
        n = make(0)
        self.assertEqual(n.belief.rho(), 0.0)
        feed(n, load_jsonl("T8_train.jsonl")[:10])
        self.assertAlmostEqual(n.belief.rho(), 10 / 40)
        self.assertTrue(n.belief.predict("music_session", [{"id": "x", "class": "collaborator"}]).head_used)
        feed(n, load_jsonl("T8_train.jsonl")[10:30])
        self.assertAlmostEqual(n.belief.rho(), 30 / 60)
        self.assertEqual(make(0, no_head=True).belief.rho(), 0.0)

    def test_replay_buffer_is_bounded(self):
        n = make(0)
        feed(n, load_jsonl("T8_train.jsonl"))
        self.assertEqual(n.belief.rb_len, 120)
        self.assertEqual(n.belief.n_obs, 120)


class ArbitrationTests(unittest.TestCase):
    def test_trained_situation_favors_mf_novel_situation_favors_mb(self):
        """docs/04 第 3.3 節: 確信の高い方に重みが寄る。訓練を重ねた状況は MF（習慣）、新しい状況は MB が優勢。"""
        n = persona("K", 0)
        familiar = n.probe({"context": "solo_practice", "concepts": ["music", "solo_work", "authorship"], "participants": []})
        novel = n.probe({"context": "music_session", "concepts": ["music", "collaboration"],
                         "participants": [{"id": "someone_new", "class": "collaborator"}]})
        self.assertGreater(familiar["omega"], 0.5)
        self.assertLess(novel["omega"], 0.5)
        self.assertGreater(familiar["omega"], novel["omega"])


class AblationTests(unittest.TestCase):
    def test_flat_value_uses_only_lowest_layer_features(self):
        n = make(0, flat_value=True)
        feed(n, load_jsonl("K.jsonl"))
        self.assertTrue(all(nm.startswith("ind:") for nm in n.value.person.names))
        self.assertTrue(all(nm.startswith(("ctx:", "concept:")) for nm in n.value.sit.names))
        self.assertTrue(all(abs(n.value.person.S[i, i]) <= 0.25 + 1e-12 for i in range(len(n.value.person.names))))
        like = n.probe({"entity": "never_seen", "class": "collaborator"})
        self.assertEqual(like["like"], 0.0)               # 類の価値を継承しない

    def test_fixed_teacher_ignores_prediction_error(self):
        n = make(0, fixed_teacher=True)
        feed(n, load_jsonl("K.jsonl")[:8])
        S0 = n.value.person.S.copy()
        n.experience(load_json("new_event.json"))
        self.assertTrue(np.array_equal(S0, n.value.person.S[: S0.shape[0], : S0.shape[0]]))
        d = n.ledger.entries[-1]["dW"]["person"]
        self.assertEqual(list(d), ["ind:friendA"])                       # 更新は個体の特徴だけ
        du = n.ledger.entries[-1]["teachers"]["person"]["friendA"]
        self.assertTrue(np.allclose(d["ind:friendA"], 0.1 * np.array(du)))

    def test_no_head_has_no_network_and_uses_gen_prior(self):
        n = make(0, no_head=True)
        feed(n, load_jsonl("T8_train.jsonl"))
        self.assertIsNone(n.belief.head)
        p = n.probe({"entity": "x", "class": "band_mate", "context": "music_session"})
        self.assertFalse(n.belief.predict("music_session", [{"id": "x", "class": "band_mate"}]).head_used)
        self.assertEqual(n.belief.cfg.head_kind, "none")
        # 未見の類の個体の事前 = m_gen（類のデータがないので）
        m_gen = n.belief.summary()["m_gen"]["credit_taken"]
        self.assertAlmostEqual(p["m_cls"]["credit_taken"], m_gen, places=12)


if __name__ == "__main__":
    unittest.main()
