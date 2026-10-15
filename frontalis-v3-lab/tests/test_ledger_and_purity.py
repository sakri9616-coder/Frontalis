"""台帳の不変量、appraise/probe の純粋性、optional の回避、価値の教師が価値ネット自身の出力に依存しないこと。"""
import copy
import unittest

import numpy as np

from helpers import deep_equal, feed, load_json, load_jsonl, make, persona


def state_of(n):
    m, a = n.get_state()
    return m, a


def same_state(x, y):
    mx, ax = state_of(x)
    my, ay = state_of(y)
    return deep_equal(mx, my) and all(np.array_equal(ax[k], ay[k]) for k in ax)


class LedgerTests(unittest.TestCase):
    def test_W_equals_sum_of_ledger_updates(self):
        for name in ("R", "K", "X"):
            n = persona(name, 0)
            for model, kal in (("person", n.value.person), ("situation", n.value.sit)):
                tot = n.ledger.totals(model)
                for nm, row in tot.items():
                    self.assertTrue(np.allclose(kal.W[kal.idx[nm]], row, atol=1e-12), f"{name} {model} {nm}")
                # 台帳にない特徴の W は 0
                for nm, w in zip(kal.names, kal.W):
                    if nm not in tot:
                        self.assertTrue(np.all(w == 0.0))

    def test_ledger_records_every_experience(self):
        n = persona("K", 0)
        self.assertEqual(len(n.ledger), 20)
        e = n.ledger.entries[0]
        for key in ("id", "Q", "f", "delta", "du", "phi", "dW", "teachers"):
            self.assertIn(key, e)
        self.assertTrue(len(n.ledger.history_of("person", "ind:k_partner01")) >= 1)


class PurityTests(unittest.TestCase):
    def test_appraise_and_probe_do_not_change_state(self):
        n = persona("R", 2)
        before = copy.deepcopy(n)
        n.appraise(load_json("new_event.json"))
        n.probe({"entity": "friendA", "class": "collaborator", "context": "music_session"})
        n.probe({"concept": "music"})
        self.assertTrue(same_state(n, before))

    def test_appraise_matches_experience_expectation(self):
        n = persona("R", 0)
        new = load_json("new_event.json")
        a = n.appraise(new)
        e = n.experience(new)
        for k in ("Q", "V_MB", "V_MF", "omega", "f", "delta", "du", "contributions", "phi", "teachers", "emotions"):
            self.assertTrue(deep_equal(a[k], e[k]), k)

    def test_avoided_optional_experience_is_recorded_but_not_learned(self):
        # 強い負の予期を作り、optional な経験を何度か与えて、回避が起きる種を使う
        n = persona("K", 0)
        e = copy.deepcopy(load_json("new_event.json")); e["optional"] = True
        found = False
        for i in range(60):
            m = copy.deepcopy(n)
            ee = copy.deepcopy(e); ee["id"] = f"opt-{i}"
            m.rng = np.random.default_rng(i)
            ap = m.experience(ee)
            if ap["approached"] is False:
                found = True
                base = copy.deepcopy(n)
                base.rng = np.random.default_rng(i)
                base.value = copy.deepcopy(m.value); base.belief = copy.deepcopy(m.belief)
                self.assertTrue(np.array_equal(m.h, n.h) and m.mood == n.mood)
                self.assertEqual(m.ledger.avoided(), [f"opt-{i}"])
                self.assertEqual(len(m.ledger), len(n.ledger) + 1)
                self.assertTrue(np.array_equal(m.value.person.W, n.value.person.W))
                self.assertTrue(np.array_equal(m.belief.N1, n.belief.N1))
                self.assertIsNone(ap["f"])
                break
        self.assertTrue(found, "回避する場合が見つからなかった")

    def test_choice_uses_state_rng_only(self):
        a, b = persona("K", 5), persona("K", 5)
        e = copy.deepcopy(load_json("new_event.json")); e["optional"] = True
        self.assertEqual(a.experience(e)["approached"], b.experience(e)["approached"])


class TeacherIndependenceTests(unittest.TestCase):
    """docs/04 補題1: 価値ネットの更新目標に、価値ネット自身の出力を入れない。
    価値の重み W と共分散 Σ を勝手に変えても、教師（Δu の帰属）・信念・欲求の状態の更新は変わらない。"""

    def test_teachers_belief_and_h_do_not_depend_on_value_state(self):
        a = persona("K", 0)
        b = copy.deepcopy(a)
        rng = np.random.default_rng(123)
        b.value.person.W = b.value.person.W + rng.normal(0, 0.5, b.value.person.W.shape)
        b.value.sit.W = b.value.sit.W + rng.normal(0, 0.5, b.value.sit.W.shape)
        new = load_json("new_event.json")
        ra, rb = a.experience(new), b.experience(new)
        self.assertTrue(deep_equal(ra["teachers"], rb["teachers"]))
        self.assertTrue(deep_equal(ra["du"], rb["du"]))
        self.assertTrue(deep_equal(ra["phi"], rb["phi"]))
        self.assertNotEqual(ra["Q"], rb["Q"])                       # 評価（読み出し）は変わる
        self.assertTrue(np.array_equal(a.h, b.h))
        ma, aa = a.belief.get_state(); mb, ab = b.belief.get_state()
        self.assertTrue(deep_equal(ma, mb) and all(np.array_equal(aa[k], ab[k]) for k in aa))

    def test_all_personas_start_from_identical_state(self):
        a, b = make(0), make(0)
        self.assertTrue(same_state(a, b))


if __name__ == "__main__":
    unittest.main()
