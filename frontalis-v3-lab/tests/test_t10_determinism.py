"""T10 決定性: 同じ種・同じ入力で、全出力と全状態が完全一致する。種が違えば（頭の初期値が違うので）状態は異なる。"""
import copy
import unittest

import numpy as np

from helpers import SEEDS, deep_equal, feed, load_json, load_jsonl, make, pmap


def inputs():
    new = load_json("new_event.json")
    opt = copy.deepcopy(new); opt["id"] = "opt-1"; opt["optional"] = True
    return load_jsonl("R.jsonl") + [new, opt] + load_jsonl("T8_train.jsonl")[:30]


def probes(n):
    return [n.probe({"entity": "friendA", "class": "collaborator", "context": "music_session"}),
            n.probe({"context": "music_session", "concepts": ["music"], "participants": []}),
            n.probe({"concept": "music"})]


def check_seed(seed):
    """同じ種・同じ入力で 2 回作って比べる（v3.2: 種を並列化。比べる内容は同じ）。"""
    xs = inputs()
    a, b = make(seed), make(seed)
    oa, ob = feed(a, xs), feed(b, xs)
    ma, aa = a.get_state(); mb, ab = b.get_state()
    return {"outputs": deep_equal(oa, ob), "probes": deep_equal(probes(a), probes(b)), "meta": deep_equal(ma, mb),
            "arrays": all(np.array_equal(aa[k], ab[k]) for k in aa)}


class T10Determinism(unittest.TestCase):
    def test_same_seed_same_result(self):
        for seed, r in zip(SEEDS, pmap(check_seed, SEEDS)):
            self.assertTrue(r["outputs"], f"seed {seed}: 出力が一致しない")
            self.assertTrue(r["probes"], f"seed {seed}: プローブが一致しない")
            self.assertTrue(r["meta"])
            self.assertTrue(r["arrays"])

    def test_appraise_does_not_change_state(self):
        xs = inputs()
        n = make(3); feed(n, xs[:25])
        m0, a0 = copy.deepcopy(n.get_state())
        for e in xs[25:30]:
            n.appraise(e)
        n.probe({"entity": "friendA", "class": "collaborator"})
        m1, a1 = n.get_state()
        self.assertTrue(deep_equal(m0, m1))
        self.assertTrue(all(np.array_equal(a0[k], a1[k]) for k in a0))

    def test_different_seed_differs(self):
        xs = inputs()[:10]
        kw = {"mixture_heads": ("mlp", "linear", "knn1", "none")}      # v3.3: 既定の mixture から mlp を外したので、種で初期値が変わる mlp を入れて確かめる
        a, b = make(0, **kw), make(1, **kw)
        feed(a, xs); feed(b, xs)
        self.assertFalse(np.array_equal(a.belief.head.W1, b.belief.head.W1))


if __name__ == "__main__":
    unittest.main()
