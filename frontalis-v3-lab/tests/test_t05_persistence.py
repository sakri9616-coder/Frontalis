"""T5 保存・復元。
経験 A を学習 → 保存 → 復元 → 経験 B と、A+B を連続で学習した結果が、全状態・全出力で完全一致（np.array_equal）。
改ざん・版違いは例外。B には optional な経験（乱数を使う選択）を含める。
"""
import copy
import json
import os
import tempfile
import unittest

import numpy as np

from helpers import SEEDS, deep_equal, feed, load_json, load_jsonl, make
from npnv3 import NPN, StateIntegrityError, StateVersionError


def split_histories():
    a = load_jsonl("K.jsonl")
    new = load_json("new_event.json")
    opt = copy.deepcopy(new); opt["id"] = "optional-1"; opt["optional"] = True
    opt2 = copy.deepcopy(load_jsonl("R.jsonl")[10]); opt2["id"] = "optional-2"; opt2["optional"] = True
    b = [new, opt, opt2] + load_jsonl("R.jsonl")[:8]
    return a, b


def states_equal(x, y):
    mx, ax = x.get_state()
    my, ay = y.get_state()
    return deep_equal(mx, my) and ax.keys() == ay.keys() and all(np.array_equal(ax[k], ay[k]) for k in ax)


class T5SaveRestore(unittest.TestCase):
    def test_exact_restore_all_seeds(self):
        a, b = split_histories()
        for seed in SEEDS:
            with tempfile.TemporaryDirectory() as d:
                n1 = make(seed); feed(n1, a); n1.save(d)
                n2 = NPN.load(d)
                self.assertTrue(states_equal(n1, n2), f"seed {seed}: 復元直後の状態が一致しない")
                out_restored = feed(n2, b)
            n3 = make(seed); feed(n3, a)
            out_continuous = feed(n3, b)
            self.assertTrue(deep_equal(out_restored, out_continuous), f"seed {seed}: 出力が一致しない")
            self.assertTrue(states_equal(n2, n3), f"seed {seed}: B の後の状態が一致しない")
        approached = [o["approached"] for o in out_continuous if o["optional"]]
        self.assertTrue(len(approached) == 2)

    def test_tamper_and_version(self):
        a, _ = split_histories()
        n = make(0); feed(n, a)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            NPN.load(d)                                         # 正常
            # (1) state.json の数値を 1 つ書き換える
            p = os.path.join(d, "state.json")
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            doc["meta"]["mood"] = doc["meta"]["mood"] + 1e-9
            with open(p, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            with self.assertRaises(StateIntegrityError):
                NPN.load(d)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            # (2) arrays.npz の配列を 1 つ書き換える
            z = dict(np.load(os.path.join(d, "arrays.npz")))
            key = "v_person_W"
            z[key] = z[key].copy(); z[key][0, 0] += 1e-12
            np.savez(os.path.join(d, "arrays.npz"), **z)
            with self.assertRaises(StateIntegrityError):
                NPN.load(d)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            # (3) 版違い
            p = os.path.join(d, "state.json")
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            doc["schema_version"] = 999
            with open(p, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            with self.assertRaises(StateVersionError):
                NPN.load(d)


if __name__ == "__main__":
    unittest.main()
