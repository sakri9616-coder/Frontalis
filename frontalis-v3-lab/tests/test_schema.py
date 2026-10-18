"""経験・プローブの検証（SPEC 第2節）。"""
import copy
import unittest

import numpy as np

from helpers import load_json, make
from npnv3 import SchemaError


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.n = make(0)
        self.ok = load_json("new_event.json")

    def bad(self, mutate):
        e = copy.deepcopy(self.ok)
        mutate(e)
        with self.assertRaises(SchemaError):
            self.n.appraise(e)

    def test_valid_experience(self):
        self.n.appraise(self.ok)

    def test_unknown_outcome_type_is_an_error(self):
        self.bad(lambda e: e["outcomes"].append({"type": "teleport", "actor": "self", "magnitude": 0.5}))

    def test_magnitude_range(self):
        self.bad(lambda e: e["outcomes"][0].update(magnitude=1.2))
        self.bad(lambda e: e["outcomes"][0].update(magnitude=-0.1))

    def test_actor_must_be_known(self):
        self.bad(lambda e: e["outcomes"][0].update(actor="stranger_x"))

    def test_unknown_words_are_allowed_with_warnings(self):
        """v3.1（SPEC-v3.1 1.6）: lexicon にない類・文脈・概念は、エラーにせず、ゼロベクトルで登録して警告を記録する。
        （v3.0 では SchemaError だった。結果の型と actor は、これまでどおり未知ならエラー。）"""
        e = copy.deepcopy(self.ok)
        e["participants"][0]["class"] = "alien"; e["context"] = "moon"; e["concepts"] = ["music", "teleport"]
        ap = self.n.appraise(e)
        self.assertEqual(len(ap["warnings"]), 3)
        self.assertEqual(self.n.warnings, [])                       # appraise は状態を変えない
        self.n.experience(e)
        self.assertEqual(len(self.n.warnings), 3)
        self.assertTrue(np.all(self.n.emb.own["cls"]["alien"] == 0.0))
        e2 = copy.deepcopy(e); e2["id"] = "again"
        self.assertEqual(self.n.appraise(e2)["warnings"], [])        # 登録済みなので警告は 1 回だけ

    def test_interpretations_schema(self):
        e = copy.deepcopy(self.ok)
        e["outcomes"].append({"interpretations": {"credit_taken": 1.0, "credit_shared": 3.0}, "actor": "friendA", "magnitude": 0.5})
        self.n.appraise(e)
        self.bad(lambda x: x["outcomes"].append({"interpretations": {"credit_taken": 1.0}, "actor": "friendA", "magnitude": 0.5}))
        self.bad(lambda x: x["outcomes"].append({"interpretations": {"credit_taken": 1.0, "teleport": 1.0}, "actor": "friendA", "magnitude": 0.5}))
        self.bad(lambda x: x["outcomes"].append({"interpretations": {"credit_taken": 0.0, "credit_shared": 1.0}, "actor": "friendA", "magnitude": 0.5}))
        self.bad(lambda x: x["outcomes"].append({"type": "connection", "interpretations": {"a": 1, "b": 1}, "actor": "friendA", "magnitude": 0.5}))

    def test_missing_and_extra_keys(self):
        self.bad(lambda e: e.pop("outcomes"))
        self.bad(lambda e: e.update(extra=1))

    def test_reserved_and_duplicate_ids(self):
        self.bad(lambda e: e["participants"][0].update(id="self"))
        self.bad(lambda e: e["participants"].append(dict(e["participants"][0])))

    def test_probe_forms(self):
        self.n.probe({"context": "music_session", "concepts": ["music"], "participants": []})
        self.n.probe({"entity": "friendA", "class": "collaborator"})
        self.n.probe({"concept": "music"})
        with self.assertRaises(SchemaError):
            self.n.probe({"entity": "friendA"})
        self.n.probe({"concept": "teleport"})                       # v3.1: 未知の語は許す（ゼロベクトル）
        self.n.probe({"cue": "type:praise"})

    def test_class_of_an_individual_cannot_change(self):
        self.n.experience(self.ok)
        e = copy.deepcopy(self.ok); e["id"] = "again"; e["participants"][0]["class"] = "mentor"
        with self.assertRaises(ValueError):
            self.n.experience(e)


if __name__ == "__main__":
    unittest.main()
