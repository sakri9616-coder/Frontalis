"""経験・プローブの検証（SPEC 第2節）。"""
import copy
import unittest

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

    def test_unknown_class_context_concept(self):
        self.bad(lambda e: e["participants"][0].update({"class": "alien"}))
        self.bad(lambda e: e.update(context="moon"))
        self.bad(lambda e: e.update(concepts=["music", "teleport"]))

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
        with self.assertRaises(SchemaError):
            self.n.probe({"concept": "teleport"})

    def test_class_of_an_individual_cannot_change(self):
        self.n.experience(self.ok)
        e = copy.deepcopy(self.ok); e["id"] = "again"; e["participants"][0]["class"] = "mentor"
        with self.assertRaises(ValueError):
            self.n.experience(e)


if __name__ == "__main__":
    unittest.main()
