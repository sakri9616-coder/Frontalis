"""CLI（python -m npnv3）の動作。"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from helpers import HIST, ROOT


def run(*args):
    return subprocess.run([sys.executable, "-m", "npnv3", *args], cwd=ROOT, capture_output=True, text=True)


class CLITests(unittest.TestCase):
    def test_init_feed_probe_show(self):
        with tempfile.TemporaryDirectory() as d:
            st = os.path.join(d, "state")
            r = run("init", st, "--seed", "3"); self.assertEqual(r.returncode, 0, r.stderr)
            r = run("feed", st, str(HIST / "R.jsonl")); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("22 件を処理", r.stdout)
            pr = os.path.join(d, "probes.json")
            with open(pr, "w", encoding="utf-8") as f:
                json.dump([{"entity": "friendA", "class": "collaborator"}, {"concept": "music"}], f)
            r = run("probe", st, pr); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("エンティティ friendA", r.stdout)
            r = run("show", st); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("人モデルの価値", r.stdout)

    def test_bad_input_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            st = os.path.join(d, "state")
            run("init", st)
            bad = os.path.join(d, "bad.jsonl")
            with open(bad, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "x", "context": "music_session", "concepts": [], "participants": [],
                                    "outcomes": [{"type": "teleport", "actor": "self", "magnitude": 0.5}]}) + "\n")
            r = run("feed", st, bad)
            self.assertEqual(r.returncode, 2)
            self.assertIn("未知の結果の型", r.stderr)


if __name__ == "__main__":
    unittest.main()
