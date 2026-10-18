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

    def test_explain_prints_the_breakdown_and_the_assignment_records(self):
        with tempfile.TemporaryDirectory() as d:
            st = os.path.join(d, "state")
            run("init", st)
            ev = []
            for k in range(4):
                ev.append({"id": f"c{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": f"s{k}"})
                ev.append({"id": f"u{k}", "context": "studio", "concepts": [], "participants": [],
                           "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": f"s{k}"})
            fp = os.path.join(d, "e.jsonl")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(e) for e in ev) + "\n")
            r = run("feed", st, fp); self.assertEqual(r.returncode, 0, r.stderr)
            r = run("explain", st, "concept:praise_from_X", "--context", "cafe_chat", "--trace", "2"); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("応答 C = Σ p(k)·C_k", r.stdout)
            self.assertIn("割当の記録", r.stdout)
            self.assertIn("MAP", r.stdout)
            r = run("probe", st, os.path.join(d, "none.json")); self.assertNotEqual(r.returncode, 0)

    def test_feed_correction_and_choice_kinds(self):
        """v3.3 第 2 部: 行ごとに kind = "correction"（訂正）| "choice"（二択）を受ける。要約・show・probe に J が出る。"""
        import yz_protocol as Y
        with tempfile.TemporaryDirectory() as d:
            st = os.path.join(d, "state")
            run("init", st)
            lines = list(Y.y_history(0, True)[:4]) + [Y.y_corrections(0)[0], Y.hesitant_choice("hc1", "x_a", "x_b"), Y.hesitant_choice("hc2", "x_a", "x_b")]
            fp = os.path.join(d, "e.jsonl")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(e) for e in lines) + "\n")
            r = run("feed", st, fp); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("訂正: j0 の結果を訂正", r.stdout)
            self.assertIn("当時の記録", r.stdout)
            self.assertIn("二択:", r.stdout)
            self.assertIn("葛藤の強さ", r.stdout)
            self.assertIn("7 件を処理", r.stdout)                   # 入力 7 行 = 経験 4 + 訂正 1 + 二択 2
            self.assertIn("経験 6 件", r.stdout)                   # 学習した経験は 6（訂正は経験の数に入らない）
            r = run("show", st); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("選択の正当化 J", r.stdout)
            self.assertIn("concept:x_", r.stdout)
            self.assertIn("訂正を受けた回数: 1", r.stdout)
            pr = os.path.join(d, "p.json")
            with open(pr, "w", encoding="utf-8") as f:
                json.dump([{"cue": "concept:x_a"}, {"context": "studio", "concepts": ["x_a"], "participants": []}], f)
            r = run("probe", st, pr); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("正当化 J=", r.stdout)
            self.assertIn("J_Q=", r.stdout)
            bad = os.path.join(d, "bad.jsonl")
            with open(bad, "w", encoding="utf-8") as f:
                f.write(json.dumps({"id": "c", "kind": "correction", "refers_to": "nope", "corrected_outcomes": []}) + "\n")
            r = run("feed", st, bad); self.assertEqual(r.returncode, 2)
            self.assertIn("参照先が見つかりません", r.stderr)

    def test_persist_mode_init_choice_summary_and_preference_command(self):
        """v3.3 第 2 部 8: init で justification_mode / lc_design を選べる。persist の二択の要約（源つきの出来事・反証への抵抗）と show の源、preference コマンド（Z8）。"""
        import yz_protocol as Y
        with tempfile.TemporaryDirectory() as d:
            st = os.path.join(d, "state")
            r = run("init", st, "--justification-mode", "persist", "--lc-design", "revised"); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("justification_mode = persist", r.stdout)
            self.assertIn("lc_design = revised", r.stdout)
            warm = []
            for i in range(2):
                warm.append({"id": f"wa{i}", "context": Y.Z_CTX, "concepts": ["project_x"], "participants": [dict(Y.ALLY)],
                             "outcomes": [{"type": "connection", "actor": "ally", "magnitude": 0.8}, {"type": "support", "actor": "ally", "magnitude": 0.6}]})
            lines = warm + [Y.hesitant_choice("p1", "pp_x", "pp_y"), Y.hesitant_choice("p2", "pp_x", "pp_y"), Y.hesitant_choice("p3", "pp_x", "pp_y")]
            fp = os.path.join(d, "e.jsonl")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(e) for e in lines) + "\n")
            r = run("feed", st, fp); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("源つきの出来事", r.stdout)                 # 再び選ぶ（reaffirm）または反する選択（counter）
            r = run("show", st); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("justification_mode = persist", r.stdout)
            self.assertIn("（源: ", r.stdout)
            self.assertIn("spread", r.stdout)
            op = os.path.join(d, "o.json")
            with open(op, "w", encoding="utf-8") as f:
                json.dump([{"context": "studio", "concepts": ["pp_x"]}, {"context": "studio", "concepts": ["pp_y"]}], f)
            r = run("preference", st, op); self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("根拠のある部分", r.stdout)
            self.assertIn("正当化 J の部分", r.stdout)
            self.assertIn("ΔQ =", r.stdout)
            self.assertIn("源:", r.stdout)
            bad = os.path.join(d, "bad.json")
            with open(bad, "w", encoding="utf-8") as f:
                json.dump([{"cue": "concept:x"}], f)
            r = run("preference", st, bad); self.assertEqual(r.returncode, 2)
            r = run("init", os.path.join(d, "s2"), "--justification-mode", "forever"); self.assertNotEqual(r.returncode, 0)

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
