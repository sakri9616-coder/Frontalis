"""v3.4 Stage 5: tests/ra_protocol.py の変更が、乱数種の守りの定数だけであることの単体テスト（リードの条件 S4-13 の (1)(2)(4)）。

  (1) 変更は PHASES と STAGE_ALLOWED の 2 行（評価の種 3400〜3439 を phase = "stage5" でだけ通す）に限る。他の関数・定数は、Stage 4 で凍結した版とバイト単位で同じ。
  (2) 環境変数による解錠を足していない。
  (4) 変更後のファイルの 2 行を元の行に戻すと、freeze-protocol-v2.json の SHA-256（a8fc4a79…c820）に一致する。AST でも、全ての関数・クラスが同じ。

評価の種も予行の種も実行しない（guard_seeds を呼ぶだけ）。
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import unittest
from pathlib import Path

from helpers import ROOT  # noqa: F401

import ra_protocol as RP  # noqa: E402

OLD1 = 'PHASES = {"dev": SEEDS_DEV, "diag": SEEDS_DIAG, "mut": SEEDS_MUTATION}'
NEW1 = 'PHASES = {"dev": SEEDS_DEV, "diag": SEEDS_DIAG, "mut": SEEDS_MUTATION, "stage5": SEEDS_EVAL}'
OLD2 = 'STAGE_ALLOWED = frozenset(SEEDS_DEV) | frozenset(SEEDS_DIAG) | frozenset(SEEDS_MUTATION)           # Stage 4'
NEW2 = 'STAGE_ALLOWED = frozenset(SEEDS_DEV) | frozenset(SEEDS_DIAG) | frozenset(SEEDS_MUTATION) | frozenset(SEEDS_EVAL)           # Stage 5'

V2 = ROOT / "tests" / "results" / "v34" / "freeze-protocol-v2.json"
V3 = ROOT / "tests" / "results" / "v34" / "freeze-protocol-v3.json"
PATH = ROOT / "tests" / "ra_protocol.py"


def lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


class GuardOnlyChange(unittest.TestCase):
    def setUp(self):
        self.new = lf(PATH.read_bytes()).decode("utf-8")
        self.assertEqual(self.new.count(NEW1), 1)
        self.assertEqual(self.new.count(NEW2), 1)
        self.reverted = self.new.replace(NEW1, OLD1).replace(NEW2, OLD2)

    def test_reverting_the_two_guard_lines_gives_the_stage4_frozen_file_byte_for_byte(self):
        v2 = json.loads(V2.read_text(encoding="utf-8"))
        old_hash = v2["files"]["frontalis-v3-lab/tests/ra_protocol.py"]
        self.assertEqual(hashlib.sha256(self.reverted.encode("utf-8")).hexdigest(), old_hash)

    def test_every_function_and_class_is_unchanged_ast(self):
        a, b = ast.parse(self.new), ast.parse(self.reverted)
        fa = {n.name: ast.dump(n) for n in ast.walk(a) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        fb = {n.name: ast.dump(n) for n in ast.walk(b) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(fa, fb)
        self.assertGreater(len(fa), 20)

    def test_guard_function_is_unchanged_and_has_no_environment_unlock(self):
        src = inspect.getsource(RP.guard_seeds)
        self.assertNotIn("environ", src)
        self.assertNotIn("getenv", src)
        self.assertNotIn("environ", self.new)
        self.assertNotIn("getenv", self.new)

    def test_guard_behaviour(self):
        RP.guard_seeds(list(range(3400, 3440)), "stage5")
        RP.guard_seeds([3490], "dev")
        RP.guard_seeds([3460], "diag")
        RP.guard_seeds([3450], "mut")
        for s in (3399, 3440, 3450, 3459, 3490, 3500, 3539, 3540):
            with self.assertRaises(PermissionError):
                RP.guard_seeds([s], "stage5")
        for phase in ("dev", "diag", "mut"):
            for s in (3400, 3439):
                with self.assertRaises(PermissionError):
                    RP.guard_seeds([s], phase)
        for s in (3500, 3539):                   # 予約の種はどの phase でも拒否
            for phase in RP.PHASES:
                with self.assertRaises(PermissionError):
                    RP.guard_seeds([s], phase)
        with self.assertRaises(ValueError):
            RP.guard_seeds([3400], "eval")
        self.assertEqual(set(RP.PHASES), {"dev", "diag", "mut", "stage5"})
        self.assertEqual(set(RP.PHASES["stage5"]), set(range(3400, 3440)))
        self.assertFalse(set(RP.STAGE_ALLOWED) & set(RP.SEEDS_RESERVED))

    def test_freeze_protocol_v3_records_old_and_new_hash_and_matches_the_files(self):
        if not V3.exists():
            self.skipTest("freeze-protocol-v3.json は本番の前に書く")
        v3 = json.loads(V3.read_text(encoding="utf-8"))
        v2 = json.loads(V2.read_text(encoding="utf-8"))
        self.assertEqual(v3["version"], 3)
        self.assertEqual(v3["ra_protocol"]["old_sha256_v2"], v2["files"]["frontalis-v3-lab/tests/ra_protocol.py"])
        self.assertEqual(v3["ra_protocol"]["new_sha256"], RP.file_sha256(PATH))
        self.assertTrue(v3["ra_protocol"]["diff_is_guard_only"])
        self.assertFalse(v3["ra_protocol"]["environment_variable_unlock"])
        for rel, h in {**v3["evaluation_code_sha256"], **v3["system_files_sha256"]}.items():
            self.assertEqual(RP.file_sha256(ROOT.parent / rel), h, rel)


if __name__ == "__main__":
    unittest.main()
