"""報告用の実行器（テストではない）: 条件づけの設計 lc_design の既定を差し替えて、全テスト（または指定したモジュール）を走らせる。

v3.3 第 2 部の作業 0: 既定の設計を spec_literal に戻し、revised（局所 MAP など。第 1 部の改訂）は実験設定として残した。CHANGELOG の M 節の比較表は、
同じテストを両方の設計で走らせて作る。既定の設計の判定は普通の `python3 -B -m unittest discover -s tests`。この実行器は、**revised を既定として全テストを走らせた場合**の
合否と数値を、既定の結果を上書きせずに残すためのもの。

  python3 tests/run_with_design.py revised                # 全テストを revised を既定にして実行する
  python3 tests/run_with_design.py revised test_w03_within_between test_u06_no_amplification_extended   # モジュールを指定
  python3 tests/run_with_design.py spec_literal --set kappa_ag=0     # 他の既定も差し替える（--set 名前=値。結果は tests/results/spec_literal+kappa_ag=0/）。
                                                                       # 第 2 部の主体性の重み κ_ag の効果を、設計の切り替えの効果から分けるために使った

やること:
  - `Config.__init__` を包み、lc_design を省いた Config（NPN(seed) など）の既定を指定の設計にする。明示的に lc_design を指定しているテスト・復元（from_dict）・replace は
    そのまま（dataclasses.replace と from_dict は全項目を渡す）。
  - テストが dump する数値の書き込み先を tests/results/<設計>/ に変える（既定の結果 tests/results/*.json は上書きしない）。
  - 各テストの合否を tests/results/<設計>/_status.json に書く（unittest の id → ok / FAIL / ERROR / expected failure）。
"""
import json
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import helpers  # noqa: E402
import npnv3.config as C  # noqa: E402


def main(argv):
    if len(argv) < 2 or argv[1] not in ("revised", "spec_literal"):
        print(__doc__)
        return 2
    design = argv[1]
    rest = argv[2:]
    sets = {}
    while "--set" in rest:
        i = rest.index("--set")
        k, v = rest[i + 1].split("=", 1)
        sets[k] = json.loads(v) if v not in ("true", "false") else (v == "true")
        del rest[i:i + 2]
    argv = argv[:2] + rest
    tag = design + "".join(f"+{k}={v}" for k, v in sets.items())
    orig = C.Config.__init__

    def init(self, *a, **kw):
        kw.setdefault("lc_design", design)
        for k, v in sets.items():
            kw.setdefault(k, v)
        orig(self, *a, **kw)

    C.Config.__init__ = init
    out_dir = helpers.RESULTS / tag
    helpers.RESULTS = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = unittest.TestLoader()
    suite = loader.discover(str(HERE)) if len(argv) == 2 else loader.loadTestsFromNames(argv[2:])
    status = {}

    class Result(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test); status[test.id()] = "ok"

        def addFailure(self, test, err):
            super().addFailure(test, err); status[test.id()] = "FAIL"

        def addError(self, test, err):
            super().addError(test, err); status[test.id()] = "ERROR"

        def addExpectedFailure(self, test, err):
            super().addExpectedFailure(test, err); status[test.id()] = "expected failure"

        def addUnexpectedSuccess(self, test):
            super().addUnexpectedSuccess(test); status[test.id()] = "unexpected success"

    t0 = time.time()
    res = unittest.TextTestRunner(verbosity=1, resultclass=Result).run(suite)
    status["_time_seconds"] = round(time.time() - t0, 1)
    status["_design"] = design
    status["_overrides"] = sets
    with open(out_dir / "_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=0)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
