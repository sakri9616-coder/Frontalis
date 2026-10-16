"""v3.4 Stage 5 のための並列ランナー（SPEC-v3.4 第 11 節のプロトコルの一部。Stage 4 で凍結）。

Windows に fork はないので、multiprocessing の "spawn" 方式（ProcessPoolExecutor(mp_context=spawn)）を使う。
  - 仕事は (関数の名前 "module:function", 引数の tuple) の列。結果は入力と同じ順序で返る（ex.map、chunksize = 1）。
  - 結果は直列（serial = True）と並列でビット単位で同じになる（仕事が決定的であること。tests/test_ra_stage4.py が開発用の種で確かめる）。
    ワーカーは BLAS/OpenMP のスレッド数を 1 にして起動する（環境変数を子プロセスにだけ引き継ぐ）。
  - 自分が起動したプロセスの PID だけを扱う。イメージ名での一括終了はしない。プールは with 句で閉じ、終了コードを info に返す。
  - 乱数種の守りは仕事の関数の側（ra_protocol.guard_seeds）にある。ここでは種を見ない。

使い方:
    from ra_parallel import run_tasks
    if __name__ == "__main__":                                  # Windows の spawn では必須
        results, info = run_tasks([("ra_protocol:run_individual", (3490, "trust", "none", "interp", 8, 0, ["C-v34"], "dev"))], workers=10)
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAX_WORKERS = 10                       # PLAN-v3.4: 並列は最大 10 プロセス（12 論理コア）
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

Task = Tuple[str, Tuple[Any, ...]]


def _resolve(name: str):
    mod, _, fn = name.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _run_one(task: Task) -> Tuple[Any, int, float]:
    """ワーカーで 1 つの仕事を実行する。返すもの: (結果, ワーカーの PID, 所要時間（秒）)。"""
    name, args = task
    t0 = time.perf_counter()
    res = _resolve(name)(*args)
    return res, os.getpid(), time.perf_counter() - t0


def _init_worker() -> None:
    for k in THREAD_ENV:
        os.environ[k] = "1"


def run_tasks(tasks: Sequence[Task], workers: Optional[int] = None, serial: bool = False) -> Tuple[List[Any], Dict[str, Any]]:
    """仕事の列を実行し、(結果の列, 情報) を返す。serial = True なら同じプロセスで順に実行する（並列との比較の基準）。
    情報: workers、pids（自分が起動したワーカーの PID）、exit_codes（プールを閉じた後の終了コード。0 以外または None は異常）、seconds（各仕事の所要時間）、wall（全体の秒）。"""
    tasks = list(tasks)
    t0 = time.perf_counter()
    if serial or workers == 1 or len(tasks) <= 1:
        outs = [_run_one(t) for t in tasks]
        info = {"workers": 1, "pids": [os.getpid()], "exit_codes": [], "serial": True}
    else:
        n = max(1, min(int(workers) if workers is not None else MAX_WORKERS, MAX_WORKERS, len(tasks)))
        saved = {k: os.environ.get(k) for k in THREAD_ENV}
        for k in THREAD_ENV:                                   # 子プロセスは起動時に親の環境を引き継ぐ
            os.environ[k] = "1"
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker) as ex:
                outs = list(ex.map(_run_one, tasks, chunksize=1))
                procs = list(getattr(ex, "_processes", {}).values())
            codes = [p.exitcode for p in procs]
            pids = [p.pid for p in procs]
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        info = {"workers": n, "pids": pids, "exit_codes": codes, "serial": False}
    info["seconds"] = [o[2] for o in outs]
    info["worker_pids_of_tasks"] = [o[1] for o in outs]
    info["wall"] = time.perf_counter() - t0
    return [o[0] for o in outs], info


def estimate_wall(task_seconds: Sequence[float], workers: int = MAX_WORKERS) -> float:
    """所要時間（秒）の列から、workers 本の貪欲な割り当て（最長の仕事から順に、最も空いたワーカーへ）の壁時計の見積もりを返す。起動・結果の受け渡しの時間は含まない。"""
    import heapq
    loads = [0.0] * max(1, int(workers))
    heapq.heapify(loads)
    for t in sorted(task_seconds, reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + float(t))
    return max(loads)
