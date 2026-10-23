"""Run the frozen v14 evaluation (npn/npn_consol.py) for seeds A..B-1 on one Colab VM, one process per seed, as many at a
time as the VM has CPUs. Each seed writes p62_eval_v14_s<seed>.json here (colab_runner.py copies new top-level files back).
This wrapper only schedules; it does not change what npn_consol.py computes.

usage (from probes/):  python p62_colab_job.py A B
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor


def run(s):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-B", "npn/npn_consol.py", "--seeds", str(s), str(s + 1),
                        "--out", f"p62_eval_v14_s{s}.json"], capture_output=True, text=True)
    return s, r.returncode, time.time() - t0, r.stderr[-300:]


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    n = os.cpu_count() or 1
    print("cpus", n, flush=True)
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(run, range(a, b)))
    for s, rc, dt, err in res:
        print(s, "rc", rc, f"{dt:.0f}s", err if rc else "", flush=True)
    sys.exit(max(rc for _, rc, _, _ in res))
