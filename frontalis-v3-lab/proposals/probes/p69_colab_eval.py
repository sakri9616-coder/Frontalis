"""Run v19 (npn/npn_dyad_transfer.py --seeds) for seeds A..B-1 on one Colab VM, one process per seed, as many at a time as the
VM has CPUs. Writes <PREFIX>_s<seed>.json here.   usage (from probes/):  python p69_colab_eval.py PREFIX A B
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run(prefix, s):
    r = subprocess.run([sys.executable, "-B", "npn/npn_dyad_transfer.py", "--seeds", str(s), str(s + 1),
                        "--out", f"{prefix}_s{s}.json"], capture_output=True, text=True)
    return s, r.returncode, r.stderr[-300:]


if __name__ == "__main__":
    prefix, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
        res = list(ex.map(lambda s: run(prefix, s), range(a, b)))
    for s, rc, err in res:
        print(s, "rc", rc, err if rc else "")
    sys.exit(max(rc for _, rc, _ in res))
