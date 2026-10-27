"""Calibrate LR_MATCH for v15 on one Colab VM (npn/npn_gate.py --calib): one process per dev seed, change times only.
Writes p63_calib_lr<LR>_s<seed>.txt here.   usage (from probes/):  python p63_colab_calib.py LR A B
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run(lr, s):
    with open(f"p63_calib_lr{lr}_s{s}.txt", "w") as fh:
        return subprocess.run([sys.executable, "-B", "npn/npn_gate.py", "--calib", str(s), str(s + 1), lr],
                              stdout=fh, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    lr, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
        rcs = list(ex.map(lambda s: run(lr, s), range(a, b)))
    print("rcs", rcs)
    sys.exit(max(rcs))
