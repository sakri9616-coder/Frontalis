"""Prototype of v15 (npn/npn_gate.py --proto) on one Colab VM: one process per seed, KAPPA given, as many at a time as
the VM has CPUs. Writes p63_proto_k<KAPPA>_s<seed>.txt here. Prototype seeds only (0-7), not evaluation seeds.

usage (from probes/):  python p63_colab_proto.py KAPPA A B
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run(k, s):
    env = {**os.environ, "NPN_GATE_KAPPA": k}
    with open(f"p63_proto_k{k}_s{s}.txt", "w") as fh:
        r = subprocess.run([sys.executable, "-B", "npn/npn_gate.py", "--proto", str(s), str(s + 1)],
                           stdout=fh, stderr=subprocess.STDOUT, env=env)
    return r.returncode


if __name__ == "__main__":
    k, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
        rcs = list(ex.map(lambda s: run(k, s), range(a, b)))
    print("rcs", rcs)
    sys.exit(max(rcs))
