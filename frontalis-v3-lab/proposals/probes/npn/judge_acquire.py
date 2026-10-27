"""Judge v16 (PREREG-v16) from result JSON files of npn_acquire.py.

usage: python judge_acquire.py --out judgment.json res1.json [res2.json ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "credit"))
from run_g import binom_two_sided  # noqa: E402


def gate(x):
    x = np.asarray(x, float)
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    p = binom_two_sided(pos, pos + neg) if pos + neg else 1.0
    return {"pos": pos, "neg": neg, "p": p, "mean": float(x.mean()), "pass": bool(pos >= 30 and p < 0.01)}


def judge(rows):
    a = lambda f: np.array([f(r) for r in rows])
    res = {"n": len(rows)}
    res["G1"] = gate(a(lambda r: r["XT0_net"]["D1"] - r["YT0_net"]["D1"]))
    res["G2"] = gate(a(lambda r: r["YT0_net"]["dV"][0] - r["XT0_net"]["dV"][0]))
    res["G3a"] = gate(a(lambda r: r["G3"]["D1_random"] - r["G3"]["D1_top"]))
    res["G3b"] = gate(a(lambda r: r["G3"]["dV_top"][0] - r["G3"]["dV_random"][0]))
    res["G4a"] = gate(a(lambda r: r["XT+_net"]["boost"][0] - r["XT-_net"]["boost"][0]))
    res["G4b"] = gate(a(lambda r: r["XT+_net"]["boost"][0] - r["XT+_net"]["boost"][1]))
    res["G4c"] = gate(a(lambda r: r["XT-_net"]["boost_after_reliable"][0] - r["XT-_net"]["boost"][0]))
    sx = int(sum(r["XT0_net"]["sane"][0] > 0 and r["XT0_net"]["sane"][1] < 0 for r in rows))
    sy = int(sum(r["YT0_net"]["sane"][0] > 0 and r["YT0_net"]["sane"][1] < 0 for r in rows))
    res["sanity"] = {"X": sx, "Y": sy, "ok": bool(sx >= 36 and sy >= 36)}
    d = {"linear": {"G1": gate(a(lambda r: r["XT0_lin"]["D1"] - r["YT0_lin"]["D1"])),
                    "G2": gate(a(lambda r: r["YT0_lin"]["dV"][0] - r["XT0_lin"]["dV"][0])),
                    "G4b": gate(a(lambda r: r["XT+_lin"]["boost"][0] - r["XT+_lin"]["boost"][1]))},
         "table_G1_mean": float(a(lambda r: r["XT0_tab"]["D1"] - r["YT0_tab"]["D1"]).mean()),
         "mirror": {"G4a": gate(a(lambda r: r["XT+_mirror"]["boost"][0] - r["XT-_mirror"]["boost"][0])),
                    "G4b": gate(a(lambda r: r["XT+_mirror"]["boost"][0] - r["XT+_mirror"]["boost"][1])),
                    "boost_Tminus_mean": float(a(lambda r: r["XT-_mirror"]["boost"][0]).mean()),
                    "net_boost_Tminus_mean": float(a(lambda r: r["XT-_net"]["boost"][0]).mean())},
         "G3_devalue": {"D1_intact_mean": float(a(lambda r: r["G3"]["D1_intact"]).mean()),
                        "D1_top_mean": float(a(lambda r: r["G3"]["D1_top"]).mean()),
                        "D1_devalue_mean": float(a(lambda r: r["G3"]["D1_devalue"]).mean()),
                        "dV_top_mean": float(a(lambda r: r["G3"]["dV_top"][0]).mean()),
                        "dV_devalue_mean": float(a(lambda r: r["G3"]["dV_devalue"][0]).mean())},
         "G2_split": {f"{f}_{k}": [float(a(lambda r: r[f"{f}T0_{k}"]["dV"][i]).mean()) for i in range(3)]
                      for f in ("X", "Y") for k in ("net", "lin")}}
    res["describe"] = d
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")     # pipeline test with fewer seeds; not a judgment
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    rows = {}
    for fn in a.files:
        rows.update(json.load(open(fn, encoding="utf-8"))["rows"])
    seeds = sorted(rows, key=int)
    if len(seeds) != 40 and not a.check:
        sys.exit(f"{len(seeds)} seeds: PREREG-v16 judges only when all 40 are present (use --check for a pipeline test)")
    res = judge([rows[s] for s in seeds])
    res["seeds"] = [int(seeds[0]), int(seeds[-1])]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: (v.get("pass", v.get("ok")) if isinstance(v, dict) else v) for k, v in res.items()
                      if k != "describe"}))
