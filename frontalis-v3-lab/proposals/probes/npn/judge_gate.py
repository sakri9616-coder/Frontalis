"""Judge v15 (DESIGN-gate v1.3 section 6) from result JSON files of npn_gate.py. match = none (LR_MATCH = 0.01).

usage: python judge_gate.py --out judgment.json res1.json [res2.json ...]
"""
import argparse
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "credit"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_g import binom_two_sided  # noqa: E402
from npn_gate import change_time, MODES, PLAIN  # noqa: E402
import npn_consol as C  # noqa: E402

MATCH = "none"


def gate(x):
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    p = binom_two_sided(pos, pos + neg) if pos + neg else 1.0
    return {"pos": pos, "neg": neg, "p": p, "mean": float(x.mean()), "pass": bool(pos >= 30 and p < 0.01)}


def judge(rows):
    g = lambda k, f: np.array([r[k][f] for r in rows])
    ct = lambda k: np.array([change_time(r[k]["series10"], C.COOP_BAD_AT) for r in rows])
    res = {"n": len(rows), "match_mode": MATCH}
    s1a = gate(g("gated_f0_g4", "coop") - g(f"{MATCH}_f0_g4", "coop"))
    s1b = gate(g("gated_f1_advC_g4", "qual") - g(f"{MATCH}_f1_advC_g4", "qual"))
    res["S1"] = {"coop": s1a, "qual_sym": s1b, "pass": s1a["pass"] and s1b["pass"]}
    med = {m: float(statistics.median(ct(f"{m}_reason_g0"))) for m in MODES}
    ratio = med["gated"] / med[MATCH]
    res["S2"] = {"median_change_time": dict(med), "ratio_gated_over_match": ratio, "pass": bool(ratio <= 1.1)}
    res["S3"] = gate(ct("gshuf_reason_g0") - ct("gated_reason_g0"))          # positive = gated changed earlier
    d4 = g("gated_f0_g4", "coop") - g("gshuf_f0_g4", "coop")
    s4 = gate(d4)
    s4["gshuf_ge_gated"] = int((d4 <= 0).sum())
    s4["reading"] = ("contributes to keeping" if s4["pass"] else
                     "timing does not contribute to keeping (speed only)" if s4["gshuf_ge_gated"] >= 20 else
                     "no evidence that timing contributes to keeping")
    res["S4"] = s4
    sane_f = int((np.array([r["gated_f0_g4"]["formed"][0] for r in rows]) >= 0.5).sum())
    sane_q = int((g("gated_f0_g4", "qual") >= 0.5).sum())
    res["sanity"] = {"formed_coop": sane_f, "qual_learned": sane_q, "ok": bool(sane_f >= 36 and sane_q >= 36)}
    med.update({m: float(statistics.median(ct(f"{m}_reason_g0"))) for m in PLAIN})
    desc = {"D1_plane": {m: {"keep_coop_mean": float(g(f"{m}_f0_g4", "coop").mean()),
                             "keep_qual_sym_mean": float(g(f"{m}_f1_advC_g4", "qual").mean()),
                             "change_time_median_g0": med[m]} for m in MODES + PLAIN}}
    fd = {m: g(f"{m}_f1_g4", "late") - g(f"{m}_f0_g4", "late") for m in MODES}
    desc["D2_follow_diff"] = {m: {"mean": float(v.mean()), "pos": int((v > 0).sum())} for m, v in fd.items()}
    desc["D3_change_time_median_g4"] = {m: float(statistics.median(ct(f"{m}_reason_g4"))) for m in MODES}
    desc["D4_gate_mean"] = {m: {"first100": float(g(f"{m}_f0_g4", "h_mean_first100").mean()),
                                "dyad": float(g(f"{m}_f0_g4", "h_mean_dyad").mean())} for m in ("gated", "gshuf")}
    desc["D5_step_ratio"] = {m: float(np.nanmean(g(f"{m}_f0_g4", "step_ratio"))) for m in MODES}
    res["describe"] = desc
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    rows = {}
    for fn in a.files:
        rows.update(json.load(open(fn, encoding="utf-8"))["rows"])
    seeds = sorted(rows, key=int)
    if len(seeds) != 40:
        print(f"WARNING: {len(seeds)} seeds, the gates (30/40, 36/40) are defined for 40")
    res = judge([rows[s] for s in seeds])
    res["seeds"] = [int(seeds[0]), int(seeds[-1])]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: (v.get("pass", v.get("ok")) if isinstance(v, dict) else v) for k, v in res.items()
                      if k != "describe"}))
