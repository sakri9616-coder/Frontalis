"""Judge v14 (DESIGN-consolidation v3 section 5) from one or more result JSON files of npn_consol.py.

usage: python judge_consol.py --out judgment.json res1.json [res2.json ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "credit"))
from run_g import binom_two_sided  # noqa: E402

MODES = ("consol", "perm", "permfix", "uniform", "none", "slow")


def gate(x):
    """x: per-seed paired differences. Pass = (x > 0) in >= 30 seeds and sign test (ties dropped) p < 0.01."""
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    p = binom_two_sided(pos, pos + neg) if pos + neg else 1.0
    return {"pos": pos, "neg": neg, "p": p, "mean": float(x.mean()), "pass": bool(pos >= 30 and p < 0.01)}


def judge(rows):
    g = lambda k, f: np.array([r[k][f] for r in rows])
    res = {"n": len(rows)}
    r1 = {}
    for ctl in ("perm", "permfix", "slow"):
        r1[f"coop_vs_{ctl}"] = gate(g("consol_f0_g4", "coop") - g(f"{ctl}_f0_g4", "coop"))
        r1[f"qual_sym_vs_{ctl}"] = gate(g("consol_f1_advC_g4", "qual") - g(f"{ctl}_f1_advC_g4", "qual"))
    for ctl in ("uniform", "none"):
        r1[f"describe_coop_vs_{ctl}"] = gate(g("consol_f0_g4", "coop") - g(f"{ctl}_f0_g4", "coop"))
        r1[f"describe_qual_sym_vs_{ctl}"] = gate(g("consol_f1_advC_g4", "qual") - g(f"{ctl}_f1_advC_g4", "qual"))
    r1["pass"] = all(r1[f"{side}_vs_{ctl}"]["pass"] for side in ("coop", "qual_sym") for ctl in ("perm", "permfix", "slow"))
    slow_formed = int((np.array([r["slow_f0_g4"]["formed"][0] for r in rows]) >= 0.5).sum())
    slow_formed_sym = int((np.array([r["slow_f1_advC_g4"]["formed"][1] for r in rows]) >= 0.5).sum())
    r1["slow_formation_ok"] = bool(slow_formed >= 36 and slow_formed_sym >= 36)
    r1["slow_formed_counts"] = [slow_formed, slow_formed_sym]
    r1["step_ratio_mean"] = {m: float(np.mean([r[f"{m}_f0_g4"]["step_ratio"] for r in rows])) for m in MODES}
    res["R1"] = r1
    diff = {m: g(f"{m}_f1_g4", "late") - g(f"{m}_f0_g4", "late") for m in MODES}
    res["R2"] = {"diff_consol": gate(diff["consol"]), "diff_consol_vs_permfix": gate(diff["consol"] - diff["permfix"])}
    res["R2"]["pass"] = res["R2"]["diff_consol"]["pass"] and res["R2"]["diff_consol_vs_permfix"]["pass"]
    d3 = g("consol_f0_g0", "coop") - g("consol_reason_g0", "coop")
    res["R3"] = {"count_ge_0.5": int((d3 >= 0.5).sum()), "mean": float(d3.mean()), "pass": bool((d3 >= 0.5).sum() >= 30),
                 "describe_g4_count_ge_0.5": int(((g("consol_f0_g4", "coop") - g("consol_reason_g4", "coop")) >= 0.5).sum())}
    ab = [r["ablation"] for r in rows]
    dc = lambda a, arm, when: abs(a[arm][when][0] - a["intact"][when][0])      # |change of the coop value| vs intact
    arr = lambda arm, when: np.array([dc(a, arm, when) for a in ab])
    r5 = {"R5a_now_formed_vs_random": gate(arr("formed_reset", "now") - arr("random_reset", "now")),
          "R5b_later_formed_vs_random_reset": gate(arr("formed_reset", "later") - arr("random_reset", "later"))}
    r5["pass"] = r5["R5a_now_formed_vs_random"]["pass"] and r5["R5b_later_formed_vs_random_reset"]["pass"]
    r5["describe"] = {f"{arm}.{when}": float(arr(arm, when).mean())
                      for arm in ("formed_reset", "formed_keep", "end_reset", "end_keep", "random_reset", "random_keep")
                      for when in ("now", "later")}
    r5["describe"]["end_vs_random_now"] = gate(arr("end_reset", "now") - arr("random_reset", "now"))
    r5["describe"]["formed_keep_vs_reset_later"] = gate(arr("formed_keep", "later") - arr("formed_reset", "later"))
    r5["describe"]["follow_500"] = {arm: float(np.mean([a[arm]["follow"] for a in ab]))
                                    for arm in ("intact", "formed_reset", "end_reset", "random_reset")}
    r5["describe"]["qual_abs_change_later"] = {arm: float(np.mean([abs(a[arm]["later"][1] - a["intact"]["later"][1])
                                                                  for a in ab]))
                                               for arm in ("formed_reset", "end_reset", "random_reset")}
    r5["describe"]["overlap_formed_end_mean"] = float(np.mean([a["overlap_formed_end"] for a in ab]))
    res["R5"] = r5
    sane_q = int((g("consol_f0_g4", "qual") >= 0.5).sum())
    sane_f = int((np.array([r["consol_f0_g4"]["formed"][0] for r in rows]) >= 0.5).sum())
    res["sanity"] = {"qual_learned": sane_q, "formed_coop": sane_f, "ok": bool(sane_q >= 36 and sane_f >= 36)}
    desc = {"D1_harm_late": {m: float(g(f"{m}_harm", "late").mean()) for m in MODES},
            "D2_coop_series_f0_g4": {m: [float(np.mean([r[f"{m}_f0_g4"]["series"][i][0] for r in rows])) for i in (9, 19, 29)]
                                     for m in MODES}}
    inter = {m: (g(f"{m}_f1_g4", "late") - g(f"{m}_f1_g0", "late")) - (g(f"{m}_f0_g4", "late") - g(f"{m}_f0_g0", "late"))
             for m in MODES}
    desc["D3_B1_interaction"] = {m: {"pos": int((v > 0).sum()), "mean": float(v.mean())} for m, v in inter.items()}
    desc["D5_reason_g0_coop_drop"] = {m: float((g(f"{m}_f0_g0", "coop") - g(f"{m}_reason_g0", "coop")).mean())
                                      for m in MODES}
    desc["D3_trust_sign_opposite"] = {m: int(((g(f"{m}_f0_g4", "T") < 0) & (g(f"{m}_f1_g4", "T") > 0)).sum()) for m in MODES}
    means = {}
    for k in rows[0]:
        if k == "ablation":
            continue
        for f in ("late", "T", "coop", "qual", "over"):
            means[f"{k}.{f}"] = round(float(g(k, f).mean()), 3)
    desc["means"] = means
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
    print(json.dumps({k: (v.get("pass", v.get("ok")) if isinstance(v, dict) else v) for k, v in res.items() if k != "describe"}))
