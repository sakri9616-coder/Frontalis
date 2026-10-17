"""Apply the draft-4 judgement rules (PREREG draft 4, sections 5, 6, 8) to cell results. On dev seeds the sign rule is
scaled to 8 of 10; on 40 seeds it is 30 of 40. Floors as in PREREG section 5."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

from measure4 import judge

EXPECT = {  # PREREG draft 4 section 6
    "PL-struct": {"J2-a": True, "J2-b": True, "repro": True},
    "PL-rand": {"J2-a": True, "J2-b": True, "J4-a": True, "J4-b": True, "repro": True},
    "PL-global": {"J2-a": False, "J2-b": False, "J4-a": False, "J4-b": False, "repro": False},
    "PL-noise": {"J1-a": False, "J2-a": False, "J2-b": False, "J4-a": False, "J4-b": False, "J3": False, "repro": False},
    "PL-state": {"J3": True},
    "PL-state0": {"J3": False},
}


def frac(n):
    return 0.8 if n <= 10 else 0.75


def contrasts(rows: List[dict]) -> Dict[str, dict]:
    n = len(rows); fr = frac(n)
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    med = lambda a: float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")
    out = {}
    out["J1-a"] = judge(list(g("D_AB_un") - g("D_AA_un")), 0.25 * med(g("D_AA_un")), fr)
    out["J2-a"] = judge(list(g("Drho_AB_un") - g("Drho_AA_un")), 0.25 * med(g("Drho_AA_un")), fr)
    out["J2-b"] = judge(list(g("Drho_AB_nc") - g("Drho_AA_nc")), 0.25 * med(g("Drho_AA_nc")), fr)
    out["J3"] = judge(list(g("swapd_AB") - g("swapd_AA")), 0.25 * med(g("swapd_AA")), fr)
    if "xp_other_un" in rows[0]:
        out["J4-a"] = judge(list(g("xp_other_un") - g("xp_same_un")), 0.05, fr)
        out["J4-b"] = judge(list(g("xp_other_nc") - g("xp_same_nc")), 0.05, fr)
    else:
        for k in ("J4-a", "J4-b"):
            out[k] = dict(passed=False, n_pos=0, n=n, median=float("nan"), floor=0.05, note="undefined (no learned representation)")
    out["J5-a"] = judge(list(g("r2_nc")), 0.05, fr)
    out["repro"] = judge(list(g("repro_un")), 0.05, fr)
    out["desc"] = {k: med(g(k)) for k in ("D_AB_un", "D_AA_un", "Drho_AB_un", "Drho_AA_un", "Drho_AB_nc", "Drho_AA_nc", "swap_AB", "swap_AA",
                                          "r2_un", "r2_nc", "repro_un", "repro_nc", "c_AB_un", "c_sh_un", "c_AA_un", "c_AB_nc", "c_sh_nc",
                                          "p_AB", "p_sh", "xp_other_un", "xp_same_un", "xp_other_nc", "xp_same_nc", "swapd_AB", "swapd_AA", "conflict_A", "conflict_B", "inter_AB", "inter_AA", "r2_rm", "r2_rm_fresh")}
    return out


def by_cell(results: Dict[str, dict]):
    cells = defaultdict(list)
    for key, d in results.items():
        for r in d["rows"]:
            cells[(r["learner"], r["innate"], r["cond"], r["fam"], r["scale"])].append(r)
    return cells


def paired(rows_x, rows_y, kx, ky, floor):
    ix = {r["seed"]: r for r in rows_x}; iy = {r["seed"]: r for r in rows_y}
    s = sorted(set(ix) & set(iy))
    return judge([ix[k][kx] - iy[k][ky] for k in s], floor, frac(len(s)))


def report(results: Dict[str, dict]) -> dict:
    cells = by_cell(results)
    out = {}
    for key, rows in sorted(cells.items()):
        out["|".join(key)] = contrasts(rows)
    # J1-b: vs C-off of the same learner / innate / family / scale
    for (L, I, C, Fm, S), rows in cells.items():
        off = cells.get((L, I, "C-off", Fm, S))
        if C != "C-off" and off:
            ix = {r["seed"]: r for r in off}
            d = [(r["D_AB_un"] - r["D_AA_un"]) - (ix[r["seed"]]["D_AB_un"] - ix[r["seed"]]["D_AA_un"]) for r in rows if r["seed"] in ix]
            out["|".join((L, I, C, Fm, S))]["J1-b"] = judge(d, 0.25 * float(np.median([r["D_AA_un"] for r in rows])), frac(len(d)))
        # J5-c vs memory/capacity controls, J4-d vs F6m and F5
        if L in ("set", "rel"):
            for ctl in ("krr", "hb", "rf"):
                cr = cells.get((ctl, I, C, Fm, S))
                if cr:
                    out["|".join((L, I, C, Fm, S))][f"J5-c_{ctl}"] = paired(rows, cr, "r2_nc", "r2_nc", 0.02)
            if Fm in ("F1", "F2") and "xp_other_nc" in rows[0]:
                for ref in ("F6m", "F5"):
                    rr = cells.get((L, I, C, ref, S))
                    if rr:
                        ix = {r["seed"]: r for r in rr}
                        d = [(r["xp_other_nc"] - r["xp_same_nc"]) - (ix[r["seed"]]["xp_other_nc"] - ix[r["seed"]]["xp_same_nc"]) for r in rows if r["seed"] in ix]
                        out["|".join((L, I, C, Fm, S))][f"J4-d_{ref}"] = judge(d, 0.05, frac(len(d)))
    # calibration
    cal = defaultdict(list)
    for key, d in results.items():
        for r in d["rows"]:
            for name, cr in r.get("calib", {}).items():
                cal[(r["learner"], r["innate"], r["cond"], r["fam"], r["scale"], name)].append(cr)
    calout = {}
    for k, rows in sorted(cal.items()):
        c = contrasts(rows)
        exp = dict(EXPECT.get(k[-1], {}))
        if k[2] == "C-off" and k[-1] == "PL-state":
            exp["J3"] = False      # in C-off the targets do not depend on the evaluative state: J3 is a built-in null
        calout["|".join(k)] = {j: dict(passed=c[j]["passed"], expected=e, ok=(c[j]["passed"] == e), n_pos=c[j]["n_pos"], median=c[j]["median"], floor=c[j]["floor"])
                               for j, e in exp.items() if j in c}
    out["_calibration"] = calout
    return out


if __name__ == "__main__":
    res = {}
    for p in sys.argv[1:-1]:
        res.update(json.load(open(p, encoding="utf-8")))
    rep = report(res)
    json.dump(rep, open(sys.argv[-1], "w", encoding="utf-8"), indent=1, default=float)
