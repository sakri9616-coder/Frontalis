"""ds1_tab.py -- mechanical tabulation for diagnostic set 1 (numpy only).

Inputs (in ../ from this file):
  p15_cells_set_IH.json, p15_cells_krr_IH.json, p15_cells_nn0_IH.json, p17_cells.json

Each input is dict name -> {"meta":..., "rows":[...]}; each row has
learner, innate, cond, fam, scale, seed and numeric fields r2_un, r2_nc,
swapd_AB, swapd_AA, swap_AB, swap_AA, kappa (len 4: A,A2,B,B2), w_A, w_B (len 5).

Writes (next to this file):
  ds1_table.json  (all numbers)
  ds1_table.md    (compact tables, 3 significant digits)
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../probes/diag5
PROBES = HERE.parent                            # .../probes

INPUTS = [
    "p15_cells_set_IH.json",
    "p15_cells_krr_IH.json",
    "p15_cells_nn0_IH.json",
    "p17_cells.json",
]


# ---------------------------------------------------------------- utilities
def finite(v):
    """Return a plain float, or None when non-finite / missing."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def med(vals):
    vals = [finite(v) for v in vals]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return finite(np.median(np.asarray(vals, dtype=float)))


def rankdata(a):
    """Average ranks (ties share the mean rank), 1-based. numpy only."""
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rho; None when undefined (n<2 or a constant vector)."""
    x = np.asarray([finite(v) for v in x], dtype=float)
    y = np.asarray([finite(v) for v in y], dtype=float)
    if x.size < 2 or x.size != y.size:
        return None
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return None
    rho = np.corrcoef(rx, ry)[0, 1]
    return finite(rho)


def g3(v):
    """3 significant digits for the markdown tables."""
    v = finite(v)
    if v is None:
        return "-"
    if v == 0:
        return "0"
    s = "%.3g" % v
    return s


# ---------------------------------------------------------------- load rows
rows = []
for fn in INPUTS:
    data = json.loads((PROBES / fn).read_text(encoding="utf-8"))
    for _name, cell in data.items():
        for r in cell.get("rows", []):
            rows.append(r)

# per-row derived quantities
derived = []
for r in rows:
    kap = np.asarray([finite(v) for v in r.get("kappa", [])], dtype=float)
    wa = np.asarray([finite(v) for v in r.get("w_A", [])], dtype=float)
    wb = np.asarray([finite(v) for v in r.get("w_B", [])], dtype=float)
    abs_kappa = float(np.mean(np.abs(kap))) if kap.size else None
    if wa.size == 5:
        m = float(np.mean(wa))
        wcv = float(np.std(wa) / m) if m != 0 else None
    else:
        wcv = None
    wl1 = float(np.sum(np.abs(wa - wb))) if wa.size and wa.size == wb.size else None
    kap_a = float(kap[0]) if kap.size >= 1 else None
    kap_b = float(kap[2]) if kap.size >= 3 else None
    kd = abs(kap_a - kap_b) if (kap_a is not None and kap_b is not None) else None
    derived.append({
        "learner": r.get("learner"),
        "innate": r.get("innate"),
        "cond": r.get("cond"),
        "fam": r.get("fam"),
        "scale": r.get("scale"),
        "seed": r.get("seed"),
        "r2_un": finite(r.get("r2_un")),
        "r2_nc": finite(r.get("r2_nc")),
        "swapd_AB": finite(r.get("swapd_AB")),
        "swapd_AA": finite(r.get("swapd_AA")),
        "j3diff": (r.get("swapd_AB") - r.get("swapd_AA"))
        if (finite(r.get("swapd_AB")) is not None and finite(r.get("swapd_AA")) is not None) else None,
        "abs_kappa": finite(abs_kappa),
        "wcv": finite(wcv),
        "wl1": finite(wl1),
        "kappa_diff": finite(kd),
    })


# ---------------------------------------------------------------- 1) per cell
groups = {}
for d in derived:
    key = (d["learner"], d["innate"], d["cond"], d["fam"], d["scale"])
    groups.setdefault(key, []).append(d)

per_cell = {}
for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
    g = groups[key]
    j3 = [d["j3diff"] for d in g if d["j3diff"] is not None]
    per_cell["|".join(str(x) for x in key)] = {
        "learner": key[0], "innate": key[1], "cond": key[2], "fam": key[3], "scale": key[4],
        "n": len(g),
        "r2_un_med": med([d["r2_un"] for d in g]),
        "r2_nc_med": med([d["r2_nc"] for d in g]),
        "swapd_AB_med": med([d["swapd_AB"] for d in g]),
        "swapd_AA_med": med([d["swapd_AA"] for d in g]),
        "J3diff_med": med(j3),
        "J3diff_gt0": int(np.sum(np.asarray(j3) > 0)) if j3 else 0,
        "abs_kappa_mean_med": med([d["abs_kappa"] for d in g]),
        "wCV_med": med([d["wcv"] for d in g]),
        "wL1_med": med([d["wl1"] for d in g]),
    }


# ---------------------------------------------------------------- 2) pairs
# seed-indexed values per (learner, innate, cond, fam, scale)
idx = {}
for d in derived:
    k = (d["learner"], d["innate"], d["cond"], d["fam"], d["scale"])
    idx.setdefault(k, {})[d["seed"]] = d


def paired_summary(pairs):
    """pairs: list of per-seed differences. -> {n, gt0, med}."""
    vals = [finite(v) for v in pairs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "gt0": 0, "med": None}
    arr = np.asarray(vals, dtype=float)
    return {"n": int(arr.size), "gt0": int(np.sum(arr > 0)), "med": finite(np.median(arr))}


# 2a) krr - set  (same innate, cond, fam, scale; paired on seed)
krr_minus_set = {}
keysA = sorted({(d["innate"], d["cond"], d["fam"], d["scale"]) for d in derived},
               key=lambda k: tuple(str(x) for x in k))
for (inn, cond, fam, scale) in keysA:
    gk = idx.get(("krr", inn, cond, fam, scale))
    gs = idx.get(("set", inn, cond, fam, scale))
    if not gk or not gs:
        continue
    common = sorted(set(gk) & set(gs))
    diffs = [gk[s]["r2_nc"] - gs[s]["r2_nc"] for s in common
             if gk[s]["r2_nc"] is not None and gs[s]["r2_nc"] is not None]
    krr_minus_set["%s|%s|%s|%s" % (inn, cond, fam, scale)] = paired_summary(diffs)

# 2b) set - nn0
set_minus_nn0 = {}
for (inn, cond, fam, scale) in keysA:
    gs = idx.get(("set", inn, cond, fam, scale))
    gn = idx.get(("nn0", inn, cond, fam, scale))
    if not gs or not gn:
        continue
    common = sorted(set(gs) & set(gn))
    diffs = [gs[s]["r2_nc"] - gn[s]["r2_nc"] for s in common
             if gs[s]["r2_nc"] is not None and gn[s]["r2_nc"] is not None]
    set_minus_nn0["%s|%s|%s|%s" % (inn, cond, fam, scale)] = paired_summary(diffs)

# 2c) within learner: C-SLp - C-off  (r2_nc and r2_un)
keysB = sorted({(d["learner"], d["innate"], d["fam"], d["scale"]) for d in derived},
               key=lambda k: tuple(str(x) for x in k))
slp_minus_off_nc = {}
slp_minus_off_un = {}
for (learner, inn, fam, scale) in keysB:
    gslp = idx.get((learner, inn, "C-SLp", fam, scale))
    goff = idx.get((learner, inn, "C-off", fam, scale))
    if not gslp or not goff:
        continue
    common = sorted(set(gslp) & set(goff))
    dnc = [gslp[s]["r2_nc"] - goff[s]["r2_nc"] for s in common
           if gslp[s]["r2_nc"] is not None and goff[s]["r2_nc"] is not None]
    dun = [gslp[s]["r2_un"] - goff[s]["r2_un"] for s in common
           if gslp[s]["r2_un"] is not None and goff[s]["r2_un"] is not None]
    lab = "%s|%s|%s|%s" % (learner, inn, fam, scale)
    slp_minus_off_nc[lab] = paired_summary(dnc)
    slp_minus_off_un[lab] = paired_summary(dun)


# ---------------------------------------------------------------- 3) Spearman
spearman_out = {}
for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
    learner, inn, cond, fam, scale = key
    if cond != "C-SLp":
        continue
    g = groups[key]
    if len(g) < 8:
        continue
    g = sorted(g, key=lambda d: str(d["seed"]))
    j3 = [d["j3diff"] for d in g]
    wl1 = [d["wl1"] for d in g]
    kd = [d["kappa_diff"] for d in g]
    spearman_out["|".join(str(x) for x in key)] = {
        "learner": learner, "innate": inn, "cond": cond, "fam": fam, "scale": scale,
        "n": len(g),
        "rho_J3diff_wL1": spearman(j3, wl1),
        "rho_J3diff_kappaDiff": spearman(j3, kd),
    }


# ---------------------------------------------------------------- write JSON
payload = {
    "per_cell": per_cell,
    "pairs": {
        "r2_nc_krr_minus_set": krr_minus_set,
        "r2_nc_set_minus_nn0": set_minus_nn0,
        "r2_nc_C-SLp_minus_C-off_within_learner": slp_minus_off_nc,
        "r2_un_C-SLp_minus_C-off_within_learner": slp_minus_off_un,
    },
    "spearman": spearman_out,
}
(HERE / "ds1_table.json").write_text(
    json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")


# ---------------------------------------------------------------- write MD
L = []
L.append("# ds1 diagnostic table")
L.append("")
L.append("Values are medians unless stated; markdown shows 3 significant digits.")
L.append("")

L.append("## 1. Per-cell medians  (learner|innate|cond|fam|scale)")
L.append("")
L.append("| learner | innate | cond | fam | scale | n | r2_un | r2_nc | swapd_AB | swapd_AA | J3diff | J3diff>0 | mean|kappa| | wCV | L1(wA-wB) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, v in per_cell.items():
    L.append("| %s | %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %d | %s | %s | %s |" % (
        v["learner"], v["innate"], v["cond"], v["fam"], v["scale"], v["n"],
        g3(v["r2_un_med"]), g3(v["r2_nc_med"]), g3(v["swapd_AB_med"]), g3(v["swapd_AA_med"]),
        g3(v["J3diff_med"]), v["J3diff_gt0"], g3(v["abs_kappa_mean_med"]),
        g3(v["wCV_med"]), g3(v["wL1_med"])))
L.append("")

L.append("## 2. Paired-by-seed differences")
L.append("")
L.append("### 2a. r2_nc(krr) - r2_nc(set)   [paired within innate|cond|fam|scale]")
L.append("")
L.append("| innate | cond | fam | scale | n | >0 | median |")
L.append("|---|---|---|---|---|---|---|")
for lab, v in krr_minus_set.items():
    a, b, c, d = lab.split("|")
    L.append("| %s | %s | %s | %s | %d | %d | %s |" % (a, b, c, d, v["n"], v["gt0"], g3(v["med"])))
L.append("")

L.append("### 2b. r2_nc(set) - r2_nc(nn0)   [paired within innate|cond|fam|scale]")
L.append("")
L.append("| innate | cond | fam | scale | n | >0 | median |")
L.append("|---|---|---|---|---|---|---|")
for lab, v in set_minus_nn0.items():
    a, b, c, d = lab.split("|")
    L.append("| %s | %s | %s | %s | %d | %d | %s |" % (a, b, c, d, v["n"], v["gt0"], g3(v["med"])))
L.append("")

L.append("### 2c. Within learner: r2_nc(C-SLp) - r2_nc(C-off)")
L.append("")
L.append("| learner | innate | fam | scale | n | >0 | median |")
L.append("|---|---|---|---|---|---|---|")
for lab, v in slp_minus_off_nc.items():
    a, b, c, d = lab.split("|")
    L.append("| %s | %s | %s | %s | %d | %d | %s |" % (a, b, c, d, v["n"], v["gt0"], g3(v["med"])))
L.append("")

L.append("### 2d. Within learner: r2_un(C-SLp) - r2_un(C-off)")
L.append("")
L.append("| learner | innate | fam | scale | n | >0 | median |")
L.append("|---|---|---|---|---|---|---|")
for lab, v in slp_minus_off_un.items():
    a, b, c, d = lab.split("|")
    L.append("| %s | %s | %s | %s | %d | %d | %s |" % (a, b, c, d, v["n"], v["gt0"], g3(v["med"])))
L.append("")

L.append("## 3. Spearman across seeds within C-SLp cells (n >= 8)")
L.append("")
L.append("| learner | innate | cond | fam | scale | n | rho(J3diff, L1(wA-wB)) | rho(J3diff, |kappa_A-kappa_B|) |")
L.append("|---|---|---|---|---|---|---|---|")
for k, v in spearman_out.items():
    L.append("| %s | %s | %s | %s | %s | %d | %s | %s |" % (
        v["learner"], v["innate"], v["cond"], v["fam"], v["scale"], v["n"],
        g3(v["rho_J3diff_wL1"]), g3(v["rho_J3diff_kappaDiff"])))
L.append("")

(HERE / "ds1_table.md").write_text("\n".join(L), encoding="utf-8")

print(str((HERE / "ds1_table.json")))
print(str((HERE / "ds1_table.md")))
