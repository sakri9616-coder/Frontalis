"""Judge v18 (PREREG-v18) from result JSON files of npn_dyad_yoked.py.

usage: python judge_dyad_yoked.py --out judgment.json res1.json [res2.json ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "credit"))
from run_g import binom_two_sided  # noqa: E402

KEYS = ("e_conf", "e_quiet", "boost_conf", "boost_quiet", "V_round", "V_roundE", "V_neutral", "V_neutralE", "V_red")
R_MIN = 0.05


def gate(x):
    x = np.asarray(x, float)
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    p = binom_two_sided(pos, pos + neg) if pos + neg else 1.0
    return {"pos": pos, "neg": neg, "p": p, "mean": float(x.mean()), "pass": bool(pos >= 30 and p < 0.01)}


def three_way(x):
    """Pass, reverse pass, or neither (DESIGN 3: C1, C1', C3)."""
    g = gate(x)
    g["reverse_pass"] = bool(g["neg"] >= 30 and g["p"] < 0.01)
    g["verdict"] = "pass" if g["pass"] else ("reverse" if g["reverse_pass"] else "neither")
    return g


def desc(x):
    x = np.asarray(x, float)
    return {"pos": int((x > 0).sum()), "neg": int((x < 0).sum()), "mean": float(x.mean())}


def m(r, name, t, k, c="yoked_noflip"):
    return r[name][c]["meas"][t][k]


def judge(rows):
    a = lambda f: np.array([f(r) for r in rows])
    res = {"n": len(rows)}
    sv = int(sum(r["X_net"]["sane"][0] > 0 and r["X_net"]["sane"][1] < 0 for r in rows))
    sq = int(sum(m(r, "X_net", "0", "e_quiet") > 0 for r in rows))
    eq = int(sum(len({r[n][c]["n_roundE"] for n in r for c in ("yoked_noflip", "yoked_flip")}) == 1 for r in rows))
    res["sanity"] = {"V": sv, "e_quiet_t0": sq, "equal_evidence": eq, "ok": bool(sv >= 36 and sq >= 36 and eq == len(rows))}
    # C1: same evidence, fall of boost_conf from t = 0 to t = 500, N - X (network, yoked, no flip)
    fall = lambda r, n, t="500": m(r, n, "0", "boost_conf") - m(r, n, t, "boost_conf")
    res["C1"] = three_way(a(lambda r: fall(r, "N_net") - fall(r, "X_net")))

    # C1': share of the change of V(round + E) attributed to the word, r = d boost_conf / d V(round + E), t = 0 -> 500
    def ratio(r, n):
        dv = m(r, n, "500", "V_roundE") - m(r, n, "0", "V_roundE")
        db = m(r, n, "500", "boost_conf") - m(r, n, "0", "boost_conf")
        return db / dv if abs(dv) >= R_MIN else np.nan

    c1p = a(lambda r: (ratio(r, "N_net") - ratio(r, "X_net")) - (ratio(r, "N_lin") - ratio(r, "X_lin")))
    ok = ~np.isnan(c1p)
    res["C1prime"] = three_way(c1p[ok])
    res["C1prime"]["excluded"] = int((~ok).sum())
    # C2: domain-specific discount at t = 1500, network - linear (X, yoked, no flip)
    gap = lambda r, n: m(r, n, "1500", "boost_quiet") - m(r, n, "1500", "boost_conf")
    res["C2"] = gate(a(lambda r: gap(r, "X_net") - gap(r, "X_lin")))

    # C3: round choices on conflict turns in t = 500..999, (flip - no flip), N - X (network, yoked)
    gain = lambda r, n, w="w500_1000": (r[n]["yoked_flip"][f"conf_follow_{w}"][0]
                                        - r[n]["yoked_noflip"][f"conf_follow_{w}"][0])
    c3 = three_way(a(lambda r: gain(r, "N_net") - gain(r, "X_net")))
    rate = lambda r, n: r[n]["yoked_flip"]["conf_follow_w500_1000"][0] / max(r[n]["yoked_flip"]["conf_follow_w500_1000"][1], 1)
    c3["window_rate_flip"] = {"X": float(a(lambda r: rate(r, "X_net")).mean()), "N": float(a(lambda r: rate(r, "N_net")).mean())}
    c3["window_saturated"] = bool(min(c3["window_rate_flip"].values()) > 0.7)
    vx500 = a(lambda r: m(r, "X_net", "500", "V_roundE") - m(r, "N_net", "500", "V_roundE"))
    c3["V_roundE500_X_ge_N"] = int((vx500 >= 0).sum())
    c3["may_read_slowed_by_upbringing"] = bool(c3["pass"] and c3["V_roundE500_X_ge_N"] >= 30)
    res["C3"] = c3

    d = {"C1_t1500": desc(a(lambda r: fall(r, "N_net", "1500") - fall(r, "X_net", "1500"))),
         "C1_lin": desc(a(lambda r: fall(r, "N_lin") - fall(r, "X_lin"))),
         "C1_on_e": desc(a(lambda r: (m(r, "N_net", "0", "e_conf") - m(r, "N_net", "500", "e_conf"))
                           - (m(r, "X_net", "0", "e_conf") - m(r, "X_net", "500", "e_conf")))),
         "ratio_mean": {n: float(np.nanmean(a(lambda r: ratio(r, n)))) for n in ("X_net", "N_net", "X_lin", "N_lin")},
         "C3_lin": desc(a(lambda r: gain(r, "N_lin") - gain(r, "X_lin"))),
         "C3_net_minus_lin": desc(a(lambda r: (gain(r, "N_net") - gain(r, "X_net")) - (gain(r, "N_lin") - gain(r, "X_lin")))),
         "C3_last500": desc(a(lambda r: gain(r, "N_net", "last500") - gain(r, "X_net", "last500"))),
         "obs0": {n: {"boost_conf_0": float(a(lambda r: m(r, n, "0", "boost_conf", "obs0_noflip")).mean()),
                      "boost_conf_1500": float(a(lambda r: m(r, n, "1500", "boost_conf", "obs0_noflip")).mean()),
                      "fall_pos": desc(a(lambda r: m(r, n, "0", "boost_conf", "obs0_noflip")
                                         - m(r, n, "1500", "boost_conf", "obs0_noflip"))),
                      "n_roundE_mean": float(a(lambda r: r[n]["obs0_noflip"]["n_roundE"]).mean()),
                      "fall_per_event": float(np.mean([(m(r, n, "0", "boost_conf", "obs0_noflip")
                                                         - m(r, n, "1500", "boost_conf", "obs0_noflip"))
                                                        / max(r[n]["obs0_noflip"]["n_roundE"], 1) for r in rows]))}
                  for n in ("X_net", "N_net")},
         "means": {n: {c: {t: {k: float(a(lambda r: m(r, n, t, k, c)).mean()) for k in KEYS} for t in ("0", "500", "1000", "1500")}
                       for c in rows[0][n] if c != "sane"} for n in rows[0]},
         "choices": {n: {c: {w: np.sum([r[n][c][w] for r in rows], 0).tolist() for w in rows[0][n][c]
                             if w.startswith(("conf_follow", "quiet_follow"))}
                         for c in rows[0][n] if c != "sane"} for n in rows[0]},
         "n_roundE_yoked_mean": float(a(lambda r: r["X_net"]["yoked_noflip"]["n_roundE"]).mean())}
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
        sys.exit(f"{len(seeds)} seeds: PREREG-v18 judges only when all 40 are present (use --check for a pipeline test)")
    res = judge([rows[s] for s in seeds])
    res["seeds"] = [int(seeds[0]), int(seeds[-1])]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: (v.get("verdict", v.get("pass", v.get("ok"))) if isinstance(v, dict) else v)
                      for k, v in res.items() if k != "describe"}, ensure_ascii=False))
