"""Run v11 F-G3a and F-align. usage: python run_f.py --seeds A B --out res.json | --smoke"""
import argparse, json
import numpy as np
import sim_features as F
from run_g import binom_two_sided, two_prop_p


def opp(a, b):
    return int(a * b < 0)        # opposite only when the product is negative; zeros counted separately


def run(seeds):
    res = {"seeds": [seeds[0], seeds[-1] + 1]}
    for beta in (0.0, 2.0, 4.0):
        m = [F.f_g3a(beta, s) for s in seeds]; t = [F.f_g3a_retest(beta, s) for s in seeds]
        res[f"b{beta:g}"] = {"opposite": sum(opp(*x) for x in m), "zeros": sum(int(x[0] == 0 or x[1] == 0) for x in m),
                             "retest_opposite": sum(opp(*x) for x in t), "retest_zeros": sum(int(x[0] == 0 or x[1] == 0) for x in t),
                             "n": len(seeds)}
    b0 = res["b0"]
    for beta in (2.0, 4.0):
        g = res[f"b{beta:g}"]
        g["p_vs_retest"] = two_prop_p(g["opposite"], g["n"], g["retest_opposite"], g["n"])
        g["pass"] = bool(g["opposite"] >= 0.75 * g["n"] and g["retest_opposite"] <= 0.25 * g["n"]
                         and g["opposite"] > g["retest_opposite"] and g["p_vs_retest"] < 0.01
                         and binom_two_sided(b0["opposite"], b0["n"]) >= 0.01)
    res["units_at_calibration_floor"] = int(sum((F.grow(0.0, F.world(s), s, s, partner=2).s <= 1e-6).sum() for s in seeds))
    al = np.array([F.f_align(s) for s in seeds])
    res["align_best_corr_per_aspect"] = {"median": np.median(al, 0).round(3).tolist(), "min": al.min(0).round(3).tolist()}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs=2, type=int); ap.add_argument("--out"); a = ap.parse_args()
    if a.smoke:
        ag = F.grow(4.0, F.world(1), 1, 1, partner=2)
        assert np.isfinite(ag.c).all() and np.abs(ag.c).max() < 10 and ag.s.min() > 0
        print("smoke ok")
    else:
        json.dump(run(list(range(*a.seeds))), open(a.out, "w", encoding="utf-8"), indent=1); print("written", a.out)
