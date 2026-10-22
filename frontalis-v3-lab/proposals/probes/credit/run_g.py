"""Run DESIGN-dyad v5.2 G3a and G3b on a given seed range and write a JSON result.

usage:
  python run_g.py --smoke                      # invariants only; prints no G quantity
  python run_g.py --seeds A B --out res.json   # seeds A..B-1 (40 per arm)
"""
import argparse
import json
import math

import numpy as np

import sim_credit as S


def binom_two_sided(k, n, p=0.5):
    """Exact two-sided binomial test p-value."""
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] + 1e-15))


def two_prop_p(k1, n1, k2, n2):
    """Two-sided two-proportion z test p-value."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)) or 1e-12
    return math.erfc(abs(k1 / n1 - k2 / n2) / se / math.sqrt(2))


def smoke():
    rng = np.random.default_rng(12345)
    ag, w = S.Agent(4, 4.0), S.World(4, np.ones(4))
    for _ in range(500):
        pres, o = w.event(rng)
        c_before = ag.c.copy()
        dc = ag.learn(pres, o)
        if pres.sum():
            z = np.where(pres == 1, ag.beta * c_before, -np.inf)
            a = np.exp(z - z[pres == 1].max()); a /= a.sum()
            assert abs((a * o)[pres == 1].sum() - o) < 1e-9, "credit must be conserved"
        assert np.all(dc[pres == 0] == 0), "absent aspects must not learn"
        assert np.all(np.abs(ag.c) < 10)
    ag0 = S.Agent(4, 0.0)
    S.live(ag0, w, 4000, np.random.default_rng(7))
    assert np.ptp(ag0.c) < 0.3, "beta = 0 must stay symmetric"
    print("smoke ok")


def run(seeds):
    res = {"seeds": [seeds[0], seeds[-1] + 1],
           "beta_star_late": {"K3": S.beta_star_late(3, [1, -1, 1]), "K5": S.beta_star_late(5, [-1, 1, 1, 1, 1])}}
    for K in (3, 5):
        for beta in S.BETAS:
            opp = [int(a != b) for a, b in (S.g3a(K, beta, s) for s in seeds)]
            rt = [int(a != b) for a, b in (S.g3a_retest(K, beta, s) for s in seeds)]
            nb = [x for s in seeds for x in S.g3b(K, beta, s)]
            res[f"G3a_K{K}_b{beta:g}"] = {"opposite": sum(opp), "n": len(opp),
                                          "retest_opposite": sum(rt), "p_retest": binom_two_sided(sum(rt), len(rt))}
            res[f"G3b_K{K}_b{beta:g}"] = {"not_blocked": int(sum(nb)), "n": len(nb)}
        for beta in (2.0, 4.0):
            g, b0 = res[f"G3a_K{K}_b{beta:g}"], res[f"G3b_K{K}_b0"]
            g["pass"] = bool(g["opposite"] >= 0.75 * g["n"] and g["p_retest"] >= 0.01
                             and res[f"G3a_K{K}_b0"]["p_retest"] >= 0.01)
            b = res[f"G3b_K{K}_b{beta:g}"]
            b["p_vs_beta0"] = two_prop_p(b["not_blocked"], b["n"], b0["not_blocked"], b0["n"])
            b["pass"] = bool(b["p_vs_beta0"] < 0.01 and b["not_blocked"] / b["n"] > b0["not_blocked"] / b0["n"])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.smoke:
        smoke()
    else:
        r = run(list(range(a.seeds[0], a.seeds[1])))
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=1)
        print("written", a.out)
