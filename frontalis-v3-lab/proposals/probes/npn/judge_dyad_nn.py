"""Judge v17 (PREREG-v17) from result JSON files of npn_dyad_nn.py.

usage: python judge_dyad_nn.py --out judgment.json res1.json [res2.json ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "credit"))
from run_g import binom_two_sided  # noqa: E402

END = "1500"
KEYS = ("e_conf", "e_quiet", "boost_conf", "boost_quiet", "V_round", "V_roundE", "V_neutral", "V_neutralE", "V_red")


def gate(x):
    x = np.asarray(x, float)
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    p = binom_two_sided(pos, pos + neg) if pos + neg else 1.0
    return {"pos": pos, "neg": neg, "p": p, "mean": float(x.mean()), "pass": bool(pos >= 30 and p < 0.01)}


def desc(x):
    """A description: counts and mean only, no pass/fail."""
    x = np.asarray(x, float)
    return {"pos": int((x > 0).sum()), "neg": int((x < 0).sum()), "mean": float(x.mean())}


def m(r, name, t, k, f="noflip"):
    return r[name][f]["meas"][t][k]


def judge(rows):
    a = lambda f: np.array([f(r) for r in rows])
    res = {"n": len(rows)}
    sv = int(sum(r["X_net"]["sane"][0] > 0 and r["X_net"]["sane"][1] < 0 for r in rows))
    sq = int(sum(m(r, "X_net", "0", "e_quiet") > 0 for r in rows))
    res["sanity"] = {"V": sv, "e_quiet_t0": sq, "ok": bool(sv >= 36 and sq >= 36)}
    # A1: the word's effect on round objects falls over the dyad (network, X.T0r, no flip)
    res["A1"] = gate(a(lambda r: m(r, "X_net", "0", "e_conf") - m(r, "X_net", END, "e_conf")))
    n1 = int(sum(m(r, "X_net", END, "e_quiet") > 0 for r in rows))
    res["A1prime"] = {"pos": n1, "pass": bool(n1 >= 36)}
    res["A1"]["boost_conf_fell"] = int(sum(m(r, "X_net", END, "boost_conf") < m(r, "X_net", "0", "boost_conf")
                                           for r in rows))
    # A2: domain-specific discount in the evaluation level, network > linear (v16 G4b re-check in the dyad)
    gap = lambda r, n: m(r, n, END, "boost_quiet") - m(r, n, END, "boost_conf")
    res["A2"] = gate(a(lambda r: gap(r, "X_net") - gap(r, "X_lin")))
    # A3: less compliant than the mirror
    res["A3"] = gate(a(lambda r: m(r, "X_mirror", END, "e_conf") - m(r, "X_net", END, "e_conf")))
    # A4: evidence seen on observation turns changes choice (flip > no flip, round choices in the last 500 turns)
    d4 = a(lambda r: r["X_net"]["flip"]["conf_round_last500"][0] - r["X_net"]["noflip"]["conf_round_last500"][0])
    g4 = gate(d4)
    g4["undecidable"] = bool(g4["pos"] + g4["neg"] < 30)
    if g4["undecidable"]:
        g4["pass"] = None
    res["A4"] = g4
    # A5: own experience against the word from the start (first 500 turns, conflict turns, following = choosing round)
    res["A5"] = gate(a(lambda r: r["N_net"]["noflip"]["conf_follow_first500"][0]
                       - r["X_net"]["noflip"]["conf_follow_first500"][0]))
    # N.T0r: does A1 also hold without the upbringing's own experience (reading rule of DESIGN 9.1)
    nt = gate(a(lambda r: m(r, "N_net", "0", "e_conf") - m(r, "N_net", END, "e_conf")))
    res["N_A1"] = nt
    res["claim_wording"] = "結果が外れる分野で" if nt["pass"] else "自分の経験と食い違う分野で"
    dx = a(lambda r: m(r, "X_net", END, "e_conf") - m(r, "X_net", "0", "e_conf"))
    dn = a(lambda r: m(r, "N_net", END, "e_conf") - m(r, "N_net", "0", "e_conf"))
    d = {"dE_conf_mean": {"X": float(dx.mean()), "N": float(dn.mean())},
         "N_fall_minus_X_fall_e_conf": desc(dx - dn),
         "N_fall_minus_X_fall_boost_conf": desc(a(lambda r: (m(r, "N_net", "0", "boost_conf") - m(r, "N_net", END, "boost_conf"))
                                                  - (m(r, "X_net", "0", "boost_conf") - m(r, "X_net", END, "boost_conf")))),
         "A2_on_e": desc(a(lambda r: (m(r, "X_net", END, "e_quiet") - m(r, "X_net", END, "e_conf"))
                           - (m(r, "X_lin", END, "e_quiet") - m(r, "X_lin", END, "e_conf")))),
         "A1_lin": desc(a(lambda r: m(r, "X_lin", "0", "e_conf") - m(r, "X_lin", END, "e_conf"))),
         "t0_spread": {k: [float(a(lambda r: m(r, "X_net", "0", k)).mean()), float(a(lambda r: m(r, "X_net", "0", k)).min()),
                           float(a(lambda r: m(r, "X_net", "0", k)).max())] for k in ("e_conf", "boost_conf")},
         "means": {n: {f: {t: {k: float(a(lambda r: m(r, n, t, k, f)).mean()) for k in KEYS} for t in ("0", END)}
                       for f in ("noflip", "flip") if f in rows[0][n]} for n in rows[0]},
         "choices": {n: {f: {c: np.sum([r[n][f][c] for r in rows], 0).tolist()
                             for c in ("conf_round_last500", "conf_round_after_flip", "conf_follow_first500",
                                       "conf_follow_last500", "quiet_follow_first500", "quiet_follow_last500")}
                         for f in ("noflip", "flip") if f in rows[0][n]} for n in rows[0]},
         "A5_lin": desc(a(lambda r: r["N_net"]["noflip"]["conf_follow_first500"][0]
                          - r["X_lin"]["noflip"]["conf_follow_first500"][0])),
         "A4_zero_diff_seeds": int((d4 == 0).sum())}
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
        sys.exit(f"{len(seeds)} seeds: PREREG-v17 judges only when all 40 are present (use --check for a pipeline test)")
    res = judge([rows[s] for s in seeds])
    res["seeds"] = [int(seeds[0]), int(seeds[-1])]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: (v.get("pass", v.get("ok")) if isinstance(v, dict) else v) for k, v in res.items()
                      if k != "describe"}, ensure_ascii=False))
