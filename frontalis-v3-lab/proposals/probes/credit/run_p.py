"""Run v12 (DESIGN-perception).

usage:
  python run_p.py --smoke
  python run_p.py --dev A B                    # learning sanity only; no P1/P2 quantity
  python run_p.py --seeds A B --out res.json
"""
import argparse
import json
import time

import numpy as np

import sim_perception as P
from run_g import binom_two_sided


def summarize(rows):
    d = np.array([r["delta"] for r in rows])
    p2 = np.array([r["p2"] for r in rows])
    return {"n": len(rows), "delta_pos": int((d > 0).sum()), "delta_neg": int((d < 0).sum()), "delta_zero": int((d == 0).sum()),
            "delta_mean": float(d.mean()), "delta_mid_mean": float(np.mean([r["delta_mid"] for r in rows])),
            "p2_pos": int((p2 > 0).sum()), "p2_neg": int((p2 < 0).sum()), "p2_zero": int((p2 == 0).sum()), "p2_mean": float(p2.mean()),
            "swap_featPa_cPb_mean": float(np.mean([r["p2_featPa_cPb"] for r in rows])),
            "swap_featPb_cPa_mean": float(np.mean([r["p2_featPb_cPa"] for r in rows])),
            "share_own_first": np.mean([r["share_own_first"] for r in rows], 0).round(3).tolist(),
            "share_own_second": np.mean([r["share_own_second"] for r in rows], 0).round(3).tolist()}


def run(seeds):
    res = {"seeds": [seeds[0], seeds[-1] + 1]}
    for kind in ("main", "sub"):
        for beta in (0.0, 4.0):
            res[f"{kind}_b{beta:g}"] = summarize([P.pair(beta, s, kind) for s in seeds])
    m4, m0 = res["main_b4"], res["main_b0"]
    def chance(pos, neg):            # beta = 0 control: sign test on the non-zero pairs only (PREREG-v12 addendum)
        return pos + neg == 0 or binom_two_sided(pos, pos + neg) >= 0.01
    res["P1_pass"] = bool(m4["delta_pos"] >= 29 and chance(m0["delta_pos"], m0["delta_neg"]))
    res["P2_pass"] = bool(m4["p2_pos"] >= 30 and chance(m0["p2_pos"], m0["p2_neg"]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev", nargs=2, type=int)
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.smoke:
        ag, w, picks, mid = P.life(4.0, 1, "main", 0)
        assert np.isfinite(ag.c).all() and np.isfinite(ag.W).all() and mid is not None
        print("smoke ok")
    elif a.dev:
        t0, bad, cm, own = time.time(), 0, [], []
        for s in range(*a.dev):
            for kind in ("main", "sub"):
                ag, w, picks, _ = P.life(4.0, s, kind, 0)
                bad += int(not (np.isfinite(ag.W).all() and np.isfinite(ag.c).all()))
                cm.append(np.abs(ag.c).max())
        print(f"agents={len(cm)} nonfinite={bad} max|c|={max(cm):.2f} time={time.time() - t0:.0f}s")
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(run(list(range(*a.seeds))), f, indent=1)
        print("written", a.out)
