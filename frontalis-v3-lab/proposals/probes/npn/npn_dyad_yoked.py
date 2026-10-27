"""Same evidence, different upbringing: blame on the partner's word and the speed of change (DESIGN-dyad-noobs v2.1, v18).

Upbringings (X.T0r, N.T0r), the evaluation network, the linear model, e and boost are those of npn_dyad_nn.py (v17) and
npn_acquire.py (v16), imported unchanged. Changes from v17 (DESIGN 2, 8.4):
  conflict turn: a round object (not red, recommended) + two distinct neutral objects (no red object);
  yoked condition: on conflict turns only the recommended round object (+E) is learned, the chosen object is not; on
  quiet turns the chosen object is learned. obs0 condition: the chosen object is learned on every turn.
The learned event uses noise[0]. Streams as v17: objects and recommendation [s,31]; choice [s,37]; outcomes [s,41] (one
uniform per shown object); observation [s,43] (drawn, unused); noise [s,47] (two vectors per turn).

usage:
  python npn_dyad_yoked.py --smoke
  python npn_dyad_yoked.py --proto A B
  python npn_dyad_yoked.py --dev A B
  python npn_dyad_yoked.py --seeds A B --out res.json
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from npn_acquire import D, RED, ROUND, SQUARE, BLUE, HIGH, Linear, Net, World, sgd, train
from npn_dyad_nn import MEAS_T, N_DYAD, NEUTRAL, P_CONF, ROUNDS, TAU, T_FLIP, V, measure, outcome, upbringing, vecs

WINDOWS = {"all": (0, N_DYAD), "first500": (0, 500), "w500_1000": (T_FLIP, 1000), "last500": (N_DYAD - 500, N_DYAD)}


def turns(seed):
    rk = np.random.default_rng([seed, 31])
    out = []
    for _ in range(N_DYAD):
        if rk.random() < P_CONF:
            a, b = rk.choice(len(NEUTRAL), 2, replace=False)
            out.append(("conf", [ROUNDS[int(rk.integers(4))], NEUTRAL[int(a)], NEUTRAL[int(b)]], 0))
        else:
            objs = [NEUTRAL[int(j)] for j in rk.choice(len(NEUTRAL), 3, replace=False)]
            out.append(("quiet", objs, int(rk.integers(3))))
    return out


def dyad(model, world, seed, flip, mode):
    tl = turns(seed)
    rch = np.random.default_rng([seed, 37])
    rou = np.random.default_rng([seed, 41])
    rob = np.random.default_rng([seed, 43])
    rn = np.random.default_rng([seed, 47])
    meas, log, n_roundE = {}, [], 0
    for t, (kind, objs, rec) in enumerate(tl):
        if t in MEAS_T:
            meas[t] = measure(model, world)
        with torch.no_grad():
            p = torch.softmax(TAU * model(vecs(world, objs, rec)), 0).numpy().astype(np.float64)
        k = min(int(np.searchsorted(np.cumsum(p), rch.random(), side="right")), 2)
        us = rou.random(3)
        rob.random()
        noise = rn.normal(0, 0.1 / np.sqrt(D), (2, D))
        idx = rec if (mode == "yoked" and kind == "conf") else k
        o = outcome(objs[idx], idx == rec, t, flip, us[idx])
        x = world.vec(objs[idx], idx == rec) + torch.tensor(noise[0], dtype=torch.float32)
        sgd(model, x, o)
        n_roundE += int(idx == rec and objs[idx][1] == ROUND)
        log.append((kind == "conf", k == rec))
    meas[N_DYAD] = measure(model, world)
    res = {"meas": {str(t): v for t, v in meas.items()}, "n_roundE": n_roundE}
    for name, (lo, hi) in WINDOWS.items():
        res[f"conf_follow_{name}"] = [sum(r[1] for r in log[lo:hi] if r[0]), sum(1 for r in log[lo:hi] if r[0])]
        res[f"quiet_follow_{name}"] = [sum(r[1] for r in log[lo:hi] if not r[0]), sum(1 for r in log[lo:hi] if not r[0])]
    return res


def per_seed(seed):
    w, out = World(seed), {}
    for name, form, make in (("X_net", "X", Net), ("N_net", "N", Net), ("X_lin", "X", Linear), ("N_lin", "N", Linear)):
        m = make(seed)
        train(m, w, upbringing(seed, form), seed, 0)
        row = {"sane": [V(m, w, (RED, SQUARE, HIGH)), V(m, w, (BLUE, ROUND, HIGH))]}
        for flip in (False, True):
            row["yoked_" + ("flip" if flip else "noflip")] = dyad(copy.deepcopy(m), w, seed, flip, "yoked")
        if make is Net:
            row["obs0_noflip"] = dyad(copy.deepcopy(m), w, seed, False, "obs0")
        out[name] = row
    return out


def summary(r):
    s = {}
    for name in r:
        y = r[name]["yoked_noflip"]["meas"]
        s[name] = {k: [round(y[t][k], 2) for t in ("0", "500", "1500")] for k in ("boost_conf", "V_roundE", "V_round")}
        f, nf = r[name]["yoked_flip"], r[name]["yoked_noflip"]
        s[name]["w500_1000 f/nf"] = [f["conf_follow_w500_1000"][0], nf["conf_follow_w500_1000"][0]]
        s[name]["last500 f/nf"] = [f["conf_follow_last500"][0], nf["conf_follow_last500"][0]]
        if "obs0_noflip" in r[name]:
            o = r[name]["obs0_noflip"]
            s[name]["obs0 boost_conf 0/1500, nE"] = [round(o["meas"]["0"]["boost_conf"], 2),
                                                     round(o["meas"]["1500"]["boost_conf"], 2), o["n_roundE"]]
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--proto", nargs=2, type=int)
    ap.add_argument("--dev", nargs=2, type=int)
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.smoke:
        t0 = time.time()
        r = per_seed(1)
        assert np.isfinite(r["X_net"]["yoked_noflip"]["meas"]["1500"]["boost_conf"])
        assert len({r[n]["yoked_noflip"]["n_roundE"] for n in r}) == 1
        print("smoke ok", f"{time.time() - t0:.0f}s")
    elif a.proto or a.dev:
        for s in range(*(a.proto or a.dev)):
            r = per_seed(s)
            if a.dev:
                ok = all(np.isfinite(v) for n in r for c in r[n] if c != "sane"
                         for t in r[n][c]["meas"].values() for v in t.values())
                print(s, "finite", ok, flush=True)
                continue
            print(s, json.dumps(summary(r)), flush=True)
    else:
        rows = {s: per_seed(s) for s in range(*a.seeds)}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "rows": rows}, fh, indent=1)
        print("written", a.out)
