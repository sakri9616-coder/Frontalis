"""Selective trust in behaviour: does the partner's word lose its effect only where it is wrong (DESIGN-dyad-nn v2.1, v17).

World, objects, word E, network V, linear model, SGD and mirror are those of npn_acquire.py (v16, imported unchanged).
Upbringing (3000 passive events, stage 0): feature events 0.8 from SEEN with the X outcomes (red -> P+, round -> P-,
0.8); partner events 0.2 praise a neutral object (blue/green x square/triangle x high/low) followed by P+ (0.8). The
partner never praises round objects (T0r). N.T0r: the same events with every feature outcome set to 0.
Dyad (1500 turns): three objects, the partner recommends one (its vector gets E). Conflict turn (0.5): a round object
(not red), a red object (not round), a neutral object; the partner recommends the round one. Quiet turn: three distinct
neutral objects, one recommended at random. Outcomes: red -> P+ (0.8), round -> P- (0.8; P+ from turn 500 in flip),
neutral -> P+ (0.8) if recommended, else 0. Choice: softmax(tau V) over the noise-free vectors, one uniform per turn.
Learning: the chosen object's vector (with E if recommended, plus noise) toward its outcome; on observation turns (0.3)
also the recommended object (+E) if it was not chosen. Mirror: target + 1 whenever the input carries E.
Streams: objects and recommendation [s,31]; choice [s,37]; outcomes [s,41] (one uniform per shown object); observation
[s,43]; noise [s,47] (two vectors per turn). Upbringing: event choice [s,3,0], outcomes [s,5,0], noise [s,7,0].

usage:
  python npn_dyad_nn.py --smoke
  python npn_dyad_nn.py --proto A B          # prototype print (prototype seeds only)
  python npn_dyad_nn.py --dev A B            # finiteness only
  python npn_dyad_nn.py --seeds A B --out res.json
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from npn_acquire import (BLUE, CLIP, D, GREEN, HIGH, LOW, LR, P, P_PARTNER, RED, ROUND, SEEN, SQUARE, TRI, Linear, Net,
                         World, feature_o, sgd, train)

N_UP, N_DYAD, T_FLIP = 3000, 1500, 500
TAU, P_CONF, P_OBS = 3.0, 0.5, 0.3
NEUTRAL = [(c, s, t) for c in (BLUE, GREEN) for s in (SQUARE, TRI) for t in (HIGH, LOW)]
ROUNDS = [(c, ROUND, t) for c in (BLUE, GREEN) for t in (HIGH, LOW)]
REDS = [(RED, s, t) for s in (SQUARE, TRI) for t in (HIGH, LOW)]
MEAS_T = (0, 500, 1000, 1500)
assert (LR, CLIP) == (0.01, 1.0)


def upbringing(seed, form):
    rc = np.random.default_rng([seed, 3, 0])
    ro = np.random.default_rng([seed, 5, 0])
    ev = []
    for _ in range(N_UP):
        if rc.random() < P_PARTNER:
            ev.append((NEUTRAL[int(rc.integers(len(NEUTRAL)))], True, 1.0 if ro.random() < P else 0.0))
        else:
            obj = SEEN[int(rc.integers(len(SEEN)))]
            o = feature_o(obj, "X", ro)
            ev.append((obj, False, o if form == "X" else 0.0))
    return ev


def turns(seed):
    """The dyad's object stream: (kind, objects, index of the recommended object). Independent of the model."""
    rk = np.random.default_rng([seed, 31])
    out = []
    for _ in range(N_DYAD):
        if rk.random() < P_CONF:
            objs = [ROUNDS[int(rk.integers(4))], REDS[int(rk.integers(4))], NEUTRAL[int(rk.integers(8))]]
            out.append(("conf", objs, 0))
        else:
            objs = [NEUTRAL[int(j)] for j in rk.choice(len(NEUTRAL), 3, replace=False)]
            out.append(("quiet", objs, int(rk.integers(3))))
    return out


def outcome(obj, rec, t, flip, u):
    if obj[0] == RED:
        return 1.0 if u < P else 0.0
    if obj[1] == ROUND:
        return (1.0 if u < P else 0.0) if (flip and t >= T_FLIP) else (-1.0 if u < P else 0.0)
    return (1.0 if u < P else 0.0) if rec else 0.0


def vecs(world, objs, rec):
    return torch.stack([world.vec(o, k == rec) for k, o in enumerate(objs)])


def V(model, world, obj, praise=False):
    with torch.no_grad():
        return float(model(world.vec(obj, praise)))


def measure(model, world):
    vr = {o: (V(model, world, o), V(model, world, o, True)) for o in NEUTRAL + ROUNDS + REDS}

    def e(rec, comp):
        d0 = vr[rec][0] - vr[comp][0]
        d1 = vr[rec][1] - vr[comp][0]
        return 1 / (1 + np.exp(-TAU * d1)) - 1 / (1 + np.exp(-TAU * d0))

    m = lambda xs: float(np.mean(xs))
    return {
        "e_conf": m([e(r, c) for r in ROUNDS for c in NEUTRAL]),
        "e_quiet": m([e(r, c) for r in NEUTRAL for c in NEUTRAL if c != r]),
        "boost_conf": m([vr[o][1] - vr[o][0] for o in ROUNDS]),
        "boost_quiet": m([vr[o][1] - vr[o][0] for o in NEUTRAL]),
        "V_round": m([vr[o][0] for o in ROUNDS]), "V_roundE": m([vr[o][1] for o in ROUNDS]),
        "V_neutral": m([vr[o][0] for o in NEUTRAL]), "V_neutralE": m([vr[o][1] for o in NEUTRAL]),
        "V_red": m([vr[o][0] for o in REDS]),
    }


def dyad(model, world, seed, flip, mirror=False):
    tl = turns(seed)
    rch = np.random.default_rng([seed, 37])
    rou = np.random.default_rng([seed, 41])
    rob = np.random.default_rng([seed, 43])
    rn = np.random.default_rng([seed, 47])
    meas, log = {}, []
    for t, (kind, objs, rec) in enumerate(tl):
        if t in MEAS_T:
            meas[t] = measure(model, world)
        with torch.no_grad():
            p = torch.softmax(TAU * model(vecs(world, objs, rec)), 0).numpy().astype(np.float64)
        u = rch.random()
        k = min(int(np.searchsorted(np.cumsum(p), u, side="right")), 2)
        us = rou.random(3)
        obs = rob.random() < P_OBS
        noise = rn.normal(0, 0.1 / np.sqrt(D), (2, D))
        learn = [k] + ([rec] if (obs and rec != k) else [])
        for j, idx in enumerate(learn):
            o = outcome(objs[idx], idx == rec, t, flip, us[idx])
            x = world.vec(objs[idx], idx == rec) + torch.tensor(noise[j], dtype=torch.float32)
            sgd(model, x, o + (1.0 if (mirror and idx == rec) else 0.0))
        log.append((kind == "conf", k == rec, objs[k][1] == ROUND and objs[k][0] != RED))
    meas[N_DYAD] = measure(model, world)

    def rate(lo, hi, col):
        sel = [r for r in log[lo:hi] if r[0]]
        return [sum(r[col] for r in sel), len(sel)]

    quiet = lambda lo, hi: [sum(r[1] for r in log[lo:hi] if not r[0]), sum(1 for r in log[lo:hi] if not r[0])]
    return {"meas": {str(t): v for t, v in meas.items()},
            "conf_round_last500": rate(N_DYAD - 500, N_DYAD, 2), "conf_round_after_flip": rate(T_FLIP, N_DYAD, 2),
            "conf_follow_first500": rate(0, 500, 1), "conf_follow_last500": rate(N_DYAD - 500, N_DYAD, 1),
            "quiet_follow_first500": quiet(0, 500), "quiet_follow_last500": quiet(N_DYAD - 500, N_DYAD)}


def per_seed(seed):
    w, out = World(seed), {}
    for name, form, make, mirror in (("X_net", "X", Net, False), ("X_lin", "X", Linear, False),
                                     ("X_mirror", "X", Net, True), ("N_net", "N", Net, False)):
        m = make(seed)
        train(m, w, upbringing(seed, form), seed, 0, mirror=mirror)
        row = {"sane": [V(m, w, (RED, SQUARE, HIGH)), V(m, w, (BLUE, ROUND, HIGH))]}
        for flip in ((False, True) if name != "N_net" else (False,)):
            row["flip" if flip else "noflip"] = dyad(copy.deepcopy(m), w, seed, flip, mirror=mirror)
        out[name] = row
    return out


def summary(r):
    s = {}
    for name in r:
        nf = r[name]["noflip"]["meas"]
        s[name] = {k: [round(nf["0"][k], 3), round(nf[str(N_DYAD)][k], 3)]
                   for k in ("e_conf", "e_quiet", "boost_conf", "boost_quiet", "V_round", "V_roundE")}
        s[name]["round_last500 nf/f"] = [r[name]["noflip"]["conf_round_last500"],
                                         r[name].get("flip", {}).get("conf_round_last500")]
        if "flip" in r[name]:
            s[name]["flip V_roundE end"] = round(r[name]["flip"]["meas"][str(N_DYAD)]["V_roundE"], 3)
    s["sane"] = [round(v, 2) for v in r["X_net"]["sane"]]
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
        assert np.isfinite(r["X_net"]["noflip"]["meas"]["1500"]["e_conf"])
        print("smoke ok", f"{time.time() - t0:.0f}s")
    elif a.proto or a.dev:
        for s in range(*(a.proto or a.dev)):
            r = per_seed(s)
            if a.dev:
                ok = all(np.isfinite(v) for n in r for f in ("noflip", "flip") if f in r[n]
                         for t in r[n][f]["meas"].values() for v in t.values())
                print(s, "finite", ok, flush=True)
                continue
            print(s, json.dumps(summary(r)), flush=True)
    else:
        rows = {s: per_seed(s) for s in range(*a.seeds)}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "rows": rows}, fh, indent=1)
        print("written", a.out)
