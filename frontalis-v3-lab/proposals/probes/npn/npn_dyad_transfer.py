"""Does the discount of the partner's word learned where the partner was wrong move to an unseen domain that the
agent itself evaluates as bad (DESIGN-dyad-transfer v2, v19)?

World (attribute vectors, E), network, linear model and SGD are those of npn_acquire.py (v16), imported unchanged.
Upbringing X2 (3000 passive events): red -> P+, round -> P-, sound low -> P- (0.8 each); red-round, red-low and
round-low objects are never shown. Partner events (0.2) praise a neutral object (blue/green x square/triangle x
high/mid) followed by P+ (0.8); the partner never praises round or low objects. X2r: the low outcomes are 0. N2: every
feature outcome is 0. One outcome uniform per event in every form, so all forms share one event stream.
Dyad (1500 turns): conflict turn (0.5): a round object (recommended) + two distinct green neutral objects; quiet turn:
three distinct green neutral objects, one recommended. Condition W: on conflict turns only the recommended round
object (+E) is learned; on quiet turns the chosen object. Condition Q: the same turns; nothing is learned on conflict
turns. Outcomes: round -> P- (0.8); a recommended neutral -> P+ (0.8); else 0. Blue neutral, low and red objects never
appear in the dyad. Streams: attributes [s,0]; upbringing events [s,3,0], outcomes [s,5,0], noise [s,7,0]; dyad objects
and recommendation [s,31]; choice [s,37]; outcomes [s,41] (one uniform per shown object); noise [s,47] (two per turn).

usage:
  python npn_dyad_transfer.py --smoke
  python npn_dyad_transfer.py --proto A B
  python npn_dyad_transfer.py --dev A B
  python npn_dyad_transfer.py --seeds A B --out res.json
"""
import argparse
import copy
import json
import time

import numpy as np
import torch

from npn_acquire import (BLUE, CLIP, D, GREEN, HIGH, LOW, LR, MID, P, P_PARTNER, RED, ROUND, SQUARE, TRI, Linear, Net,
                         World, sgd, train)

N_UP, N_DYAD, TAU, P_CONF = 3000, 1500, 3.0, 0.5
GREEN_N = [(GREEN, s, t) for s in (SQUARE, TRI) for t in (HIGH, MID)]
BLUE_N = [(BLUE, s, t) for s in (SQUARE, TRI) for t in (HIGH, MID)]
NEUTRAL = GREEN_N + BLUE_N
ROUNDS = [(c, ROUND, t) for c in (BLUE, GREEN) for t in (HIGH, MID)]
LOWS = [(c, s, LOW) for c in (BLUE, GREEN) for s in (SQUARE, TRI)]
REDS = [(RED, s, t) for s in (SQUARE, TRI) for t in (HIGH, MID)]
SHOWN = NEUTRAL + ROUNDS + LOWS + REDS
DOMAINS = {"low": LOWS, "blueN": BLUE_N, "red": REDS, "round": ROUNDS, "greenN": GREEN_N}
MEAS_T = (0, 500, 1000, 1500)
assert (LR, CLIP, P, P_PARTNER) == (0.01, 1.0, 0.8, 0.2)


def up_outcome(obj, form, u):
    if obj[0] == RED:
        return (1.0 if u < P else 0.0) if form != "N2" else 0.0
    if obj[1] == ROUND:
        return (-1.0 if u < P else 0.0) if form != "N2" else 0.0
    if obj[2] == LOW:
        return (-1.0 if u < P else 0.0) if form == "X2" else 0.0
    return 0.0


def upbringing(seed, form):
    rc = np.random.default_rng([seed, 3, 0])
    ro = np.random.default_rng([seed, 5, 0])
    ev = []
    for _ in range(N_UP):
        partner = rc.random() < P_PARTNER
        u = ro.random()
        if partner:
            ev.append((NEUTRAL[int(rc.integers(len(NEUTRAL)))], True, 1.0 if u < P else 0.0))
        else:
            obj = SHOWN[int(rc.integers(len(SHOWN)))]
            ev.append((obj, False, up_outcome(obj, form, u)))
    return ev


def turns(seed):
    rk = np.random.default_rng([seed, 31])
    out = []
    for _ in range(N_DYAD):
        if rk.random() < P_CONF:
            a, b = rk.choice(len(GREEN_N), 2, replace=False)
            out.append(("conf", [ROUNDS[int(rk.integers(4))], GREEN_N[int(a)], GREEN_N[int(b)]], 0))
        else:
            objs = [GREEN_N[int(j)] for j in rk.choice(len(GREEN_N), 3, replace=False)]
            out.append(("quiet", objs, int(rk.integers(3))))
    return out


def outcome(obj, rec, u):
    if obj[1] == ROUND:
        return -1.0 if u < P else 0.0
    return (1.0 if u < P else 0.0) if rec else 0.0


def V(model, world, obj, praise=False):
    with torch.no_grad():
        return float(model(world.vec(obj, praise)))


def measure(model, world):
    vr = {o: (V(model, world, o), V(model, world, o, True)) for o in SHOWN}

    def e(rec, comp):
        sig = lambda d: 1 / (1 + np.exp(-TAU * d))
        return sig(vr[rec][1] - vr[comp][0]) - sig(vr[rec][0] - vr[comp][0])

    out = {}
    for name, objs in DOMAINS.items():
        out[f"boost_{name}"] = float(np.mean([vr[o][1] - vr[o][0] for o in objs]))
        out[f"V_{name}"] = float(np.mean([vr[o][0] for o in objs]))
        out[f"VE_{name}"] = float(np.mean([vr[o][1] for o in objs]))
        out[f"e_{name}"] = float(np.mean([e(r, c) for r in objs for c in BLUE_N if c != r]))
    return out


def overlap(net, world):
    """Cosine between (round - blue neutral) and (low - blue neutral) in the mean second-layer activation."""
    with torch.no_grad():
        h = {k: torch.stack([net.hidden(world.vec(o)) for o in v]).mean(0) for k, v in DOMAINS.items()}
    a, b = h["round"] - h["blueN"], h["low"] - h["blueN"]
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def dyad(model, world, seed, cond):
    tl = turns(seed)
    rch = np.random.default_rng([seed, 37])
    rou = np.random.default_rng([seed, 41])
    rn = np.random.default_rng([seed, 47])
    meas, follow = {}, [0, 0, 0, 0]
    for t, (kind, objs, rec) in enumerate(tl):
        if t in MEAS_T:
            meas[t] = measure(model, world)
        with torch.no_grad():
            x = torch.stack([world.vec(o, j == rec) for j, o in enumerate(objs)])
            p = torch.softmax(TAU * model(x), 0).numpy().astype(np.float64)
        k = min(int(np.searchsorted(np.cumsum(p), rch.random(), side="right")), 2)
        us = rou.random(3)
        noise = rn.normal(0, 0.1 / np.sqrt(D), (2, D))
        c = 0 if kind == "conf" else 2
        follow[c] += int(k == rec)
        follow[c + 1] += 1
        if kind == "conf" and cond == "Q":
            continue
        idx = rec if kind == "conf" else k
        xi = world.vec(objs[idx], idx == rec) + torch.tensor(noise[0], dtype=torch.float32)
        sgd(model, xi, outcome(objs[idx], idx == rec, us[idx]))
    meas[N_DYAD] = measure(model, world)
    return {"meas": {str(t): v for t, v in meas.items()}, "follow_conf": follow[:2], "follow_quiet": follow[2:]}


def per_seed(seed):
    w, out = World(seed), {}
    bases = {}
    for name, form, make in (("X2_net", "X2", Net), ("X2r_net", "X2r", Net), ("N2_net", "N2", Net),
                             ("X2_lin", "X2", Linear)):
        m = make(seed)
        train(m, w, upbringing(seed, form), seed, 0)
        bases[name] = m
        out[name] = {"sane": [V(m, w, (RED, SQUARE, HIGH)), V(m, w, (BLUE, ROUND, HIGH)), V(m, w, (BLUE, SQUARE, LOW))]}
        if make is Net:
            out[name]["overlap"] = overlap(m, w)
    for name, cond in (("X2_net", "W"), ("X2_net", "Q"), ("X2r_net", "W"), ("N2_net", "W"), ("X2_lin", "W")):
        out[name][cond] = dyad(copy.deepcopy(bases[name]), w, seed, cond)
    return out


def M(r, name, cond, dom="low", t="1500"):
    ms = r[name][cond]["meas"]
    d = lambda k: ms[t][f"boost_{k}"] - ms["0"][f"boost_{k}"]
    return -(d(dom) - d("blueN"))


def summary(r):
    s = {"sane X2": [round(v, 2) for v in r["X2_net"]["sane"]], "overlap X2/X2r/N2":
         [round(r[n]["overlap"], 2) for n in ("X2_net", "X2r_net", "N2_net")]}
    for name, cond in (("X2_net", "W"), ("X2_net", "Q"), ("X2r_net", "W"), ("N2_net", "W"), ("X2_lin", "W")):
        ms = r[name][cond]["meas"]
        s[f"{name}.{cond}"] = {"M_low": round(M(r, name, cond), 3), "M_red": round(M(r, name, cond, "red"), 3),
                               "boost 0/1500 low,blueN,red,round": [[round(ms[t][f"boost_{k}"], 2)
                                                                     for k in ("low", "blueN", "red", "round")]
                                                                    for t in ("0", "1500")]}
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
        assert np.isfinite(M(r, "X2_net", "W"))
        print("smoke ok", f"{time.time() - t0:.0f}s", json.dumps(summary(r)))
    elif a.proto or a.dev:
        for s in range(*(a.proto or a.dev)):
            r = per_seed(s)
            if a.dev:
                ok = all(np.isfinite(v) for n in r for c in ("W", "Q") if c in r[n]
                         for t in r[n][c]["meas"].values() for v in t.values())
                print(s, "finite", ok, flush=True)
                continue
            print(s, json.dumps(summary(r)), flush=True)
    else:
        rows = {s: per_seed(s) for s in range(*a.seeds)}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "rows": rows}, fh, indent=1)
        print("written", a.out)
