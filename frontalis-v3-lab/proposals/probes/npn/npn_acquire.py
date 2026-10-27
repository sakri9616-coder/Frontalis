"""Acquired evaluative features, blocking, ablation and the partner's word as evidence (DESIGN-acquire v2.3, v16).

Objects are combinations of colour (red, blue, green), shape (round, square, triangle) and sound (high, mid, low); the
object vector is the sum of three fixed random attribute vectors (D24, std 1/sqrt(24) per component) plus noise
(std 0.1/sqrt(24)). The partner's word E (praise) is one more fixed vector added to the object vector.
Evaluation network V: MLP D24 -> 32 -> 32 -> 1 (ReLU, biases, PyTorch default init seeded), trained after each event
toward the innate outcome o (P+ = +1, P- = -1, else 0) by squared error, SGD lr 0.01, grad-norm clip 1.0.
Upbringings: X: red -> P+ (0.8), round -> P- (0.8); Y: round -> P+, red -> P-. Red-and-round and sound "mid" are never
shown. Partner (T+ / T-), 20% of events: praise an object from S_good (green triangles) or S_round (round, not red),
50/50. T+: S_good -> P+ (0.8), S_round -> the object's usual outcome. T-: P+ or P- with equal probability.
Baselines: linear regression (same events), table (mean per object and praise), mirror (praise events trained toward
o + 1). Streams: attribute vectors [s,0]; event choice [s,3]; outcomes [s,5]; noise [s,7]; random ablation [s,23].

usage:
  python npn_acquire.py --smoke
  python npn_acquire.py --proto A B          # prototype print (prototype seeds only)
  python npn_acquire.py --dev A B            # learning sanity only (finite)
  python npn_acquire.py --seeds A B --out res.json
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
D, H = 24, 32
LR, CLIP = 0.01, 1.0
N_UP, N_SHARED, N_RELIABLE, N_DEVALUE = 3000, 200, 400, 200
P, P_PARTNER, N_ABL = 0.8, 0.2, 4
RED, BLUE, GREEN = 0, 1, 2
ROUND, SQUARE, TRI = 0, 1, 2
HIGH, MID, LOW = 0, 1, 2
SEEN = [(c, s, t) for c in range(3) for s in range(3) for t in (HIGH, LOW) if not (c == RED and s == ROUND)]
S_GOOD = [(GREEN, TRI, HIGH), (GREEN, TRI, LOW)]
S_ROUND = [(BLUE, ROUND, HIGH), (BLUE, ROUND, LOW), (GREEN, ROUND, HIGH), (GREEN, ROUND, LOW)]


class World:
    def __init__(self, seed):
        g = np.random.default_rng([seed, 0])
        self.attr = g.normal(0, 1 / np.sqrt(D), (3, 3, D))
        self.E = g.normal(0, 1 / np.sqrt(D), D)

    def vec(self, obj, praise=False, rng=None):
        v = self.attr[0, obj[0]] + self.attr[1, obj[1]] + self.attr[2, obj[2]] + (self.E if praise else 0)
        if rng is not None:
            v = v + rng.normal(0, 0.1 / np.sqrt(D), D)
        return torch.tensor(v, dtype=torch.float32)


def feature_o(obj, form, ro):
    good, bad = (obj[0] == RED, obj[1] == ROUND) if form == "X" else (obj[1] == ROUND, obj[0] == RED)
    if good:
        return 1.0 if ro.random() < P else 0.0
    if bad:
        return -1.0 if ro.random() < P else 0.0
    return 0.0


class Net(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.l1, self.l2, self.out = nn.Linear(D, H), nn.Linear(H, H), nn.Linear(H, 1)
        self.mask = torch.ones(H)

    def hidden(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x)))) * self.mask

    def forward(self, x):
        return self.out(self.hidden(x)).squeeze(-1)


class Linear(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(D, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class Table:
    def __init__(self):
        self.s, self.n = {}, {}

    def learn(self, key, o):
        self.s[key] = self.s.get(key, 0.0) + o
        self.n[key] = self.n.get(key, 0) + 1

    def value(self, key):
        return self.s[key] / self.n[key] if key in self.n else 0.0


def sgd(model, x, target):
    model.zero_grad()
    loss = (model(x) - target) ** 2
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(-LR * p.grad)


def events(seed, form, partner, n, stage, reliable=False):
    """The event list of one stage: (object, praised, o). The same seed gives the same object sequence for every form."""
    rc = np.random.default_rng([seed, 3, stage])
    ro = np.random.default_rng([seed, 5, stage])
    ev = []
    for _ in range(n):
        is_partner = partner is not None and rc.random() < P_PARTNER
        u = rc.random()
        if is_partner:
            obj = (S_GOOD if u < 0.5 else S_ROUND)[int(rc.integers(2 if u < 0.5 else 4))]
            if partner == "T+" or reliable:
                o = (1.0 if ro.random() < P else 0.0) if obj in S_GOOD else feature_o(obj, form, ro)
            else:
                o = 1.0 if ro.random() < 0.5 else -1.0
            ev.append((obj, True, o))
        else:
            obj = SEEN[int(rc.integers(len(SEEN)))]
            ev.append((obj, False, feature_o(obj, form, ro)))
    return ev


def train(model, world, ev, seed, stage, mirror=False):
    rn = np.random.default_rng([seed, 7, stage])
    for obj, pr, o in ev:
        target = o + (1.0 if (mirror and pr) else 0.0)
        if isinstance(model, Table):
            model.learn((obj, pr), target)
        else:
            sgd(model, world.vec(obj, pr, rn), target)


def V(model, world, obj, praise=False):
    if isinstance(model, Table):
        return model.value((obj, praise))
    with torch.no_grad():
        return float(model(world.vec(obj, praise)))


def boost(model, world, obj):
    return V(model, world, obj, True) - V(model, world, obj, False)


def shared(seed):
    return [((RED, TRI, LOW), False, 1.0)] * N_SHARED


def raise_after_shared(model, world, seed, objs):
    before = {o: V(model, world, o) for o in objs}
    train(model, world, shared(seed), seed, 10)
    return {o: V(model, world, o) - before[o] for o in objs}


def contrib_units(net, world):
    with torch.no_grad():
        red = torch.stack([net.hidden(world.vec(o)) for o in SEEN if o[0] == RED]).mean(0)
        other = torch.stack([net.hidden(world.vec(o)) for o in SEEN if o[0] != RED]).mean(0)
        c = net.out.weight[0] * (red - other)
    order = [int(j) for j in torch.argsort(c, descending=True) if c[j] > 0]
    return order[:N_ABL], c


D1_OBJS = ((RED, SQUARE, MID), (BLUE, ROUND, MID))
G2_OBJS = ((GREEN, TRI, LOW), (GREEN, TRI, HIGH), (GREEN, SQUARE, LOW))
G4_OBJS = ((GREEN, SQUARE, MID), (BLUE, ROUND, MID))


def d1(model, world):
    return V(model, world, D1_OBJS[0]) - V(model, world, D1_OBJS[1])


def per_seed(seed):
    w, out = World(seed), {}
    # G1, G2, sanity: X.T0 and Y.T0 (network, linear, table)
    for form in ("X", "Y"):
        up = events(seed, form, None, N_UP, 0)
        for kind in ("net", "lin", "tab"):
            m = Net(seed) if kind == "net" else (Linear(seed) if kind == "lin" else Table())
            train(m, w, up, seed, 0)
            # sanity: [value that should be > 0, value that should be < 0]
            sane = [(RED, SQUARE, HIGH), (BLUE, ROUND, HIGH)] if form == "X" else [(BLUE, ROUND, HIGH), (RED, TRI, HIGH)]
            row = {"D1": d1(m, w), "sane": [V(m, w, o) for o in sane]}
            if kind == "net" and form == "X":
                base = copy.deepcopy(m)
            dv = raise_after_shared(m, w, seed, G2_OBJS)
            row["dV"] = [dv[o] for o in G2_OBJS]
            out[f"{form}T0_{kind}"] = row
    # G3 on X.T0
    top, c = contrib_units(base, w)
    ra = np.random.default_rng([seed, 23])
    rnd = [int(j) for j in ra.choice(H, N_ABL, replace=False)]
    g3 = {"top": top, "random": rnd, "c": c.tolist()}
    for name, units in (("top", top), ("random", rnd)):
        m = copy.deepcopy(base)
        m.mask = torch.ones(H)
        m.mask[units] = 0.0
        g3[f"D1_{name}"] = d1(m, w)
        g3[f"dV_{name}"] = [raise_after_shared(m, w, seed, G2_OBJS)[o] for o in G2_OBJS]
    m = copy.deepcopy(base)
    reds = [o for o in SEEN if o[0] == RED]
    rd = np.random.default_rng([seed, 3, 20])
    train(m, w, [(reds[int(rd.integers(len(reds)))], False, 0.0) for _ in range(N_DEVALUE)], seed, 20)
    g3["D1_devalue"] = d1(m, w)
    g3["dV_devalue"] = [raise_after_shared(m, w, seed, G2_OBJS)[o] for o in G2_OBJS]
    g3["D1_intact"] = d1(base, w)
    out["G3"] = g3
    # G4: X.T+ and X.T- (network, mirror, linear)
    for partner in ("T+", "T-"):
        up = events(seed, "X", partner, N_UP, 0)
        for kind in ("net", "mirror", "lin"):
            m = Linear(seed) if kind == "lin" else Net(seed)
            train(m, w, up, seed, 0, mirror=(kind == "mirror"))
            row = {"boost": [boost(m, w, o) for o in G4_OBJS], "V": [V(m, w, o, pr) for o in G4_OBJS for pr in (False, True)]}
            if partner == "T-":
                train(m, w, events(seed, "X", partner, N_RELIABLE, 30, reliable=True), seed, 30,
                      mirror=(kind == "mirror"))
                row["boost_after_reliable"] = [boost(m, w, o) for o in G4_OBJS]
            out[f"X{partner}_{kind}"] = row
    return out


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
        assert np.isfinite(r["XT0_net"]["D1"]) and np.isfinite(r["XT+_net"]["boost"][0])
        print("smoke ok", f"{time.time() - t0:.0f}s")
    elif a.proto or a.dev:
        for s in range(*(a.proto or a.dev)):
            r = per_seed(s)
            if a.dev:
                print(s, "finite", all(np.isfinite(v) for k in r for v in (r[k].get("D1", 0),)), flush=True)
                continue
            g3 = r["G3"]
            print(s, json.dumps({
                "D1 X/Y net": [round(r["XT0_net"]["D1"], 2), round(r["YT0_net"]["D1"], 2)],
                "D1 X/Y lin": [round(r["XT0_lin"]["D1"], 2), round(r["YT0_lin"]["D1"], 2)],
                "sane X/Y": [[round(v, 2) for v in r["XT0_net"]["sane"]], [round(v, 2) for v in r["YT0_net"]["sane"]]],
                "G2 dV X/Y net": [[round(v, 2) for v in r["XT0_net"]["dV"]], [round(v, 2) for v in r["YT0_net"]["dV"]]],
                "G2 dV X/Y lin": [[round(v, 2) for v in r["XT0_lin"]["dV"]], [round(v, 2) for v in r["YT0_lin"]["dV"]]],
                "G3 D1 intact/top/rand/deval": [round(g3["D1_intact"], 2), round(g3["D1_top"], 2), round(g3["D1_random"], 2),
                                                round(g3["D1_devalue"], 2)],
                "G3 dV top/rand/deval": [round(g3["dV_top"][0], 2), round(g3["dV_random"][0], 2), round(g3["dV_devalue"][0], 2)],
                "G4 boost T+ net/mir/lin": [[round(v, 2) for v in r[f"XT+_{k}"]["boost"]] for k in ("net", "mirror", "lin")],
                "G4 boost T- net/mir": [[round(v, 2) for v in r[f"XT-_{k}"]["boost"]] for k in ("net", "mirror")],
                "G4c T- net after": [round(v, 2) for v in r["XT-_net"]["boost_after_reliable"]],
            }), flush=True)
    else:
        rows = {s: per_seed(s) for s in range(*a.seeds)}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "rows": rows}, fh, indent=1)
        print("written", a.out)
