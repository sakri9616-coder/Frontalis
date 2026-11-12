"""Tool check and leaf-leak observation (DESIGN-assert-3, version 3.1; exploration only, prototype seeds 0-7, numpy).

Elements are fixed random vectors (D24, std 1/sqrt(24)): surfaces g1..g5, content attributes a1..a3 and b1..b3,
tastes (sweet, bitter, bland), reactions (approve, disapprove). A content event is a_i + b_j; other events are one
element. Noise (std 0.1/sqrt(24)) is added to the learned input only (teacher values are read without noise).
An episode is one turn; its last event is a leaf.
Rule S: target(e) = innate signal of e if any (sweet +1, bitter -1, bland 0) + current V(next event) if any; no
update when neither exists. Loss (V - target)^2, SGD lr 0.01, total grad-norm clip 1.0 (also for the linear model).
Learners: S (MLP 24-32-32-1), F (same, but a leaf used as a teacher is read from the frozen copy made at the end of
formation), SL (linear), FL (linear with the frozen-leaf teacher), ST (table, step 0.1, no clip).
Formation: 0.8 content -> taste (X: attribute a decides, U: attribute b decides), 0.2 approve -> sweet or
disapprove -> bitter. Contents (a2,b3) and (a3,b2) never appear in formation.
Part 1: g1..g4 -> content. From the end of part 1, separately:
  part 2 (1000 turns): 1/3 g5 -> (a1,b1) -> (a2,b2), else as part 1;
  branch 2a (333 turns): (a1,b1) -> (a2,b2) only;  branch 2b (333 turns): (a3,b3) -> (a2,b2) only;
  part 3 (4000 turns): a partner reacts to contents (Pb by b, Pa by a); half reaction -> content, half the reverse.
Streams: elements [s,0]; event choice [s,3,stage]; noise [s,7,stage]; initial weights [s,11].

usage:
  python npn_leafleak.py --smoke
  python npn_leafleak.py --seeds 0 8 --out ../p70_leafleak_proto.json
  python npn_leafleak.py --summary ../p70_leafleak_proto.json
"""
import argparse
import copy
import json
import time

import numpy as np

D, H = 24, 32
LR, CLIP, TAB_STEP = 0.01, 1.0, 0.1
NOISE = 0.1 / np.sqrt(D)
N1, N2, NB, N3 = 600, 1000, 333, 4000
MEASURE_T = list(range(0, 1001, 50)) + list(range(1250, 4001, 250))
N_FORMS = (3000, 9000)
INNATE = (1.0, -1.0, 0.0)  # sweet, bitter, bland
CONTENTS = [(i, j) for i in range(3) for j in range(3)]
HELD_OUT = [(1, 2), (2, 1)]
FORMED = [c for c in CONTENTS if c not in HELD_OUT]
OBJ_CONTENT = [(1, 2), (2, 1), (0, 2), (2, 0)]  # g1..g4
CHAIN = [("g", 4), ("c", 0, 0), ("c", 1, 1)]
BRANCH = {"p2a": [("c", 0, 0), ("c", 1, 1)], "p2b": [("c", 2, 2), ("c", 1, 1)]}
OWN4 = [(0, 0), (0, 1), (1, 0), (1, 1)]
PARTNER = {
    "Pb": [((0, 0), 0), ((1, 0), 0), ((2, 0), 0), ((0, 1), 1), ((1, 1), 1)],
    "Pa": [((0, 0), 0), ((0, 1), 0), ((0, 2), 0), ((1, 0), 1), ((1, 1), 1)],
}
KINDS = ("S", "F", "SL", "FL", "ST")


class World:
    def __init__(self, seed):
        g = np.random.default_rng([seed, 0])
        s = 1 / np.sqrt(D)
        self.el = {"g": g.normal(0, s, (5, D)), "a": g.normal(0, s, (3, D)), "b": g.normal(0, s, (3, D)),
                   "t": g.normal(0, s, (3, D)), "r": g.normal(0, s, (2, D))}

    def vec(self, e):
        if e[0] == "c":
            return self.el["a"][e[1]] + self.el["b"][e[2]]
        return self.el[e[0]][e[1]]


def _norm(grads):
    return np.sqrt(sum(float((g * g).sum()) for g in grads))


class MLP:
    def __init__(self, seed):
        g = np.random.default_rng([seed, 11])

        def lin(n_out, n_in):
            k = 1 / np.sqrt(n_in)
            return g.uniform(-k, k, (n_out, n_in)), g.uniform(-k, k, n_out)

        self.W1, self.b1 = lin(H, D)
        self.W2, self.b2 = lin(H, H)
        w3, b3 = lin(1, H)
        self.w3, self.b3 = w3[0], float(b3[0])

    def _grads(self, x):
        z1 = self.W1 @ x + self.b1
        h1 = np.maximum(z1, 0)
        z2 = self.W2 @ h1 + self.b2
        h2 = np.maximum(z2, 0)
        d2 = self.w3 * (z2 > 0)
        d1 = (self.W2.T @ d2) * (z1 > 0)
        return float(self.w3 @ h2 + self.b3), [h2, np.array(1.0), np.outer(d2, h1), d2, np.outer(d1, x), d1]

    def value(self, e, world):
        return self._grads(world.vec(e))[0]

    def grad_norm(self, e, world):
        return _norm(self._grads(world.vec(e))[1])

    def learn(self, e, world, target, noise):
        v, gr = self._grads(world.vec(e) + noise)
        g = 2 * (v - target)
        s = LR * min(1.0, CLIP / (abs(g) * _norm(gr) + 1e-6)) * g
        self.w3 -= s * gr[0]
        self.b3 -= s
        self.W2 -= s * gr[2]
        self.b2 -= s * gr[3]
        self.W1 -= s * gr[4]
        self.b1 -= s * gr[5]


class Lin:
    def __init__(self, seed):
        g = np.random.default_rng([seed, 11])
        k = 1 / np.sqrt(D)
        self.w, self.b = g.uniform(-k, k, D), float(g.uniform(-k, k))

    def value(self, e, world):
        return float(self.w @ world.vec(e) + self.b)

    def grad_norm(self, e, world):
        x = world.vec(e)
        return float(np.sqrt(x @ x + 1))

    def learn(self, e, world, target, noise):
        x = world.vec(e) + noise
        g = 2 * (float(self.w @ x + self.b) - target)
        s = LR * min(1.0, CLIP / (abs(g) * np.sqrt(x @ x + 1) + 1e-6)) * g
        self.w -= s * x
        self.b -= s


class Tab:
    def __init__(self, seed=None):
        self.v = {}

    def value(self, e, world):
        return self.v.get(e, 0.0)

    def grad_norm(self, e, world):
        return None

    def learn(self, e, world, target, noise):
        v = self.v.get(e, 0.0)
        self.v[e] = v + TAB_STEP * (target - v)


def run_turn(model, world, events, rn, frozen=None):
    """Rule S over one episode. With `frozen`, a leaf used as a teacher is read from the frozen copy."""
    last = len(events) - 1
    for i, e in enumerate(events):
        noise = rn.normal(0, NOISE, D)
        target = None
        if e[0] == "t":
            target = INNATE[e[1]]
        if i < last:
            teacher = frozen if (frozen is not None and i + 1 == last) else model
            target = (target or 0.0) + teacher.value(events[i + 1], world)
        if target is not None:
            model.learn(e, world, target, noise)


def formation_turns(seed, form, n):
    rc = np.random.default_rng([seed, 3, 0])
    out = []
    for _ in range(n):
        u, k = rc.random(), int(rc.integers(7 * 2))
        if u < 0.8:
            c = FORMED[k % 7]
            out.append([("c",) + c, ("t", c[0] if form == "X" else c[1])])
        else:
            out.append([("r", k % 2), ("t", k % 2)])
    return out


def part1_turns(seed, n):
    rc = np.random.default_rng([seed, 3, 1])
    return [[("g", o), ("c",) + OBJ_CONTENT[o]] for o in (int(rc.integers(4)) for _ in range(n))]


def part2_turns(seed, n):
    rc = np.random.default_rng([seed, 3, 2])
    out = []
    for _ in range(n):
        u, o = rc.random(), int(rc.integers(4))
        out.append(list(CHAIN) if u < 1 / 3 else [("g", o), ("c",) + OBJ_CONTENT[o]])
    return out


def part3_turns(seed, partner, n):
    rc = np.random.default_rng([seed, 3, 3])
    out = []
    for _ in range(n):
        k, u = int(rc.integers(5)), rc.random()
        c, r = PARTNER[partner][k]
        out.append([("r", r), ("c",) + c] if u < 0.5 else [("c",) + c, ("r", r)])
    return out


def snapshot(model, world):
    r5 = lambda v: round(v, 5)
    return {"c": [r5(model.value(("c",) + c, world)) for c in CONTENTS],
            "g": [r5(model.value(("g", i), world)) for i in range(5)],
            "t": [r5(model.value(("t", i), world)) for i in range(3)],
            "r": [r5(model.value(("r", i), world)) for i in range(2)]}


def per_seed(seed, n_forms=N_FORMS):
    w, out = World(seed), {}
    for nf in n_forms:
        for form in ("X", "U"):
            turns0 = formation_turns(seed, form, nf)
            for kind in ("S", "SL", "ST"):
                m = {"S": MLP, "SL": Lin, "ST": Tab}[kind](seed)
                rn = np.random.default_rng([seed, 7, 0])
                for ev in turns0:
                    run_turn(m, w, ev, rn)
                variants = [(kind, m, None)]
                if kind != "ST":
                    variants.append(("F" if kind == "S" else "FL", copy.deepcopy(m), copy.deepcopy(m)))
                for name, m0, frozen in variants:
                    row = {"form_end": snapshot(m0, w)}
                    m1 = copy.deepcopy(m0)
                    rn = np.random.default_rng([seed, 7, 1])
                    for ev in part1_turns(seed, N1):
                        run_turn(m1, w, ev, rn, frozen)
                    row["p1_end"] = snapshot(m1, w)
                    gn = [m1.grad_norm(("c",) + c, w) for c in OWN4]
                    row["grad_norm_own4"] = None if gn[0] is None else float(np.mean(gn))
                    m2 = copy.deepcopy(m1)
                    rn = np.random.default_rng([seed, 7, 2])
                    for ev in part2_turns(seed, N2):
                        run_turn(m2, w, ev, rn, frozen)
                    row["p2_end"] = snapshot(m2, w)
                    for stage, (key, ev) in enumerate(BRANCH.items(), 4):
                        mb = copy.deepcopy(m1)
                        rn = np.random.default_rng([seed, 7, stage])
                        for _ in range(NB):
                            run_turn(mb, w, ev, rn, frozen)
                        row[key + "_end"] = snapshot(mb, w)
                    for partner in ("Pb", "Pa"):
                        m3 = copy.deepcopy(m1)
                        rn = np.random.default_rng([seed, 7, 3])
                        series = [snapshot(m3, w)]
                        for t, ev in enumerate(part3_turns(seed, partner, N3), 1):
                            run_turn(m3, w, ev, rn, frozen)
                            if t in MEASURE_T:
                                series.append(snapshot(m3, w))
                        row["p3_" + partner] = series
                    out[f"{nf}_{form}_{name}"] = row
    return out


# ---------- summary (DESIGN-assert-3 sections 5 and 9) ----------

def cv(snap, c):
    return snap["c"][CONTENTS.index(c)]


def own_other(snap, form):
    a = (cv(snap, (0, 0)) + cv(snap, (0, 1))) / 2 - (cv(snap, (1, 0)) + cv(snap, (1, 1))) / 2
    b = (cv(snap, (0, 0)) + cv(snap, (1, 0))) / 2 - (cv(snap, (0, 1)) + cv(snap, (1, 1))) / 2
    return (a, b) if form == "X" else (b, a)


def form_dev(row, form):
    e = row["form_end"]
    d = [abs(cv(e, c) - INNATE[c[0] if form == "X" else c[1]]) for c in FORMED]
    d += [abs(e["t"][i] - INNATE[i]) for i in range(3)] + [abs(e["r"][0] - 1), abs(e["r"][1] + 1)]
    return max(d)


def m2(xs):
    return f"{np.mean(xs):+.2f}" if len(xs) else "  n/a"


def pos(xs):
    return f"+{int((np.array(xs) > 0).sum())}/{len(xs)}"


def summarize(path):
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    seeds = sorted(rows, key=int)
    fmt = lambda v: " ".join(f"{y:+.2f}" for y in v)
    for nf in N_FORMS:
        print(f"\n===== formation {nf} =====")
        R = lambda s, form, k: rows[s][f"{nf}_{form}_{k}"]
        ok = {k: {(s, f): form_dev(R(s, f, k), f) < 0.2 for s in seeds for f in ("X", "U")} for k in KINDS}
        cells = lambda k: [(s, f) for s in seeds for f in ("X", "U") if ok[k][(s, f)]]
        pairs = lambda k: [s for s in seeds if ok[k][(s, "X")] and ok[k][(s, "U")]]
        print("-- formation end: max |V - expected| (7 formed contents, tastes, reactions); cells within 0.2 are kept")
        for k in KINDS:
            dev = [form_dev(R(s, f, k), f) for s in seeds for f in ("X", "U")]
            print(f"  {k:2s} mean {np.mean(dev):.2f} max {np.max(dev):.2f} kept {len(cells(k))}/16, seed pairs {len(pairs(k))}/8")
        print("-- held-out contents at formation end: V(a2,b3), V(a3,b2)   (X expects -1, 0; U expects 0, -1)")
        for k in ("S", "SL", "ST"):
            for form in ("X", "U"):
                e = [R(s, f, k)["form_end"] for s, f in cells(k) if f == form]
                print(f"  {k:2s} {form}: {m2([cv(x, (1, 2)) for x in e])}  {m2([cv(x, (2, 1)) for x in e])}")
        print("-- part 1: Q1 = V(g1) - V(g2)   (expected X about -1, U about +1, ST 0)")
        for k in KINDS:
            for form in ("X", "U"):
                q = [R(s, f, k)["p1_end"]["g"][0] - R(s, f, k)["p1_end"]["g"][1] for s, f in cells(k) if f == form]
                print(f"  {k:2s} {form}: {m2(q)} (sd {np.std(q) if q else 0:.2f})")
        for key, src, sh_a, sh_b, none in (("p2a", (0, 0), (0, 2), (2, 0), (2, 2)), ("p2b", (2, 2), (2, 0), (0, 2), (0, 0))):
            print(f"-- branch {key}: {BRANCH[key][0][1:]} -> (1, 1) only, {NB} turns. drop = before - after")
            print("   kind | source | target leaf | taste-attr shared | other attr shared | none | common(7 formed) | sel mean | seed-pair sign")
            for k in KINDS:
                acc = {n: [] for n in ("src", "tgt", "own", "oth", "none", "com")}
                selp = []
                for s in seeds:
                    per = {}
                    for form in ("X", "U"):
                        if not ok[k][(s, form)]:
                            continue
                        r = R(s, form, k)
                        b, a = r["p1_end"], r[key + "_end"]
                        drop = lambda c: cv(b, c) - cv(a, c)
                        own, oth = (sh_a, sh_b) if form == "X" else (sh_b, sh_a)
                        for n, v in (("src", drop(src)), ("tgt", drop((1, 1))), ("own", drop(own)), ("oth", drop(oth)),
                                     ("none", drop(none)), ("com", np.mean([drop(c) for c in FORMED]))):
                            acc[n].append(v)
                        per[form] = drop(own) - drop(oth)
                    if len(per) == 2:
                        selp.append((per["X"] + per["U"]) / 2)
                print(f"   {k:2s} | {m2(acc['src'])} | {m2(acc['tgt'])} | {m2(acc['own'])} | {m2(acc['oth'])} | {m2(acc['none'])} | {m2(acc['com'])} | {m2(selp)} | {pos(selp) if selp else 'n/a'}"
                      f" | shared-minus-none {m2([(o + t) / 2 - n for o, t, n in zip(acc['own'], acc['oth'], acc['none'])])}")
                if k in ("S", "SL"):
                    print(f"        per seed pair: {fmt(selp)}")
        print("-- part 2 with objects (description): common drop over 7 formed contents; object value, live-leaf minus frozen-leaf learner")
        for k, kf in (("S", "F"), ("SL", "FL")):
            com, diff = [], []
            for s, form in cells(k):
                r = R(s, form, k)
                com.append(np.mean([cv(r["p1_end"], c) - cv(r["p2_end"], c) for c in FORMED]))
                gi = 2 if form == "X" else 3
                diff.append(r["p2_end"]["g"][gi] - R(s, form, kf)["p2_end"]["g"][gi])
            print(f"  {k:2s}: common drop {m2(com)} (max {np.max(com):+.2f}) | {k} - {kf} at the object {m2(diff)}")
        print("-- part 3: series at t =", " ".join(str(t) for t in MEASURE_T))
        for k in KINDS:
            for form, partner in (("X", "Pb"), ("U", "Pa"), ("X", "Pa"), ("U", "Pb")):
                conflict = (form, partner) in (("X", "Pb"), ("U", "Pa"))
                ser = [R(s, f, k)["p3_" + partner] for s, f in cells(k) if f == form]
                if not ser:
                    print(f"  {k:2s} {form}x{partner}: no kept cells")
                    continue
                T = len(ser[0])
                pick = [0, 2, 4, 10, 20, 24, 32]  # t = 0, 100, 200, 500, 1000, 2000, 4000
                pr = [np.mean([x[t]["r"][0] - x[t]["r"][1] for x in ser]) for t in range(T)]
                own = [np.mean([own_other(x[t], form)[0] for x in ser]) for t in range(T)]
                oth = [np.mean([own_other(x[t], form)[1] for x in ser]) for t in range(T)]
                print(f"  {k:2s} {form}x{partner} ({'conflict' if conflict else 'agree'}), n={len(ser)}, shown at t = 0 100 200 500 1000 2000 4000")
                print(f"     Pr    {fmt([pr[i] for i in pick])}")
                print(f"     Own   {fmt([own[i] for i in pick])}")
                print(f"     Other {fmt([oth[i] for i in pick])}")
                if conflict:
                    u = ((0, 2), (2, 2)) if form == "X" else ((2, 0), (2, 2))
                    unt = [np.mean([cv(x[t], u[0]) - cv(x[t], u[1]) for x in ser]) for t in range(T)]
                    print(f"     Untch {fmt([unt[i] for i in pick])}")
                    half = []
                    for x in ser:
                        o = [own_other(x[t], form)[0] for t in range(T)]
                        hit = [t for t in range(T) if o[t] < o[0] / 2]
                        half.append(([0] + MEASURE_T[1:] if MEASURE_T[0] else MEASURE_T)[hit[0]] if hit else None)
                    gn = [R(s, f, k)["grad_norm_own4"] for s, f in cells(k) if f == form]
                    print(f"     half-time of Own per seed: {half}" + ("" if gn[0] is None else f" | grad norm at start {np.mean(gn):.2f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.smoke:
        t0 = time.time()
        r = per_seed(999, n_forms=(3000,))
        ok = all(np.isfinite(v) for row in r.values() for v in row["p2_end"]["c"] + row["p2b_end"]["c"] + row["p3_Pb"][-1]["c"])
        print("smoke finite", ok, "series length", len(r["3000_X_S"]["p3_Pb"]), f"{time.time() - t0:.0f}s")
    elif a.summary:
        summarize(a.summary)
    else:
        t0 = time.time()
        rows = {}
        for s in range(*a.seeds):
            rows[str(s)] = per_seed(s)
            print("seed", s, f"{time.time() - t0:.0f}s", flush=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "rows": rows}, fh)
        print("written", a.out)
