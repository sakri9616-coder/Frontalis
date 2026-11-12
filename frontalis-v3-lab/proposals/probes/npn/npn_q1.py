"""Exploration 5: a familiarity network U carries the amount of experience (DESIGN-q1 v1.1, from SELECT-q1).

Prototype seeds only, numpy only. World, formation, V (MLP 24-32-32-1), rule S and the q = 1 dyad with partner M are
those of npn_reaction, without replay or record. U: MLP 24-32-1 with a sigmoid output, cross-entropy, lr 0.01;
positives = both events of the turn (clean vectors), one negative per turn = a normal random vector with the spread of
a sum of k elements, k drawn from {1, 2, 3}. u(x) = 1 - U(x) = sigmoid(-z). Rule S step for "X -> Y" (Y not a taste)
is multiplied by m = 2 u(X) / (u(X) + u(Y)) (after clipping; tables: step x m). u is read before the turn's learning.
Learners: VU, VX (U swapped with the other individual of the seed), S0 (no U), TU (aligned table + network U),
TC0..TC3 (aligned table + counted u: 1/(1+n), 1/(2+0.1n), 1/(2+0.01n), 1/(2+0.003n)), T0m (aligned table, no U);
tables at steps 0.1 / 0.03 / 0.01. Individuals P2, Q2 only.

usage:
  python npn_q1.py --time 999
  python npn_q1.py --regress ../p75_reaction_rep_a.json
  python npn_q1.py --buildcheck 16
  python npn_q1.py --base 24 32 --out ../p76_q1_base.json
  python npn_q1.py --seeds 16 20 --out ../p76_q1_a.json
  python npn_q1.py --summary ../p76_q1_a.json ../p76_q1_b.json --basefile ../p76_q1_base.json
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npn_choice as C  # noqa: E402
import npn_reaction as R  # noqa: E402
from npn_choice import A, B, NN, INDIVS, World, item, formation_turns, wished_side, snap, softmax2, own, follow, pr, form_dev  # noqa: E402
from npn_leafleak import D, LR, CLIP, NOISE, INNATE, MLP, _norm, run_turn  # noqa: E402

STEPS = R.STEPS
TIMES = R.T_REP
T_READ = R.T_READ
T_EARLY = [100, 300, 1000]
IND = ("P2", "Q2")
U_LR, U_H = 0.01, 32
COUNTS = {"0": (1.0, 1.0), "1": (2.0, 0.1), "2": (2.0, 0.01), "3": (2.0, 0.003)}  # u = 1 / (a + c n), fixed (CRITIQUE-q1-1 D1)


class UNet:
    def __init__(self, seed):
        g = np.random.default_rng([seed, 41])
        k1, k2 = 1 / np.sqrt(D), 1 / np.sqrt(U_H)
        self.W1, self.b1 = g.uniform(-k1, k1, (U_H, D)), g.uniform(-k1, k1, U_H)
        self.w2, self.b2 = g.uniform(-k2, k2, U_H), float(g.uniform(-k2, k2))

    def z(self, x):
        return float(self.w2 @ np.maximum(self.W1 @ x + self.b1, 0) + self.b2)

    def u(self, e, w):
        return 1.0 / (1.0 + np.exp(self.z(w.vec(e))))

    def _learn(self, x, y):
        z1 = self.W1 @ x + self.b1
        h = np.maximum(z1, 0)
        p = 1 / (1 + np.exp(-(self.w2 @ h + self.b2)))
        dz = p - y
        dh = dz * self.w2 * (z1 > 0)
        self.w2 -= U_LR * dz * h
        self.b2 -= U_LR * dz
        self.W1 -= U_LR * np.outer(dh, x)
        self.b1 -= U_LR * dh

    def step(self, events, w, rneg):
        for e in events:
            self._learn(w.vec(e), 1.0)
        k = int(rneg.integers(1, 4))
        self._learn(rneg.normal(0, np.sqrt(k / D), D), 0.0)


class Cnt:
    def __init__(self, f="0"):
        self.n, self.a, self.c = {}, *COUNTS[f]

    def u(self, e, w):
        return 1.0 / (self.a + self.c * self.n.get(e, 0))

    def step(self, events, w, rneg):
        for e in events:
            self.n[e] = self.n.get(e, 0) + 1


class Const:
    """Build check: equal familiarity everywhere (m = 1)."""
    def u(self, e, w):
        return 0.5

    def step(self, events, w, rneg):
        pass


def learn_m(model, e, w, target, noise, m):
    if isinstance(model, R.TabS):
        v = model.v.get(e, 0.0)
        model.v[e] = v + model.step * m * (target - v)
        return
    v, gr = model._grads(w.vec(e) + noise)
    g = 2 * (v - target)
    s = LR * min(1.0, CLIP / (abs(g) * _norm(gr) + 1e-6)) * g * m
    model.w3 -= s * gr[0]
    model.b3 -= s
    model.W2 -= s * gr[2]
    model.b2 -= s * gr[3]
    model.W1 -= s * gr[4]
    model.b1 -= s * gr[5]


def run_turn_m(model, w, events, rn, um):
    """Rule S (as npn_leafleak.run_turn) with the familiarity ratio on non-taste teachers."""
    last = len(events) - 1
    us = [um.u(e, w) for e in events] if um is not None else None
    for i, e in enumerate(events):
        noise = rn.normal(0, NOISE, D)
        target, m = None, 1.0
        if e[0] == "t":
            target = INNATE[e[1]]
        if i < last:
            target = (target or 0.0) + model.value(events[i + 1], w)
            if um is not None and events[i + 1][0] != "t":
                m = 2 * us[i] / (us[i] + us[i + 1])
        if target is not None:
            learn_m(model, e, w, target, noise, m)


def ustat(um, w, full=False):
    if um is None:
        return None
    r = lambda e: round(float(um.u(e, w)), 7)
    out = {"c": [[r(item(d, k)) for k in range(2)] for d in range(3)],
           "rb": [[r(("rb", i, d)) for d in range(3)] for i in range(2)],
           "m": [[[r(("m", item(d, k), s)) for s in range(2)] for k in range(2)] for d in range(3)],
           "t": [r(("t", i)) for i in range(3)], "c3": [r(("c", 2, A)), r(("c", 5, B))]}
    if full:
        out["all"] = {repr(e): r(e) for e in R.all_events()}
    return out


def m_s(us, d):
    a, b = us["c"][d][0], us["rb"][1][d]
    return 2 * a / (a + b)


def dyad(V0, U0, w, seed, partner="M"):
    m = copy.deepcopy(V0)
    um = copy.deepcopy(U0) if U0 is not None else None
    rc = np.random.default_rng([seed, 3, 1])
    rs = np.random.default_rng([seed, 29, 1])
    rn = np.random.default_rng([seed, 7, 1])
    rneg = np.random.default_rng([seed, 41, 1])
    series, useries, log = [snap(m, w)], [ustat(um, w)], []
    for t in range(1, C.N_DYAD + 1):
        d, wish, order, u = int(rc.integers(3)), rc.random() < C.P_WISH, rc.random(), rs.random()
        ws = wished_side(partner, d, seed)
        if wish:
            opts = [("m", item(d, k), 0 if k == ws else 1) for k in range(2)]
            p = softmax2(m.value(opts[0], w), m.value(opts[1], w))
            ch = 0 if u < p[0] else 1
            react = ("rb", 0 if ch == ws else 1, d)
            ev = [opts[ch], react] if order < 0.5 else [react, item(d, ch)]
            log.append((d, 1, int(ch == ws)))
        else:
            p = softmax2(m.value(item(d, 0), w), m.value(item(d, 1), w))
            ch = 0 if u < p[0] else 1
            react = ("rb", 0 if ch == ws else 1, d)
            ev = [item(d, ch), react] if order < 0.5 else [react, item(d, ch)]
            log.append((d, 0, int(ch == 0)))
        run_turn_m(m, w, ev, rn, um)
        if um is not None:
            um.step(ev, w, rneg)
        if t in TIMES:
            series.append(snap(m, w))
            useries.append(ustat(um, w))
    if isinstance(m, R.Tab):
        assert len(m.v) == len(R.all_events()), "a table learned an event outside the aligned cells"
    return {"series": series, "u": useries, "log": log}


def form(seed, indiv, w):
    cd, rd = INDIVS[indiv]
    record = formation_turns(seed, cd, rd)
    net, U = MLP(seed), UNet(seed)
    cnts = {f: Cnt(f) for f in COUNTS}
    rn = np.random.default_rng([seed, 7, 0])
    rneg = np.random.default_rng([seed, 41, 0])
    U3 = None
    for t, ev in enumerate(record, 1):
        run_turn(net, w, ev, rn)
        U.step(ev, w, rneg)
        for c in cnts.values():
            c.step(ev, w, None)
        if t == 3000:
            U3 = copy.deepcopy(U)
    return net, U, U3, cnts


def per_seed(seed, base=False):
    w = World(seed)
    F = {i: form(seed, i, w) for i in IND}
    out = {}
    for indiv in IND:
        net, U, U3, cnts = F[indiv]
        other = F[IND[1 - IND.index(indiv)]][1]
        tabs = {st: R.aligned_tab(net, w, sz) for st, sz in STEPS.items()}
        out[f"{indiv}_form"] = {"v": snap(net, w), "U9000": ustat(U, w, True), "U3000": ustat(U3, w), "cnt": ustat(cnts["0"], w, True)}
        runs = []
        if not base:
            runs += [("VU", net, U), ("VX", net, other), ("S0", net, None)]
            runs += [(f"TU{st}", tabs[st], U) for st in STEPS]
            runs += [(f"TC{f}_{st}", tabs[st], cnts[f]) for f in COUNTS for st in STEPS]
        else:
            runs += [(f"TC0_{st}", tabs[st], cnts["0"]) for st in STEPS]
        runs += [(f"T0m{st}", tabs[st], None) for st in STEPS]
        for kind, V0, U0 in runs:
            out[f"{indiv}_{kind}"] = dyad(V0, U0, w, seed)
    return out


def regress(prev):
    old = json.load(open(prev, encoding="utf-8"))["rows"]["8"]
    seed, indiv = 8, "P2"
    w = World(seed)
    net = form(seed, indiv, w)[0]
    md = float(np.max(np.abs(R.flat(snap(net, w)) - R.flat(old[f"{indiv}_form_mlp"]))))
    for key, V0 in (("S0", net), ("T0m10", R.aligned_tab(net, w, 0.1))):
        new = dyad(V0, None, w, seed)["series"]
        ref = old[f"{indiv}_q1_{key}_M"]["series"]
        md = max(md, max(float(np.max(np.abs(R.flat(x) - R.flat(y)))) for x, y in zip(new, ref)))
    print(f"[gate 3] regression seed 8 P2 formation V, S0 x M, T0m10 x M vs p75: max |diff| {md:.2e} -> {'OK' if md < 1e-6 else 'FAIL'}")


def buildcheck(seed):
    """CRITIQUE-q1-1 D6b: constant u gives m = 1 -> VU equals S0, and the table equals T0m."""
    w = World(seed)
    net = form(seed, "P2", w)[0]
    md = 0.0
    for V0 in (net, R.aligned_tab(net, w, 0.1)):
        a = dyad(V0, None, w, seed)["series"]
        b = dyad(V0, Const(), w, seed)["series"]
        md = max(md, max(float(np.max(np.abs(R.flat(x) - R.flat(y)))) for x, y in zip(a, b)))
    print(f"[build] constant u (network and table) equals no-U runs, seed {seed} P2: max |diff| {md:.2e} -> {'OK' if md < 1e-6 else 'FAIL'}")


# ---------- summary (DESIGN-q1 v1.1 sections 4 and 8) ----------

def cellmap(indiv):
    """keep cell = content thick, reaction thin; copy cell = content thin, reaction thick."""
    cd, rd = INDIVS[indiv]
    return {"keep": cd, "copy": rd}


def summarize(paths, basefile):
    rows = {}
    for p in paths:
        rows.update(json.load(open(p, encoding="utf-8"))["rows"])
    base = json.load(open(basefile, encoding="utf-8"))["rows"] if basefile else None
    times = [0] + [t for t in TIMES if t > 0]
    ix = [times.index(t) for t in T_READ]
    avg = lambda sr, f: float(np.mean([f(sr[j]) for j in ix]))
    at = lambda sr, f, t: float(f(sr[times.index(t)]))
    seeds = sorted(rows, key=int)
    allc = [(s, i) for s in seeds for i in IND]
    keep = [c for c in allc if form_dev(rows[c[0]][f"{c[1]}_form"]["v"]) < 0.2]
    both = sorted({s for s, i in keep if (s, "P2") in keep and (s, "Q2") in keep}, key=int)
    print(f"[5] kept {len(keep)}/{len(allc)}; seeds with both P2 and Q2: {len(both)} {both}; excluded {[c for c in allc if c not in keep]}")
    enough = len(both) >= 6
    gate = {}
    for nm, lab in (("U9000", "gate 1 (formation end)"), ("U3000", "gate 2 (short formation, 3000)"), ("cnt", "reference TC0")):
        kv = [m_s(rows[s][f"{i}_form"][nm], cellmap(i)["keep"]) for s, i in keep]
        cv = [m_s(rows[s][f"{i}_form"][nm], cellmap(i)["copy"]) for s, i in keep]
        gate[nm] = (np.mean(kv) <= 0.5 and np.mean(cv) >= 1.2)
        print(f"[{lab}] m_s keep {np.mean(kv):.3f} per cell {np.round(kv, 2).tolist()} | copy {np.mean(cv):.3f} per cell {np.round(cv, 2).tolist()} -> {'OK' if gate[nm] else 'FAIL'}")
    # u ratio thin / thick for all events, same seed (P2 vs Q2)
    print("    u(thin)/u(thick) by event, geometric mean over seeds with both (counted ratio in brackets)")
    for key in rows[both[0]]["P2_form"]["U9000"]["all"] if both else []:
        r_, rc_ = [], []
        for s in both:
            a, b = rows[s]["P2_form"], rows[s]["Q2_form"]
            r_.append(np.log(a["U9000"]["all"][key] / b["U9000"]["all"][key]))
            rc_.append(np.log(a["cnt"]["all"][key] / b["cnt"]["all"][key]))
        print(f"      {key:40s} P2/Q2 u ratio {np.exp(np.mean(r_)):6.2f}   [counted {np.exp(np.mean(rc_)):6.2f}]   U9000 u (P2) {np.mean([rows[s]['P2_form']['U9000']['all'][key] for s in both]):.3f}")
    tools_ok = gate["U9000"] and gate["U3000"] and enough

    def cv_(kind, s, i, nm, j, mode="avg", t=None):
        sr = rows[s][f"{i}_{kind}"]["series"]
        d = cellmap(i)[nm]
        f = (lambda x: own(x, d), lambda x: follow(x, d, 1), lambda x: pr(x, d))[j]
        return avg(sr, f) if mode == "avg" else at(sr, f, t)

    if base is not None:
        bkeep = [(s, i) for s in sorted(base, key=int) for i in IND if form_dev(base[s][f"{i}_form"]["v"]) < 0.2]
        ok4 = True
        for kind in [f"TC0_{st}" for st in STEPS] + [f"T0m{st}" for st in STEPS]:
            for nm in ("keep", "copy"):
                for j, lab in ((0, "Own"), (1, "Follow")):
                    per = {}
                    for s, i in bkeep:
                        sr = base[s][f"{i}_{kind}"]["series"]
                        d = cellmap(i)[nm]
                        per.setdefault(s, []).append(avg(sr, (lambda x: own(x, d)) if j == 0 else (lambda x: follow(x, d, 1))))
                    pm = np.array([np.mean(v) for v in per.values()])
                    lo, hi = pm.mean() - 2 * pm.std(), pm.mean() + 2 * pm.std()
                    val = np.mean([cv_(kind, s, i, nm, j) for s, i in keep])
                    ok = lo <= val <= hi
                    ok4 &= ok
                    print(f"[gate 4] {kind} {nm} {lab}: {val:+.2f} band [{lo:+.2f}, {hi:+.2f}] -> {'OK' if ok else 'OFF'}")
        tools_ok &= ok4
    print(f"gates all OK: {tools_ok}" + ("" if tools_ok else " -> conditions reported as 'not read (gate)'"))
    verdict = lambda ok: ("holds" if ok else "does not hold") if tools_ok else "not read (gate)"

    kinds = ["VU", "VX", "S0"] + [f"TU{st}" for st in STEPS] + [f"TC{f}_{st}" for f in COUNTS for st in STEPS] + [f"T0m{st}" for st in STEPS]
    tbl = {}
    print("\n--- cells (5-point mean): Own / Follow / Pr")
    for kind in kinds:
        for nm in ("keep", "copy"):
            tbl[(kind, nm)] = np.array([[cv_(kind, s, i, nm, j) for j in range(3)] for s, i in keep])
        H = tbl[(kind, "keep")][:, 0] - tbl[(kind, "copy")][:, 0]
        tbl[(kind, "H")] = H
        k_, c_ = tbl[(kind, "keep")].mean(0), tbl[(kind, "copy")].mean(0)
        mid = abs(np.mean([cv_(kind, s, i, "keep", 0, "at", 2000) for s, i in keep]) - np.mean([cv_(kind, s, i, "keep", 0, "at", 3000) for s, i in keep])) > 0.2
        print(f"  {kind:8s} keep {k_[0]:+.2f} / {k_[1]:.2f} / {k_[2]:+.2f} | copy {c_[0]:+.2f} / {c_[1]:.2f} / {c_[2]:+.2f} | H {H.mean():+.2f}{'  MID(keep Own)' if mid else ''}")
    ks = [s for s, i in keep]

    def seedcount(x, sign):
        sm = {}
        for s, v in zip(ks, x):
            sm.setdefault(s, []).append(v)
        smv = np.array([np.mean(v) for v in sm.values()])
        return int(sum(smv * sign > 0)), len(smv)

    def against(getx, refs, fl):
        parts, sg, ok = [], set(), True
        for r in refs:
            x = getx(r)
            sign = 1 if x.mean() >= 0 else -1
            sg.add(sign)
            np_ = int(sum(x * sign > 0))
            ns, ms = seedcount(x, sign)
            ok &= abs(x.mean()) >= fl and np_ >= 0.75 * len(x) and ns >= 0.75 * ms
            parts.append(f"{x.mean():+.2f}")
        return ok and len(sg) == 1, " / ".join(parts)

    print("\n===== tool checks K1, K2 (expected to hold) =====")
    d1 = tbl[("VU", "H")] - tbl[("S0", "H")]
    n1, m1 = seedcount(d1, +1)
    print(f"K1 H(VU) - H(S0) {d1.mean():+.2f}, seeds positive {n1}/{m1} -> {verdict(d1.mean() >= 0.5 and n1 >= 0.75 * m1)}")
    vx = [k for k, (s, i) in enumerate(keep) if s in both]
    hx = tbl[("VX", "H")][vx]
    sm = {}
    for k in vx:
        sm.setdefault(keep[k][0], []).append(tbl[("VX", "H")][k])
    smv = np.array([np.mean(v) for v in sm.values()])
    n2, m2 = int(sum(smv < 0)), len(smv)
    print(f"K2 H(VX) {hx.mean() if len(hx) else float('nan'):+.2f} (H(VU) {tbl[('VU', 'H')].mean():+.2f}), seeds negative {n2}/{m2} -> {verdict(len(hx) > 0 and hx.mean() <= -0.5 and n2 >= 0.75 * m2)}")
    print("\n===== K3 (findings can come only from here) =====")
    for nm in ("keep", "copy"):
        for j, lab, fl in ((0, "Own", 0.3), (1, "Follow", 0.15), (2, "Pr", 0.5)):
            ok, tx = against(lambda st: tbl[("VU", nm)][:, j] - tbl[(f"TU{st}", nm)][:, j], list(STEPS), fl)
            print(f"  VU - TU  {nm:4s} {lab:6s}: {tx} -> {('DIFF' if ok else 'no diff') if tools_ok else 'not read (gate)'}")
    for nm in ("keep", "copy"):
        for j, lab, fl in ((0, "Own", 0.3), (1, "Follow", 0.15)):
            refs = [(f, st) for f in COUNTS for st in STEPS]
            ok, tx = against(lambda r: tbl[(f"TU{r[1]}", nm)][:, j] - tbl[(f"TC{r[0]}_{r[1]}", nm)][:, j], refs, fl)
            note = " (Follow from marked options: extrapolated u; not read as a difference)" if lab == "Follow" else ""
            print(f"  TU - TC  {nm:4s} {lab:6s}: {tx} -> {('DIFF' if ok else 'no diff (count speed explains)') if tools_ok else 'not read (gate)'}{note}")
    print("\n===== listening (keep-cell Pr; N) and K4 =====")
    for kind in ("VU", "TU10", "TU03", "TU01", "S0", "TC0_10"):
        prk = [np.mean([cv_(kind, s, i, "keep", 2, "at", t) for s, i in keep]) for t in T_EARLY]
        pn = [avg(rows[s][f"{i}_{kind}"]["series"], lambda x: pr(x, NN)) for s, i in keep]
        fn = [avg(rows[s][f"{i}_{kind}"]["series"], lambda x: follow(x, NN, int(s) % 2)) for s, i in keep]
        print(f"  {kind:7s} keep Pr t=100/300/1000 {' '.join(f'{x:+.2f}' for x in prk)} | 5-pt {tbl[(kind, 'keep')][:, 2].mean():+.2f} | N Pr_N {np.mean(pn):+.2f} Follow_N {np.mean(fn):.2f}")
    k4o, k4p = tbl[("VU", "keep")][:, 0].mean(), tbl[("VU", "keep")][:, 2].mean()
    okp, txp = against(lambda st: tbl[("VU", "keep")][:, 2] - tbl[(f"TU{st}", "keep")][:, 2], list(STEPS), 0.5)
    if k4o >= 1.0 and k4p < 0.5:
        k4 = "kept by erasing the partner's meaning (as the table; not a finding)"
    elif k4o >= 1.0 and k4p >= 0.5 and okp:
        k4 = "candidate: kept while still listening"
    elif k4o >= 1.0:
        k4 = "kept; Pr >= 0.5 but not a difference against all TU"
    else:
        k4 = "not kept (Own < 1.0)"
    print(f"K4 VU keep Own {k4o:+.2f} Pr {k4p:+.2f} (VU - TU Pr {txp}) -> {k4 if tools_ok else 'not read (gate)'}")
    print("\n--- description")
    for kind in ("VU", "VX", "TU10"):
        for nm in ("keep", "copy"):
            ms = np.array([[m_s(x, cellmap(i)[nm]) for x in rows[s][f"{i}_{kind}"]["u"]] for s, i in keep])
            print(f"  {kind} m_s {nm}: {C.fmt(ms.mean(0))}")
    for nm, getter in (("bitter", lambda x, i: x["c"][cellmap(i)["keep"]][1]), ("bland", lambda x, i: x["c3"][cellmap(i)["keep"]]),
                       ("sweet taste", lambda x, i: x["t"][0])):
        ser = np.array([[getter(x, i) for x in rows[s][f"{i}_VU"]["u"]] for s, i in keep])
        print(f"  VU u({nm}, keep domain): {C.fmt(ser.mean(0))}")
    for kind in ("VU", "S0", "TU10", "TC0_10"):
        for nm in ("keep", "copy"):
            ty = []
            for s, i in keep:
                fin = rows[s][f"{i}_{kind}"]["series"][-1]
                d = cellmap(i)[nm]
                ty.append(R.classify(own(fin, d), follow(fin, d, 1), R.absmax_ab(fin)))
            print(f"  types {kind} {nm}: {dict((t, ty.count(t)) for t in sorted(set(ty)))}")
    print("  per-seed H (VU / S0 / T0m10): " + " | ".join(f"{s}: {np.mean([tbl[('VU', 'H')][k] for k, c in enumerate(keep) if c[0] == s]):+.2f}/{np.mean([tbl[('S0', 'H')][k] for k, c in enumerate(keep) if c[0] == s]):+.2f}/{np.mean([tbl[('T0m10', 'H')][k] for k, c in enumerate(keep) if c[0] == s]):+.2f}" for s in sorted(set(ks), key=int)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=int)
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--base", nargs=2, type=int)
    ap.add_argument("--out")
    ap.add_argument("--regress")
    ap.add_argument("--buildcheck", type=int)
    ap.add_argument("--summary", nargs="+")
    ap.add_argument("--basefile")
    a = ap.parse_args()
    if a.time is not None:
        t0 = time.time()
        per_seed(a.time)
        print("one seed", f"{time.time() - t0:.0f}s")
    elif a.regress:
        regress(a.regress)
    elif a.buildcheck is not None:
        buildcheck(a.buildcheck)
    elif a.summary:
        summarize(a.summary, a.basefile)
    else:
        t0 = time.time()
        rng = a.base or a.seeds
        rows = {}
        for s in range(*rng):
            rows[str(s)] = per_seed(s, base=a.base is not None)
            print("seed", s, f"{time.time() - t0:.0f}s", flush=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": rng, "base": a.base is not None, "times": [0] + [t for t in TIMES if t > 0], "rows": rows}, fh)
        print("written", a.out)
