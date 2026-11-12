"""Exploration 4: the partner also reacts after unwished choices, and tables start from the network (DESIGN-reaction).

Prototype seeds only, numpy only. World, formation, V, rule S, replay and wish turns are those of npn_choice.
Change (q = 1): on a no-wish turn the chosen unmarked option is followed by the partner's reaction (approve if it is
the partner's preferred side, else disapprove); half [chosen content, reaction + b_d], half [reaction + b_d, chosen
content], using the order number already drawn. q = 0 is the previous world.
Aligned tables (TBm, T0m): a table whose 29 cells are copied from the paired network at formation end.
Learners: RB (MLP, raw-record replay), S0 (MLP, no replay), TBm, T0m.

usage:
  python npn_reaction.py --time 999
  python npn_reaction.py --seeds 0 8 --out ../p74_reaction_proto.json
  python npn_reaction.py --summary ../p74_reaction_proto.json --prev ../p73_choice_proto.json
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
from npn_choice import (A, B, NN, INDIVS, T_MEASURE, World, item, formation_turns, wished_side, snap, softmax2,  # noqa: E402
                        Replay, own, follow, pr, form_dev)
from npn_leafleak import MLP, Tab, run_turn  # noqa: E402

N_DYAD = C.N_DYAD
STEPS = {"10": 0.1, "03": 0.03, "01": 0.01}  # table step sizes, fixed before running (CRITIQUE-reaction-1 D1)
CONTROL = False  # RBnw / TBnw (wish turns not learned). Off: one seed took 55 s with it, over the 5-minute line (D4)
RUNS = {1: [("RB", "M"), ("RB", "K"), ("S0", "M")] + ([("RBnw", "M")] if CONTROL else [])
           + [r for st in STEPS for r in ((f"TBm{st}", "M"), (f"TBm{st}", "K"), (f"T0m{st}", "M"))]
           + ([(f"TBnw{st}", "M") for st in STEPS] if CONTROL else []),
        0: [r for st in STEPS for r in ((f"TBm{st}", "M"), (f"TBm{st}", "K"), (f"T0m{st}", "M"))]}
REGRESSION_SEED = 0
TABLE_OF = {"RB": "TBm", "S0": "T0m", "RBnw": "TBnw"}
# replication (DESIGN-reaction-rep): q = 1 only, control on, extra measuring times; read the mean of t >= 2000
RUNS_REP = {1: [("RB", "M"), ("RB", "K"), ("S0", "M"), ("RBnw", "M")]
               + [r for st in STEPS for r in ((f"TBm{st}", "M"), (f"TBm{st}", "K"), (f"T0m{st}", "M"), (f"TBnw{st}", "M"))],
            0: [("RB", "M")] + [(f"TBm{st}", "M") for st in STEPS]}  # C6 (CRITIQUE-reaction-rep-1, 3-6)
T_REP = sorted(T_MEASURE + [2250, 2500, 2750])
T_READ = [2000, 2250, 2500, 2750, 3000]
TIMES = list(T_MEASURE)


class TabS(Tab):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def learn(self, e, world, target, noise):
        v = self.v.get(e, 0.0)
        self.v[e] = v + self.step * (target - v)


def all_events():
    ev = [("c", k + (3 if d == B else 0), d) for d in (A, B) for k in range(3)]
    ev += [("t", i) for i in range(3)]
    ev += [("rb", i, d) for i in range(2) for d in range(3)]
    ev += [("n", k) for k in range(2)]
    ev += [("m", item(d, k), s) for d in range(3) for k in range(2) for s in range(2)]
    return ev


def aligned_tab(net, w, step):
    t = TabS(step)
    for e in all_events():
        t.v[e] = float(net.value(e, w))
    return t


def dyad(kind, partner, V0, record, w, seed, q):
    """kind: RB / S0 / RBnw (network) or TBm.. / T0m.. / TBnw.. (aligned tables). *nw: wish turns choose but do not learn."""
    m = copy.deepcopy(V0)
    rep = Replay("RB", record) if kind.startswith(("RB", "TB")) else None
    learn_wish = not kind.startswith(("RBnw", "TBnw"))
    rc = np.random.default_rng([seed, 3, 1])
    rs = np.random.default_rng([seed, 29, 1])
    rn = np.random.default_rng([seed, 7, 1])
    rn2 = np.random.default_rng([seed, 7, 2])
    rr = np.random.default_rng([seed, 17, 1])
    series, log = [snap(m, w)], []
    for t in range(1, N_DYAD + 1):
        d, wish, order, u = int(rc.integers(3)), rc.random() < C.P_WISH, rc.random(), rs.random()
        ws = wished_side(partner, d, seed)
        if wish:
            opts = [("m", item(d, k), 0 if k == ws else 1) for k in range(2)]
            p = softmax2(m.value(opts[0], w), m.value(opts[1], w))
            ch = 0 if u < p[0] else 1
            follow_ = ch == ws
            react = ("rb", 0 if follow_ else 1, d)
            if learn_wish:
                run_turn(m, w, [opts[ch], react] if order < 0.5 else [react, item(d, ch)], rn)
            log.append((d, 1, int(follow_)))
        else:
            p = softmax2(m.value(item(d, 0), w), m.value(item(d, 1), w))
            ch = 0 if u < p[0] else 1
            if q:
                react = ("rb", 0 if ch == ws else 1, d)
                run_turn(m, w, [item(d, ch), react] if order < 0.5 else [react, item(d, ch)], rn)
            log.append((d, 0, int(ch == 0)))
        if rep is not None:
            run_turn(m, w, rep.pick(rr), rn2)
        if t in TIMES:
            series.append(snap(m, w))
    if isinstance(m, Tab):
        assert len(m.v) == len(all_events()), "a table learned an event outside the aligned cells"
    return {"series": series, "log": log}


def per_seed(seed, rep=False):
    w = World(seed)
    out = {}
    for indiv, (cd, rd) in INDIVS.items():
        record = formation_turns(seed, cd, rd)
        net = MLP(seed)
        rn = np.random.default_rng([seed, 7, 0])
        for ev in record:
            run_turn(net, w, ev, rn)
        tabs = {st: aligned_tab(net, w, sz) for st, sz in STEPS.items()}
        out[f"{indiv}_form_mlp"] = snap(net, w)
        out[f"{indiv}_form_tabm"] = snap(tabs["10"], w)
        runs = {q: list(r) for q, r in (RUNS_REP if rep else RUNS).items()}
        if seed == REGRESSION_SEED and not rep:
            runs[0] += [("RB", "M"), ("S0", "M")]
        for q, lst in runs.items():
            for kind, partner in lst:
                V0 = tabs[kind[-2:]] if kind.startswith("T") else net
                out[f"{indiv}_q{q}_{kind}_{partner}"] = dyad(kind, partner, V0, record, w, seed, q)
    return out


# ---------- summary (DESIGN-reaction sections 5 and 7) ----------

def flat(s):
    return np.concatenate([np.ravel(s["u"]), np.ravel(s["m"]), np.ravel(s["rb"]), np.ravel(s["t"]), np.ravel(s["c3"])])


def own_n(s, ws):
    return s["u"][NN][ws] - s["u"][NN][1 - ws]


def absmax_ab(s):
    return max(abs(s["u"][d][k]) for d in (A, B) for k in range(2))


def classify(o, f, am):
    if am > 3:
        return "collapse-drift"
    if o <= 0.5 and f >= 0.7:
        return "copy"
    if o >= 1.0 and f >= 0.7:
        return "comply"
    if o >= 1.0 and f <= 0.3:
        return "keep"
    if o <= 0.5 and 0.3 < f < 0.7:
        return "collapse-shrink"
    return "mixed"


def cells_of(keep):
    """(seed, indiv, domain, content thick, reaction thick)."""
    out = []
    for s, i in keep:
        cd, rd = INDIVS[i]
        for d in (A, B):
            out.append((s, i, d, int(d == cd), int(d == rd)))
    return out


REF3 = {(1, 1): (1.73, 0.70), (0, 1): (0.26, 0.92), (1, 0): (1.96, 0.06), (0, 0): (1.66, 0.13)}  # CRITIQUE-reaction-1 B2/B3, step 0.1
KEYS = ((1, 1), (0, 1), (1, 0), (0, 0))


def summarize(path, prev):
    data = json.load(open(path, encoding="utf-8"))["rows"]
    old = json.load(open(prev, encoding="utf-8"))["rows"] if prev else None
    seeds = sorted(data, key=int)
    allc = [(s, i) for s in seeds for i in INDIVS]
    dev = {c: form_dev(data[c[0]][f"{c[1]}_form_mlp"]) for c in allc}
    keep = [c for c in allc if dev[c] < 0.2]
    print(f"[5] formation end (mlp) kept {len(keep)}/{len(allc)}; excluded {[c for c in allc if c not in keep]}")
    if old is not None:
        okeep = [c for c in allc if form_dev(old[c[0]][f"{c[1]}_form_mlp"]) < 0.2 and form_dev(old[c[0]][f"{c[1]}_form_tab"]) < 0.2]
        print(f"    same cells as p73: {okeep == keep}")
    cells = cells_of(keep)
    ser = lambda s, i, q, k, p: data[s][f"{i}_q{q}_{k}_{p}"]["series"]

    # [1] aligned tables at t = 0 equal the network on all 29 values
    md = 0.0
    for s, i in keep:
        a = data[s][f"{i}_form_mlp"]
        md = max(md, float(np.max(np.abs(flat(a) - flat(data[s][f"{i}_form_tabm"])))))
        for q in (0, 1):
            for k, p in RUNS[q]:
                if k.startswith("T"):
                    md = max(md, float(np.max(np.abs(flat(ser(s, i, q, k, p)[0]) - flat(a)))))
    print(f"[1] aligned tables vs network at t=0 (29 values each): max |diff| {md:.2e} -> {'OK' if md < 0.01 else 'FAIL'}")

    # [2] regression against p73
    if old is not None:
        s = str(REGRESSION_SEED)
        md2, lg = 0.0, True
        for i in INDIVS:
            for k in ("RB", "S0"):
                a, b = data[s][f"{i}_q0_{k}_M"], old[s][f"{i}_{k}_M"]
                md2 = max(md2, max(float(np.max(np.abs(flat(x) - flat(y)))) for x, y in zip(a["series"], b["series"])))
                lg &= [tuple(x) for x in a["log"]] == [tuple(x) for x in b["log"]]
        print(f"[2] regression q=0 seed {s} RB/S0 x M vs p73: max |diff| {md2:.2e}, logs equal {lg} -> {'OK' if md2 < 1e-6 and lg else 'FAIL'}")

    # [3] hand calculation (TBm10, q=1, M)
    print("[3] TBm10 q=1 M vs critique B2/B3 (tolerance Own 0.25, Follow 0.10)")
    allok = True
    for key in KEYS:
        cc = [c for c in cells if (c[3], c[4]) == key]
        o = np.mean([own(ser(s, i, 1, "TBm10", "M")[-1], d) for s, i, d, *_ in cc])
        f = np.mean([follow(ser(s, i, 1, "TBm10", "M")[-1], d, 1) for s, i, d, *_ in cc])
        p_ = np.mean([pr(ser(s, i, 1, "TBm10", "M")[-1], d) for s, i, d, *_ in cc])
        ok = abs(o - REF3[key][0]) <= 0.25 and abs(f - REF3[key][1]) <= 0.10
        allok &= ok
        print(f"    cell {key}: Own {o:+.2f} (ref {REF3[key][0]:+.2f}) Follow {f:.2f} (ref {REF3[key][1]:.2f}) Pr {p_:+.2f} -> {'OK' if ok else 'OFF'}")
    o = np.mean([own(ser(s, i, 1, "T0m10", "M")[-1], d) for s, i, d, *_ in cells])
    f = np.mean([follow(ser(s, i, 1, "T0m10", "M")[-1], d, 1) for s, i, d, *_ in cells])
    p_ = np.mean([pr(ser(s, i, 1, "T0m10", "M")[-1], d) for s, i, d, *_ in cells])
    print(f"    T0m10 q=1 (all cells): Own {o:+.2f} (critique ~0.95) Follow {f:.2f} (~0.10) Pr {p_:+.2f} (~-0.85) | [3] {'OK' if allok else 'OFF'}")

    # [6] p73 S0 collapse split
    if old is not None:
        ab, nonly = 0, 0
        for s, i in keep:
            fin = old[s][f"{i}_S0_M"]["series"][-1]
            amab, amn = absmax_ab(fin), max(abs(x) for x in fin["u"][NN])
            ab += int(amab > 3)
            nonly += int(amab <= 3 and amn > 3)
        print(f"[6] p73 S0 x M |V|>3: A/B drift {ab}/{len(keep)}, N-only {nonly}/{len(keep)}")

    # [4] starting-point contribution (q = 0)
    print("[4] q=0 starting-point check")
    me = {1: [], 0: []}
    for s, i, d, ct, rt in cells:
        f0 = data[s][f"{i}_form_mlp"]
        me[ct].append([f0["m"][d][1][0] - f0["u"][d][1], f0["m"][d][0][1] - f0["u"][d][0]])
    for ct in (1, 0):
        x = np.array(me[ct])
        print(f"    t=0 mark effect (bitter+w+, sweet+w-), content-{'thick' if ct else 'thin'}: mean {np.round(x.mean(0), 3)}, max|.|>=0.3 in {int(sum(np.max(np.abs(x), 1) >= 0.3))}/{len(x)}")
    if old is not None:
        pairs = [(s, a, b, d) for s in seeds for a, b, d in (("P", "Q2", A), ("Q", "P2", B)) if (s, a) in keep and (s, b) in keep]
        srcs = [("RB (p73)", lambda s, i: old[s][f"{i}_RB_M"]["series"]), ("TB (p73, from 0)", lambda s, i: old[s][f"{i}_TB_M"]["series"])]
        srcs += [(f"TBm{st} (q=0)", (lambda st: lambda s, i: ser(s, i, 0, f"TBm{st}", "M"))(st)) for st in STEPS]
        av = {}
        for lab, g in srcs:
            dd = np.array([follow(g(s, a)[-1], d, 1) - follow(g(s, b)[-1], d, 1) for s, a, b, d in pairs])
            av[lab] = dd
            print(f"    (a) reaction-thick, content-thick minus thin, Follow_M(3000): {lab} {dd.mean():+.3f} (neg {sum(dd < 0)}/{len(dd)})")
        for st in STEPS:
            x = av["RB (p73)"] - av[f"TBm{st} (q=0)"]
            print(f"        RB pair-diff minus TBm{st} pair-diff: {x.mean():+.3f} (neg {sum(x < 0)}/{len(x)})")
        rbm = [follow(old[s][f"{i}_RB_M"]["series"][-1], NN, int(s) % 2) for s, i in keep]
        rbk = [follow(old[s][f"{i}_RB_K"]["series"][-1], NN, int(s) % 2) for s, i in keep]
        print(f"    (b) N Follow_N(3000): RB (p73) M {np.mean(rbm):.2f} K {np.mean(rbk):.2f}; t=0 shared start {np.mean([follow(data[s][f'{i}_form_mlp'], NN, int(s) % 2) for s, i in keep]):.2f}")
        for st in STEPS:
            fm = [follow(ser(s, i, 0, f"TBm{st}", "M")[-1], NN, int(s) % 2) for s, i in keep]
            fk = [follow(ser(s, i, 0, f"TBm{st}", "K")[-1], NN, int(s) % 2) for s, i in keep]
            print(f"        TBm{st} (q=0) M {np.mean(fm):.2f} K {np.mean(fk):.2f} | RB - TBm{st}: M {np.mean(rbm) - np.mean(fm):+.2f} K {np.mean(rbk) - np.mean(fk):+.2f}")
        g = {key: [] for key in KEYS}
        for s, i, d, ct, rt in cells:
            g[(ct, rt)].append(follow(old[s][f"{i}_RB_M"]["series"][-1], d, 1))
        print("    q=0 RB (p73) Follow_M by cell: " + " | ".join(f"{k}: {np.mean(v):.2f}" for k, v in g.items()))
        for st in STEPS:
            for k in ("TBm", "T0m"):
                g = {key: [] for key in KEYS}
                for s, i, d, ct, rt in cells:
                    g[(ct, rt)].append(follow(ser(s, i, 0, f"{k}{st}", "M")[-1], d, 1))
                fl = sum(pr(ser(s, i, 0, f"{k}{st}", "M")[-1], d) < -0.3 for s, i, d, *_ in cells)
                print(f"    q=0 {k}{st} Follow_M by cell: " + " | ".join(f"{kk}: {np.mean(v):.2f}" for kk, v in g.items()) + f" | reverse flag (all cells) {fl}/{len(cells)}")

    # main: q = 1, partner M
    print("\n===== q=1, partner M, t=3000. cells (content thick, reaction thick). DIFF = same sign and past the floor against all 3 table steps =====")
    ti = {t: j for j, t in enumerate([0] + [t for t in T_MEASURE if t > 0])}

    def block(k, cc):
        ss = [ser(s, i, 1, k, "M") for s, i, d, ct, rt in cc]
        ds = [c[2] for c in cc]
        return {"own": np.array([[own(x, d) for x in sr] for sr, d in zip(ss, ds)]),
                "fol": np.array([[follow(x, d, 1) for x in sr] for sr, d in zip(ss, ds)]),
                "pr": np.array([[pr(x, d) for x in sr] for sr, d in zip(ss, ds)]),
                "am": [absmax_ab(sr[-1]) for sr in ss]}

    def line(lab, r):
        types = [classify(o, f, a) for o, f, a in zip(r["own"][:, -1], r["fol"][:, -1], r["am"])]
        cnt = {t: types.count(t) for t in sorted(set(types))}
        mid = []
        if abs(r["own"][:, ti[2000]].mean() - r["own"][:, -1].mean()) > 0.2:
            mid.append("MID(Own)")
        if abs(r["fol"][:, ti[2000]].mean() - r["fol"][:, -1].mean()) > 0.1:
            mid.append("MID(Follow)")
        print(f"    {lab:7s}: Own {r['own'][:, -1].mean():+.2f} Follow {r['fol'][:, -1].mean():.2f} Pr {r['pr'][:, -1].mean():+.2f} | {cnt} | reversal {int(sum(r['own'][:, -1] < -0.3))} reverse-reading {int(sum(r['pr'][:, -1] < -0.3))} {' '.join(mid)}")

    for net in ("RB", "RBnw", "S0") if CONTROL else ("RB", "S0"):
        tb = TABLE_OF[net]
        print(f"--- {net} vs {tb}(10/03/01)")
        for key in KEYS:
            cc = [c for c in cells if (c[3], c[4]) == key]
            n = len(cc)
            rn_ = block(net, cc)
            print(f"  cell {key} n={n}")
            line(net, rn_)
            print(f"           Own series {C.fmt(rn_['own'].mean(0))}")
            print(f"           Fol series {C.fmt(rn_['fol'].mean(0))}")
            res = {"own": [], "fol": []}
            for st in STEPS:
                rt_ = block(f"{tb}{st}", cc)
                line(f"{tb}{st}", rt_)
                for nm in ("own", "fol"):
                    dd = rn_[nm][:, -1] - rt_[nm][:, -1]
                    sd = {}
                    for c, x in zip(cc, dd):
                        sd.setdefault(c[0], []).append(x)
                    sm = np.array([np.mean(v) for v in sd.values()])
                    res[nm].append((dd.mean(), max(sum(dd > 0), sum(dd < 0)), max(sum(sm > 0), sum(sm < 0)), len(sm)))
            for nm, lim in (("own", 0.3), ("fol", 0.15)):
                r = res[nm]
                sgn = {np.sign(x[0]) for x in r}
                ok = len(sgn) == 1 and all(abs(x[0]) >= lim and x[1] >= 0.75 * n for x in r)
                print(f"    d{nm} vs steps 10/03/01: " + " / ".join(f"{x[0]:+.2f} (pairs {x[1]}/{n}, seeds {x[2]}/{x[3]})" for x in r) + f" -> {'DIFF' if ok else 'no diff'}")
    # RB minus RBnw directly (does learning on wish turns move the unwished choice?)
    print("--- RB - RBnw (network), and TBm - TBnw (table), Own at 3000 by cell" + ("" if CONTROL else ": control not run"))
    for key in KEYS if CONTROL else ():
        cc = [c for c in cells if (c[3], c[4]) == key]
        x = block("RB", cc)["own"][:, -1] - block("RBnw", cc)["own"][:, -1]
        y = [(block(f"TBm{st}", cc)["own"][:, -1] - block(f"TBnw{st}", cc)["own"][:, -1]).mean() for st in STEPS]
        print(f"  cell {key}: RB-RBnw {x.mean():+.2f} (neg {sum(x < 0)}/{len(x)}) | TBm-TBnw 10/03/01 " + " / ".join(f"{v:+.2f}" for v in y))
    print("\n--- by individual group (PQ: same kind both domains, P2Q2: crossed), q=1 M, t=3000, Own / Follow")
    for grp, ids in (("PQ", ("P", "Q")), ("P2Q2", ("P2", "Q2"))):
        for key in KEYS:
            cc = [c for c in cells if (c[3], c[4]) == key and c[1] in ids]
            if not cc:
                continue
            parts = []
            for k in ("RB", "TBm10", "TBm03", "TBm01", "S0", "T0m10"):
                r = block(k, cc)
                parts.append(f"{k} {r['own'][:, -1].mean():+.2f}/{r['fol'][:, -1].mean():.2f}")
            print(f"  {grp} cell {key} n={len(cc)}: " + " | ".join(parts))
    print("\n--- K (A/B, t=3000) and N (q=1)")
    on0 = [own_n(data[s][f"{i}_form_mlp"], int(s) % 2) for s, i in keep]
    print(f"  N at t=0: Own_N {np.mean(on0):+.2f}")
    for k in ["RB"] + [f"TBm{st}" for st in STEPS]:
        oK = np.mean([own(ser(s, i, 1, k, "K")[-1], d) for s, i, d, *_ in cells])
        fK = np.mean([follow(ser(s, i, 1, k, "K")[-1], d, 0) for s, i, d, *_ in cells])
        res = {}
        for p in ("M", "K"):
            res[p] = {"own": np.array([own_n(ser(s, i, 1, k, p)[-1], int(s) % 2) for s, i in keep]),
                      "fol": np.array([follow(ser(s, i, 1, k, p)[-1], NN, int(s) % 2) for s, i in keep]),
                      "pr": np.array([pr(ser(s, i, 1, k, p)[-1], NN) for s, i in keep])}
        dfo, doo = res["M"]["fol"] - res["K"]["fol"], res["M"]["own"] - res["K"]["own"]
        print(f"  {k:6s}: K A/B Own {oK:+.2f} Follow {fK:.2f} | N M: Own_N {res['M']['own'].mean():+.2f} Follow_N {res['M']['fol'].mean():.2f} Pr_N {res['M']['pr'].mean():+.2f}"
              f" | N K: Own_N {res['K']['own'].mean():+.2f} Follow_N {res['K']['fol'].mean():.2f} Pr_N {res['K']['pr'].mean():+.2f}"
              f" | M-K Follow_N {dfo.mean():+.2f} (neg {sum(dfo < 0)}/{len(dfo)}) Own_N {doo.mean():+.2f} (neg {sum(doo < 0)}/{len(doo)})")
    am = [absmax_ab(ser(s, i, 1, "S0", "M")[-1]) for s, i in keep]
    print(f"  S0 q=1 M: A/B |V| max mean {np.mean(am):.2f}, max {np.max(am):.2f}, >3 in {sum(x > 3 for x in am)}/{len(am)}")
    for k in ("RB", "TBm10", "TBm01"):
        fo, gs = [], []
        for s, i, d, *_ in cells:
            lg = data[s][f"{i}_q1_{k}_M"]["log"][-500:]
            fo += [f for dd, wf, f in lg if wf and dd == d]
            gs += [f for dd, wf, f in lg if not wf and dd == d]
        print(f"  {k} q=1 M last 500 actual: follow {np.mean(fo):.2f}, chose sweet when unwished {np.mean(gs):.2f}")



# ---------- replication summary (DESIGN-reaction-rep v1.1, section 2 and 6) ----------

REF_T0M = (0.95, 0.10)


def regress(prev):
    """Cheap regression: seed 0, individual P, q=1 RB x M and TBm10 x M with the replication settings vs p74."""
    old = json.load(open(prev, encoding="utf-8"))["rows"]["0"]
    TIMES[:] = T_REP
    seed, indiv = 0, "P"
    w = World(seed)
    cd, rd = INDIVS[indiv]
    record = formation_turns(seed, cd, rd)
    net = MLP(seed)
    rn = np.random.default_rng([seed, 7, 0])
    for ev in record:
        run_turn(net, w, ev, rn)
    tab = aligned_tab(net, w, 0.1)
    times = [0] + [t for t in T_REP if t > 0]
    otimes = [0] + [t for t in T_MEASURE if t > 0]
    md = 0.0
    for kind, V0 in (("RB", net), ("TBm10", tab)):
        new = dyad(kind, "M", V0, record, w, seed, 1)["series"]
        ref = old[f"{indiv}_q1_{kind}_M"]["series"]
        for j, t in enumerate(otimes):
            md = max(md, float(np.max(np.abs(flat(new[times.index(t)]) - flat(ref[j])))))
    print(f"[R] regression seed 0 P, q=1 RB/TBm10 x M with replication settings vs p74: max |diff| {md:.2e} -> {'OK' if md < 1e-6 else 'FAIL'}")


def summarize_rep(paths):
    rows, times = {}, None
    for p in paths:
        dd = json.load(open(p, encoding="utf-8"))
        rows.update(dd["rows"])
        times = dd["times"]
    idx = [times.index(t) for t in T_READ]
    seeds = sorted(rows, key=int)
    allc = [(s, i) for s in seeds for i in INDIVS]
    keep = [c for c in allc if form_dev(rows[c[0]][f"{c[1]}_form_mlp"]) < 0.2]
    kseeds = sorted({s for s, i in keep}, key=int)
    print(f"[5] kept {len(keep)}/{len(allc)} cells, seeds with any kept cell {len(kseeds)}: excluded {[c for c in allc if c not in keep]}")
    enough = len(kseeds) >= 6
    print(f"    seeds >= 6: {enough}" + ("" if enough else " -> every candidate is 'not confirmed (too few)'"))
    cells = cells_of(keep)
    ser = lambda s, i, q, k, p: rows[s][f"{i}_q{q}_{k}_{p}"]["series"]
    avg = lambda sr, f: float(np.mean([f(sr[j]) for j in idx]))

    md = 0.0
    for s, i in keep:
        a = rows[s][f"{i}_form_mlp"]
        for q, lst in RUNS_REP.items():
            for k, p in lst:
                if k.startswith("T"):
                    md = max(md, float(np.max(np.abs(flat(ser(s, i, q, k, p)[0]) - flat(a)))))
    print(f"[1] aligned tables vs network at t=0 (29 values): max |diff| {md:.2e} -> {'OK' if md < 0.01 else 'FAIL'}")
    mm = 0.0
    for s, i in keep:
        for st in STEPS:
            sr = ser(s, i, 1, f"TBnw{st}", "M")
            mm = max(mm, max(float(np.max(np.abs(np.ravel(x["m"]) - np.ravel(sr[0]["m"])))) for x in sr))
    print(f"[NW] TBnw marked options never change: max |change| {mm:.2e} -> {'OK' if mm < 1e-9 else 'FAIL'}")
    ok3 = True
    print("[3] TBm10 q=1 M, 5-point mean, vs critique B2/B3 (Own +-0.25, Follow +-0.10)")
    for key in KEYS:
        cc = [c for c in cells if (c[3], c[4]) == key]
        o = np.mean([avg(ser(s, i, 1, "TBm10", "M"), lambda x: own(x, d)) for s, i, d, *_ in cc])
        f = np.mean([avg(ser(s, i, 1, "TBm10", "M"), lambda x: follow(x, d, 1)) for s, i, d, *_ in cc])
        ok = abs(o - REF3[key][0]) <= 0.25 and abs(f - REF3[key][1]) <= 0.10
        ok3 &= ok
        print(f"    cell {key} n={len(cc)}: Own {o:+.2f} (ref {REF3[key][0]:+.2f}) Follow {f:.2f} (ref {REF3[key][1]:.2f}) -> {'OK' if ok else 'OFF'}")
    o = np.mean([avg(ser(s, i, 1, "T0m10", "M"), lambda x: own(x, d)) for s, i, d, *_ in cells])
    f = np.mean([avg(ser(s, i, 1, "T0m10", "M"), lambda x: follow(x, d, 1)) for s, i, d, *_ in cells])
    ok = abs(o - REF_T0M[0]) <= 0.25 and abs(f - REF_T0M[1]) <= 0.10
    ok3 &= ok
    print(f"    T0m10 all cells: Own {o:+.2f} (ref {REF_T0M[0]:+.2f}) Follow {f:.2f} (ref {REF_T0M[1]:.2f}) -> {'OK' if ok else 'OFF'}")
    tools_ok = md < 0.01 and mm < 1e-9 and ok3 and enough
    print(f"    instruments all OK: {tools_ok}" + ("" if tools_ok else " -> candidates are 'not confirmed (instrument)'"))

    def test(diffs, keys, floor, sign):
        """diffs: list over the 3 steps of arrays over pairs; keys: seed of each pair. Pass when every step has mean*sign >= floor
        and pairs and seed-means agree in sign at >= 3/4."""
        parts, ok = [], True
        for dd in diffs:
            dd = np.asarray(dd)
            sm = {}
            for s, x in zip(keys, dd):
                sm.setdefault(s, []).append(x)
            smv = np.array([np.mean(v) for v in sm.values()])
            np_ = int(sum(dd * sign > 0))
            ns = int(sum(smv * sign > 0))
            ok &= dd.mean() * sign >= floor and np_ >= 0.75 * len(dd) and ns >= 0.75 * len(smv)
            parts.append(f"{dd.mean():+.2f} (pairs {np_}/{len(dd)}, seeds {ns}/{len(smv)})")
        return ok, " / ".join(parts)

    verdict = lambda ok: ("confirmed" if ok else "not confirmed") if tools_ok else "not confirmed (instrument)"
    print("\n===== candidates (5-point mean, t = 2000..3000) =====")
    # C1
    c1 = True
    print("C1 S0 - T0m (Own negative, Follow positive), by cell; steps 10/03/01")
    for key in KEYS:
        cc = [c for c in cells if (c[3], c[4]) == key]
        ks = [c[0] for c in cc]
        so = [avg(ser(s, i, 1, "S0", "M"), lambda x: own(x, d)) for s, i, d, *_ in cc]
        sf = [avg(ser(s, i, 1, "S0", "M"), lambda x: follow(x, d, 1)) for s, i, d, *_ in cc]
        do = [np.array(so) - np.array([avg(ser(s, i, 1, f"T0m{st}", "M"), lambda x: own(x, d)) for s, i, d, *_ in cc]) for st in STEPS]
        df = [np.array(sf) - np.array([avg(ser(s, i, 1, f"T0m{st}", "M"), lambda x: follow(x, d, 1)) for s, i, d, *_ in cc]) for st in STEPS]
        oko, to = test(do, ks, 0.3, -1)
        okf, tf = test(df, ks, 0.15, +1)
        c1 &= oko and okf
        print(f"  cell {key} n={len(cc)}: S0 Own {np.mean(so):+.2f} Follow {np.mean(sf):.2f} | dOwn {to} | dFollow {tf}")
    print(f"  C1 -> {verdict(c1)} (expected by construction; not a discovery)")
    # C2
    om = np.array([avg(ser(s, i, 1, "RB", "M"), lambda x: own_n(x, int(s) % 2)) for s, i in keep])
    ok_ = np.array([avg(ser(s, i, 1, "RB", "K"), lambda x: own_n(x, int(s) % 2)) for s, i in keep])
    pm = np.array([avg(ser(s, i, 1, "RB", "M"), lambda x: pr(x, NN)) for s, i in keep])
    pk = np.array([avg(ser(s, i, 1, "RB", "K"), lambda x: pr(x, NN)) for s, i in keep])
    fm = np.array([avg(ser(s, i, 1, "RB", "M"), lambda x: follow(x, NN, int(s) % 2)) for s, i in keep])
    fk = np.array([avg(ser(s, i, 1, "RB", "K"), lambda x: follow(x, NN, int(s) % 2)) for s, i in keep])
    ks = [s for s, i in keep]
    c2, t2 = test([om - ok_], ks, 0.3, -1)
    _, tp = test([pm - pk], ks, 0.0, -1)
    tabz = max(abs(avg(ser(s, i, 1, f"TBm{st}", "M"), lambda x: own_n(x, int(s) % 2)) - avg(ser(s, i, 1, f"TBm{st}", "K"), lambda x: own_n(x, int(s) % 2))) for s, i in keep for st in STEPS)
    print(f"C2 RB N: Own_N M {om.mean():+.2f} K {ok_.mean():+.2f}, M-K {t2} | Pr_N M {pm.mean():+.2f} K {pk.mean():+.2f}, M-K {tp} | Follow_N M {fm.mean():.2f} K {fk.mean():.2f} | tables max |M-K| {tabz:.1e}")
    print(f"  C2 -> {verdict(c2)}")
    # C3, C5
    for name, key, nm, floor, sign in (("C3", (1, 1), "fol", 0.15, -1), ("C5", (0, 1), "own", 0.3, 0)):
        cc = [c for c in cells if (c[3], c[4]) == key]
        ks = [c[0] for c in cc]
        f = (lambda d: (lambda x: follow(x, d, 1))) if nm == "fol" else (lambda d: (lambda x: own(x, d)))
        rb = np.array([avg(ser(s, i, 1, "RB", "M"), f(d)) for s, i, d, *_ in cc])
        diffs = [rb - np.array([avg(ser(s, i, 1, f"TBm{st}", "M"), f(d)) for s, i, d, *_ in cc]) for st in STEPS]
        if sign == 0:
            okp, tx = test(diffs, ks, floor, +1)
            okn, _ = test(diffs, ks, floor, -1)
            ok, lab = okp or okn, ("deeper copy in RB" if okn else ("shallower in RB" if okp else "no diff"))
        else:
            ok, tx = test(diffs, ks, floor, sign)
            lab = "RB follows less" if ok else "no diff"
        tabs = " / ".join(f"{np.mean([avg(ser(s, i, 1, f'TBm{st}', 'M'), f(d)) for s, i, d, *_ in cc]):+.2f}" for st in STEPS)
        print(f"{name} cell {key} {nm} n={len(cc)}: RB {rb.mean():+.2f} | TBm 10/03/01 {tabs} | RB - TBm {tx} -> {lab} ({verdict(ok)})")
    print("  C3 is expected from shared reaction elements; not a discovery")
    # C4
    print("C4 (RB - RBnw) - (TBm - TBnw), Own, by cell; steps 10/03/01")
    for key in KEYS:
        cc = [c for c in cells if (c[3], c[4]) == key]
        ks = [c[0] for c in cc]
        g = lambda k: np.array([avg(ser(s, i, 1, k, "M"), lambda x: own(x, d)) for s, i, d, *_ in cc])
        net = g("RB") - g("RBnw")
        diffs = [net - (g(f"TBm{st}") - g(f"TBnw{st}")) for st in STEPS]
        okp, tx = test(diffs, ks, 0.3, +1)
        okn, _ = test(diffs, ks, 0.3, -1)
        lab = "moved: supports" if okp else ("moved: lowers" if okn else "not moved")
        print(f"  cell {key} n={len(cc)}: RB-RBnw {net.mean():+.2f} | {tx} -> {lab}{'' if tools_ok else ' (instrument)'}")
    # C6
    pairs = [(s, a, b, d) for s in seeds for a, b, d in (("P", "Q2", A), ("Q", "P2", B)) if (s, a) in keep and (s, b) in keep]
    ks = [p[0] for p in pairs]
    pd = lambda k: np.array([avg(ser(s, a, 0, k, "M"), lambda x: follow(x, d, 1)) - avg(ser(s, b, 0, k, "M"), lambda x: follow(x, d, 1)) for s, a, b, d in pairs])
    rbp = pd("RB")
    c6, t6 = test([rbp - pd(f"TBm{st}") for st in STEPS], ks, 0.15, -1)
    print(f"C6 q=0 reaction-thick, content-thick minus thin Follow_M: RB {rbp.mean():+.3f} (neg {sum(rbp < 0)}/{len(rbp)}) | RB - TBm 10/03/01 {t6} -> {verdict(c6)}")
    # types (description)
    print("\n--- types at 3000 by cell (q=1 M): RB / S0")
    for key in KEYS:
        cc = [c for c in cells if (c[3], c[4]) == key]
        out = []
        for k in ("RB", "S0", "TBm10"):
            ty = [classify(own(ser(s, i, 1, k, "M")[-1], d), follow(ser(s, i, 1, k, "M")[-1], d, 1), absmax_ab(ser(s, i, 1, k, "M")[-1])) for s, i, d, *_ in cc]
            out.append(f"{k} {dict((t, ty.count(t)) for t in sorted(set(ty)))}")
        print(f"  cell {key}: " + " | ".join(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=int)
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    ap.add_argument("--summary")
    ap.add_argument("--prev")
    ap.add_argument("--rep", action="store_true", help="replication run (DESIGN-reaction-rep)")
    ap.add_argument("--summary-rep", nargs="+")
    ap.add_argument("--regress")
    a = ap.parse_args()
    if a.time is not None:
        t0 = time.time()
        r = per_seed(a.time)
        fin = all(np.isfinite(v) for k, row in r.items() if "_form_" not in k for v in np.ravel(row["series"][-1]["u"]))
        print("one seed", f"{time.time() - t0:.0f}s", "finite", fin)
    elif a.summary:
        summarize(a.summary, a.prev)
    elif a.summary_rep:
        summarize_rep(a.summary_rep)
    elif a.regress:
        regress(a.regress)
    else:
        if a.rep:
            TIMES[:] = T_REP
        t0 = time.time()
        rows = {}
        for s in range(*a.seeds):
            rows[str(s)] = per_seed(s, rep=a.rep)
            print("seed", s, f"{time.time() - t0:.0f}s", flush=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "rep": a.rep, "times": [0] + [t for t in TIMES if t > 0], "rows": rows}, fh)
        print("written", a.out)
