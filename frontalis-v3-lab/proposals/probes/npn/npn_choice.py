"""Exploration 3: choices and the partner's wishes with approval / disapproval (DESIGN-choice v1.1).

Prototype seeds only, numpy only. Arm "split" world of npn_anchor (A: a1..a3 + b1, B: a4..a6 + b2) plus a new
domain N (contents n1 + b3, n2 + b3) and wish marks w+ / w- (new vectors from stream [s,2]). Reaction events carry
the domain mark (reaction + b_d). Individuals differ only in formation proportions: content-thick domain and
reaction-thick domain (P: A/A, Q: B/B, P2: A/B, Q2: B/A; thick = 0.9).
Formation (9000, shared by all learners of an individual): 0.8 content -> taste, 0.2 reaction + b_d -> taste
(approve -> sweet, disapprove -> bitter). Dyad (3000): domain A/B/N 1/3 each, two options; wish with prob 0.5
(partner M: bitter in A/B, K: sweet; both the same fixed side in N). Wish turn: options carry w+ (wished) / w-;
choice softmax(tau V); follow -> approve, else disapprove; half [chosen marked option, reaction + b_d], half
[reaction + b_d, chosen content unmarked]; rule S. No-wish turn: choose between unmarked options; no learning.
Replay (RB, RU, TB): one formation turn (a pair: prev -> V(next), taste -> innate) per dyad turn.
Learners: S0 / RB / RU (MLP) and T0 / TB (table, one value per event, step 0.1) as baselines.
Every turn draws the same number of random numbers; replay uses its own noise stream.

usage:
  python npn_choice.py --time 999
  python npn_choice.py --seeds 0 8 --out ../p73_choice_proto.json
  python npn_choice.py --summary ../p73_choice_proto.json
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npn_anchor as AN  # noqa: E402
from npn_leafleak import D, MLP, Tab, run_turn  # noqa: E402

N_FORM, N_DYAD = 9000, 3000
TAU, P_WISH, P_THICK = 3.0, 0.5, 0.9
INNATE = (1.0, -1.0, 0.0)
LEARNERS = ("S0", "RB", "RU", "T0", "TB")
PARTNERS = ("M", "K")
A, B, NN = 0, 1, 2
INDIVS = {"P": (A, A), "Q": (B, B), "P2": (A, B), "Q2": (B, A)}  # (content-thick, reaction-thick)
T_MEASURE = [0, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]


class World(AN.World):
    def __init__(self, seed):
        super().__init__(seed)
        g = np.random.default_rng([seed, 2])
        s = 1 / np.sqrt(D)
        self.el["n"] = g.normal(0, s, (2, D))
        self.el["w"] = g.normal(0, s, (2, D))  # 0: w+, 1: w-

    def vec(self, e):
        k = e[0]
        if k == "c":
            return self.el["a"][e[1]] + self.el["b"][e[2]]
        if k == "n":
            return self.el["n"][e[1]] + self.el["b"][2]
        if k == "rb":
            return self.el["r"][e[1]] + self.el["b"][e[2]]
        if k == "m":
            return self.vec(e[1]) + self.el["w"][e[2]]
        return self.el[k][e[1]]


def item(d, k):
    """Option k (0 = sweet / n1, 1 = bitter / n2) of domain d."""
    if d == NN:
        return ("n", k)
    return ("c", k + (3 if d == B else 0), d)


def formation_turns(seed, cdom, rdom, n=N_FORM):
    rc = np.random.default_rng([seed, 3, 0])
    out = []
    for _ in range(n):
        u, k, w1, w2 = rc.random(), int(rc.integers(6)), rc.random(), rc.random()
        dc = cdom if w1 < P_THICK else 1 - cdom
        dr = rdom if w2 < P_THICK else 1 - rdom
        if u < 0.8:
            out.append([("c", k % 3 + (3 if dc == B else 0), dc), ("t", k % 3)])
        else:
            out.append([("rb", k % 2, dr), ("t", k % 2)])
    return out


def wished_side(partner, d, seed):
    if d == NN:
        return seed % 2
    return 1 if partner == "M" else 0


def snap(m, w):
    v = lambda e: round(float(m.value(e, w)), 5)
    return {"u": [[v(item(d, k)) for k in range(2)] for d in range(3)],
            "m": [[[v(("m", item(d, k), s)) for s in range(2)] for k in range(2)] for d in range(3)],
            "rb": [[v(("rb", i, d)) for d in range(3)] for i in range(2)],
            "t": [v(("t", i)) for i in range(3)],
            "c3": [v(("c", 2, A)), v(("c", 5, B))]}


def softmax2(v0, v1):
    z = np.exp(TAU * (np.array([v0, v1]) - max(v0, v1)))
    return z / z.sum()


class Replay:
    def __init__(self, kind, record):
        self.kind, self.record = kind, record
        self.kinds = {}
        for t in record:
            self.kinds.setdefault(t[0], []).append(t)
        self.keys = list(self.kinds)

    def pick(self, rr):
        u1, u2 = rr.random(), rr.random()
        if self.kind == "RU":
            lst = self.kinds[self.keys[int(u1 * len(self.keys))]]
            return lst[int(u2 * len(lst))]
        return self.record[int(u1 * len(self.record))]


def dyad(kind, partner, V0, record, w, seed):
    m = copy.deepcopy(V0)
    rep = Replay("RU" if kind == "RU" else "RB", record) if kind in ("RB", "RU", "TB") else None
    rc = np.random.default_rng([seed, 3, 1])
    rs = np.random.default_rng([seed, 29, 1])
    rn = np.random.default_rng([seed, 7, 1])
    rn2 = np.random.default_rng([seed, 7, 2])
    rr = np.random.default_rng([seed, 17, 1])
    series, log = [snap(m, w)], []
    for t in range(1, N_DYAD + 1):
        d, wish, order, u = int(rc.integers(3)), rc.random() < P_WISH, rc.random(), rs.random()
        if wish:
            ws = wished_side(partner, d, seed)
            opts = [("m", item(d, k), 0 if k == ws else 1) for k in range(2)]
            p = softmax2(m.value(opts[0], w), m.value(opts[1], w))
            ch = 0 if u < p[0] else 1
            follow = ch == ws
            react = ("rb", 0 if follow else 1, d)
            run_turn(m, w, [opts[ch], react] if order < 0.5 else [react, item(d, ch)], rn)
            log.append((d, 1, int(follow)))
        else:
            p = softmax2(m.value(item(d, 0), w), m.value(item(d, 1), w))
            log.append((d, 0, int(u < p[0])))
        if rep is not None:
            run_turn(m, w, rep.pick(rr), rn2)
        if t in T_MEASURE:
            series.append(snap(m, w))
    return {"series": series, "log": log}


def per_seed(seed):
    w = World(seed)
    out = {}
    for indiv, (cd, rd) in INDIVS.items():
        record = formation_turns(seed, cd, rd)
        base = {}
        for net in ("mlp", "tab"):
            V0 = MLP(seed) if net == "mlp" else Tab()
            rn = np.random.default_rng([seed, 7, 0])
            for ev in record:
                run_turn(V0, w, ev, rn)
            base[net] = V0
            out[f"{indiv}_form_{net}"] = snap(V0, w)
        for kind in LEARNERS:
            for partner in PARTNERS:
                out[f"{indiv}_{kind}_{partner}"] = dyad(kind, partner, base["tab" if kind.startswith("T") else "mlp"], record, w, seed)
    return out


# ---------- summary (DESIGN-choice section 8) ----------

def own(s, d):
    return s["u"][d][0] - s["u"][d][1]


def follow(s, d, ws):
    return float(softmax2(s["m"][d][0][0 if ws == 0 else 1], s["m"][d][1][0 if ws == 1 else 1])[ws])


def pr(s, d):
    return s["rb"][0][d] - s["rb"][1][d]


def form_dev(s):
    dev = [abs(s["u"][d][k] - INNATE[k]) for d in (A, B) for k in range(2)]
    dev += [abs(s["c3"][i]) for i in range(2)] + [abs(s["t"][i] - INNATE[i]) for i in range(3)]
    dev += [abs(s["rb"][0][d] - 1) for d in (A, B)] + [abs(s["rb"][1][d] + 1) for d in (A, B)]
    return max(dev)


def classify(own_c, fol, absmax, deaf, recv):
    """Priority: collapse (|V| > 3) > copy > comply > keep-deaf > keep-receive > keep-N-unreadable > collapse > mixed."""
    if absmax > 3:
        return "collapse"
    if own_c <= 0.5 and fol >= 0.7:
        return "copy"
    if own_c >= 1.0 and fol >= 0.7:
        return "comply"
    if own_c >= 1.0 and fol <= 0.3:
        return "keep-deaf" if deaf else ("keep-receive" if recv else "keep-Nunread")
    if own_c <= 0.5 and 0.3 < fol < 0.7:
        return "collapse"
    return "mixed"


def fmt(v):
    return " ".join(f"{y:+.2f}" for y in v)


def summarize(path):
    data = json.load(open(path, encoding="utf-8"))
    rows, seeds = data["rows"], sorted(data["rows"], key=int)
    print("t =", " ".join(str(t) for t in T_MEASURE))
    for net in ("mlp", "tab"):
        cells = [(s, i) for s in seeds for i in INDIVS]
        dev = {c: form_dev(rows[c[0]][f"{c[1]}_form_{net}"]) for c in cells}
        print(f"-- formation end ({net}): kept {sum(v < 0.2 for v in dev.values())}/{len(cells)}, max dev mean {np.mean(list(dev.values())):.3f}")
    cells = [(s, i) for s in seeds for i in INDIVS]
    ok = {c: form_dev(rows[c[0]][f"{c[1]}_form_mlp"]) < 0.2 and form_dev(rows[c[0]][f"{c[1]}_form_tab"]) < 0.2 for c in cells}
    keep = [c for c in cells if ok[c]]
    print(f"-- cells kept (both mlp and tab within 0.2): {len(keep)}/{len(cells)}; excluded cells are dropped for every learner and partner")
    me = [[rows[s][f"{i}_form_mlp"]["m"][d][1][0] - rows[s][f"{i}_form_mlp"]["u"][d][1],
           rows[s][f"{i}_form_mlp"]["m"][d][0][1] - rows[s][f"{i}_form_mlp"]["u"][d][0]] for s, i in keep for d in (A, B)]
    print(f"-- t=0 mark effects (mlp; bitter+w+, sweet+w-): mean {np.round(np.mean(me, 0), 3)} | max |.| >= 0.3 in {sum(max(abs(x) for x in e) >= 0.3 for e in me)}/{len(me)}")
    fN0 = [follow(rows[s][f"{i}_form_mlp"], NN, int(s) % 2) for s, i in keep]
    pN0 = [pr(rows[s][f"{i}_form_mlp"], NN) for s, i in keep]
    print(f"-- t=0 N (mlp): Follow_N per cell {np.round(fN0, 2).tolist()} | Pr_N mean {np.mean(pN0):+.2f} (sd {np.std(pN0):.2f})")
    for kind in LEARNERS:
        net = "tab" if kind.startswith("T") else "mlp"
        print(f"\n===== {kind} ({net}) =====")
        res = {}
        for partner in PARTNERS:
            agg = {}
            for s, i in keep:
                cd, rd = INDIVS[i]
                ser = rows[s][f"{i}_{kind}_{partner}"]["series"]
                wsA, wsN = (1 if partner == "M" else 0), int(s) % 2
                for nm, f in (("own_c", lambda x: own(x, cd)), ("own_cn", lambda x: own(x, 1 - cd)),
                              ("fol_c", lambda x: follow(x, cd, wsA)), ("fol_cn", lambda x: follow(x, 1 - cd, wsA)),
                              ("pr_r", lambda x: pr(x, rd)), ("pr_rn", lambda x: pr(x, 1 - rd)),
                              ("fol_N", lambda x: follow(x, NN, wsN)), ("pr_N", lambda x: pr(x, NN))):
                    agg.setdefault(nm, []).append([f(x) for x in ser])
                lg = rows[s][f"{i}_{kind}_{partner}"]["log"][-500:]
                ft = [f for d, wf, f in lg if wf and d == cd]
                agg.setdefault("act_c", []).append(np.mean(ft) if ft else np.nan)
                fin = ser[-1]
                agg.setdefault("absmax", []).append(max(abs(x) for row in fin["u"] for x in row))
            res[partner] = agg
            print(f"  partner {partner} (n={len(keep)}): content-thick c / content-thin cn / reaction-thick r / reaction-thin rn")
            for nm in ("own_c", "own_cn", "fol_c", "fol_cn", "pr_r", "pr_rn", "fol_N", "pr_N"):
                print(f"     {nm:7s} {fmt(np.mean(agg[nm], 0))}")
            print(f"     actual follow rate, last 500 turns, content-thick domain: {np.nanmean(agg['act_c']):.2f}")
        M, K = res["M"], res["K"]
        n = len(keep)
        tc, tn, flag = [], [], 0
        for j in range(n):
            deaf = (M["fol_N"][j][-1] < K["fol_N"][j][-1] - 0.3) and (M["pr_N"][j][-1] < K["pr_N"][j][-1] - 0.5)
            recv = M["fol_N"][j][-1] >= 0.7 and K["fol_N"][j][-1] >= 0.7
            tc.append(classify(M["own_c"][j][-1], M["fol_c"][j][-1], M["absmax"][j], deaf, recv))
            tn.append(classify(M["own_cn"][j][-1], M["fol_cn"][j][-1], M["absmax"][j], deaf, recv))
            flag += int(M["pr_r"][j][-1] < -0.3)
        cnt = lambda xs: {t: xs.count(t) for t in sorted(set(xs))}
        print(f"  TYPES (M, t=3000) content-thick: {cnt(tc)} | content-thin: {cnt(tn)} | reverse flag (Pr_reaction-thick < -0.3): {flag}/{n}")
        # 2x2: Follow of M at t=3000 by (content thick?, reaction thick?) for domains A and B
        grid = {(a, b): [] for a in (1, 0) for b in (1, 0)}
        for j, (s, i) in enumerate(keep):
            cd, rd = INDIVS[i]
            ser = rows[s][f"{i}_{kind}_M"]["series"][-1]
            for d in (A, B):
                grid[(int(d == cd), int(d == rd))].append(follow(ser, d, 1))
        print("  Follow_M(3000) by (content thick, reaction thick): " + " | ".join(f"{k}: {np.mean(v):.2f} (n={len(v)})" for k, v in grid.items()))
        dN = [M["fol_N"][j][-1] - K["fol_N"][j][-1] for j in range(n)]
        dP = [M["pr_N"][j][-1] - K["pr_N"][j][-1] for j in range(n)]
        print(f"  N (3000): Follow_N M {np.mean([x[-1] for x in M['fol_N']]):.2f} K {np.mean([x[-1] for x in K['fol_N']]):.2f} | M - K Follow_N {np.mean(dN):+.2f} (+{sum(x > 0 for x in dN)}/{n}) Pr_N {np.mean(dP):+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=int)
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.time is not None:
        t0 = time.time()
        r = per_seed(a.time)
        fin = all(np.isfinite(v) for k, row in r.items() if "_form_" not in k for v in np.ravel(row["series"][-1]["u"]))
        print("one seed", f"{time.time() - t0:.0f}s", "finite", fin)
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
