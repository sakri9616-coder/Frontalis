"""Exploration 6: values as log-odds with cross-entropy; the margin |z| carries the amount (DESIGN-logit v1.1).

Prototype seeds only, numpy only. World, formation record, partner M (q = 1) and individuals P2 / Q2 are those of
npn_q1 / npn_reaction. VL: MLP 24-32-32-1 whose output is a log-odds z; p = sigmoid(z); read value v = 2p - 1.
Rule S with cross-entropy: target of a taste = (innate + 1) / 2; target of "X -> Y" = p(Y) (read without noise);
dL/dz = p - target; the same clipping as before. No multiplier, no U, no taste exception.
Worlds: one-way (reaction always after the choice), round (q = 1 of npn_q1), nolearn (round order, but turns whose
reaction comes first are chosen and not learned; separates rewriting from pressure frequency, CRITIQUE-logit-1 D3).
Tables (CRITIQUE-logit-1 D1): log-odds of the 29 events copied from VL; one move = LR n^2 min(1, 1/(|p - t| n)) (p - t),
the same clipping as VL. TLe: n per event = VL gradient norm at formation end. TLlo / TLmid / TLhi: n = median over
the 24 events learned in the dyad, x 1/3, 1, 3. S0: the old rule-S MLP.
Probes from the formation-end copy: pressure [sweet content, disapprove + b_d] and reason [sweet content, bitter taste].

usage (jobs run on Colab from probes/):
  python npn_logit.py --time 999
  python npn_logit.py --regress ../p75_reaction_rep_a.json
  python npn/npn_logit.py --base 40 48 --out p77_logit_base.json
  python npn/npn_logit.py --basecheck p77_logit_base.json
  python npn/npn_logit.py --seeds 48 56 --out p77_logit_main.json
  python npn/npn_logit.py --gates p77_logit_main.json --basefile p77_logit_base.json
  python npn/npn_logit.py --main p77_logit_main.json --basefile p77_logit_base.json
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
from npn_leafleak import D, CLIP, NOISE, INNATE, MLP, _norm, run_turn  # noqa: E402

VL_LR = 0.01
NSCALE = 1.0
IND = ("P2", "Q2")
TIMES = list(range(0, 1001, 25)) + [1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
T_BAR = [100, 300, 1000, 3000]
T_END = [2000, 2250, 2500, 2750, 3000]
N_PROBE = 1000
EVENTS = R.all_events()
DYAD_EVENTS = [item(d, k) for d in range(3) for k in range(2)] + [("m", item(d, k), s) for d in range(3) for k in range(2) for s in range(2)] \
    + [("rb", i, d) for i in range(2) for d in range(3)]
TABS = ("TLlo", "TLmid", "TLhi", "TLe")
WORLDS = ("one", "round", "nolearn")


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


class VL(MLP):
    nclip = 0
    nlearn = 0

    def z(self, e, w):
        return MLP.value(self, e, w)

    def p(self, e, w):
        return sig(self.z(e, w))

    def value(self, e, w):
        return 2 * self.p(e, w) - 1

    def learn_p(self, e, w, tgt, noise):
        z, gr = self._grads(w.vec(e) + noise)
        g = sig(z) - tgt
        c = CLIP / (abs(g) * _norm(gr) + 1e-6)
        self.nlearn += 1
        self.nclip += int(c < 1.0)
        s = VL_LR * min(1.0, c) * g
        self.w3 -= s * gr[0]
        self.b3 -= s
        self.W2 -= s * gr[2]
        self.b2 -= s * gr[3]
        self.W1 -= s * gr[4]
        self.b1 -= s * gr[5]

    def gnorm(self, e, w):
        return _norm(self._grads(w.vec(e))[1])


class TLz:
    """Table of log-odds with the clipping of VL; n is a number or a dict per event."""
    nclip = 0
    nlearn = 0

    def __init__(self, zs, n):
        self.v, self.n = dict(zs), n

    def z(self, e, w):
        return self.v[e]

    def p(self, e, w):
        return sig(self.v[e])

    def value(self, e, w):
        return 2 * sig(self.v[e]) - 1

    def learn_p(self, e, w, tgt, noise):
        n = self.n[e] if isinstance(self.n, dict) else self.n
        g = sig(self.v[e]) - tgt
        c = CLIP / (abs(g) * n + 1e-6)
        self.nlearn += 1
        self.nclip += int(c < 1.0)
        self.v[e] = self.v[e] - VL_LR * n * n * min(1.0, c) * g


def run_turn_L(model, w, events, rn):
    last = len(events) - 1
    for i, e in enumerate(events):
        noise = rn.normal(0, NOISE, D)
        tgt = None
        if e[0] == "t":
            tgt = (INNATE[e[1]] + 1) / 2
        if i < last:
            assert e[0] != "t"
            tgt = model.p(events[i + 1], w)
        if tgt is not None:
            model.learn_p(e, w, tgt, noise)


def zsnap(m, w):
    if not hasattr(m, "z"):
        return None
    zz = lambda e: round(float(m.z(e, w)), 4)
    ab = [item(d, k) for d in (A, B) for k in range(2)] + [("rb", i, d) for i in range(2) for d in (A, B)]
    return {"c": [[zz(item(d, k)) for k in range(2)] for d in range(3)], "rb": [[zz(("rb", i, d)) for d in range(3)] for i in range(2)],
            "max": round(max(abs(float(m.z(e, w))) for e in ab), 4)}


def turnfn(m):
    return run_turn if (isinstance(m, MLP) and not isinstance(m, VL)) else run_turn_L


def dyad(model, w, seed, world, partner="M"):
    m = copy.deepcopy(model)
    turn = turnfn(m)
    c0, l0 = getattr(m, "nclip", 0), getattr(m, "nlearn", 0)
    rc = np.random.default_rng([seed, 3, 1])
    rs = np.random.default_rng([seed, 29, 1])
    rn = np.random.default_rng([seed, 7, 1])
    series, zs, log = [snap(m, w)], [zsnap(m, w)], []
    for t in range(1, C.N_DYAD + 1):
        d, wish, order, u = int(rc.integers(3)), rc.random() < C.P_WISH, rc.random(), rs.random()
        ws = wished_side(partner, d, seed)
        if wish:
            opts = [("m", item(d, k), 0 if k == ws else 1) for k in range(2)]
            p = softmax2(m.value(opts[0], w), m.value(opts[1], w))
            ch = 0 if u < p[0] else 1
            first = opts[ch]
            log.append((d, 1, int(ch == ws)))
        else:
            p = softmax2(m.value(item(d, 0), w), m.value(item(d, 1), w))
            ch = 0 if u < p[0] else 1
            first = item(d, ch)
            log.append((d, 0, int(ch == 0)))
        react = ("rb", 0 if ch == ws else 1, d)
        if world == "one" or order < 0.5:
            turn(m, w, [first, react], rn)
        elif world == "round":
            turn(m, w, [react, item(d, ch)], rn)
        if t in TIMES:
            series.append(snap(m, w))
            zs.append(zsnap(m, w))
    if isinstance(m, TLz):
        assert len(m.v) == len(EVENTS)
    nl = getattr(m, "nlearn", 0) - l0
    return {"series": series, "z": zs, "log": log, "clip": (getattr(m, "nclip", 0) - c0) / nl if nl else None}


def probe(model, w, seed, d, kind):
    m = copy.deepcopy(model)
    turn = turnfn(m)
    c0, l0 = getattr(m, "nclip", 0), getattr(m, "nlearn", 0)
    rn = np.random.default_rng([seed, 51, d, 0 if kind == "pressure" else 1])
    ev = [item(d, 0), ("rb", 1, d)] if kind == "pressure" else [item(d, 0), ("t", 1)]
    v0 = float(m.value(item(d, 0), w))
    vs, T = [v0], None
    for t in range(1, N_PROBE + 1):
        turn(m, w, ev, rn)
        v = float(m.value(item(d, 0), w))
        if T is None and v <= v0 / 2:
            T = t
        if t % 10 == 0:
            vs.append(round(v, 5))
    nl = getattr(m, "nlearn", 0) - l0
    return {"v": vs, "T": T if T is not None else N_PROBE + 1, "clip": (getattr(m, "nclip", 0) - c0) / nl if nl else None}


def gate1_ok(m, w):
    s = snap(m, w)
    for d in (A, B):
        if not (s["u"][d][0] >= 0.5 and s["u"][d][1] <= -0.5 and s["rb"][0][d] >= 0.5 and s["rb"][1][d] <= -0.5):
            return False
    if max(abs(x) for x in s["c3"]) > 0.3:
        return False
    return all(abs(s["t"][i] - INNATE[i]) <= 0.2 for i in range(3))


def form(seed, indiv, w):
    cd, rd = INDIVS[indiv]
    record = formation_turns(seed, cd, rd)
    vl, s0 = VL(seed), MLP(seed)
    rn = np.random.default_rng([seed, 7, 0])
    for ev in record:
        run_turn_L(vl, w, ev, rn)
    rn = np.random.default_rng([seed, 7, 0])
    for ev in record:
        run_turn(s0, w, ev, rn)
    return vl, s0


def tables(vl, w):
    zs = {e: float(vl.z(e, w)) for e in EVENTS}
    ne = {e: NSCALE * vl.gnorm(e, w) for e in EVENTS}
    nmed = float(np.median([ne[e] for e in DYAD_EVENTS]))
    out = {"TLlo": TLz(zs, nmed / 3), "TLmid": TLz(zs, nmed), "TLhi": TLz(zs, 3 * nmed), "TLe": TLz(zs, ne)}
    return out, nmed


def per_seed(seed, base=False):
    w = World(seed)
    out = {}
    for indiv in IND:
        vl, s0 = form(seed, indiv, w)
        tb, nmed = tables(vl, w)
        out[f"{indiv}_form"] = {"v": snap(vl, w), "z": {repr(e): round(float(vl.z(e, w)), 5) for e in EVENTS},
                                "n": {repr(e): float(vl.gnorm(e, w)) for e in EVENTS}, "nmed": nmed, "s0": snap(s0, w),
                                "g1": gate1_ok(vl, w), "s0ok": form_dev(snap(s0, w)) < 0.2,
                                "clip": vl.nclip / vl.nlearn}
        cd = INDIVS[indiv][0]
        cells = {"thick": cd, "thin": 1 - cd}
        learners = dict(VL=vl, **tb) if base else dict(VL=vl, S0=s0, **tb)
        for nm, m in learners.items():
            for cn, d in cells.items():
                for kind in ("pressure", "reason"):
                    out[f"{indiv}_probe_{nm}_{cn}_{kind}"] = probe(m, w, seed, d, kind)
        runs = [(k, wd) for k in TABS for wd in WORLDS]
        if not base:
            runs = [("VL", wd) for wd in WORLDS] + [("S0", "one")] + runs
        for nm, wd in runs:
            out[f"{indiv}_{nm}_{wd}"] = dyad(learners[nm], w, seed, wd)
    return out


def regress(prev):
    old = json.load(open(prev, encoding="utf-8"))["rows"]["8"]
    seed, indiv = 8, "P2"
    w = World(seed)
    _, s0 = form(seed, indiv, w)
    md = float(np.max(np.abs(R.flat(snap(s0, w)) - R.flat(old[f"{indiv}_form_mlp"]))))
    new = dyad(s0, w, seed, "round")["series"]
    ref = old[f"{indiv}_q1_S0_M"]["series"]
    rt = [0] + [t for t in R.T_REP if t > 0]
    for j, t in enumerate(rt):
        md = max(md, float(np.max(np.abs(R.flat(new[TIMES.index(t)]) - R.flat(ref[j])))))
    print(f"[gate 4] regression seed 8 P2 S0 formation and round x M vs p75: max |diff| {md:.2e} -> {'OK' if md < 1e-6 else 'FAIL'}")


# ---------- summaries ----------

def load(paths):
    rows = {}
    for p in paths:
        rows.update(json.load(open(p, encoding="utf-8"))["rows"])
    return rows


def kept(rows):
    seeds = sorted(rows, key=int)
    keep = [(s, i) for s in seeds for i in IND if rows[s][f"{i}_form"]["g1"]]
    both = [s for s in seeds if (s, "P2") in keep and (s, "Q2") in keep]
    return seeds, keep, both


def cells_of(i):
    cd = INDIVS[i][0]
    return {"thick": cd, "thin": 1 - cd}


def rho_at(sr, d, t):
    return own(sr[TIMES.index(t)], d) / own(sr[0], d)


def rho(sr, d, ts):
    return float(np.mean([rho_at(sr, d, t) for t in ts]))


def ytime(sr, d):
    for t in TIMES:
        if rho_at(sr, d, t) <= 0.3:
            return t
    return 3001


def kpairs(rows, both):
    """Gate 2 / K1 pairs: same seed, same domain; (thick individual, thin individual, domain)."""
    return [(s, "P2", "Q2", A) for s in both] + [(s, "Q2", "P2", B) for s in both]


def tp_ratio(rows, pairs, nm):
    return np.array([rows[s][f"{a}_probe_{nm}_thick_pressure"]["T"] / rows[s][f"{b}_probe_{nm}_thin_pressure"]["T"] for s, a, b, d in pairs], float)


def gm(x):
    return float(np.exp(np.mean(np.log(x))))


def basecheck(path):
    """Base seeds only: gate 1 definition, the Tp(TLe)/Tp(VL) gate, and the expectations written into DESIGN-logit section 7."""
    rows = load([path])
    seeds, keep, both = kept(rows)
    print(f"[base gate 1] kept {len(keep)}/{2 * len(seeds)}; seeds with both {len(both)} -> {'OK' if len(both) >= 6 else 'FAIL (raise VL_LR to 0.03 once)'}")
    tp = {nm: np.array([rows[s][f"{i}_probe_{nm}_{cn}_pressure"]["T"] for s, i in keep for cn in ("thick", "thin")], float) for nm in ("VL",) + TABS}
    r = gm(tp["TLe"] / tp["VL"])
    print(f"[base speed gate] Tp(TLe)/Tp(VL) geometric mean {r:.2f} -> {'OK' if 1 / 3 <= r <= 3 else 'OUT (match n once)'}; Tp(TLlo/mid/hi)/Tp(VL) " + " / ".join(f"{gm(tp[k] / tp['VL']):.2f}" for k in TABS[:3]))
    pairs = kpairs(rows, both)
    for nm in ("VL",) + TABS:
        x = tp_ratio(rows, pairs, nm)
        print(f"  K1 pairs {nm}: Tp thick/thin geometric mean {gm(x):.2f} (sd of log {np.std(np.log(x)):.2f}, n={len(x)})")
    for nm in ("VL",) + TABS:
        T = {(cn, kd): np.array([rows[s][f"{i}_probe_{nm}_{cn}_{kd}"]["T"] for s, i in keep], float) for cn in ("thick", "thin") for kd in ("pressure", "reason")}
        print(f"  probes {nm}: Tp thick {gm(T[('thick', 'pressure')]):.0f} thin {gm(T[('thin', 'pressure')]):.0f} | Tr thick {gm(T[('thick', 'reason')]):.0f} | thick Tp/Tr {gm(T[('thick', 'pressure')] / T[('thick', 'reason')]):.2f}")
    dz = []
    for s in both:
        za, zb = rows[s]["P2_form"]["z"], rows[s]["Q2_form"]["z"]
        for d, thick, thin in ((A, za, zb), (B, zb, za)):
            for k in range(2):
                dz.append(abs(thick[repr(item(d, k))]) - abs(thin[repr(item(d, k))]))
    mz = {cn: np.mean([abs(rows[s][f"{i}_form"]["z"][repr(item(cells_of(i)[cn], 0))]) for s, i in keep]) for cn in ("thick", "thin")}
    print(f"  margins |z| sweet thick {mz['thick']:.2f} thin {mz['thin']:.2f}; same-seed difference {np.mean(dz):+.2f}; VL formation clip fraction {np.mean([rows[s][f'{i}_form']['clip'] for s, i in keep]):.2f}")
    for k in TABS:
        parts = []
        for wd in WORLDS:
            rb = {cn: np.mean([rho(rows[s][f"{i}_{k}_{wd}"]["series"], cells_of(i)[cn], T_BAR) for s, i in keep]) for cn in ("thick", "thin")}
            re = np.mean([rho(rows[s][f"{i}_{k}_{wd}"]["series"], cells_of(i)["thick"], T_END) for s, i in keep])
            yt = np.median([ytime(rows[s][f"{i}_{k}_{wd}"]["series"], cells_of(i)["thick"]) for s, i in keep])
            parts.append(f"{wd}: rho_bar {rb['thick']:+.2f}/{rb['thin']:+.2f} rho_end thick {re:+.2f} yield(thick, median) {yt:.0f}")
        E = np.mean([rho(rows[s][f"{i}_{k}_round"]["series"], cells_of(i)["thick"], T_END) - rho(rows[s][f"{i}_{k}_nolearn"]["series"], cells_of(i)["thick"], T_END) for s, i in keep])
        F = np.mean([rho(rows[s][f"{i}_{k}_nolearn"]["series"], cells_of(i)["thick"], T_END) - rho(rows[s][f"{i}_{k}_one"]["series"], cells_of(i)["thick"], T_END) for s, i in keep])
        print(f"  {k}: " + " | ".join(parts) + f" | E {E:+.2f} frequency part {F:+.2f}")


def gates(paths, basefile):
    rows = load(paths)
    seeds, keep, both = kept(rows)
    print(f"[gate 1] kept {len(keep)}/{2 * len(seeds)}; seeds with both {len(both)} {both} -> {'OK' if len(both) >= 6 else 'FAIL (add seeds from 56)'}")
    dz = []
    for s in both:
        za, zb = rows[s]["P2_form"]["z"], rows[s]["Q2_form"]["z"]
        for d, thick, thin in ((A, za, zb), (B, zb, za)):
            for k in range(2):
                e = repr(item(d, k))
                dz.append(abs(thick[e]) - abs(thin[e]))
    dz = np.array(dz)
    g2 = len(dz) > 0 and dz.mean() >= 1.0 and sum(dz > 0) >= 0.75 * len(dz)
    print(f"[gate 2] margin difference thick - thin (same seed, same content): mean {dz.mean() if len(dz) else float('nan'):+.2f}, positive {int(sum(dz > 0))}/{len(dz)} -> {'OK' if g2 else 'FAIL'}")
    print("    reaction log-odds by cell (approve / disapprove): " + " | ".join(
        f"{cn} {np.mean([rows[s][f'{i}_form']['z'][repr(('rb', 0, cells_of(i)[cn]))] for s, i in keep]):+.2f} / {np.mean([rows[s][f'{i}_form']['z'][repr(('rb', 1, cells_of(i)[cn]))] for s, i in keep]):+.2f}" for cn in ("thick", "thin")))
    dr = {wd: sum(max(x["max"] for x in rows[s][f"{i}_VL_{wd}"]["z"]) > rows[s][f"{i}_VL_{wd}"]["z"][0]["max"] + 1 for s, i in keep) for wd in WORLDS}
    g3 = all(v <= len(keep) / 10 for v in dr.values())
    print(f"[gate 3] VL drift (max |z| over A/B contents and reactions exceeds its t=0 value + 1): " + ", ".join(f"{wd} {v}/{len(keep)}" for wd, v in dr.items()) + f" -> {'OK' if g3 else 'FAIL'}")
    mv = 0.0
    for s, i in keep:
        for k in TABS:
            sr = rows[s][f"{i}_{k}_one"]["series"]
            mv = max(mv, max(float(np.max(np.abs(np.ravel(x["rb"]) - np.ravel(sr[0]["rb"])))) for x in sr))
    tl = TLz({("c", 0, 0): 1.3}, 13.0)
    tl.learn_p(("c", 0, 0), None, 0.2, None)
    g_ = sig(1.3) - 0.2
    hand = 1.3 - VL_LR * 169 * min(1.0, CLIP / (abs(g_) * 13 + 1e-6)) * g_
    g5 = mv == 0.0 and abs(tl.v[("c", 0, 0)] - hand) < 1e-12
    print(f"[gate 5] one-way tables never move reactions: max |change| {mv:.1e}; TL one clipped step equals hand {abs(tl.v[('c', 0, 0)] - hand):.1e} -> {'OK' if g5 else 'FAIL'}")
    g6 = True
    if basefile:
        base = load([basefile])
        bseeds, bkeep, _ = kept(base)
        for k in ("TLmid", "TLe"):
            for cn in ("thick", "thin"):
                per = {}
                for s, i in bkeep:
                    per.setdefault(s, []).append(rho(base[s][f"{i}_{k}_one"]["series"], cells_of(i)[cn], T_BAR))
                pm = np.array([np.mean(v) for v in per.values()])
                lo, hi = pm.mean() - 2 * pm.std(), pm.mean() + 2 * pm.std()
                val = np.mean([rho(rows[s][f"{i}_{k}_one"]["series"], cells_of(i)[cn], T_BAR) for s, i in keep])
                ok = lo <= val <= hi
                g6 &= ok
                print(f"[gate 6] {k} one-way rho_bar {cn}: {val:+.2f} band [{lo:+.2f}, {hi:+.2f}] (only checks the seed set) -> {'OK' if ok else 'OFF'}")
    allok = len(both) >= 6 and g2 and g3 and g5 and g6
    print(f"gates (1, 2, 3, 5, 6; gate 4 is the separate regression) all OK: {allok}")


def mainsum(paths, basefile):
    rows = load(paths)
    seeds, keep, both = kept(rows)
    ks = [s for s, i in keep]
    avg = lambda sr, f: float(np.mean([f(sr[TIMES.index(t)]) for t in T_END]))

    def seedcount(x, sign, keys):
        sm = {}
        for s, v in zip(keys, x):
            sm.setdefault(s, []).append(v)
        smv = np.array([np.mean(v) for v in sm.values()])
        return int(sum(smv * sign > 0)), len(smv)

    def against(getx, fl, keys=None):
        keys = keys or ks
        parts, sg, ok = [], set(), True
        for k in TABS:
            x = np.asarray(getx(k), float)
            sign = 1 if x.mean() >= 0 else -1
            sg.add(sign)
            np_ = int(sum(x * sign > 0))
            ns, ms = seedcount(x, sign, keys)
            ok &= abs(x.mean()) >= fl and np_ >= 0.75 * len(x) and ns >= 0.75 * ms
            parts.append(f"{x.mean():+.2f}")
        return ok and len(sg) == 1, " / ".join(parts)

    q = {}
    print("--- worlds (5-point means unless stated); yield = first time rho <= 0.3 (3001 = never)")
    for nm in ("VL", "S0") + TABS:
        for wd in WORLDS:
            key = f"{nm}_{wd}"
            if f"{keep[0][1]}_{key}" not in rows[keep[0][0]]:
                continue
            kk = [(s, i) for s, i in keep if nm != "S0" or rows[s][f"{i}_form"]["s0ok"]]
            r = {"keys": [s for s, i in kk]}
            g = lambda cn, f: np.array([f(rows[s][f"{i}_{key}"], cells_of(i)[cn], s) for s, i in kk])
            for cn in ("thick", "thin"):
                r[f"rbar_{cn}"] = g(cn, lambda x, d, s: rho(x["series"], d, T_BAR))
                r[f"rend_{cn}"] = g(cn, lambda x, d, s: rho(x["series"], d, T_END))
                r[f"yt_{cn}"] = g(cn, lambda x, d, s: ytime(x["series"], d))
                r[f"pr_{cn}"] = g(cn, lambda x, d, s: avg(x["series"], lambda y: pr(y, d)))
                r[f"fol_{cn}"] = g(cn, lambda x, d, s: avg(x["series"], lambda y: follow(y, d, 1)))
                if nm != "S0":
                    r[f"zpr_{cn}"] = g(cn, lambda x, d, s: float(np.mean([z["rb"][0][d] - z["rb"][1][d] for z in [x["z"][TIMES.index(t)] for t in T_END]])))
                    r[f"zown_{cn}"] = g(cn, lambda x, d, s: float(np.mean([z["c"][d][0] - z["c"][d][1] for z in [x["z"][TIMES.index(t)] for t in T_END]])))
            r["G"] = r["rbar_thick"] - r["rbar_thin"]
            r["folN"] = np.array([avg(rows[s][f"{i}_{key}"]["series"], lambda y: follow(y, NN, int(s) % 2)) for s, i in kk])
            r["prN"] = np.array([avg(rows[s][f"{i}_{key}"]["series"], lambda y: pr(y, NN)) for s, i in kk])
            types = []
            for j, (s, i) in enumerate(kk):
                sr, zz = rows[s][f"{i}_{key}"]["series"], rows[s][f"{i}_{key}"]["z"]
                re, prk, fn = r["rend_thick"][j], r["pr_thick"][j], r["folN"][j]
                vs = [avg(sr, lambda x: x["u"][d][k]) for d in (A, B) for k in range(2)] + [avg(sr, lambda x: x["rb"][k][d]) for d in (A, B) for k in range(2)]
                if re >= 0.7 and prk >= 0.5 and fn >= 0.7:
                    ty = "listened-kept"
                elif re >= 0.7:
                    ty = "plugged-kept" + ("(Pr)" if prk < 0.5 else "") + ("(N)" if fn < 0.7 else "")
                elif re <= 0.3:
                    ty = "yielded-copy" if r["fol_thick"][j] >= 0.7 else "yielded"
                elif max(vs) - min(vs) <= 0.3:
                    ty = "shrink"
                elif zz[0] is not None and max(x["max"] for x in zz) > zz[0]["max"] + 1:
                    ty = "drift"
                else:
                    ty = "mid"
                types.append(ty)
            r["types"] = {t: types.count(t) for t in sorted(set(types))}
            r["listened"] = sum(t == "listened-kept" for t in types)
            q[key] = r
            clip = np.mean([rows[s][f"{i}_{key}"]["clip"] for s, i in kk if rows[s][f"{i}_{key}"]["clip"] is not None]) if nm != "S0" else float("nan")
            zp = f" zPr thick {r['zpr_thick'].mean():+.2f} zOwn thick {r['zown_thick'].mean():+.2f} |" if nm != "S0" else ""
            print(f"  {key:12s} n={len(kk)} yield thick/thin (median) {np.median(r['yt_thick']):.0f}/{np.median(r['yt_thin']):.0f} | rho_bar {r['rbar_thick'].mean():+.2f}/{r['rbar_thin'].mean():+.2f} G {r['G'].mean():+.2f} | rho_end thick {r['rend_thick'].mean():+.2f} | Pr thick {r['pr_thick'].mean():+.2f} |{zp} Follow thick {r['fol_thick'].mean():.2f} | N Pr {r['prN'].mean():+.2f} Follow {r['folN'].mean():.2f} | clip {clip:.2f} | {r['types']}")
    print("\n--- probes (T: first turn v(sweet) <= half; 1001 = not reached)")
    pairs = kpairs(rows, both)
    for nm in ("VL", "S0") + TABS:
        T = {(cn, kd): np.array([rows[s][f"{i}_probe_{nm}_{cn}_{kd}"]["T"] for s, i in keep], float) for cn in ("thick", "thin") for kd in ("pressure", "reason")}
        x = tp_ratio(rows, pairs, nm)
        print(f"  {nm:6s} Tp thick {gm(T[('thick', 'pressure')]):.0f} thin {gm(T[('thin', 'pressure')]):.0f} | Tr thick {gm(T[('thick', 'reason')]):.0f} | pairs Tp thick/thin {gm(x):.2f} (>1 in {int(sum(x > 1))}/{len(x)}) | thick Tp/Tr {gm(T[('thick', 'pressure')] / T[('thick', 'reason')]):.2f}")
    print("\n===== conditions (read only if all gates passed) =====")
    x = tp_ratio(rows, pairs, "VL")
    if basefile:
        base = load([basefile])
        _, _, bboth = kept(base)
        bx = np.log(tp_ratio(base, kpairs(base, bboth), "VL"))
        lo, hi = np.exp(bx.mean() - 2 * bx.std() / np.sqrt(len(bx))), np.exp(bx.mean() + 2 * bx.std() / np.sqrt(len(bx)))
        k1 = lo <= gm(x) <= hi and sum(x > 1) >= 0.75 * len(x)
        print(f"K1 (tool) VL pairs Tp thick/thin {gm(x):.2f}, >1 in {int(sum(x > 1))}/{len(x)}; base band [{lo:.2f}, {hi:.2f}] -> {'holds' if k1 else 'does not hold'}")
    for cn in ("thick", "thin"):
        ok, tx = against(lambda k: q["VL_one"][f"yt_{cn}"] - q[f"{k}_one"][f"yt_{cn}"], 100)
        print(f"K2 one-way yield time {cn}: VL - TLlo/TLmid/TLhi/TLe {tx} -> {'DIFF' if ok else 'no diff'} (floor 100 turns)")
        ok, tx = against(lambda k: q["VL_one"][f"rbar_{cn}"] - q[f"{k}_one"][f"rbar_{cn}"], 0.15)
        print(f"K2 one-way rho_bar {cn}: {tx} -> {'DIFF' if ok else 'no diff'}")
    ok, tx = against(lambda k: q["VL_one"]["G"] - q[f"{k}_one"]["G"], 0.15)
    print(f"K2 one-way G: {tx} -> {'DIFF' if ok else 'no diff'}")
    ok, tx = against(lambda k: q["VL_one"]["zpr_thick"] - q[f"{k}_one"]["zpr_thick"], 1.0)
    print(f"K3 one-way Pr thick in log-odds: {tx} -> {'DIFF' if ok else 'no diff'} (floor 1.0 in z)")
    ok, tx = against(lambda k: q["VL_one"]["pr_thick"] - q[f"{k}_one"]["pr_thick"], 0.5)
    print(f"K3 one-way Pr thick (v): {tx} -> {'DIFF' if ok else 'no diff'}")
    ok, tx = against(lambda k: q["VL_one"]["folN"] - q[f"{k}_one"]["folN"], 0.15)
    print(f"K3 one-way Follow_N: {tx} -> {'DIFF' if ok else 'no diff'}")
    okr, txr = against(lambda k: q["VL_one"]["rend_thick"] - q[f"{k}_one"]["rend_thick"], 0.3)
    lis = q["VL_one"]["listened"]
    print(f"K3 listened-kept in VL one-way {lis}/{len(keep)}; rho_end thick VL - tables {txr} -> {'candidate' if (lis >= 0.75 * len(keep) and okr) else 'not a candidate'}")
    E = {k: q[f"{k}_round"]["rend_thick"] - q[f"{k}_nolearn"]["rend_thick"] for k in ("VL",) + TABS}
    Fq = {k: q[f"{k}_nolearn"]["rend_thick"] - q[f"{k}_one"]["rend_thick"] for k in ("VL",) + TABS}
    ok, tx = against(lambda k: E["VL"] - E[k], 0.3)
    print(f"K4 E = rho_end thick (round - nolearn): VL {E['VL'].mean():+.2f}, tables " + " / ".join(f"{E[k].mean():+.2f}" for k in TABS) + f" | VL - tables {tx} -> {'DIFF' if ok else 'no diff'} | frequency part (nolearn - one) VL {Fq['VL'].mean():+.2f}, tables " + " / ".join(f"{Fq[k].mean():+.2f}" for k in TABS))
    lr = {k: np.log(np.array([rows[s][f"{i}_probe_{k}_thick_pressure"]["T"] / rows[s][f"{i}_probe_{k}_thick_reason"]["T"] for s, i in keep], float)) for k in ("VL",) + TABS}
    ok, tx = against(lambda k: lr["VL"] - lr[k], np.log(1.5))
    print(f"K5 (description) thick Tp/Tr: VL {np.exp(lr['VL'].mean()):.2f}; log-ratio VL - tables {tx} -> {'DIFF' if ok else 'no diff'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=int)
    ap.add_argument("--regress")
    ap.add_argument("--base", nargs=2, type=int)
    ap.add_argument("--basecheck")
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--nscale", type=float, default=1.0)
    ap.add_argument("--out")
    ap.add_argument("--gates", nargs="+")
    ap.add_argument("--basefile")
    ap.add_argument("--main", nargs="+")
    a = ap.parse_args()
    VL_LR, NSCALE = a.lr, a.nscale
    if a.time is not None:
        t0 = time.time()
        per_seed(a.time)
        print("one seed", f"{time.time() - t0:.0f}s")
    elif a.regress:
        regress(a.regress)
    elif a.basecheck:
        basecheck(a.basecheck)
    elif a.gates:
        gates(a.gates, a.basefile)
    elif a.main:
        mainsum(a.main, a.basefile)
    else:
        t0 = time.time()
        rng = a.base or a.seeds
        rows = {}
        for sd in range(*rng):
            rows[str(sd)] = per_seed(sd, base=a.base is not None)
            print("seed", sd, f"{time.time() - t0:.0f}s", flush=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": rng, "lr": VL_LR, "nscale": NSCALE, "times": TIMES, "rows": rows}, fh)
        print("written", a.out)
