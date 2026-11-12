"""Exploration 2: replay from memory supports evaluations while only the partner's reactions arrive (DESIGN-replay).

Prototype seeds only, numpy only. World, individuals, partner and rule S as in npn_anchor.py (arms shared / split,
P: domain A 0.9, Q: B 0.9). V is the plain MLP. Formation (shared by all learners, no replay) lasts N_FORM turns.
Memory: a pair = (prev vector, next vector, innate signal s of next, flag h). Vectors are scaled by sqrt(24) for G.
G is a VAE (50 -> 64 -> 6+6, 6 -> 64 -> 50, ReLU, recon / (2 * 0.3^2) + KL, Adam 1e-3, one step per pair).
Replay: after each real turn, one pair is recalled and V learns it with rule S (prev -> V(next); next -> s if h).
Learners: S0 no replay; RB uniform from the raw formation record; RBc same, record grows with later real pairs;
RU raw record, uniform over prev kinds; RGf recall from G frozen at the end of formation; RGc G keeps learning
from each real pair and the pair it recalled that turn.
Phases after formation: dyad 3000 -> departure 500 (replay only, memory frozen); reason test 300 (sweet content ->
bitter, real) -> 500 replay only.
Streams as npn_anchor; replay choice [s,17,stage]; G init [s,19]; G noise [s,23,stage].

usage:
  python npn_replay.py --gcheck 999
  python npn_replay.py --smoke
  python npn_replay.py --seeds 0 8 --out ../p72_replay_proto.json
  python npn_replay.py --summary ../p72_replay_proto.json
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from npn_leafleak import D, MLP, NOISE  # noqa: E402
from npn_anchor import A, B, C, ARMS, World, content, dyad_turns, formation_turns, reason_turns, run_turn  # noqa: E402

N_FORM, N_DYAD, N_LEAVE, N_REASON = 9000, 3000, 500, 300
SCALE = np.sqrt(D)
GH, GZ, GSIG, GLR, GBETA, GSTEPS = 64, 3, 0.3, 1e-3, 1.0, 3
COS_OK = 0.95
INNATE = (1.0, -1.0, 0.0)
LEARNERS = ("S0", "RB", "RBc", "RU", "RGf", "RGc")
T_MEASURE = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]


class RWorld:
    """World that also resolves recalled vectors ("v", array)."""

    def __init__(self, base):
        self.base, self.el = base, base.el

    def vec(self, e):
        return e[1] if e[0] == "v" else self.base.vec(e)


class VAE:
    def __init__(self, seed):
        g = np.random.default_rng([seed, 19])

        def lin(o, i):
            k = 1 / np.sqrt(i)
            return [g.uniform(-k, k, (o, i)), np.zeros(o)]

        n_in = 2 * D + 2
        self.p = lin(GH, n_in) + lin(2 * GZ, GH) + lin(GH, GZ) + lin(n_in, GH)
        self.m = [np.zeros_like(x) for x in self.p]
        self.v = [np.zeros_like(x) for x in self.p]
        self.t = 0

    def decode(self, z):
        W3, b3, W4, b4 = self.p[4:]
        h = np.maximum(W3 @ z + b3, 0)
        return W4 @ h + b4

    def train(self, x, rng):
        W1, b1, W2, b2, W3, b3, W4, b4 = self.p
        h1p = W1 @ x + b1
        h1 = np.maximum(h1p, 0)
        o = W2 @ h1 + b2
        mu, lv = o[:GZ], np.clip(o[GZ:], -8, 8)
        eps = rng.normal(0, 1, GZ)
        sd = np.exp(lv / 2)
        z = mu + sd * eps
        h2p = W3 @ z + b3
        h2 = np.maximum(h2p, 0)
        xh = W4 @ h2 + b4
        dxh = (xh - x) / GSIG ** 2
        gW4, gb4 = np.outer(dxh, h2), dxh
        dh2 = (W4.T @ dxh) * (h2p > 0)
        gW3, gb3 = np.outer(dh2, z), dh2
        dz = W3.T @ dh2
        dmu = dz + GBETA * mu
        dlv = dz * eps * sd / 2 + GBETA * 0.5 * (np.exp(lv) - 1)
        do = np.concatenate([dmu, dlv])
        gW2, gb2 = np.outer(do, h1), do
        dh1 = (W2.T @ do) * (h1p > 0)
        gW1, gb1 = np.outer(dh1, x), dh1
        grads = [gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4]
        self.t += 1
        for i, gr in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * gr
            self.v[i] = 0.999 * self.v[i] + 0.001 * gr * gr
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= GLR * mh / (np.sqrt(vh) + 1e-8)

    def recall(self, rng):
        x = self.decode(rng.normal(0, 1, GZ))
        return x


def pack(xp, xn, s, h):
    return np.concatenate([xp * SCALE, xn * SCALE, [s, h]])


def unpack(x):
    return x[:D] / SCALE, x[D:2 * D] / SCALE, float(np.clip(x[2 * D], -1, 1)), bool(x[2 * D + 1] > 0.5)


def pair_of(turn, w):
    """Memory pair of a real turn [prev, next]."""
    prev, nxt = turn
    s, h = (INNATE[nxt[1]], 1.0) if nxt[0] == "t" else (0.0, 0.0)
    return {"key": prev, "x": pack(w.vec(prev), w.vec(nxt), s, h)}


def replay_pair(model, rw, x, rn):
    xp, xn, s, h = unpack(x)
    ev = [("v", xp), ("v", xn)]
    n0, n1 = rn.normal(0, NOISE, D), rn.normal(0, NOISE, D)
    target = model.value(ev[1], rw)
    model.learn(ev[0], rw, target, n0)
    if h:
        model.learn(ev[1], rw, s, n1)


class Memory:
    def __init__(self, kind, record, g, seed):
        self.kind, self.record = kind, list(record)
        self.g = copy.deepcopy(g) if g is not None else None
        self.keys = {}
        for p in self.record:
            self.keys.setdefault(p["key"], []).append(p)
        self.rng_g = np.random.default_rng([seed, 23, 1])

    def recall(self, rc):
        if self.kind in ("RB", "RBc"):
            return self.record[int(rc.integers(len(self.record)))]["x"]
        if self.kind == "RU":
            ks = list(self.keys)
            lst = self.keys[ks[int(rc.integers(len(ks)))]]
            return lst[int(rc.integers(len(lst)))]["x"]
        return self.g.recall(rc)

    def store(self, p, recalled):
        if self.kind == "RBc":
            self.record.append(p)
        elif self.kind == "RGc":
            self.g.train(p["x"], self.rng_g)
            if recalled is not None:
                self.g.train(recalled, self.rng_g)


def snap(m, w, arm):
    r5 = lambda v: round(float(v), 5)
    return {"c": [[r5(m.value(content(arm, d, k), w)) for k in range(3)] for d in range(3)],
            "r": [r5(m.value(("r", i), w)) for i in range(2)],
            "t": [r5(m.value(("t", i), w)) for i in range(3)]}


def g_quality(g, w, arm, seed, n=2000, tag=9):
    """Recall profile. ret: share of recalls that are a formation-type pair (prev = content of A/B or a reaction,
    next = a taste, both cosine >= COS_OK). cons: among those, h set, the taste is the rule's and |s - rule| < 0.5.
    dom: thick/thin counts among formation-type content recalls. dyad: share of partner-type pairs.
    margin: mean (best - second best) cosine of prev."""
    protos = {("c", d, k): w.vec(content(arm, d, k)) for d in range(3) for k in range(3)}
    protos.update({("r", i): w.vec(("r", i)) for i in range(2)})
    protos.update({("t", i): w.vec(("t", i)) for i in range(3)})
    keys = list(protos)
    M = np.array([protos[k] / np.linalg.norm(protos[k]) for k in keys])
    rng = np.random.default_rng([seed, 23, tag])
    ret, cons, dyad, dom, marg = 0, 0, 0, [0, 0, 0], []
    for _ in range(n):
        xp, xn, s, h = unpack(g.recall(rng))
        cp, cn = M @ (xp / (np.linalg.norm(xp) + 1e-12)), M @ (xn / (np.linalg.norm(xn) + 1e-12))
        op = np.sort(cp)
        marg.append(float(op[-1] - op[-2]))
        ip, i_n = int(np.argmax(cp)), int(np.argmax(cn))
        kp, kn = keys[ip], keys[i_n]
        if cp[ip] < COS_OK or cn[i_n] < COS_OK:
            continue
        if (kp[0] == "r" and kn[0] == "c" and kn[1] != C) or (kp[0] == "c" and kp[1] != C and kn[0] == "r"):
            dyad += 1
            continue
        if kn[0] != "t" or kp[0] == "t" or (kp[0] == "c" and kp[1] == C):
            continue
        ret += 1
        want = kp[2] if kp[0] == "c" else kp[1]
        if kp[0] == "c":
            dom[kp[1]] += 1
        cons += int(h and kn[1] == want and abs(s - INNATE[want]) < 0.5)
    return {"ret": ret / n, "cons": cons / max(ret, 1), "dom": dom, "dyad": dyad / n, "margin": float(np.mean(marg))}


def run_learner(kind, V0, mem0, w, rw, arm, seed):
    row = {}
    # dyad then departure
    m = copy.deepcopy(V0)
    mem = None if kind == "S0" else Memory(kind, mem0["record"], mem0["g"], seed)
    rn, rr = np.random.default_rng([seed, 7, 1]), np.random.default_rng([seed, 17, 1])
    series = [snap(m, w, arm)]
    for t, ev in enumerate(dyad_turns(seed, arm), 1):
        run_turn(m, w, ev, rn)
        rec = None
        if mem is not None:
            rec = mem.recall(rr)
            replay_pair(m, rw, rec, rn)
            mem.store(pair_of(ev, w), rec)
        if t in T_MEASURE:
            series.append(snap(m, w, arm))
        if kind == "RGc" and t in (1000, 2000, 3000):
            row.setdefault("gprof", {})[str(t)] = g_quality(mem.g, w, arm, seed, n=500, tag=t)
    row["dyad"] = series
    leave = []
    for t in range(1, N_LEAVE + 1):
        if mem is not None:
            replay_pair(m, rw, mem.recall(rr), rn)
        if t % 100 == 0:
            leave.append(snap(m, w, arm))
    row["leave"] = leave
    # reason test then replay only
    m = copy.deepcopy(V0)
    mem = None if kind == "S0" else Memory(kind, mem0["record"], mem0["g"], seed)
    rn, rr = np.random.default_rng([seed, 7, 2]), np.random.default_rng([seed, 17, 2])
    for ev in reason_turns(seed, arm):
        run_turn(m, w, ev, rn)
        if mem is not None:
            rec = mem.recall(rr)
            replay_pair(m, rw, rec, rn)
            mem.store(pair_of(ev, w), rec)
    row["reason_end"] = snap(m, w, arm)
    if kind == "RGc":
        row.setdefault("gprof", {})["reason"] = g_quality(mem.g, w, arm, seed, n=500, tag=4000)
    for _ in range(N_LEAVE):
        if mem is not None:
            replay_pair(m, rw, mem.recall(rr), rn)
    row["reason_after"] = snap(m, w, arm)
    return row


def formation(seed, arm, indiv, w, nform=N_FORM):
    V = MLP(seed)
    g = VAE(seed)
    rn, rg = np.random.default_rng([seed, 7, 0]), np.random.default_rng([seed, 23, 0])
    record = []
    for ev in formation_turns(seed, arm, indiv, nform):
        run_turn(V, w, ev, rn)
        p = pair_of(ev, w)
        record.append(p)
        for _ in range(GSTEPS):
            g.train(p["x"], rg)
    return V, {"record": record, "g": g}


def per_seed(seed, nform=N_FORM, learners=LEARNERS):
    w = World(seed)
    rw = RWorld(w)
    out = {}
    for arm in ARMS:
        for indiv in ("P", "Q"):
            V0, mem0 = formation(seed, arm, indiv, w, nform)
            base = {"form_end": snap(V0, w, arm), "gq": g_quality(mem0["g"], w, arm, seed)}
            for kind in learners:
                out[f"{arm}_{indiv}_{kind}"] = dict(base, **run_learner(kind, V0, mem0, w, rw, arm, seed))
    return out


# ---------- summary (DESIGN-replay section 5) ----------

THICK, THIN = {"P": A, "Q": B}, {"P": B, "Q": A}


def own(s, d):
    return s["c"][d][0] - s["c"][d][1]


def form_dev(s):
    dev = [abs(s["c"][d][k] - INNATE[k]) for d in (A, B) for k in range(3)]
    dev += [abs(s["t"][i] - INNATE[i]) for i in range(3)] + [abs(s["r"][0] - 1), abs(s["r"][1] + 1)]
    return max(dev)


def fmt(v):
    return " ".join(f"{y:+.2f}" for y in v)


def g_ok(gq, indiv):
    th, tn = gq["dom"][THICK[indiv]], gq["dom"][THIN[indiv]]
    ratio = th / tn if tn else float("inf")
    return gq["ret"] >= 0.6 and gq["cons"] >= 0.8, ratio


def summarize(path):
    data = json.load(open(path, encoding="utf-8"))
    rows, seeds = data["rows"], sorted(data["rows"], key=int)
    print("formation turns:", data.get("nform"), "| G:", data.get("g"), "| t =", " ".join(str(t) for t in T_MEASURE))
    cells = [(s, i) for s in seeds for i in "PQ"]
    for arm in ARMS:
        print(f"\n===== arm {arm} =====")
        R = lambda s, i, k: rows[s][f"{arm}_{i}_{k}"]
        vok = {(s, i): form_dev(R(s, i, "S0")["form_end"]) < 0.2 for s, i in cells}
        print(f"-- V formation end kept {sum(vok.values())}/16, pairs {sum(vok[(s, 'P')] and vok[(s, 'Q')] for s in seeds)}/8")
        print("-- G recall at formation end per seed (P | Q): return, consistency, thick/thin ratio, margin (* = below a floor)")
        gok = {}
        for s in seeds:
            parts = []
            for i in "PQ":
                gq = R(s, i, "S0")["gq"]
                ok, ratio = g_ok(gq, i)
                gok[(s, i)] = ok
                parts.append(f"{gq['ret']:.2f} {gq['cons']:.2f} {ratio:.1f} {gq['margin']:.2f}{'' if ok else '*'}")
            print(f"  seed {s}: " + " | ".join(parts))
        print(f"  G kept {sum(gok.values())}/16, pairs {sum(gok[(s, 'P')] and gok[(s, 'Q')] for s in seeds)}/8")
        print("-- RGc memory profile (all cells): stage, return, thick/thin, partner-pair share")
        for tag in ("1000", "2000", "3000", "reason"):
            ps = [(R(s, i, "RGc")["gprof"][tag], i) for s, i in cells]
            th = sum(p["dom"][THICK[i]] for p, i in ps)
            tn = sum(p["dom"][THIN[i]] for p, i in ps)
            print(f"  {tag:6s} ret {np.mean([p['ret'] for p, _ in ps]):.2f} thick/thin {th / max(tn, 1):.1f} partner {np.mean([p['dyad'] for p, _ in ps]):.2f}")
        res = {}
        for k in LEARNERS:
            pairs_v = [s for s in seeds if vok[(s, "P")] and vok[(s, "Q")]]
            pairs_g = [s for s in pairs_v if gok[(s, "P")] and gok[(s, "Q")]]
            labels = (("kept", pairs_g), ("all", pairs_v)) if k.startswith("RG") else (("kept", pairs_v),)
            for label, use in labels:
                if not use:
                    print(f"  {k} [{label}]: no pairs")
                    continue
                Rth, Rtn, Hp, Dp, Pr, oth, otn, oc, rec, rs = [], [], [], [], [], [], [], [], [], []
                for s in use:
                    hs, ds, ow = [], [], []
                    for i in "PQ":
                        r = R(s, i, k)
                        ser = r["dyad"]
                        th = np.array([own(x, THICK[i]) for x in ser])
                        tn = np.array([own(x, THIN[i]) for x in ser])
                        Rth.append(th / th[0]); Rtn.append(tn / tn[0])
                        hs.append(th / th[0] - tn / tn[0]); ds.append(((tn[0] - tn) - (th[0] - th)) / 2)
                        ow.append(th[-1])
                        oth.append(th); otn.append(tn)
                        Pr.append([x["r"][0] - x["r"][1] for x in ser])
                        oc.append([own(x, C) for x in ser])
                        lt, ln = own(r["leave"][-1], THICK[i]), own(r["leave"][-1], THIN[i])
                        rec.append([(lt - th[-1]) / (th[0] - th[-1] + 1e-9), (ln - tn[-1]) / (tn[0] - tn[-1] + 1e-9)])
                        f0, re, ra = r["form_end"]["c"], r["reason_end"]["c"], r["reason_after"]["c"]
                        rs.append([f0[THICK[i]][0], re[THICK[i]][0], ra[THICK[i]][0], f0[THIN[i]][0], re[THIN[i]][0], ra[THIN[i]][0]])
                    Hp.append((hs[0] + hs[1]) / 2); Dp.append((ds[0] + ds[1]) / 2)
                    res.setdefault(k + label, {})[s] = ((hs[0][-1] + hs[1][-1]) / 2, (ds[0][-1] + ds[1][-1]) / 2, float(np.mean(ow)))
                Rm = (np.mean(Rth, 0) + np.mean(Rtn, 0)) / 2
                ts = int(np.argmin(np.abs(Rm - 0.5)))
                h3, d3 = [h[-1] for h in Hp], [d[-1] for d in Dp]
                print(f"  {k:3s} [{label}] pairs {len(use)} | H(3000) {np.mean(h3):+.2f} +{sum(x > 0 for x in h3)}/{len(h3)}"
                      f" | D(3000) {np.mean(d3):+.2f} +{sum(x > 0 for x in d3)}/{len(d3)}"
                      f" | Own_thick(3000) {np.mean([o[-1] for o in oth]):+.2f} | t* {T_MEASURE[ts]} H(t*) {np.mean([h[ts] for h in Hp]):+.2f}")
                print(f"      H         {fmt(np.mean(Hp, 0))}")
                print(f"      Own_thick {fmt(np.mean(oth, 0))}")
                print(f"      Own_thin  {fmt(np.mean(otn, 0))}")
                print(f"      Pr        {fmt(np.mean(Pr, 0))}")
                print(f"      Own_C     {fmt(np.mean(oc, 0))}")
                print(f"      leave recovery fraction thick {np.mean([r[0] for r in rec]):+.2f} thin {np.mean([r[1] for r in rec]):+.2f}")
                rm = np.mean(rs, 0)
                print(f"      reason V(sweet): thick {rm[0]:+.2f} -> {rm[1]:+.2f} -> {rm[2]:+.2f} | thin {rm[3]:+.2f} -> {rm[4]:+.2f} -> {rm[5]:+.2f}")
        print("-- paired differences at t = 3000 (H, Own_thick)")
        for k1, k2 in (("RGfkept", "RUkept"), ("RGckept", "RUkept"), ("RGfall", "RUkept"), ("RGcall", "RUkept"), ("RBkept", "RUkept"),
                       ("RGfkept", "S0kept"), ("RGckept", "S0kept"), ("RGfall", "S0kept"), ("RGcall", "S0kept"), ("RBkept", "S0kept")):
            if k1 not in res or k2 not in res:
                continue
            common = [s for s in res[k1] if s in res[k2]]
            dh = [res[k1][s][0] - res[k2][s][0] for s in common]
            do = [res[k1][s][2] - res[k2][s][2] for s in common]
            print(f"  {k1} - {k2}: H {np.mean(dh):+.2f} +{sum(x > 0 for x in dh)}/{len(dh)} | Own_thick {np.mean(do):+.2f} +{sum(x > 0 for x in do)}/{len(do)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcheck", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.gcheck is not None:
        t0 = time.time()
        w = World(a.gcheck)
        for arm in ARMS:
            for indiv in "PQ":
                V0, mem0 = formation(a.gcheck, arm, indiv, w)
                gq = g_quality(mem0["g"], w, arm, a.gcheck)
                print(arm, indiv, gq, "V dev", round(form_dev(snap(V0, w, arm)), 3), f"{time.time() - t0:.0f}s", flush=True)
    elif a.smoke:
        t0 = time.time()
        r = per_seed(999)
        print("smoke finite", all(np.isfinite(v) for row in r.values() for v in np.ravel(row["dyad"][-1]["c"])), f"{time.time() - t0:.0f}s")
    elif a.summary:
        summarize(a.summary)
    else:
        t0 = time.time()
        rows = {}
        for s in range(*a.seeds):
            rows[str(s)] = per_seed(s)
            print("seed", s, f"{time.time() - t0:.0f}s", flush=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "nform": N_FORM, "learners": LEARNERS, "g": [GSIG, GZ, GH, GLR, GBETA, GSTEPS], "rows": rows}, fh)
        print("written", a.out)
