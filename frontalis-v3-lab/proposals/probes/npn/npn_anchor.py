"""Exploration 1: can importance-weighted plasticity keep thick domains and yield thin ones (DESIGN-anchor v1.1).

Prototype seeds only, numpy only. Builds on npn_leafleak.py (same elements, MLP, linear model, rule S).
Contents are a_i + b_j. Domains: b1 = A, b2 = B, b3 = C (C never appears in formation or dyad).
Arm "shared": both domains use a1 (sweet), a2 (bitter), a3 (bland). Arm "split": A uses a1..a3, B uses a4..a6
(sweet, bitter, bland); a4..a6 are new vectors from stream [s,1]. Domain C uses a1..a3 in both arms.
Formation: 0.8 content -> taste (P: domain A 0.9 / B 0.1; Q: A 0.1 / B 0.9), 0.2 approve -> sweet or
disapprove -> bitter. Dyad (no innate signal): content (A/B half, sweet/bitter half); the partner approves bitter
and disapproves sweet contents; half reaction -> content, half content -> reaction.
Reason test (300 turns from the end of formation): the sweet content of A or B -> bitter.
Learners: S0 plain SGD; SA importance Omega += (dV/dtheta)^2 at the noisy learned input for every learned event
(no decay), step scaled by f = 1/(1 + lam Omega / mean over the array); SP same f under a fixed per-array
permutation drawn at birth; SU plain with lr * c (c = mean over P and Q of SA's step ratio, separately for
formation and dyad); L0 / LA linear plain / importance; TC table with step 1/(n+1).
Streams: elements [s,0] (+ a4..a6 [s,1]); formation [s,3,0]; dyad [s,3,1]; reason [s,3,2]; noise [s,7,stage];
init [s,11]; permutation [s,13].

usage:
  python npn_anchor.py --smoke
  python npn_anchor.py --seeds 0 8 --nform 3000 --out ../p71_anchor_proto.json
  python npn_anchor.py --summary ../p71_anchor_proto.json
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npn_leafleak as LL  # noqa: E402
from npn_leafleak import CLIP, D, LR, MLP, Lin, _norm, run_turn  # noqa: E402

LAM = 10.0
N_DYAD, N_REASON = 3000, 300
P_THICK = 0.9
T_MEASURE = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]
KINDS = ("S0", "SA", "SP", "SU", "L0", "LA", "TC")
ARMS = ("shared", "split")
A, B, C = 0, 1, 2
DOM = "ABC"


class World(LL.World):
    def __init__(self, seed):
        super().__init__(seed)
        g = np.random.default_rng([seed, 1])
        self.el["a"] = np.vstack([self.el["a"], g.normal(0, 1 / np.sqrt(D), (3, D))])


def aset(arm, d):
    """Attribute indices (sweet, bitter, bland) of domain d."""
    return (3, 4, 5) if (arm == "split" and d == B) else (0, 1, 2)


def content(arm, d, k):
    return ("c", aset(arm, d)[k], d)


class Plastic:
    def init_plastic(self, mode, seed, params):
        self.mode = mode  # plain | imp | perm | uniform
        self.c = 1.0
        self.omega = [np.zeros(np.asarray(p, dtype=float).shape) for p in params]
        g = np.random.default_rng([seed, 13])
        self.perm = [g.permutation(np.asarray(p).size) for p in params]
        self.reset_ratio()

    def reset_ratio(self):
        self.r_step, self.r_func, self.r_n = 0.0, 0.0, 0

    def scales(self):
        out = []
        for om, pm in zip(self.omega, self.perm):
            m = om.mean()
            f = 1.0 / (1.0 + LAM * om / m) if m > 0 else np.ones_like(om)
            if self.mode == "perm":
                f = f.reshape(-1)[pm].reshape(om.shape)
            out.append(f)
        return out

    def step(self, og, g):
        s = LR * min(1.0, CLIP / (abs(g) * _norm(og) + 1e-6)) * g
        if self.mode == "plain":
            return [s * x for x in og]
        if self.mode == "uniform":
            return [self.c * s * x for x in og]
        for om, x in zip(self.omega, og):
            om += np.asarray(x) ** 2
        f = self.scales()
        fg = [fi * x for fi, x in zip(f, og)]
        n2 = sum(float((np.asarray(x) ** 2).sum()) for x in og) + 1e-12
        self.r_step += _norm(fg) / np.sqrt(n2)
        self.r_func += sum(float((fi * np.asarray(x) ** 2).sum()) for fi, x in zip(f, og)) / n2
        self.r_n += 1
        return [s * x for x in fg]

    def eff_plasticity(self, xs):
        f = self.scales()
        r = []
        for x in xs:
            og = self.out_grads(x)
            r.append(_norm([fi * gi for fi, gi in zip(f, og)]) / (_norm(og) + 1e-12))
        return float(np.mean(r))


class PMLP(MLP, Plastic):
    def __init__(self, seed, mode):
        MLP.__init__(self, seed)
        self.init_plastic(mode, seed, [self.w3, np.array(self.b3), self.W2, self.b2, self.W1, self.b1])

    def out_grads(self, x):
        return self._grads(x)[1]

    def learn(self, e, world, target, noise):
        v, og = self._grads(world.vec(e) + noise)
        u = self.step(og, 2 * (v - target))
        self.w3 -= u[0]
        self.b3 -= float(u[1])
        self.W2 -= u[2]
        self.b2 -= u[3]
        self.W1 -= u[4]
        self.b1 -= u[5]


class PLin(Lin, Plastic):
    def __init__(self, seed, mode):
        Lin.__init__(self, seed)
        self.init_plastic(mode, seed, [self.w, np.array(self.b)])

    def out_grads(self, x):
        return [x, np.array(1.0)]

    def learn(self, e, world, target, noise):
        x = world.vec(e) + noise
        u = self.step([x, np.array(1.0)], 2 * (float(self.w @ x + self.b) - target))
        self.w -= u[0]
        self.b -= float(u[1])


class CountTab:
    def __init__(self, seed=None):
        self.v, self.n = {}, {}

    def value(self, e, world):
        return self.v.get(e, 0.0)

    def learn(self, e, world, target, noise):
        n, v = self.n.get(e, 0), self.v.get(e, 0.0)
        self.v[e] = v + (target - v) / (n + 1)
        self.n[e] = n + 1


def formation_turns(seed, arm, indiv, n):
    rc = np.random.default_rng([seed, 3, 0])
    pa = P_THICK if indiv == "P" else 1 - P_THICK
    out = []
    for _ in range(n):
        u, k, w = rc.random(), int(rc.integers(6)), rc.random()
        if u < 0.8:
            out.append([content(arm, A if w < pa else B, k % 3), ("t", k % 3)])
        else:
            out.append([("r", k % 2), ("t", k % 2)])
    return out


def dyad_turns(seed, arm, n=N_DYAD):
    rc = np.random.default_rng([seed, 3, 1])
    out = []
    for _ in range(n):
        d, k, order = int(rc.integers(2)), int(rc.integers(2)), rc.random()
        c, r = content(arm, d, k), ("r", 0 if k == 1 else 1)  # approve bitter, disapprove sweet
        out.append([r, c] if order < 0.5 else [c, r])
    return out


def reason_turns(seed, arm, n=N_REASON):
    rc = np.random.default_rng([seed, 3, 2])
    return [[content(arm, int(rc.integers(2)), 0), ("t", 1)] for _ in range(n)]


def snap(m, w, arm):
    r5 = lambda v: round(float(v), 5)
    return {"c": [[r5(m.value(content(arm, d, k), w)) for k in range(3)] for d in range(3)],
            "r": [r5(m.value(("r", i), w)) for i in range(2)],
            "t": [r5(m.value(("t", i), w)) for i in range(3)]}


def make(kind, seed):
    if kind == "TC":
        return CountTab()
    mode = {"S0": "plain", "SA": "imp", "SP": "perm", "SU": "uniform", "L0": "plain", "LA": "imp"}[kind]
    return (PLin if kind in ("L0", "LA") else PMLP)(seed, mode)


def run_one(kind, seed, arm, indiv, w, nform, c=(1.0, 1.0)):
    m = make(kind, seed)
    m.c = c[0]
    rn = np.random.default_rng([seed, 7, 0])
    for ev in formation_turns(seed, arm, indiv, nform):
        run_turn(m, w, ev, rn)
    row = {"form_end": snap(m, w, arm)}
    if kind in ("SA", "SP", "LA"):
        row["eff_plast"] = [m.eff_plasticity([w.vec(content(arm, d, k)) for k in range(3)]) for d in range(3)]
        row["ratio_form"] = [m.r_step / m.r_n, m.r_func / m.r_n]
        m.reset_ratio()
    if kind != "TC":
        m.c = c[1]
    md = copy.deepcopy(m)
    rn = np.random.default_rng([seed, 7, 1])
    series = [snap(md, w, arm)]
    for t, ev in enumerate(dyad_turns(seed, arm), 1):
        run_turn(md, w, ev, rn)
        if t in T_MEASURE:
            series.append(snap(md, w, arm))
    row["dyad"] = series
    if kind in ("SA", "SP", "LA"):
        row["ratio_dyad"] = [md.r_step / md.r_n, md.r_func / md.r_n]
    mr = copy.deepcopy(m)
    rn = np.random.default_rng([seed, 7, 2])
    for ev in reason_turns(seed, arm):
        run_turn(mr, w, ev, rn)
    row["reason_end"] = snap(mr, w, arm)
    return row


def per_seed(seed, nform):
    w, out = World(seed), {}
    for arm in ARMS:
        for kind in KINDS:
            if kind == "SU":
                sa = [out[f"{arm}_{i}_SA"] for i in ("P", "Q")]
                c = (np.mean([r["ratio_form"][0] for r in sa]), np.mean([r["ratio_dyad"][0] for r in sa]))
            else:
                c = (1.0, 1.0)
            for indiv in ("P", "Q"):
                out[f"{arm}_{indiv}_{kind}"] = run_one(kind, seed, arm, indiv, w, nform, c)
    return out


# ---------- summary (DESIGN-anchor sections 5 and 8) ----------

def own(s, d):
    return s["c"][d][0] - s["c"][d][1]


def form_dev(row):
    e = row["form_end"]
    dev = [abs(e["c"][d][k] - (1, -1, 0)[k]) for d in (A, B) for k in range(3)]
    dev += [abs(e["t"][i] - (1, -1, 0)[i]) for i in range(3)] + [abs(e["r"][0] - 1), abs(e["r"][1] + 1)]
    return max(dev)


def fmt(v):
    return " ".join(f"{y:+.2f}" for y in v)


THICK, THIN = {"P": A, "Q": B}, {"P": B, "Q": A}


def summarize(path):
    data = json.load(open(path, encoding="utf-8"))
    rows, seeds = data["rows"], sorted(data["rows"], key=int)
    print("formation turns:", data.get("nform"))
    print("t =", " ".join(str(t) for t in T_MEASURE))
    for arm in ARMS:
        print(f"\n===== arm {arm} =====")
        R = lambda s, i, k: rows[s][f"{arm}_{i}_{k}"]
        ok, readable = {}, {}
        print("-- formation end: max |V - rule| (6 contents of A and B, tastes, reactions); kept if < 0.2")
        for k in KINDS:
            dev = {(s, i): form_dev(R(s, i, k)) for s in seeds for i in "PQ"}
            ok[k] = {key: v < 0.2 for key, v in dev.items()}
            pairs = [s for s in seeds if ok[k][(s, "P")] and ok[k][(s, "Q")]]
            readable[k] = len(pairs) >= 6
            th = [max(abs(R(s, i, k)["form_end"]["c"][THIN[i]][j] - (1, -1, 0)[j]) for j in range(3)) for s in seeds for i in "PQ"]
            tk = [max(abs(R(s, i, k)["form_end"]["c"][THICK[i]][j] - (1, -1, 0)[j]) for j in range(3)) for s in seeds for i in "PQ"]
            print(f"  {k:2s} max-dev mean {np.mean(list(dev.values())):.3f} | thick err {np.mean(tk):.3f} thin err {np.mean(th):.3f}"
                  f" | kept {sum(ok[k].values())}/16 pairs {len(pairs)}/8 {'' if readable[k] else '-> NOT READABLE'}")
        print("-- importance learners: effective plasticity at formation end (thick / thin / C), step and function ratios")
        for k in ("SA", "SP", "LA"):
            ep = np.array([[R(s, i, k)["eff_plast"][THICK[i]], R(s, i, k)["eff_plast"][THIN[i]], R(s, i, k)["eff_plast"][C]] for s in seeds for i in "PQ"])
            rf = np.mean([R(s, i, k)["ratio_form"] for s in seeds for i in "PQ"], 0)
            rd = np.mean([R(s, i, k)["ratio_dyad"] for s in seeds for i in "PQ"], 0)
            print(f"  {k}: thick {ep[:, 0].mean():.3f} thin {ep[:, 1].mean():.3f} C {ep[:, 2].mean():.3f} thin/thick {np.mean(ep[:, 1] / ep[:, 0]):.2f}"
                  f" | form step {rf[0]:.3f} func {rf[1]:.3f} | dyad step {rd[0]:.3f} func {rd[1]:.3f}")
        Hst, tstar = {}, {}
        print("-- dyad (pairs kept in both P and Q)")
        for k in KINDS:
            pairs = [s for s in seeds if ok[k][(s, "P")] and ok[k][(s, "Q")]]
            if not pairs:
                print(f"  {k}: no pairs")
                continue
            Rth, Rtn, Hp, Dp, Pr, lvl, oc = [], [], [], [], [], [], []
            for s in pairs:
                hs, ds = [], []
                for i in "PQ":
                    ser = R(s, i, k)["dyad"]
                    th, tn = [own(x, THICK[i]) for x in ser], [own(x, THIN[i]) for x in ser]
                    rth, rtn = np.array(th) / th[0], np.array(tn) / tn[0]
                    Rth.append(rth); Rtn.append(rtn)
                    hs.append(rth - rtn)
                    ds.append(((tn[0] - np.array(tn)) - (th[0] - np.array(th))) / 2)
                    Pr.append([x["r"][0] - x["r"][1] for x in ser])
                    lvl.append([np.mean([x["c"][d][j] - ser[0]["c"][d][j] for d in (A, B) for j in range(3)]) for x in ser])
                    oc.append([own(x, C) for x in ser])
                Hp.append((hs[0] + hs[1]) / 2); Dp.append((ds[0] + ds[1]) / 2)
            Rm = (np.mean(Rth, 0) + np.mean(Rtn, 0)) / 2
            ts = int(np.argmin(np.abs(Rm - 0.5)))
            tstar[k] = ts
            Hst[k] = {s: h[ts] for s, h in zip(pairs, Hp)}
            h_ts, d_ts = [h[ts] for h in Hp], [d[ts] for d in Dp]
            print(f"  {k:2s} pairs {len(pairs)} {'' if readable[k] else '(NOT READABLE)'} | t* = {T_MEASURE[ts]} (mean R {Rm[ts]:+.2f})"
                  f" | H(t*) {np.mean(h_ts):+.2f} +{sum(x > 0 for x in h_ts)}/{len(h_ts)} | D(t*) {np.mean(d_ts):+.2f} +{sum(x > 0 for x in d_ts)}/{len(d_ts)}")
            print(f"     H per pair at t*: {fmt(h_ts)}")
            print(f"     H       {fmt(np.mean(Hp, 0))}")
            print(f"     D       {fmt(np.mean(Dp, 0))}")
            print(f"     R_thick {fmt(np.mean(Rth, 0))}")
            print(f"     R_thin  {fmt(np.mean(Rtn, 0))}")
            print(f"     Pr      {fmt(np.mean(Pr, 0))}")
            print(f"     level   {fmt(np.mean(lvl, 0))}")
            print(f"     Own_C   {fmt(np.mean(oc, 0))}")
        print("-- paired differences of H at each learner's own t*")
        for k1, k2 in (("SA", "SP"), ("SA", "SU"), ("SA", "S0"), ("SA", "LA"), ("LA", "L0")):
            if k1 not in Hst or k2 not in Hst:
                continue
            common = [s for s in Hst[k1] if s in Hst[k2]]
            d = [Hst[k1][s] - Hst[k2][s] for s in common]
            print(f"  {k1} - {k2}: mean {np.mean(d):+.2f} +{sum(x > 0 for x in d)}/{len(d)}")
        print("-- reason test: drop of V(sweet, thick) and V(sweet, thin) after 300 turns -> bitter; ratio thick/thin")
        for k in KINDS:
            dt, dn = [], []
            for s in seeds:
                for i in "PQ":
                    if ok[k][(s, i)]:
                        r = R(s, i, k)
                        dt.append(r["form_end"]["c"][THICK[i]][0] - r["reason_end"]["c"][THICK[i]][0])
                        dn.append(r["form_end"]["c"][THIN[i]][0] - r["reason_end"]["c"][THIN[i]][0])
            if dt:
                print(f"  {k:2s} thick {np.mean(dt):+.2f} thin {np.mean(dn):+.2f} ratio {np.mean(dt) / np.mean(dn):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--nform", type=int, default=3000)
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.smoke:
        t0 = time.time()
        r = per_seed(999, 3000)
        fin = all(np.isfinite(v) for row in r.values() for v in np.ravel(row["dyad"][-1]["c"]))
        print("smoke finite", fin, "SA form ratio", np.round(r["split_P_SA"]["ratio_form"], 3), f"{time.time() - t0:.0f}s")
    elif a.summary:
        summarize(a.summary)
    else:
        t0 = time.time()
        rows = {}
        for s in range(*a.seeds):
            rows[str(s)] = per_seed(s, a.nform)
            print("seed", s, f"{time.time() - t0:.0f}s", flush=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "nform": a.nform, "rows": rows}, fh)
        print("written", a.out)
