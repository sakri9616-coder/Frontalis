"""Feasibility probe P2 for DESIGN draft 2 (dev seeds 77380-77385 only; not a test of the hypothesis).

What is new compared with P1 (which regressed one designer value function R):
  * The world returns multi-channel BODY-STATE changes (K channels), not a value. Nothing says what is good.
  * Innate structure I-H: homeostatic band per channel; discomfort d_k = max(|b_k| - tol, 0)^2 kept per channel.
  * The personality's own evaluation f of an experience is computed AT THE TIME of the experience from its CURRENT
    state (cycle-on): history-dependent channel weights w (sensitisation), reference to its own expectation V,
    interpretation = attribution of the outcome to the two present entities by its own belief, salience-weighted learning.
    f is stored and becomes the learning target of the value head (an NN updated inside the cycle).
  * cycle-off: the same experiences, but targets come from the frozen innate evaluator (w = 1, no reference,
    uniform attribution, no salience). The difference isolates the cycle (philosophy section 6).
  * History classes defined a priori: A = early phase only with entity cluster G1, B = early phase only with G2,
    then an identical shared phase. A' = same history as A, different minibatch noise (test-retest).
  * Worlds: F1 (trait x context interaction), F6 (additive), F5 (structure-free permutation).
Measured (no developer value label anywhere):
  divergence between classes on unseen probes vs test-retest noise; cycle-on vs cycle-off;
  subspace patching (history-difference subspace found on TRAIN inputs, applied on UNSEEN probes) vs matched random
  subspace and vs same-class (A vs A') subspace; next-learning difference (same new batch -> update vectors);
  own body-change prediction R^2 on unseen entities (label = world consequence, not a value).
Also measures GPU throughput (personality-steps per second) for the compute estimate.
Run: PYTHONUTF8=1 python -B p2_cycle_world.py
"""
from __future__ import annotations

import json
import sys
import zlib
import time
from pathlib import Path

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parent / "p2_results.json"
K, DT, DC = 3, 4, 3
N_TR_PER_CL, N_UN_PER_CL, N_CTX = 80, 40, 12
D_OBS = 32 + 16 + 16 + 8          # entity obs + id + context obs + distractors
H = 64
N_STEPS, N_EARLY, BS = 3000, 1200, 32
TOL, DECAY = 0.4, 0.7


def make_world(seed, family):
    rng = np.random.default_rng([seed, 11])
    n_ent = 2 * (N_TR_PER_CL + N_UN_PER_CL)
    cl = np.repeat([0, 1], N_TR_PER_CL + N_UN_PER_CL)
    mu = rng.normal(size=DT); mu = 1.0 * mu / np.linalg.norm(mu)
    T = rng.normal(size=(n_ent, DT)) * 0.8 + np.where(cl[:, None] == 0, mu, -mu)
    C = rng.normal(size=(N_CTX, DC))
    Pt = rng.normal(size=(DT, 32)) / np.sqrt(DT); Pc = rng.normal(size=(DC, 16)) / np.sqrt(DC)
    Xe = np.concatenate([np.tanh(T @ Pt) + 0.3 * rng.normal(size=(n_ent, 32)), rng.normal(size=(n_ent, 16))], 1)
    Xc = np.tanh(C @ Pc) + 0.2 * rng.normal(size=(N_CTX, 16))
    B = rng.normal(size=(K, DT, DC)) / np.sqrt(DT * DC) * 2.0
    A = rng.normal(size=(K, DT)) * 0.5
    inter = np.einsum("ed,kdc,xc->exk", T, B, C)            # (ent, ctx, K)
    add = np.einsum("ed,kd->ek", T, A)[:, None, :] + 0.5 * rng.normal(size=(N_CTX, K))[None]
    if family == "F1":
        M = 0.7 * np.tanh(inter) + 0.2 * add
    elif family == "F6":
        M = 0.5 * np.tanh(add)
    else:                                                    # F5: same marginal as F1, structure destroyed
        M0 = 0.7 * np.tanh(inter) + 0.2 * add
        M = rng.permutation(M0.reshape(-1, K)).reshape(M0.shape)
    prop = rng.normal(size=n_ent)                            # latent propensity to be the actor
    idx_tr = {c: np.where(cl == c)[0][:N_TR_PER_CL] for c in (0, 1)}
    idx_un = {c: np.where(cl == c)[0][N_TR_PER_CL:] for c in (0, 1)}
    return dict(Xe=Xe.astype(np.float32), Xc=Xc.astype(np.float32), M=M.astype(np.float32), prop=prop,
                tr=idx_tr, un=idx_un)


def make_sequence(w, rng_shared, rng_early, early_cluster):
    """Two entities + context per experience. Early phase: only the class's cluster; shared phase identical across classes."""
    seq = []
    pool_e = w["tr"][early_cluster]
    for _ in range(N_EARLY):
        e = rng_early.choice(pool_e, 2, replace=False); seq.append((e[0], e[1], rng_early.integers(N_CTX), rng_early.random(), rng_early.normal(size=K)))
    pool = np.concatenate([w["tr"][0], w["tr"][1]])
    for _ in range(N_STEPS - N_EARLY):
        e = rng_shared.choice(pool, 2, replace=False); seq.append((e[0], e[1], rng_shared.integers(N_CTX), rng_shared.random(), rng_shared.normal(size=K)))
    return seq


def obs(w, e, c, rng=None):
    n = len(e)
    dis = np.zeros((n, 8), np.float32) if rng is None else rng.normal(size=(n, 8)).astype(np.float32)
    return np.concatenate([w["Xe"][e], w["Xc"][c], dis], 1)


class Batched:
    """P independent MLP personalities (encoder 2x64, belief head K, value head 1), stacked for bmm."""

    def __init__(self, P, seeds):
        g = torch.Generator(device="cpu")
        Ws = []
        for s in seeds:                                      # init depends only on the pair seed (A, B, A' share init)
            g.manual_seed(int(s))
            Ws.append([torch.randn(D_OBS, H, generator=g) * np.sqrt(2 / D_OBS), torch.zeros(1, H),
                       torch.randn(H, H, generator=g) * np.sqrt(2 / H), torch.zeros(1, H),
                       torch.randn(H, K, generator=g) * np.sqrt(1 / H), torch.zeros(1, K),
                       torch.randn(H, 1, generator=g) * np.sqrt(1 / H), torch.zeros(1, 1)])
        self.p = [torch.stack([W[i] for W in Ws]).to(DEV).requires_grad_(True) for i in range(8)]
        self.opt = torch.optim.Adam(self.p, lr=1e-3, weight_decay=0.0)

    def hidden(self, x, p=None):                             # x: (P, n, D)
        W1, b1, W2, b2 = (p or self.p)[:4]
        return torch.relu(torch.bmm(torch.relu(torch.bmm(x, W1) + b1), W2) + b2)

    def heads(self, h, p=None):
        p = p or self.p
        return torch.bmm(h, p[4]) + p[5], (torch.bmm(h, p[6]) + p[7]).squeeze(-1)

    def snapshot(self):
        return [t.detach().clone() for t in self.p]


def train(worlds_per_p, seqs, cycle_on, init_seeds, mb_seeds, timing_only=False):
    P = len(seqs)
    net = Batched(P, init_seeds)
    gens = [np.random.default_rng([int(s), 99]) for s in mb_seeds]
    b = torch.zeros(P, K, device=DEV)
    wts = torch.ones(P, K, device=DEV)
    run = torch.full((P,), 1e-3, device=DEV)
    cyc = torch.tensor(cycle_on, dtype=torch.float32, device=DEV)
    # storage of items: x for the two entities, target per entity, belief target, salience
    X = torch.zeros(P, N_STEPS, 2, D_OBS, device=DEV)
    TV = torch.zeros(P, N_STEPS, 2, device=DEV)
    DB = torch.zeros(P, N_STEPS, K, device=DEV)
    SAL = torch.ones(P, N_STEPS, device=DEV)
    t0 = time.time()
    for t in range(N_STEPS):
        xs, dbs = [], []
        for p in range(P):
            e1, e2, c, u, nz = seqs[p][t]
            w = worlds_per_p[p]
            actor = e1 if u < 1 / (1 + np.exp(-(w["prop"][e1] - w["prop"][e2]))) else e2
            xs.append(obs(w, np.array([e1, e2]), np.array([c, c]), gens[p]))
            dbs.append(w["M"][actor, c] + 0.1 * nz)
        x = torch.tensor(np.stack(xs), device=DEV)            # (P, 2, D)
        db = torch.tensor(np.stack(dbs), dtype=torch.float32, device=DEV)
        with torch.no_grad():
            h = net.hidden(x)
            pb, v = net.heads(h)                               # (P,2,K), (P,2)
            b = DECAY * b + db
            d = torch.clamp(b.abs() - TOL, min=0) ** 2         # per-channel discomfort (innate, not summed)
            # cycle-on evaluative state: weights w (sensitisation), reference, attribution, salience
            mag = d.mean(1)
            run = run + 0.05 * (mag - run)
            target_w = torch.clamp(d / run[:, None].clamp(min=1e-6), 0, 3)
            upd = (mag > 1e-9).float()[:, None] * cyc[:, None]
            wts = wts + 0.01 * upd * (target_w - wts); wts = 3 * wts / wts.sum(1, keepdim=True)
            w_use = cyc[:, None] * wts + (1 - cyc[:, None]) * torch.ones_like(wts)
            innate = -(w_use * d).sum(1)                       # this personality's weighting of its own channels
            expect = v.sum(1)
            f = innate + cyc * 0.5 * (innate - expect)         # reference to its own expectation (cycle-on only)
            ll = -((db[:, None, :] - pb) ** 2).sum(-1) / (2 * 0.1)
            r = torch.softmax(ll, dim=1)                       # interpretation: who caused it, by its own belief
            r = cyc[:, None] * r + (1 - cyc[:, None]) * 0.5
            sal = 1 + cyc * torch.clamp(f.abs(), max=2.0)
            X[:, t] = x; TV[:, t] = r * f[:, None]; DB[:, t] = db; SAL[:, t] = sal
        # one gradient step on a minibatch of past items
        n = t + 1
        idx = torch.tensor(np.stack([g.integers(0, n, BS) for g in gens]), device=DEV)
        xb = torch.gather(X, 1, idx[:, :, None, None].expand(-1, -1, 2, D_OBS)).reshape(P, BS * 2, D_OBS)
        tv = torch.gather(TV, 1, idx[:, :, None].expand(-1, -1, 2)).reshape(P, BS * 2)
        dbb = torch.gather(DB, 1, idx[:, :, None].expand(-1, -1, K))
        sal = torch.gather(SAL, 1, idx)
        h = net.hidden(xb)
        pb, v = net.heads(h)
        pb = pb.reshape(P, BS, 2, K); v = v.reshape(P, BS, 2)
        mix = -torch.logsumexp(-((dbb[:, :, None, :] - pb) ** 2).sum(-1) / (2 * 0.1), dim=2).mean()
        lv = ((sal[:, :, None] * (v - tv.reshape(P, BS, 2)) ** 2).mean())
        net.opt.zero_grad(); (P * (mix * 0.1 + lv)).backward(); net.opt.step()
        if timing_only and t == 199:
            torch.cuda.synchronize() if DEV == "cuda" else None
            return (time.time() - t0) / 200
    return net, wts


def probes(w):
    e = np.concatenate([w["un"][0], w["un"][1]])
    ee, cc = np.meshgrid(e, np.arange(N_CTX), indexing="ij")
    ee, cc = ee.ravel(), cc.ravel()
    return torch.tensor(obs(w, ee, cc), device=DEV), w["M"][ee, cc], (np.isin(ee, w["un"][0])).astype(int)


def train_inputs(w):
    e = np.concatenate([w["tr"][0], w["tr"][1]])
    ee, cc = np.meshgrid(e, np.arange(N_CTX), indexing="ij")
    return torch.tensor(obs(w, ee.ravel(), cc.ravel()), device=DEV)


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).mean() / ((y - y.mean(0)) ** 2).mean())


def main():
    seeds = list(range(77380, 77386))
    fams = ["F1", "F6", "F5"]
    worlds = {(s, f): make_world(s, f) for s in seeds for f in fams}
    rows, meta = [], []
    # personalities: for each seed x family x cycle: A, B, A'
    for s in seeds:
        for f in fams:
            w = worlds[(s, f)]
            for cyc in (1, 0):
                for role in ("A", "B", "A2"):
                    rs = np.random.default_rng([s, 21]); re = np.random.default_rng([s, 22 if role != "B" else 23])
                    seq = make_sequence(w, rs, re, 0 if role != "B" else 1)
                    meta.append(dict(seed=s, fam=f, cyc=cyc, role=role)); rows.append(seq)
    P = len(rows)
    print("personalities", P, file=sys.stderr)
    # throughput: 200 steps at P and at 4P
    tp = {}
    for mult in (1, 4):
        sub = (rows * mult)[: P * mult]
        mm = (meta * mult)[: P * mult]
        sec = train([worlds[(m["seed"], m["fam"])] for m in mm], sub, [m["cyc"] for m in mm],
                    [m["seed"] for m in mm], [zlib.crc32(repr((m["seed"], m["fam"], m["cyc"], m["role"], i)).encode()) for i, m in enumerate(mm)],
                    timing_only=True)
        tp[P * mult] = sec
        print("sec/step", P * mult, sec, file=sys.stderr)
    t0 = time.time()
    net, wts = train([worlds[(m["seed"], m["fam"])] for m in meta], rows, [m["cyc"] for m in meta],
                     [m["seed"] * 10 + m["cyc"] for m in meta],
                     [zlib.crc32(repr((m["seed"], m["fam"], m["cyc"], m["role"])).encode()) for m in meta])
    train_sec = time.time() - t0
    print("trained", train_sec, file=sys.stderr)
    p = net.snapshot()
    res = []
    for s in seeds:
        for f in fams:
            w = worlds[(s, f)]
            Xp, Mp, isG1 = probes(w); Xt = train_inputs(w)
            for cyc in (1, 0):
                ix = {m["role"]: i for i, m in enumerate(meta) if m["seed"] == s and m["fam"] == f and m["cyc"] == cyc}
                sl = lambda i: [t[i:i + 1] for t in p]
                with torch.no_grad():
                    Hp = {r: net.hidden(Xp[None], sl(i))[0] for r, i in ix.items()}
                    Ht = {r: net.hidden(Xt[None], sl(i))[0] for r, i in ix.items()}
                    V = {r: net.heads(Hp[r][None], sl(i))[1][0].cpu().numpy() for r, i in ix.items()}
                    PB = {r: net.heads(Hp[r][None], sl(i))[0][0].cpu().numpy() for r, i in ix.items()}
                scale = np.mean([np.abs(V[r]).mean() for r in V]) + 1e-9
                d_between = np.abs(V["A"] - V["B"]).mean() / scale
                d_retest = np.abs(V["A"] - V["A2"]).mean() / scale
                # direction of the class difference by cluster (reported, not scored)
                gap_g1 = float((V["A"] - V["B"])[isG1 == 1].mean()); gap_g2 = float((V["A"] - V["B"])[isG1 == 0].mean())

                def patch(src, dst, basis_from):
                    Dtr = (Ht[basis_from[0]] - Ht[basis_from[1]]).cpu().numpy()
                    U = np.linalg.svd(Dtr - Dtr.mean(0), full_matrices=False)[2][:4].T
                    return U

                def close_gap(U, src="A", dst="B"):
                    Ut = torch.tensor(U, dtype=torch.float32, device=DEV)
                    Hpat = Hp[dst] + (Hp[src] - Hp[dst]) @ Ut @ Ut.T
                    with torch.no_grad():
                        vpat = net.heads(Hpat[None], sl(ix[dst]))[1][0].cpu().numpy()
                    return 1 - np.abs(V[src] - vpat).mean() / (np.abs(V[src] - V[dst]).mean() + 1e-12)

                U_ab = patch("A", "B", ("A", "B"))
                U_aa = patch("A", "B", ("A", "A2"))                 # same-class difference subspace
                Dtr = (Ht["A"] - Ht["B"]).cpu().numpy(); var_ab = np.linalg.svd(Dtr - Dtr.mean(0), compute_uv=False)
                Hall = Ht["B"].cpu().numpy(); _, sv, Vt = np.linalg.svd(Hall - Hall.mean(0), full_matrices=False)
                rr = np.random.default_rng([s, 31])
                rand_close = []
                for _ in range(5):                                  # random 4-dim subspace inside the top-16 PCs of B (variance-matched-ish)
                    Q = np.linalg.qr(Vt[:16].T @ rr.normal(size=(16, 4)))[0]
                    rand_close.append(close_gap(Q))
                # next learning (philosophy section 6): the SAME new experiences (pairs of unseen G1 entities, 3 contexts) are
                # evaluated by EACH personality with ITS OWN current state (w, expectation, attribution); that evaluation is the
                # target of 20 SGD steps; we compare the resulting update vectors of the value on the probes.
                fs_e = w["un"][0][:10]
                pairs = [(fs_e[i], fs_e[(i + 1) % 10], c) for i in range(10) for c in range(3)]
                e1 = np.array([a for a, _, _ in pairs]); e2 = np.array([b_ for _, b_, _ in pairs]); cc = np.array([c for _, _, c in pairs])
                act = np.where(w["prop"][e1] >= w["prop"][e2], e1, e2)
                dbn = torch.tensor(w["M"][act, cc], device=DEV)
                Xn = torch.tensor(np.stack([obs(w, e1, cc), obs(w, e2, cc)], 1), device=DEV)       # (30, 2, D)
                dn = torch.clamp(dbn.abs() - TOL, min=0) ** 2
                upd = {}
                for r in ("A", "B", "A2"):
                    q = [t[ix[r]:ix[r] + 1].clone().requires_grad_(True) for t in p]
                    with torch.no_grad():
                        hh = net.hidden(Xn.reshape(1, 60, D_OBS), q); pbn, vn = net.heads(hh, q)
                        pbn = pbn.reshape(30, 2, K); vn = vn.reshape(30, 2)
                        wu = wts[ix[r]] if cyc else torch.ones(K, device=DEV)
                        inn = -(wu[None] * dn).sum(1)
                        fn = inn + cyc * 0.5 * (inn - vn.sum(1))
                        rn = torch.softmax(-((dbn[:, None, :] - pbn) ** 2).sum(-1) / 0.2, 1) if cyc else torch.full((30, 2), 0.5, device=DEV)
                        tgt = (rn * fn[:, None]).reshape(60)
                    o = torch.optim.SGD(q, lr=0.05)
                    for _ in range(20):
                        _, vv = net.heads(net.hidden(Xn.reshape(1, 60, D_OBS), q), q)
                        loss = ((vv[0] - tgt) ** 2).mean(); o.zero_grad(); loss.backward(); o.step()
                    with torch.no_grad():
                        vnew = net.heads(net.hidden(Xp[None], q), q)[1][0].cpu().numpy()
                    upd[r] = vnew - V[r]
                cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                res.append(dict(seed=s, fam=f, cyc=cyc, d_between=float(d_between), d_retest=float(d_retest),
                                gap_g1=gap_g1, gap_g2=gap_g2,
                                close_ab=float(close_gap(U_ab)), close_aa=float(close_gap(U_aa)),
                                close_rand=float(np.mean(rand_close)),
                                upd_cos_AB=cos(upd["A"], upd["B"]), upd_cos_AA=cos(upd["A"], upd["A2"]),
                                own_state_r2_A=r2(PB["A"], Mp), own_state_r2_B=r2(PB["B"], Mp),
                                w_A=wts[ix["A"]].cpu().numpy().round(3).tolist(), w_B=wts[ix["B"]].cpu().numpy().round(3).tolist()))
    out = dict(throughput_sec_per_step=tp, train_sec=train_sec, P=P, steps=N_STEPS, rows=res)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    keys = ["d_between", "d_retest", "close_ab", "close_aa", "close_rand", "upd_cos_AB", "upd_cos_AA", "own_state_r2_A", "gap_g1", "gap_g2"]
    for f in fams:
        for cyc in (1, 0):
            sel = [r for r in res if r["fam"] == f and r["cyc"] == cyc]
            print(json.dumps({"fam": f, "cyc": cyc, **{k: [round(float(np.mean([r[k] for r in sel])), 3),
                                                            round(float(np.min([r[k] for r in sel])), 3),
                                                            round(float(np.max([r[k] for r in sel])), 3)] for k in keys}}))
    print(json.dumps({"throughput_sec_per_step": tp, "train_sec": train_sec, "P": P}))


if __name__ == "__main__":
    main()
