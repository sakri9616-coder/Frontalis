"""Batched simulator for the draft-2 world (dev seeds only). Shared by probes P3 and later.

Everything for all personalities runs as batched tensor ops on the GPU (no per-personality Python loop).
Randomness is split into independent, per-personality streams so that test-retest twins can differ in exactly one source:
  env  : outcome noise and distractor input noise (the world's randomness)
  sgd  : minibatch indices (the learner's randomness)
  init : initial weights
A personality is a dict(world=(seed, fam), early=0|1, env=int, sgd=int, init=int, cfg=dict).
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.use_deterministic_algorithms(True)

K, DT, DC = 3, 4, 3
N_TR, N_UN, N_CTX = 80, 40, 12          # per cluster
D_E, D_C, D_DIS = 48, 16, 8
D_OBS = D_E + D_C + D_DIS
TOL, DECAY = 0.4, 0.7


def make_world(seed: int, fam: str) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng([seed, 11])
    n_ent = 2 * (N_TR + N_UN)
    cl = np.repeat([0, 1], N_TR + N_UN)
    mu = rng.normal(size=DT); mu = mu / np.linalg.norm(mu)
    T = rng.normal(size=(n_ent, DT)) * 0.8 + np.where(cl[:, None] == 0, mu, -mu)
    C = rng.normal(size=(N_CTX, DC))
    Pt = rng.normal(size=(DT, 32)) / np.sqrt(DT); Pc = rng.normal(size=(DC, 16)) / np.sqrt(DC)
    Xe = np.concatenate([np.tanh(T @ Pt) + 0.3 * rng.normal(size=(n_ent, 32)), rng.normal(size=(n_ent, 16))], 1)
    Xc = np.tanh(C @ Pc) + 0.2 * rng.normal(size=(N_CTX, 16))
    B = rng.normal(size=(K, DT, DC)) / np.sqrt(DT * DC) * 2.0
    A = rng.normal(size=(K, DT)) * 0.5
    inter = np.einsum("ed,kdc,xc->exk", T, B, C)
    add = np.einsum("ed,kd->ek", T, A)[:, None, :] + 0.5 * rng.normal(size=(N_CTX, K))[None]
    M0 = 0.7 * np.tanh(inter) + 0.2 * add
    if fam == "F1":
        M = M0
    elif fam == "F6":
        M = 0.5 * np.tanh(add)
    else:
        M = rng.permutation(M0.reshape(-1, K)).reshape(M0.shape)
    ids = np.arange(n_ent)
    return dict(Xe=Xe.astype(np.float32), Xc=Xc.astype(np.float32), M=M.astype(np.float32),
                prop=rng.normal(size=n_ent).astype(np.float32),
                tr0=ids[cl == 0][:N_TR], tr1=ids[cl == 1][:N_TR], un0=ids[cl == 0][N_TR:], un1=ids[cl == 1][N_TR:])


def _pairs(rng, pool, n):
    i1 = rng.integers(0, len(pool), n)
    i2 = (i1 + 1 + rng.integers(0, len(pool) - 1, n)) % len(pool)
    return pool[i1], pool[i2]


def make_sequence(w, seed, early, n_steps, n_early, history="HC-E"):
    """Experience order is part of the HISTORY (not noise): early phase from cluster `early`, shared phase identical
    for both classes (it depends only on the world seed)."""
    rs = np.random.default_rng([seed, 21]); re = np.random.default_rng([seed, 22 + early])
    if history == "HC-X":      # early phase: all training entities, but only contexts 0-5 (class A) or 6-11 (class B)
        pool_e = np.concatenate([w["tr0"], w["tr1"]])
        a1, a2 = _pairs(re, pool_e, n_early); ac = re.integers(0, N_CTX // 2, n_early) + (N_CTX // 2) * early; au = re.random(n_early)
    else:
        pool_e = w["tr0"] if early == 0 else w["tr1"]
        a1, a2 = _pairs(re, pool_e, n_early); ac = re.integers(0, N_CTX, n_early); au = re.random(n_early)
    pool = np.concatenate([w["tr0"], w["tr1"]])
    m = n_steps - n_early
    b1, b2 = _pairs(rs, pool, m); bc = rs.integers(0, N_CTX, m); bu = rs.random(m)
    return np.concatenate([a1, b1]), np.concatenate([a2, b2]), np.concatenate([ac, bc]), np.concatenate([au, bu])


class Pop:
    """P personalities. Encoders: shared (one) or separate (belief encoder, value encoder)."""

    def __init__(self, P, init_seeds, hidden=64, separate=False, vdim=1, ebot=0):
        self.separate = separate; self.vdim = vdim; self.ebot = ebot
        names = ["W1", "b1", "W2", "b2"]
        shapes = {"W1": (D_OBS, hidden), "b1": (1, hidden), "W2": (hidden, hidden), "b2": (1, hidden)}
        self.p: Dict[str, torch.Tensor] = {}
        encs = ["enc", "venc"] if separate else ["enc"]
        g = torch.Generator(device="cpu")
        stacks: Dict[str, List[torch.Tensor]] = {}
        for s in init_seeds:
            g.manual_seed(int(s))
            for e in encs:
                for n in names:
                    fan = shapes[n][0]
                    t = torch.randn(*shapes[n], generator=g) * np.sqrt(2 / fan) if n.startswith("W") else torch.zeros(*shapes[n])
                    stacks.setdefault(f"{e}.{n}", []).append(t)
            vin = ebot if ebot else hidden
            extra = ((("Wu", (hidden, ebot)), ("bu", (1, ebot))) if ebot else ())
            for n, shp in (("Wb", (hidden, K)), ("bb", (1, K)), ("Wv", (vin, vdim)), ("bv", (1, vdim))) + extra:
                t = torch.randn(*shp, generator=g) * np.sqrt(1 / hidden) if n.startswith("W") else torch.zeros(*shp)
                stacks.setdefault(n, []).append(t)
        self.p = {k: torch.stack(v).to(DEV).requires_grad_(True) for k, v in stacks.items()}

    def enc(self, x, which="enc", p=None):
        p = p or self.p
        if which == "venc" and not self.separate:
            which = "enc"
        h = torch.relu(torch.bmm(x, p[f"{which}.W1"]) + p[f"{which}.b1"])
        return torch.relu(torch.bmm(h, p[f"{which}.W2"]) + p[f"{which}.b2"])

    def belief(self, x, p=None):
        p = p or self.p
        return torch.bmm(self.enc(x, "enc", p), p["Wb"]) + p["bb"]

    def vrep(self, x, p=None):
        """The layer the value readout reads: the evaluation layer u (ebot units) if present, else the last hidden layer."""
        p = p or self.p
        h = self.enc(x, "venc", p)
        return torch.relu(torch.bmm(h, p["Wu"]) + p["bu"]) if self.ebot else h

    def value(self, x, p=None, h=None):
        p = p or self.p
        if h is None:
            h = self.vrep(x, p)
        u = torch.bmm(h, p["Wv"]) + p["bv"]
        return u.sum(-1)                     # scalar value = sum of the evaluation channels (vdim = 1: the single head)

    def value_vec(self, x, p=None, h=None):
        p = p or self.p
        if h is None:
            h = self.vrep(x, p)
        return torch.bmm(h, p["Wv"]) + p["bv"]

    def slice(self, i):
        return {k: v[i:i + 1].detach() for k, v in self.p.items()}


DEFAULT = dict(cycle=1, w_mode="ema", w_rate=0.01, ref=0.5, attrib=1, salience=1, sal_max=2.0, norm_targets=0,
               separate=0, lr=1e-3, bs=32, spn=1, n_steps=3000, n_early=1200, clip=0.0, wd=0.0, aff=0.0, mask_id=0, mc=0, ebot=0, history="HC-E")


def run(pers: List[Dict[str, Any]], worlds: Dict, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train all personalities (same cfg) in one batch. Returns the population and per-personality final states."""
    c = {**DEFAULT, **cfg}
    P = len(pers); N = c["n_steps"]; BS = c["bs"]; SPN = c["spn"]
    wkeys = sorted({q["world"] for q in pers}); widx = {k: i for i, k in enumerate(wkeys)}
    XE = torch.tensor(np.stack([worlds[k]["Xe"] for k in wkeys]), device=DEV)
    XC = torch.tensor(np.stack([worlds[k]["Xc"] for k in wkeys]), device=DEV)
    MM = torch.tensor(np.stack([worlds[k]["M"] for k in wkeys]), device=DEV)
    PR = torch.tensor(np.stack([worlds[k]["prop"] for k in wkeys]), device=DEV)
    wid = torch.tensor([widx[q["world"]] for q in pers], device=DEV)
    seqs = [make_sequence(worlds[q["world"]], q["world"][0], q["early"], N, c["n_early"], c["history"]) for q in pers]
    E1 = torch.tensor(np.stack([s[0] for s in seqs]), device=DEV); E2 = torch.tensor(np.stack([s[1] for s in seqs]), device=DEV)
    CC = torch.tensor(np.stack([s[2] for s in seqs]), device=DEV); UU = torch.tensor(np.stack([s[3] for s in seqs]), dtype=torch.float32, device=DEV)
    envr = [np.random.default_rng([q["env"], 31]) for q in pers]
    NZ = torch.tensor(np.stack([r.normal(size=(N, K)) for r in envr]), dtype=torch.float32, device=DEV)
    DIS = torch.tensor(np.stack([r.normal(size=(N, 2, D_DIS)) for r in envr]), dtype=torch.float32, device=DEV)
    UMB = torch.tensor(np.stack([np.random.default_rng([q["sgd"], 41]).random((N * SPN, BS), dtype=np.float32) for q in pers]), device=DEV)
    MC = int(c["mc"]); VD = K if MC else 1
    pop = Pop(P, [q["init"] for q in pers], separate=bool(c["separate"]), vdim=VD, ebot=int(c["ebot"]))
    opt = torch.optim.AdamW(list(pop.p.values()), lr=c["lr"], weight_decay=c["wd"]) if c["wd"] > 0 else torch.optim.Adam(list(pop.p.values()), lr=c["lr"])
    ar = torch.arange(P, device=DEV)
    X = torch.zeros(P, N, 2, D_OBS, device=DEV); TV = torch.zeros(P, N, 2, VD, device=DEV)
    DB = torch.zeros(P, N, K, device=DEV); SAL = torch.ones(P, N, device=DEV)
    b = torch.zeros(P, K, device=DEV); wts = torch.ones(P, K, device=DEV); slow = torch.ones(P, K, device=DEV)
    cum = torch.zeros(P, K, device=DEV); run_mag = torch.full((P,), 1e-3, device=DEV); fscale = torch.ones(P, device=DEV)
    cyc = float(c["cycle"])
    nonfinite = 0
    t0 = time.time()
    for t in range(N):
        e1, e2, cc = E1[:, t], E2[:, t], CC[:, t]
        xe = torch.stack([XE[wid, e1], XE[wid, e2]], 1)                    # (P,2,48)
        xc = XC[wid, cc][:, None, :].expand(-1, 2, -1)
        x = torch.cat([xe, xc, DIS[:, t]], -1)
        if c["mask_id"]:
            x = x.clone(); x[..., 32:48] = 0.0       # feature-only learner: no access to the identity token
        p1 = torch.sigmoid(PR[wid, e1] - PR[wid, e2])
        actor = torch.where(UU[:, t] < p1, e1, e2)
        db = MM[wid, actor, cc] + 0.1 * NZ[:, t]
        with torch.no_grad():
            pb = pop.belief(x); vvec = pop.value_vec(x); v = vvec.sum(-1)
            b = DECAY * b + db
            d = torch.clamp(b.abs() - TOL, min=0) ** 2
            if cyc:
                mag = d.mean(1); run_mag = run_mag + 0.05 * (mag - run_mag)
                tgt = torch.clamp(d / run_mag[:, None].clamp(min=1e-6), 0, 3)
                act = (mag > 1e-9).float()[:, None]
                if c["w_mode"] == "ema":
                    wts = wts + c["w_rate"] * act * (tgt - wts)
                elif c["w_mode"] == "two":
                    wts = wts + 0.01 * act * (tgt - wts); slow = slow + 0.001 * act * (wts - slow)
                elif c["w_mode"] == "cum":
                    cum = cum + d
                wts = K * wts / wts.sum(1, keepdim=True)
                if c["w_mode"] == "two":
                    w_use = K * slow / slow.sum(1, keepdim=True)
                elif c["w_mode"] == "cum":
                    w_use = K * (cum + 1e-3) / (cum + 1e-3).sum(1, keepdim=True)
                else:
                    w_use = wts
            else:
                w_use = torch.ones_like(wts)
            innate_k = -(w_use * d)                                   # (P,K) per-channel felt signal
            innate = innate_k.sum(1)
            f = innate + cyc * c["ref"] * (innate - v.sum(1))
            f_k = innate_k + cyc * c["ref"] * (innate_k - vvec.sum(1))    # per-channel version (mc)
            if c["norm_targets"]:
                fscale = fscale + 0.01 * (f.abs() - fscale)
                f = f / fscale.clamp(min=1e-3)
                f_k = f_k / fscale.clamp(min=1e-3)[:, None]
            if cyc and c["attrib"]:
                logit = -((db[:, None, :] - pb) ** 2).sum(-1) / 0.2
                if c["aff"] > 0:   # expectation-consistent (affective) attribution: a good/bad outcome is attributed more to the
                    # entity this personality already values higher/lower (its own value V_j)
                    logit = logit + c["aff"] * torch.sign(f)[:, None] * v          # v is in normalised units when norm_targets=1
                r = torch.softmax(logit, 1)
            else:
                r = torch.full((P, 2), 0.5, device=DEV)
            sal = 1 + (cyc * c["salience"]) * torch.clamp(f.abs(), max=c["sal_max"])
            X[:, t] = x; TV[:, t] = (r[:, :, None] * f_k[:, None, :]) if MC else (r * f[:, None])[:, :, None]; DB[:, t] = db; SAL[:, t] = sal
        n = t + 1
        for j in range(SPN):
            u = UMB[:, t * SPN + j]
            idx = (u * n).long().clamp(max=n - 1)
            xb = X[ar[:, None], idx].reshape(P, BS * 2, D_OBS)
            tv = TV[ar[:, None], idx].reshape(P, BS, 2, VD)
            dbb = DB[ar[:, None], idx]; sb = SAL[ar[:, None], idx]
            pbb = pop.belief(xb).reshape(P, BS, 2, K); vv = pop.value_vec(xb).reshape(P, BS, 2, VD)
            mix = -torch.logsumexp(-((dbb[:, :, None, :] - pbb) ** 2).sum(-1) / 0.2, dim=2).mean(1)
            lv = (sb[:, :, None, None] * (vv - tv) ** 2).sum(-1).mean((1, 2))
            loss = (0.1 * mix + lv).sum()
            opt.zero_grad(); loss.backward()
            if c["clip"] > 0:
                for prm in [q_ for q_ in pop.p.values() if q_.grad is not None]:          # per-personality gradient-norm clipping (one norm per personality)
                    gn = prm.grad.flatten(1).norm(dim=1).clamp(min=1e-12)
                    prm.grad.mul_((c["clip"] / gn).clamp(max=1.0).view(-1, *([1] * (prm.grad.dim() - 1))))
            opt.step()
        if not torch.isfinite(loss):
            nonfinite += 1
    secs = time.time() - t0
    return dict(pop=pop, w_use=w_use.detach(), wts=wts.detach(), secs=secs, nonfinite=nonfinite, cfg=c,
                fscale=fscale.detach(), X=X, DB=DB)


def probe_inputs(w, which="un", mask_id=0):
    e = np.concatenate([w[which + "0"], w[which + "1"]]) if which == "un" else np.concatenate([w["tr0"], w["tr1"]])
    ee, cc = np.meshgrid(e, np.arange(N_CTX), indexing="ij")
    x = np.concatenate([w["Xe"][ee.ravel()], w["Xc"][cc.ravel()], np.zeros((ee.size, D_DIS), np.float32)], 1)
    if mask_id:
        x[:, 32:48] = 0.0
    return torch.tensor(x, device=DEV), w["M"][ee.ravel(), cc.ravel()], ee.ravel(), cc.ravel()
