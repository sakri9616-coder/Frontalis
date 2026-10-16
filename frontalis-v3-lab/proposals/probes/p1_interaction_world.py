"""Feasibility probe P1 (design input only; not a test of the hypothesis; seeds 77390-77394 reserved for probes).

Question: in a world whose innate signal depends on an entity x context INTERACTION that the designer does not hand to the
model, and whose inputs are raw-ish random projections (no named axes), do
  (a) the measures proposed in DESIGN.md discriminate (representation change, transfer, intervention), and
  (b) the measures fail as they should on a NULL world (no structure; memorisation only)?
Models: ridge on raw input (linear), random ReLU features + ridge (nonlinear capacity, no representation learning),
kNN on raw input (episodic memory), MLP (Adam). Data size N and steps S are varied separately.

Run: PYTHONUTF8=1 python -B p1_interaction_world.py  (CPU, ~minutes)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
OUT = Path(__file__).resolve().parent / "p1_results.json"

DT, DC, DU = 4, 3, 2          # observable entity traits, context latent, unobservable entity traits
N_ENT_TR, N_ENT_TE, N_CTX = 80, 40, 12


def make_world(seed: int, null: bool):
    rng = np.random.default_rng([seed, 1])
    n_ent = N_ENT_TR + N_ENT_TE
    T = rng.normal(size=(n_ent, DT))                  # observable (through a random nonlinear projection) traits
    U = rng.normal(size=(n_ent, DU))                  # traits observable only through experience
    C = rng.normal(size=(N_CTX, DC))
    Pt = rng.normal(size=(DT, 32)) / np.sqrt(DT)
    Pc = rng.normal(size=(DC, 16)) / np.sqrt(DC)
    ID = rng.normal(size=(n_ent, 16))                 # random identity token (no semantics)
    Xe = np.tanh(T @ Pt) + 0.3 * rng.normal(size=(n_ent, 32))
    Xe = np.concatenate([Xe, ID], axis=1)             # 48 dims per entity
    Xc = np.tanh(C @ Pc) + 0.2 * rng.normal(size=(N_CTX, 16))
    B = rng.normal(size=(DT, DC)) / np.sqrt(DT * DC) * 2.0
    D = rng.normal(size=(DU, DC)) / np.sqrt(DU * DC) * 2.0
    a = rng.normal(size=DT) * 0.3
    Z = T @ B @ C.T + U @ D @ C.T                     # hidden interaction variable (designer's variable; used only post hoc)
    R = np.tanh(Z + (T @ a)[:, None])                 # innate scalar signal, mean over noise
    if null:                                          # null world: same marginal, no structure (pure lookup)
        R = rng.permutation(R.ravel()).reshape(R.shape)
        Z = np.zeros_like(Z)
    return dict(Xe=Xe, Xc=Xc, R=R, Z=Z, U=U)


def sample(world, ents, ctxs, n, rng, noise=0.1):
    e = rng.choice(ents, size=n)
    c = rng.choice(ctxs, size=n)
    X = np.concatenate([world["Xe"][e], world["Xc"][c], rng.normal(size=(n, 8))], axis=1)   # 8 distractor dims
    y = world["R"][e, c] + noise * rng.normal(size=n)
    return X.astype(np.float32), y.astype(np.float32), e, c


def grid(world, ents, ctxs):
    e, c = np.meshgrid(ents, ctxs, indexing="ij")
    e, c = e.ravel(), c.ravel()
    X = np.concatenate([world["Xe"][e], world["Xc"][c], np.zeros((len(e), 8))], axis=1).astype(np.float32)
    return X, world["R"][e, c].astype(np.float32), world["Z"][e, c], e, c


def ridge(Xtr, ytr, lam):
    X1 = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    w = np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ ytr)
    return lambda X: np.concatenate([X, np.ones((len(X), 1))], 1) @ w


class MLP(torch.nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.l1 = torch.nn.Linear(d, h); self.l2 = torch.nn.Linear(h, h); self.out = torch.nn.Linear(h, 1)

    def hidden(self, x):
        return torch.relu(self.l2(torch.relu(self.l1(x))))

    def forward(self, x, mask=None):
        h = self.hidden(x)
        if mask is not None:
            h = h @ mask
        return self.out(h).squeeze(-1)


def train_mlp(Xtr, ytr, steps, seed, wd=1e-4, lr=1e-3, model=None, bs=128):
    torch.manual_seed(seed)
    m = model if model is not None else MLP(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    X, y = torch.tensor(Xtr), torch.tensor(ytr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (min(bs, len(X)),), generator=g, device="cpu")
        loss = ((m(X[idx]) - y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).mean() / max(((y - y.mean()) ** 2).mean(), 1e-12))


def probe_r2(H, z, rng):
    """Linear decodability of a variable from a representation, 2-fold on probe items (designer variable: post hoc only)."""
    n = len(z); idx = rng.permutation(n); a, b = idx[: n // 2], idx[n // 2:]
    f = ridge(H[a], z[a], 1.0)
    return r2(f(H[b]), z[b])


def run(seed, null):
    rng = np.random.default_rng([seed, 2])
    w = make_world(seed, null)
    tr_e, te_e = np.arange(N_ENT_TR), np.arange(N_ENT_TR, N_ENT_TR + N_ENT_TE)
    ctxs = np.arange(N_CTX)
    Xg, yg, zg, eg, cg = grid(w, te_e, ctxs)              # unseen entities x all (seen) contexts
    out = []
    for N in (1500, 6000):
        Xtr, ytr, _, _ = sample(w, tr_e, ctxs, N, rng)
        res = {"N": N, "null": null, "seed": seed}
        best = max((r2(ridge(Xtr, ytr, l)(Xg), yg), l) for l in (0.1, 1, 10, 100))   # note: tuned on test grid = optimistic for baselines
        res["linear"] = best[0]
        Wrf = np.random.default_rng([seed, 3]).normal(size=(Xtr.shape[1], 1024)) / np.sqrt(Xtr.shape[1])
        F = lambda X: np.maximum(X @ Wrf, 0)
        res["randfeat"] = max(r2(ridge(F(Xtr), ytr, l)(F(Xg)), yg) for l in (0.1, 1, 10, 100))
        d = (Xg ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None, :] - 2 * Xg @ Xtr.T
        nn = np.argsort(d, axis=1)[:, :5]
        res["knn"] = r2(ytr[nn].mean(1), yg)
        for S in (300, 3000):
            m0 = MLP(Xtr.shape[1]); torch.manual_seed(seed); m0 = MLP(Xtr.shape[1])
            init = {k: v.clone() for k, v in m0.state_dict().items()}
            with torch.no_grad():
                H0 = m0.hidden(torch.tensor(Xg)).cpu().numpy()
            m = train_mlp(Xtr, ytr, S, seed, model=m0)
            with torch.no_grad():
                H1 = m.hidden(torch.tensor(Xg)).cpu().numpy()
                pred = m(torch.tensor(Xg)).cpu().numpy()
            key = f"mlp_S{S}"
            res[key] = r2(pred, yg)
            prng = np.random.default_rng([seed, 4])
            if not null:
                res[key + "_decZ_before"] = probe_r2(H0, zg, prng)
                res[key + "_decZ_after"] = probe_r2(H1, zg, prng)
            # basis-free change: linear CKA between init and trained hidden representation on unseen probes
            def cka(A, Bm):
                A = A - A.mean(0); Bm = Bm - Bm.mean(0)
                return float(np.linalg.norm(A.T @ Bm) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(Bm.T @ Bm) + 1e-12))
            res[key + "_cka_init_vs_trained"] = cka(H0, H1)
            # intervention: remove the top-k principal directions of the representation CHANGE (H1 - H0) found on TRAIN inputs
            with torch.no_grad():
                Ht0 = MLP(Xtr.shape[1]); Ht0.load_state_dict(init)
                dH = (m.hidden(torch.tensor(Xtr)) - Ht0.hidden(torch.tensor(Xtr))).cpu().numpy()
            dH = dH - dH.mean(0)
            _, _, Vt = np.linalg.svd(dH, full_matrices=False)
            for k in (2, 8):
                V = Vt[:k].T
                P = torch.tensor(np.eye(V.shape[0]) - V @ V.T, dtype=torch.float32)
                # mean-preserving ablation: project out directions, add back their mean on train inputs
                with torch.no_grad():
                    Htr = m.hidden(torch.tensor(Xtr))
                    mu = Htr.mean(0)
                    Hg = m.hidden(torch.tensor(Xg))
                    Hab = (Hg - mu) @ P + mu
                    pab = m.out(Hab).squeeze(-1).cpu().numpy()
                    # control: ablate k random directions
                    Rq = np.linalg.qr(np.random.default_rng([seed, 5, k]).normal(size=(V.shape[0], k)))[0]
                    Pr = torch.tensor(np.eye(V.shape[0]) - Rq @ Rq.T, dtype=torch.float32)
                    prr = m.out((Hg - mu) @ Pr + mu).squeeze(-1).cpu().numpy()
                res[key + f"_ablate_change{k}"] = r2(pab, yg)
                res[key + f"_ablate_random{k}"] = r2(prr, yg)
            # next learning: few-shot on 10 novel entities (3 contexts each), measure gain on their OTHER 9 contexts.
            if S == 3000:
                fs_e = te_e[:10]
                fs_c = np.random.default_rng([seed, 6]).permutation(N_CTX)
                shot_c, eval_c = fs_c[:3], fs_c[3:]
                Xs, ys, _, _, _ = grid(w, fs_e, shot_c)
                Xe_, ye_, _, _, _ = grid(w, fs_e, eval_c)
                def fewshot(model_state, mask=None):
                    mm = MLP(Xtr.shape[1]); mm.load_state_dict(model_state)
                    with torch.no_grad():
                        before = r2(mm(torch.tensor(Xe_), mask).cpu().numpy(), ye_)
                    opt = torch.optim.SGD(mm.parameters(), lr=0.05)
                    for _ in range(20):
                        loss = ((mm(torch.tensor(Xs), mask) - torch.tensor(ys)) ** 2).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad():
                        after = r2(mm(torch.tensor(Xe_), mask).cpu().numpy(), ye_)
                    return before, after
                res["fewshot_trained"] = fewshot({k: v.clone() for k, v in m.state_dict().items()})
                res["fewshot_init"] = fewshot(init)
        out.append(res)
    return out


def main():
    t0 = time.time()
    rows = []
    for seed in range(77390, 77393):   # reduced from 5 seeds: CPU contention, see README-probes.md
        for null in (False, True):
            rows += run(seed, null)
            print(seed, null, f"{time.time()-t0:.0f}s", file=sys.stderr)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    # summary: mean over seeds
    keys = sorted({k for r in rows for k in r if k not in ("N", "null", "seed")})
    for null in (False, True):
        for N in (1500, 6000):
            sel = [r for r in rows if r["null"] == null and r["N"] == N]
            line = {k: np.round(np.mean([np.array(r[k], dtype=float) for r in sel if k in r], axis=0), 3).tolist()
                    for k in keys if any(k in r for r in sel)}
            print(json.dumps({"null": null, "N": N, **line}))


if __name__ == "__main__":
    main()
