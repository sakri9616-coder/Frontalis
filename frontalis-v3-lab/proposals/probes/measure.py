"""Measurements for the dev-seed probes (draft-2 PREREG quantities, computed on dev seeds only, plus diagnostics)."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from sim import DEV, K, D_OBS, N_CTX, TOL, probe_inputs


def _r2(pred, y):
    return float(1 - ((pred - y) ** 2).mean() / ((y - y.mean(0)) ** 2).mean())


def interaction_share(dv: np.ndarray, n_e: int) -> float:
    """Share of the variance of dv (entity x context grid, flattened row-major) not explained by additive row+column effects."""
    G = dv.reshape(n_e, N_CTX)
    tot = ((G - G.mean()) ** 2).sum()
    res = G - G.mean(1, keepdims=True) - G.mean(0, keepdims=True) + G.mean()
    return float((res ** 2).sum() / max(tot, 1e-12))


def next_learning(pop, i: int, w, cyc: int, wuse: torch.Tensor, fscale: float, ref: float, Xp: torch.Tensor,
                  norm: bool, clip: float, lr=0.05, steps=20, mask_id=0, mc=0) -> np.ndarray:
    """Same new experiences (pairs of unseen G1 entities x 3 contexts), evaluated by THIS personality with its own state,
    learned for `steps` SGD steps. Returns the change of its value on the probe inputs."""
    q = {k: v.clone().requires_grad_(True) for k, v in pop.slice(i).items()}
    fs = w["un0"][:10]
    pairs = [(fs[a], fs[(a + 1) % 10], c) for a in range(10) for c in range(3)]
    e1 = np.array([a for a, _, _ in pairs]); e2 = np.array([b for _, b, _ in pairs]); cc = np.array([c for _, _, c in pairs])
    act = np.where(w["prop"][e1] >= w["prop"][e2], e1, e2)
    dbn = torch.tensor(w["M"][act, cc], device=DEV)
    mk = lambda e: np.concatenate([w["Xe"][e], w["Xc"][cc], np.zeros((len(e), 8), np.float32)], 1)
    Xn_np = np.stack([mk(e1), mk(e2)], 1)
    if mask_id:
        Xn_np[..., 32:48] = 0.0
    Xn = torch.tensor(Xn_np, device=DEV).reshape(1, 60, D_OBS)
    with torch.no_grad():
        v0 = pop.value(Xp[None], q)[0]
        pb = pop.belief(Xn, q).reshape(30, 2, K); vvec = pop.value_vec(Xn, q).reshape(30, 2, -1)
        d = torch.clamp(dbn.abs() - TOL, min=0) ** 2
        inn_k = -(wuse[None] * d)
        f_k = inn_k + cyc * ref * (inn_k - vvec.sum(1)) if mc else None
        inn = inn_k.sum(1)
        f = inn + cyc * ref * (inn - vvec.sum((1, 2)))
        if norm:
            f = f / max(fscale, 1e-3)
            f_k = f_k / max(fscale, 1e-3) if mc else None
        r = torch.softmax(-((dbn[:, None, :] - pb) ** 2).sum(-1) / 0.2, 1) if cyc else torch.full((30, 2), 0.5, device=DEV)
        tgt = (r[:, :, None] * f_k[:, None, :]).reshape(60, -1) if mc else (r * f[:, None]).reshape(60, 1)
    o = torch.optim.SGD(list(q.values()), lr=lr)
    for _ in range(steps):
        loss = ((pop.value_vec(Xn, q)[0] - tgt) ** 2).sum(-1).mean()
        o.zero_grad(); loss.backward()
        if clip > 0:
            gn = torch.sqrt(sum((g.grad ** 2).sum() for g in q.values() if g.grad is not None))
            for g in q.values():
                if g.grad is not None:
                    g.grad.mul_(min(1.0, clip / (float(gn) + 1e-12)))
        o.step()
    with torch.no_grad():
        return (pop.value(Xp[None], q)[0] - v0).cpu().numpy()


def das_subspace(pop, i_src, i_dst, Hs, Hd, k, steps=300, seed=0):
    """Optimise a k-dim orthonormal subspace U (on TRAINING inputs only) so that moving dst's representation toward src's
    inside U makes dst's value match src's value (interchange-intervention style; Geiger et al., abstract-level hint)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    Z = torch.randn(Hs.shape[1], k, generator=g).to(DEV).requires_grad_(True)
    qs, qd = pop.slice(i_src), pop.slice(i_dst)
    with torch.no_grad():
        vs = pop.value(None, qs, h=Hs[None])[0]
    o = torch.optim.Adam([Z], lr=0.05)
    for _ in range(steps):
        U, _ = torch.linalg.qr(Z)
        vp = pop.value(None, qd, h=(Hd + (Hs - Hd) @ U @ U.T)[None])[0]
        loss = (vp - vs).abs().mean()
        o.zero_grad(); loss.backward(); o.step()
    U, _ = torch.linalg.qr(Z.detach())
    return U.cpu().numpy()


def evaluate(res: Dict[str, Any], pers: List[Dict[str, Any]], worlds: Dict, k_sub=4, das=False) -> List[Dict[str, Any]]:
    pop = res["pop"]; c = res["cfg"]
    out = []
    groups = sorted({q["world"] for q in pers})
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)) if np.isfinite(a).all() and np.isfinite(b).all() else float("nan")
    for wk in groups:
        w = worlds[wk]
        ix = {q["role"]: i for i, q in enumerate(pers) if q["world"] == wk}
        Xp, Mp, ee, cc = probe_inputs(w, "un", c["mask_id"]); Xt, _, _, _ = probe_inputs(w, "tr", c["mask_id"])
        n_e = len(np.unique(ee))
        V, Hp, Ht, PB = {}, {}, {}, {}
        with torch.no_grad():
            for r, i in ix.items():
                q = pop.slice(i)
                Hp[r] = pop.vrep(Xp[None], q)[0]; Ht[r] = pop.vrep(Xt[None], q)[0]
                V[r] = pop.value(None, q, h=Hp[r][None])[0].cpu().numpy()
                PB[r] = pop.belief(Xp[None], q)[0].cpu().numpy()
        scale = np.mean([np.abs(V[r]).mean() for r in V]) + 1e-12
        D = lambda a, b: float(np.abs(V[a] - V[b]).mean() / scale)
        row = dict(world=wk, D_AB=D("A", "B"), D_exact=D("A", "A_exact"), D_env=D("A", "A_env"), D_sgd=D("A", "A_sgd"),
                   D_both=D("A", "A_both"), own_r2=_r2(PB["A"], Mp))
        # head swap: how much of the A-B value gap is in the representation vs the readout
        with torch.no_grad():
            qa = pop.slice(ix["A"])
            vAonB = pop.value(None, qa, h=Hp["B"][None])[0].cpu().numpy()
        gap = np.abs(V["A"] - V["B"]).mean() + 1e-12
        row["gap_from_repr"] = float(np.abs(V["A"] - vAonB).mean() / gap)       # A's head on B's repr vs A: representation part
        # representation difference geometry (train inputs)
        Dtr = (Ht["A"] - Ht["B"]).cpu().numpy(); Dtr = Dtr - Dtr.mean(0)
        sv = np.linalg.svd(Dtr, compute_uv=False); ev = sv ** 2
        row["diff_pr"] = float(ev.sum() ** 2 / (ev ** 2).sum()); row["diff_top4"] = float(ev[:4].sum() / ev.sum())
        Hb = Ht["B"].cpu().numpy(); evb = np.linalg.svd(Hb - Hb.mean(0), compute_uv=False) ** 2
        row["repr_pr_B"] = float(evb.sum() ** 2 / (evb ** 2).sum())
        # value-relevant share of the representation difference: projection of (H_A - H_B) on A's readout direction
        wv = pop.p["Wv"][ix["A"]].sum(-1).detach().cpu().numpy(); wv = wv / (np.linalg.norm(wv) + 1e-12)
        row["diff_on_readout"] = float(((Dtr @ wv) ** 2).sum() / (ev.sum() + 1e-12))

        def close(U, src="A", dst="B"):
            Ut = torch.tensor(U, dtype=torch.float32, device=DEV)
            Hpat = Hp[dst] + (Hp[src] - Hp[dst]) @ Ut @ Ut.T
            with torch.no_grad():
                vp = pop.value(None, pop.slice(ix[dst]), h=Hpat[None])[0].cpu().numpy()
            return float(1 - np.abs(V[src] - vp).mean() / (np.abs(V[src] - V[dst]).mean() + 1e-12))

        def sub(a, b, k):
            M_ = (Ht[a] - Ht[b]).cpu().numpy(); M_ = M_ - M_.mean(0)
            return np.linalg.svd(M_, full_matrices=False)[2][:k].T

        for k in sorted({k_sub, min(16, Ht['A'].shape[1] - 1)}):
            row[f"close_AB_k{k}"] = close(sub("A", "B", k))
            row[f"close_Aboth_k{k}"] = close(sub("A", "A_both", k))
            rr = np.random.default_rng([wk[0], 31, k]); _, _, Vt = np.linalg.svd(Hb - Hb.mean(0), full_matrices=False)
            m_ = min(max(16, 2 * k), Vt.shape[0])
            row[f"close_rand_k{k}"] = float(np.mean([close(np.linalg.qr(Vt[:m_].T @ rr.normal(size=(m_, min(k, m_))))[0]) for _ in range(5)]))
        if das:
            U_ab = das_subspace(pop, ix["A"], ix["B"], Ht["A"], Ht["B"], k_sub, seed=wk[0])
            U_aa = das_subspace(pop, ix["A"], ix["A_both"], Ht["A"], Ht["A_both"], k_sub, seed=wk[0] + 1)
            row["das_close_AB"] = close(U_ab); row["das_close_Aboth"] = close(U_aa)
            # same optimisation for a RANDOM pairing target: B's representation pushed toward a shuffled copy of A's
            # (destroys input correspondence) -> control for what the optimiser achieves without a real A-B relation
            perm = torch.randperm(Ht["A"].shape[0], generator=torch.Generator(device="cpu").manual_seed(int(wk[0]))).to(DEV)
            U_sh = das_subspace(pop, ix["A"], ix["B"], Ht["A"][perm], Ht["B"], k_sub, seed=wk[0] + 2)
            row["das_close_shuf"] = close(U_sh)
            row["das_train_close_AB"] = float(1 - (np.abs(pop.value(None, pop.slice(ix["A"]), h=Ht["A"][None])[0].detach().cpu().numpy()
                                                  - pop.value(None, pop.slice(ix["B"]), h=(Ht["B"] + (Ht["A"] - Ht["B"]) @ torch.tensor(U_ab, device=DEV) @ torch.tensor(U_ab, device=DEV).T)[None])[0].detach().cpu().numpy()).mean()
                                              / (np.abs(pop.value(None, pop.slice(ix["A"]), h=Ht["A"][None])[0].detach().cpu().numpy() - pop.value(None, pop.slice(ix["B"]), h=Ht["B"][None])[0].detach().cpu().numpy()).mean() + 1e-12)))
        row["inter_AB"] = interaction_share(V["A"] - V["B"], n_e); row["inter_Aboth"] = interaction_share(V["A"] - V["A_both"], n_e)
        # next learning
        upd = {}
        for r in ("A", "B", "A_both"):
            i = ix[r]
            upd[r] = next_learning(pop, i, w, c["cycle"], res["w_use"][i], float(res["fscale"][i]), c["ref"], Xp,
                                   bool(c["norm_targets"]), clip=1.0 if c["norm_targets"] else 0.0, mask_id=c["mask_id"], mc=c["mc"])
        row["nl_cos_AB"] = cos(upd["A"], upd["B"]); row["nl_cos_Aboth"] = cos(upd["A"], upd["A_both"])
        row["nl_finite"] = int(all(np.isfinite(u).all() for u in upd.values()))
        row["w_A"] = res["w_use"][ix["A"]].cpu().numpy().round(3).tolist(); row["w_B"] = res["w_use"][ix["B"]].cpu().numpy().round(3).tolist()
        out.append(row)
    return out


def summarize(rows: List[Dict[str, Any]], keys: List[str]):
    fams = sorted({r["world"][1] for r in rows})
    s = {}
    for f in fams:
        sel = [r for r in rows if r["world"][1] == f]
        s[f] = {k: round(float(np.nanmean([r.get(k, np.nan) for r in sel])), 3) for k in keys}
        s[f]["n_AB_gt_both"] = int(sum(r["D_AB"] > r["D_both"] for r in sel))
        s[f]["n_H2a"] = int(sum(r["close_AB_k4"] > r["close_rand_k4"] for r in sel))
        s[f]["n_H2b"] = int(sum(r["close_AB_k4"] > r["close_Aboth_k4"] for r in sel))
        s[f]["n_H3a"] = int(sum((1 - r["nl_cos_AB"]) > (1 - r["nl_cos_Aboth"]) for r in sel))
        s[f]["n_H4"] = int(sum(r["inter_AB"] > r["inter_Aboth"] for r in sel))
        s[f]["n"] = len(sel)
    return s
