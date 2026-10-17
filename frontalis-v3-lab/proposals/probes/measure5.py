"""v5 measurements (PREREG-v5-DRAFT section 4). Imports draft-4 code; changes nothing in it.

J3v5 (state-only swap): the receiver B' learns 30 new experiences (measure4.new_experiences) under ONE donor quantity, the
donor's channel weights w (the quantity wgate uses). Everything else is the receiver's own: belief, value (bootstrap term),
sig2, kappa, fscale, and its body state b, which is carried through the 30 experiences as in training
(b <- DECAY*b + db, hinge on |b|; sim4.run lines 510-511). The donor's w enters twice, exactly as in training under wgate:
in the innate target -(w*d) and as the per-channel plasticity weight of the receiver's update.
u_X = change of the receiver's evaluation on P_un after learning with donor X's w."""
from __future__ import annotations

import numpy as np
import torch

import sim4
import measure4 as m4

_t = m4._t


def swap_v5(receiver, donor, w, seed, ref=0.5, steps=20, lr=0.05):
    e1, e2, c, db = m4.new_experiences(w, seed)
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    dbt = _t(db); wd = donor.state["w"]; rs = receiver.state
    with torch.no_grad():
        pb_r = receiver.PB(S, Pp, C).reshape(30, 2, sim4.KB); v_r = receiver.V(S, Pp, C).reshape(30, 2, sim4.KB)
        b = receiver.st["b"][receiver.i].clone(); d = torch.zeros_like(dbt)
        for j in range(30):                                   # body state carried as in training
            b = sim4.DECAY * b + dbt[j]
            d[j] = torch.clamp(b.abs() - sim4.TOL, min=0) ** 2
        ll = -((dbt[:, None, :] - pb_r) ** 2 / (2 * rs["sig2"][None, None, :])).sum(-1)
        inn = -(wd[None] * d); fs = max(rs["fscale"], 1e-3)
        f_k = (inn + ref * (inn - v_r.sum(1))) / fs; f = f_k.sum(1)
        r = torch.softmax(ll + rs["kappa"] * torch.sign(f)[:, None] * v_r.sum(-1) / fs, 1)
        tgt = (r[:, :, None] * f_k[:, None, :]).reshape(60, sim4.KB)
    s, p, cc = sim4.probe_set(w, "un")
    q = {k: v.clone().requires_grad_(True) for k, v in receiver.p.items()}
    X = receiver.rows(S, Pp, C); Xp = receiver.rows(s, p, cc)
    with torch.no_grad():
        v0 = receiver.net.value_vec(Xp, q)[0]
    o = torch.optim.SGD(list(q.values()), lr=lr)
    for _ in range(steps):
        loss = (wd[None] * (receiver.net.value_vec(X, q)[0] - tgt) ** 2).sum(-1).mean()
        o.zero_grad(); loss.backward()
        gs = [g for g in q.values() if g.grad is not None]
        gn = float(torch.sqrt(sum((g.grad ** 2).sum() for g in gs)))
        for g in gs:
            g.grad.mul_(min(1.0, 1.0 / (gn + 1e-12)))
        o.step()
    with torch.no_grad():
        return (receiver.net.value_vec(Xp, q)[0] - v0).cpu().numpy().ravel()


def j3v5(vw, w, seed):
    """-> (swapd_AB, swapd_AA) with receiver B' and donors A, B, A'."""
    B2 = vw["B2"]
    uA, uB, uA2 = (swap_v5(B2, vw[r], w, seed) for r in ("A", "B", "A2"))
    n = np.abs(uA).sum() + 1e-12
    return float(np.abs(uA - uB).sum() / n), float(np.abs(uA - uA2).sum() / n)


TPTS = (4000, 12800, 20000, 32000)


def w_course(db):
    """Channel weights w(t) (sensitisation, cycle on) reconstructed from one personality's outcome record db (T, KB) at TPTS."""
    b = np.zeros(sim4.KB); cum = np.zeros(sim4.KB); out = {}
    for t in range(len(db)):
        b = sim4.DECAY * b + db[t]
        cum += np.clip(np.abs(b) - sim4.TOL, 0, None) ** 2
        if t + 1 in TPTS:
            out[t + 1] = (cum + 1e-3) / (cum + 1e-3).sum() * sim4.KB
    return out
