"""Cause investigation, diagnostics on the saved NN-SET / I-H / C-SLp state at 32,000 experiences (dev seeds 77380-77389,
family F1). Frozen code is imported; the swap is re-implemented here only to decompose and ablate it.
Outputs diag5/d2_results.json."""
import json, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import sim4, measure4 as m4
from sim4 import DEV, KB, TOL
from cells import planted_views
from d1_finish import setup, SEEDS

_t = m4._t


def swap_var(receiver, donor, w, seed, ref=0.5, donor_v=True, use_w=True, use_kappa=True, steps=20, lr=0.05, parts=False):
    """measure4.swap_update for I-H, cycle on, with switches. donor_v False: the bootstrap term uses the RECEIVER's value
    (only w, kappa, fscale come from the donor). use_w False: w := 1. use_kappa False: kappa := 0."""
    e1, e2, c, db = m4.new_experiences(w, seed)
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    dbt = _t(db)
    base_d = donor.base if isinstance(donor, m4.Planted) else donor
    with torch.no_grad():
        pb_r = receiver.PB(S, Pp, C).reshape(30, 2, KB)
        v_d = (base_d if donor_v else receiver).V(S, Pp, C).reshape(30, 2, KB)
        stt = donor.state
        d = torch.clamp(dbt.abs() - TOL, min=0) ** 2
        ll = -((dbt[:, None, :] - pb_r) ** 2 / (2 * receiver.state["sig2"][None, None, :])).sum(-1)
        wv = stt["w"][None] if use_w else torch.ones(1, KB, device=DEV)
        inn = -(wv * d)
        fs = max(stt["fscale"], 1e-3)
        f_inn = (1 + ref) * inn / fs; f_boot = -ref * v_d.sum(1) / fs
        f_k = f_inn + f_boot; f = f_k.sum(1)
        a_ = torch.sign(f)[:, None] * v_d.sum(-1) / fs
        r = torch.softmax(ll + (stt["kappa"] if use_kappa else 0.0) * a_, 1)
        tgt = (r[:, :, None] * f_k[:, None, :]).reshape(60, KB)
    if parts:
        return dict(d_pos=float((d > 0).float().mean()), inn_share=float(f_inn.abs().sum() / (f_inn.abs().sum() + f_boot.abs().sum())),
                    r_max=float(r.max(1).values.mean()))
    s, p, cc = sim4.probe_set(w, "un")
    q = {k: v.clone().requires_grad_(True) for k, v in receiver.p.items()}
    X = receiver.rows(S, Pp, C); Xp = receiver.rows(s, p, cc)
    with torch.no_grad():
        v0 = receiver.net.value_vec(Xp, q)[0]
    o = torch.optim.SGD(list(q.values()), lr=lr)
    for _ in range(steps):
        loss = ((receiver.net.value_vec(X, q)[0] - tgt) ** 2).sum(-1).mean()
        o.zero_grad(); loss.backward()
        gs = [g for g in q.values() if g.grad is not None]
        gn = float(torch.sqrt(sum((g.grad ** 2).sum() for g in gs)))
        for g in gs:
            g.grad.mul_(min(1.0, 1.0 / (gn + 1e-12)))
        o.step()
    with torch.no_grad():
        return (receiver.net.value_vec(Xp, q)[0] - v0).cpu().numpy().ravel()


def j3diff(vw, w, seed, **kw):
    B2 = vw["B2"]
    uA = swap_var(B2, vw["A"], w, seed, **kw); uB = swap_var(B2, vw["B"], w, seed, **kw); uA2 = swap_var(B2, vw["A2"], w, seed, **kw)
    n = np.abs(uA).sum() + 1e-12
    return float(np.abs(uA - uB).sum() / n - np.abs(uA - uA2).sum() / n), float(np.abs(uA - uA2).sum() / n)


def body_hinge_frac(db):
    """db: (T, KB) outcome record; body state b_t = 0.7 b_{t-1} + db_t; fraction of hinge-active (channel, step)."""
    b = np.zeros(KB); act_b = 0; cum = np.zeros(KB); traj = {}
    for t in range(len(db)):
        b = sim4.DECAY * b + db[t]
        dd = np.clip(np.abs(b) - TOL, 0, None) ** 2
        act_b += (dd > 0).sum(); cum += dd
        if t + 1 in (1000, 4000, 12800, 20000, 32000):
            traj[t + 1] = (cum + 1e-3) / (cum + 1e-3).sum() * KB
    return act_b / db.size, float((np.abs(db) > TOL).mean()), traj


if __name__ == "__main__":
    scr = sys.argv[1]
    worlds, pers, cfg = setup()
    c = {**sim4.DEFAULT, **cfg}
    st = sim4.load_state(scr, pers, worlds, cfg)
    st["cfg"] = c; st["wkeys"] = sorted({q["world"] for q in pers}); st["wid"] = None
    print("hash", sim4.state_hash(st), flush=True)
    vws = m4.views(st, pers, worlds)
    idx = {(q["world"], q["role"]): i for i, q in enumerate(pers)}
    out = []
    for (s, f, cond), vw in vws.items():
        if f != "F1":
            continue
        w = worlds[(s, f)]
        pv = planted_views(vw, w, s)
        row = dict(seed=s)
        # state content
        wA, wA2, wB, wA3 = (vw[r].state["w"].cpu().numpy() for r in ("A", "A2", "B", "A3"))
        row.update(wA=wA.round(3).tolist(), wCV=float(wA.std() / wA.mean()), L1_wAB=float(np.abs(wA - wB).sum()), L1_wAA2=float(np.abs(wA - wA2).sum()),
                   L1_wA3perm=float(np.abs(wA3 - pv["PL-state"]["B"].state["w"].cpu().numpy()).sum()),
                   kappa=[vw[r].state["kappa"] for r in ("A", "A2", "B", "B2")], fscale=[vw[r].state["fscale"] for r in ("A", "A2", "B")])
        # body-state hinge in training vs db-only hinge (swap), and w trajectory, from the stored outcome record
        tr = {}
        for r in ("A", "A2", "B"):
            db = st["buf"]["DB"][idx[((s, f), r)]].cpu().numpy()
            fb, fd, traj = body_hinge_frac(db)
            tr[r] = traj; row[f"hinge_body_{r}"] = float(fb); row[f"hinge_db_{r}"] = float(fd)
        row["wtraj_L1_AB"] = {t: float(np.abs(tr["A"][t] - tr["B"][t]).sum()) for t in tr["A"]}
        row["wtraj_L1_AA2"] = {t: float(np.abs(tr["A"][t] - tr["A2"][t]).sum()) for t in tr["A"]}
        row["swap_parts_A"] = swap_var(vw["B2"], vw["A"], w, s, parts=True)
        # J3 ablations: frozen-equivalent, state-only (receiver's own V), no bootstrap (ref 0), w removed, kappa removed
        for nm, kw in {"v0": {}, "state_only": dict(donor_v=False), "ref0": dict(ref=0.0), "no_w": dict(use_w=False), "no_kappa": dict(use_kappa=False)}.items():
            row[f"J3_real_{nm}"], row[f"J3_AA_{nm}"] = j3diff(vw, w, s, **kw)
            row[f"J3_PLstate_{nm}"], _ = j3diff(pv["PL-state"], w, s, **kw)
            row[f"J3_PLstate0_{nm}"], _ = j3diff(pv["PL-state0"], w, s, **kw)
        # belief fit vs transfer (role A)
        A = vw["A"]
        row["r2_tr"] = m4.own_r2(A, "tr"); row["r2_un"] = m4.own_r2(A, "un"); row["r2_nc"] = m4.own_r2(A, "nc")
        out.append(row); print(s, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k.startswith("J3_") or k in ("wCV", "L1_wAB", "L1_wAA2", "r2_tr", "r2_nc")}, flush=True)
    summ = {}
    for k in out[0]:
        if isinstance(out[0][k], float):
            a = np.array([r[k] for r in out]); summ[k] = dict(median=float(np.median(a)), n_pos=int((a > 0).sum()), n=len(a))
    (HERE / "d2_results.json").write_text(json.dumps(dict(rows=out, summary=summ), indent=1, default=float), encoding="utf-8")
    for k, v in summ.items():
        print(k, v)
