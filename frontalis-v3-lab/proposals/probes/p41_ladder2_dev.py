"""P41: dev dry run of the revised rungs 3 and 4 (LADDER section 7), dev seeds 77380-77389 only. Same architecture and runs as
P39 (p39_ladder_dev.py, imported unchanged); only the two measurements change.
  rung 3' the imprint carries over to unseen contexts: H, S, E of the evaluation on P_nc (novel contexts 12-15, never trained),
          and the same on the value-feature change on P_nc.
  rung 4' the swap of rung 4, but the receiver's value head is refit with the 60 new-experience rows weighted x1e4, so that the
          receiver starts the 30 new experiences with (almost) the same evaluation on exactly the rows it learns from. The
          residual that matters (on those rows) is reported as resid_X; the old in-sample residual as resid_all.
Writes p41_ladder2_dev.json. Not a judgement."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim5, sim4, measure4 as m4
import p37_imprint as P
import p39_ladder_dev as L
from cells import pers_list

W_X = 1e4


def imprint(vw, which, K=4):
    """H, S, E on P_<which> for the evaluation and for the value-feature change (mean of K)."""
    s, p, c = sim4.probe_set(vw["A#0"].w, which)
    V, F = {}, {}
    for n, _, _ in P.QUAD:
        vs, fs = [], []
        for m in range(K):
            v = vw[f"{n}#{m}"]; X = v.rows(s, p, c)
            with torch.no_grad():
                vs.append(v.net.value_vec(X, v.p)[0].cpu().numpy())
                fs.append((v.net.vrep(X, v.p) - v.net.vrep(X, L.zero_adapter(v.p)))[0].cpu().numpy())
        V[n], F[n] = np.mean(vs, 0), np.mean(fs, 0)
    out = {}
    for tag, Z in (("", V), ("f", F)):
        scale = np.mean([np.abs(z).mean() for z in Z.values()]) * Z["A"].shape[1] + 1e-12
        D = lambda a, b: m4.dist(m4.cent(Z[a]), m4.cent(Z[b]), scale)
        out.update({f"H{tag}_{which}": D("A", "B"), f"S{tag}_{which}": D("A", "A_sgd"), f"E{tag}_{which}": D("A", "A_env")})
    return out


def swap_feat2(rec, don, w, seed, ref=0.5, steps=20, lr=0.05, lam=1e-3):
    e1, e2, c, db = m4.new_experiences(w, seed)
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    dbt = m4._t(db); st = rec.state
    q = {k: v.detach().clone() for k, v in rec.p.items()}
    q["v.U"], q["v.Vt"] = don.p["v.U"].detach().clone(), don.p["v.Vt"].detach().clone()
    fit = [sim4.probe_set(w, "tr"), sim4.probe_set(w, "un"), (S, Pp, C)]
    Xf = torch.cat([rec.rows(*a) for a in fit], 1)
    nX = len(S)
    with torch.no_grad():
        V0 = rec.net.value_vec(Xf, rec.p)[0]
        Phi = torch.cat([rec.net.vrep(Xf, q)[0], torch.ones_like(V0[:, :1])], 1)
        wt = torch.ones(Phi.shape[0], 1, device=Phi.device); wt[-nX:] = W_X
        Wh = torch.linalg.solve(Phi.T @ (wt * Phi) + lam * torch.eye(Phi.shape[1], device=Phi.device), Phi.T @ (wt * V0))
        q["h.Wv"], q["h.bv"] = Wh[:-1][None].clone(), Wh[-1:][None].clone()
        R = (Phi @ Wh - V0).abs()
        resid_X = float(R[-nX:].sum() / (V0[-nX:].abs().sum() + 1e-12)); resid_all = float(R.sum() / (V0.abs().sum() + 1e-12))
        pb_r = rec.PB(S, Pp, C).reshape(30, 2, sim4.KB); v_r = rec.V(S, Pp, C).reshape(30, 2, sim4.KB)
        d = torch.clamp(dbt.abs() - sim4.TOL, min=0) ** 2
        ll = -((dbt[:, None, :] - pb_r) ** 2 / (2 * st["sig2"][None, None, :])).sum(-1)
        inn = -(st["w"][None] * d)
        f_k = inn + ref * (inn - v_r.sum(1)); fs = max(st["fscale"], 1e-3); f_k = f_k / fs; f = f_k.sum(1)
        r = torch.softmax(ll + st["kappa"] * torch.sign(f)[:, None] * v_r.sum(-1) / fs, 1)
        tgt = (r[:, :, None] * f_k[:, None, :]).reshape(60, sim4.KB)
    s, p, cc = sim4.probe_set(w, "un")
    X = rec.rows(S, Pp, C); Xp = rec.rows(s, p, cc)
    train = ("h.Wv", "h.bv", "v.U", "v.Vt")
    for k in train:
        q[k].requires_grad_(True)
    with torch.no_grad():
        v0 = rec.net.value_vec(Xp, q)[0]
    o = torch.optim.SGD([q[k] for k in train], lr=lr)
    for _ in range(steps):
        loss = ((rec.net.value_vec(X, q)[0] - tgt) ** 2).sum(-1).mean()
        o.zero_grad(); loss.backward()
        gn = float(torch.sqrt(sum((q[k].grad ** 2).sum() for k in train)))
        for k in train:
            q[k].grad.mul_(min(1.0, 1.0 / (gn + 1e-12)))
        o.step()
    with torch.no_grad():
        return (rec.net.value_vec(Xp, q)[0] - v0).cpu().numpy().ravel(), resid_X, resid_all


def rung4b(vw, w, seed, K=4):
    U = {X: [] for X in ("A", "B", "A_sgd")}; rx, ra = [], []
    for m in range(K):
        for X in U:
            u, a, b = swap_feat2(vw[f"B2#{m}"], vw[f"{X}#{m}"], w, seed); U[X].append(u); rx.append(a); ra.append(b)
    uA, uB, uS = (np.mean(U[X], 0) for X in ("A", "B", "A_sgd"))
    n = np.abs(uA).sum() + 1e-12
    return dict(swapd_AB=float(np.abs(uA - uB).sum() / n), swapd_AA=float(np.abs(uA - uS).sum() / n), resid_X=float(np.median(rx)), resid_all=float(np.median(ra)))


def main(seeds=None):
    seeds = seeds or list(range(77380, 77390)); L.SEEDS = seeds; t0 = time.time()
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in seeds}
    L.ADAPTER["on"] = False
    pre_p = pers_list(seeds, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim5.run(pre_p, worlds, dict(learner="set", **P.BASE))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    L.ADAPTER["on"] = True
    pers = pers_list(seeds, ("F1",), L.roles(4), "", ["C-SLp"])
    idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
    st = sim5.run(pers, worlds, dict(learner="nn0", partner_scale=0.0, **P.BASE), init_params={k: v[idx].clone() for k, v in shared.items()})
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        r = dict(seed=s); r.update(imprint(vw, "un")); r.update(imprint(vw, "nc")); r.update(rung4b(vw, worlds[(s, f)], s)); rows.append(r)
    med = lambda k: float(np.median([r[k] for r in rows]))
    cnt = lambda a, b: sum(r[a] > r[b] for r in rows)
    summ = {k: med(k) for k in rows[0] if k != "seed"}
    summ.update(n_H_gt_S_nc=cnt("H_nc", "S_nc"), n_H_gt_E_nc=cnt("H_nc", "E_nc"), n_Hf_gt_Sf_nc=cnt("Hf_nc", "Sf_nc"), n_Hf_gt_Ef_nc=cnt("Hf_nc", "Ef_nc"),
                d4_n_pos=sum(r["swapd_AB"] > r["swapd_AA"] for r in rows), d4_median=float(np.median([r["swapd_AB"] - r["swapd_AA"] for r in rows])))
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    commit = __import__("subprocess").run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    out = dict(summary=summ, rows=rows, nonfinite=st["nonfinite"], env=env, commit=commit, secs=time.time() - t0, W_X=W_X)
    (HERE / "p41_ladder2_dev.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=float)); print(env, round(out["secs"]), "s")


if __name__ == "__main__":
    main()
