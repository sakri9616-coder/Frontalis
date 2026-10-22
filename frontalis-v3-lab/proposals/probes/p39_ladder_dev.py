"""P39: dev dry run of the whole ladder (LADDER.md), dev seeds 77380-77389 only, mid scale, F1, I-H, C-SLp, no choice scenes.

Architecture (LADDER section 1): per-world shared encoder pre-trained on a common stream (no individual history, C-off) and
frozen; each individual adds a rank-4 adapter U V^T to the 2nd layer W2 of BOTH encoders (belief b and value v; correction of
2026-10-22, see LADDER section 6) and learns adapter + linear heads; evaluation = mean of 4 (adapter, head) sets learned from
the same history with 4 SGD streams. Reference arm o3 = the same without adapter (V3).
Roles (same init): A (early 0, env 1), B (early 1, env 1), A_sgd (A with other SGD streams), A_env (A with other outcome
samples), B2 (B with other SGD streams; receiver of rung 4).
  rung 1  H, S, E on the evaluation (p37_imprint.table)
  rung 2  the same on the value-feature change dphi = phi_adapter - phi_shared (P_un, mean of 4)
  rung 3  R2_nc (beliefs, mean of 4 members of A): adapter minus o3, paired by seed
  rung 4  receiver B2 takes donor X's value adapter, its value head is refit (ridge) so its evaluation matches its own original
          one, then it learns the 30 new experiences (measure4.new_experiences; head + value adapter, 20 SGD steps, lr 0.05,
          clip 1); u_X = change of its evaluation on P_un, mean over the 4 members. swapd_AB = |uA-uB|/|uA|, swapd_AA with
          donor A_sgd. d = swapd_AB - swapd_AA.
Writes p39_ladder_dev.json. Not a judgement: the ladder's pass patterns are printed for orientation only."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim5, sim4, measure4 as m4
import p37_imprint as P
from cells import pers_list

SEEDS = list(range(77380, 77390))
RANK = 4
ROLES5 = P.QUAD + (("B2", 1, 1),)
SG5 = dict(P.SG, B2=(4, 8, 9, 0))
ADAPTER = {"on": False}


class AdaPop(sim5.NNPop):
    """NNPop with an optional per-personality low-rank adapter on W2 of both encoders (only when ADAPTER['on'])."""

    def __init__(self, kind, Pn, init_seeds, cfg, init_params=None):
        super().__init__(kind, Pn, init_seeds, cfg, init_params)
        self.ada = ADAPTER["on"] and kind == "nn0"
        if self.ada:
            H = self.H
            for e in ("b", "v"):
                U = torch.zeros(Pn, H, RANK, device=sim4.DEV)
                Vt = torch.stack([torch.randn(RANK, H, generator=torch.Generator().manual_seed(int(s) * 7 + (991 if e == "v" else 337)))
                                  for s in init_seeds]).to(sim4.DEV) / np.sqrt(H)
                for k, v in ((f"{e}.U", U), (f"{e}.Vt", Vt)):
                    self.p[k] = v.requires_grad_(True); self.trainable.append(self.p[k])

    def _mlp(self, x, e, p):
        if f"{e}.U" in p:
            h = torch.relu(torch.bmm(x, p[f"{e}.W1"]) + p[f"{e}.b1"])
            return torch.relu(torch.bmm(h, p[f"{e}.W2"] + torch.bmm(p[f"{e}.U"], p[f"{e}.Vt"])) + p[f"{e}.b2"])
        return super()._mlp(x, e, p)


sim5.NNPop = AdaPop


def roles(K=4):
    return tuple((f"{n}#{m}", early, env, SG5[n][m]) for n, early, env in ROLES5 for m in range(K))


def zero_adapter(p):
    return {k: (torch.zeros_like(v) if k.endswith(".U") else v) for k, v in p.items()}


def feat_rows(vw, w, K=4):
    """rung 2: centred value-feature change (mean of K) per role, and its H, S, E."""
    s, p, c = sim4.probe_set(w, "un")
    F = {}
    for n, _, _ in P.QUAD:
        acc = []
        for m in range(K):
            v = vw[f"{n}#{m}"]; X = v.rows(s, p, c)
            with torch.no_grad():
                acc.append((v.net.vrep(X, v.p) - v.net.vrep(X, zero_adapter(v.p)))[0].cpu().numpy())
        F[n] = np.mean(acc, 0)
    scale = np.mean([np.abs(f).mean() for f in F.values()]) * F["A"].shape[1] + 1e-12
    D = lambda a, b: m4.dist(m4.cent(F[a]), m4.cent(F[b]), scale)
    return dict(Hf=D("A", "B"), Sf=D("A", "A_sgd"), Ef=D("A", "A_env"), dphi_abs=float(np.mean([np.abs(f).mean() for f in F.values()])))


def swap_feat(rec, don, w, seed, ref=0.5, steps=20, lr=0.05, lam=1e-3):
    """rung 4 for one member: donor's value adapter, value head refit to the receiver's own evaluation, then 30 new experiences."""
    e1, e2, c, db = m4.new_experiences(w, seed)
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    dbt = m4._t(db); st = rec.state
    q = {k: v.detach().clone() for k, v in rec.p.items()}
    q["v.U"], q["v.Vt"] = don.p["v.U"].detach().clone(), don.p["v.Vt"].detach().clone()
    fit = [sim4.probe_set(w, "tr"), sim4.probe_set(w, "un"), (S, Pp, C)]
    Xf = torch.cat([rec.rows(*a) for a in fit], 1)
    with torch.no_grad():
        V0 = rec.net.value_vec(Xf, rec.p)[0]
        Phi = torch.cat([rec.net.vrep(Xf, q)[0], torch.ones_like(V0[:, :1])], 1)
        Wh = torch.linalg.solve(Phi.T @ Phi + lam * torch.eye(Phi.shape[1], device=Phi.device), Phi.T @ V0)
        q["h.Wv"], q["h.bv"] = Wh[:-1][None].clone(), Wh[-1:][None].clone()
        resid = float((Phi @ Wh - V0).abs().sum() / (V0.abs().sum() + 1e-12))
        # targets exactly as measure4.swap_update with the receiver's own state (C-SLp, I-H)
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
        return (rec.net.value_vec(Xp, q)[0] - v0).cpu().numpy().ravel(), resid


def rung4(vw, w, seed, K=4):
    U = {X: [] for X in ("A", "B", "A_sgd")}; res = []
    for m in range(K):
        for X in U:
            u, rr = swap_feat(vw[f"B2#{m}"], vw[f"{X}#{m}"], w, seed); U[X].append(u); res.append(rr)
    uA, uB, uS = (np.mean(U[X], 0) for X in ("A", "B", "A_sgd"))
    n = np.abs(uA).sum() + 1e-12
    return dict(swapd_AB=float(np.abs(uA - uB).sum() / n), swapd_AA=float(np.abs(uA - uS).sum() / n), vmatch_resid=float(np.median(res)))


def run_arm(adapter, worlds, shared, wi):
    ADAPTER["on"] = adapter
    pers = pers_list(SEEDS, ("F1",), roles(4), "", ["C-SLp"])
    idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
    st = sim5.run(pers, worlds, dict(learner="nn0", partner_scale=0.0, **P.BASE), init_params={k: v[idx].clone() for k, v in shared.items()})
    out = {}
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        E = {r: dict(V=v.get("un")["V"], r2_un=m4.own_r2(v, "un") if r.endswith("#0") else None) for r, v in vw.items()}
        row = P.table({s: E}, 4)[0]
        row["r2_nc_A"] = float(np.mean([m4.own_r2(vw[f"A#{m}"], "nc") for m in range(4)]))
        if adapter:
            row.update(feat_rows(vw, worlds[(s, f)])); row.update(rung4(vw, worlds[(s, f)], s))
        out[s] = row
    nf = st["nonfinite"]; del st; torch.cuda.empty_cache()
    return out, nf


def main():
    t0 = time.time()
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in SEEDS}
    ADAPTER["on"] = False
    pre_p = pers_list(SEEDS, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim5.run(pre_p, worlds, dict(learner="set", **P.BASE))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    ada, nf_a = run_arm(True, worlds, shared, wi)
    o3, nf_o = run_arm(False, worlds, shared, wi)
    med = lambda rows, k: float(np.median([r[k] for r in rows]))
    A, O = [ada[s] for s in SEEDS], [o3[s] for s in SEEDS]
    d3 = [ada[s]["r2_nc_A"] - o3[s]["r2_nc_A"] for s in SEEDS]
    d4 = [r["swapd_AB"] - r["swapd_AA"] for r in A]
    pattern = dict(
        rung1=dict(H_over_S=med(A, "H") / med(A, "S"), n_H_gt_S=sum(r["H"] > r["S"] for r in A), n_H_gt_E=sum(r["H"] > r["E"] for r in A), r2_un_A=med(A, "r2_un_A")),
        rung2=dict(H_over_S=med(A, "Hf") / med(A, "Sf"), n_H_gt_S=sum(r["Hf"] > r["Sf"] for r in A), n_H_gt_E=sum(r["Hf"] > r["Ef"] for r in A)),
        rung3=dict(median_diff=float(np.median(d3)), n_pos=sum(x > 0 for x in d3)),
        rung4=dict(median_d=float(np.median(d4)), n_pos=sum(x > 0 for x in d4), floor_25pct_AA=0.25 * med(A, "swapd_AA"), vmatch_resid=med(A, "vmatch_resid")),
        o3_rung1=dict(H_over_S=med(O, "H") / med(O, "S"), H=med(O, "H"), S=med(O, "S"), E=med(O, "E")))
    commit = __import__("subprocess").run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    out = dict(pattern=pattern, ada=A, o3=O, nonfinite=dict(ada=nf_a, o3=nf_o), env=env, commit=commit, secs=time.time() - t0, rank=RANK)
    (HERE / "p39_ladder_dev.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(pattern, indent=1, default=float)); print(env, round(out["secs"]), "s")


if __name__ == "__main__":
    main()
