"""Cause-investigation experiments with sim5 switches (dev seeds 77380-77389 only, family F1, I-H).
Usage: python -B d3_exp.py <name> <learner> <scale> <roles: A|6> '<json extra cfg>' [conds comma-separated]
roles A : role A only, reports belief R2 on P_tr / P_un / P_nc.
roles 6 : roles A A2 B B2 A3 A4, full draft-4 measurement (no DAS) + J3 calibration views PL-state / PL-state0.
Appends to diag5/d3_results.json under <name>."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import sim5, sim4, measure4 as m4, judge4
from cells import ROLES, SCALE, COND, pers_list, planted_views

SEEDS = list(range(77380, 77390))
_t = m4._t


def swap_v5(receiver, donor, w, seed, gate, ref=0.5, steps=20, lr=0.05):
    """Candidate v5 swap (I-H, cycle on): only the donor's EVALUATIVE STATE (w, kappa, fscale) is transferred; the
    bootstrap term uses the receiver's own value (diag5 d2: the donor's value network carries retest noise that swamps
    the state). gate: the donor's w also gates the per-channel plasticity, as in sim5 wgate."""
    e1, e2, c, db = m4.new_experiences(w, seed)
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    dbt = _t(db); stt = donor.state
    with torch.no_grad():
        pb_r = receiver.PB(S, Pp, C).reshape(30, 2, sim4.KB); v_r = receiver.V(S, Pp, C).reshape(30, 2, sim4.KB)
        d = torch.clamp(dbt.abs() - sim4.TOL, min=0) ** 2
        ll = -((dbt[:, None, :] - pb_r) ** 2 / (2 * receiver.state["sig2"][None, None, :])).sum(-1)
        inn = -(stt["w"][None] * d); fs = max(stt["fscale"], 1e-3)
        f_k = (inn + ref * (inn - v_r.sum(1))) / fs; f = f_k.sum(1)
        r = torch.softmax(ll + stt["kappa"] * torch.sign(f)[:, None] * v_r.sum(-1) / fs, 1)
        tgt = (r[:, :, None] * f_k[:, None, :]).reshape(60, sim4.KB)
        gw = stt["w"][None] if gate else 1.0
    s, p, cc = sim4.probe_set(w, "un")
    q = {k: v.clone().requires_grad_(True) for k, v in receiver.p.items()}
    X = receiver.rows(S, Pp, C); Xp = receiver.rows(s, p, cc)
    with torch.no_grad():
        v0 = receiver.net.value_vec(Xp, q)[0]
    o = torch.optim.SGD(list(q.values()), lr=lr)
    for _ in range(steps):
        loss = (gw * (receiver.net.value_vec(X, q)[0] - tgt) ** 2).sum(-1).mean()
        o.zero_grad(); loss.backward()
        gs = [g for g in q.values() if g.grad is not None]
        gn = float(torch.sqrt(sum((g.grad ** 2).sum() for g in gs)))
        for g in gs:
            g.grad.mul_(min(1.0, 1.0 / (gn + 1e-12)))
        o.step()
    with torch.no_grad():
        return (receiver.net.value_vec(Xp, q)[0] - v0).cpu().numpy().ravel()


def j3v5(vw, w, seed, gate):
    B2 = vw["B2"]
    uA, uB, uA2 = (swap_v5(B2, vw[r], w, seed, gate) for r in ("A", "B", "A2"))
    n = np.abs(uA).sum() + 1e-12
    return float(np.abs(uA - uB).sum() / n), float(np.abs(uA - uA2).sum() / n)


def main():
    name, learner, scale, roles, extra = sys.argv[1:6]
    conds = sys.argv[6].split(",") if len(sys.argv) > 6 else ["C-off", "C-SLp"]
    extra = json.loads(extra)
    lr = json.loads((HERE.parent / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    R = (("A", 0, 1, 3),) if roles == "A" else ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in SEEDS}
    pers = pers_list(SEEDS, ("F1",), R, "", conds)
    cfg = dict(learner=learner, innate="IH", **SCALE[scale], **({"lr": lr} if learner in ("set", "nn0") else {}), **extra)
    if learner not in ("set", "rel", "pre"):   # as p15_prod.py: the P16 partner scale for every other learner
        cfg["partner_scale"] = json.loads((HERE.parent / "p16_pscale.json").read_text(encoding="utf-8")).get(learner, 1.0)
    t0 = time.time()
    st = sim5.run(pers, worlds, cfg)
    vws = m4.views(st, pers, worlds)
    rows = []
    for (s, f, cond), vw in vws.items():
        A = vw["A"]
        r = dict(seed=s, cond=cond, r2_tr=m4.own_r2(A, "tr"), r2_un=m4.own_r2(A, "un"), r2_nc=m4.own_r2(A, "nc"),
                 kappa=A.state["kappa"], wCV=float(A.state["w"].std() / A.state["w"].mean()))
        if roles != "A":
            g = m4.measure_group(vw, worlds[(s, f)], s, "IH", COND[cond]["cycle"], das=False)
            r.update({k: g[k] for k in ("D_AB_un", "D_AA_un", "Drho_AB_un", "Drho_AA_un", "Drho_AB_nc", "Drho_AA_nc", "swapd_AB", "swapd_AA",
                                        "xp_other_un", "xp_same_un", "xp_other_nc", "xp_same_nc", "repro_un")})
            r["L1_wAB"] = float((vw["A"].state["w"] - vw["B"].state["w"]).abs().sum()); r["L1_wAA2"] = float((vw["A"].state["w"] - vw["A2"].state["w"]).abs().sum())
            if A.nn:
                gate = bool(cfg.get("wgate", 0))
                r["v5_swapd_AB"], r["v5_swapd_AA"] = j3v5(vw, worlds[(s, f)], s, gate)
                pv = planted_views(vw, worlds[(s, f)], s)
                for nm in ("PL-state", "PL-state0"):
                    gp = m4.measure_group(pv[nm], worlds[(s, f)], s, "IH", COND[cond]["cycle"], das=False)
                    r[f"{nm}_swapd_AB"] = gp["swapd_AB"]; r[f"{nm}_swapd_AA"] = gp["swapd_AA"]
                    r[f"{nm}_v5_swapd_AB"], r[f"{nm}_v5_swapd_AA"] = j3v5(pv[nm], worlds[(s, f)], s, gate)
        rows.append(r)
    summ = {}
    for cond in conds:
        rs = [r for r in rows if r["cond"] == cond]
        d = {k: float(np.median([r[k] for r in rs])) for k in ("r2_tr", "r2_un", "r2_nc")}
        if roles != "A":
            c = judge4.contrasts(rs)
            d.update({j: (c[j]["n_pos"], round(c[j]["median"], 4), round(c[j]["floor"], 4), c[j]["passed"]) for j in ("J1-a", "J2-a", "J2-b", "J3", "J4-a", "J4-b")})
            if "PL-state_swapd_AB" in rs[0]:
                d["L1_wAB"] = float(np.median([r["L1_wAB"] for r in rs])); d["L1_wAA2"] = float(np.median([r["L1_wAA2"] for r in rs]))
                for nm in ("", "PL-state_", "PL-state0_"):
                    x = np.array([r[f"{nm}v5_swapd_AB"] - r[f"{nm}v5_swapd_AA"] for r in rs])
                    d[f"J3v5_{nm or 'real_'}"] = (int((x > 0).sum()), round(float(np.median(x)), 4), round(0.25 * float(np.median([r[f"{nm}v5_swapd_AA"] for r in rs])), 4))
                for nm in ("PL-state", "PL-state0"):
                    x = np.array([r[f"{nm}_swapd_AB"] - r[f"{nm}_swapd_AA"] for r in rs])
                    d[f"calib_{nm}_J3"] = (int((x > 0).sum()), round(float(np.median(x)), 4), round(0.25 * float(np.median([r[f"{nm}_swapd_AA"] for r in rs])), 4))
        summ[cond] = d
    out_p = HERE / "d3_results.json"
    allr = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    allr[name] = dict(cfg={k: v for k, v in cfg.items()}, secs=time.time() - t0, nonfinite=st["nonfinite"], summary=summ, rows=rows)
    out_p.write_text(json.dumps(allr, indent=1, default=float), encoding="utf-8")
    print(name, json.dumps(summ, default=float))


if __name__ == "__main__":
    main()
