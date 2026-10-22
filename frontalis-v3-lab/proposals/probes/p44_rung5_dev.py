"""P44: dev dry run of rung 5 (interpretation; LADDER section 8). Dev seeds 77380-77389 only, mid scale.
World: the fog world (sim6, outcome noise 0.6 = P43 rule with the learning check, user decision 2026-10-22). Condition C-S+5
(common fixed kappa +5, user decision). Architecture as v8 (p39_ladder_dev: shared frozen encoder + rank-4 adapters + linear
heads, 4 sets per individual). Roles A, B, A_sgd, A_env, B2.
  rung 1 (sanity in the fog world)  H, S, E on P_un (p41_ladder2_dev.imprint) and R2_un of A#0.
  rung 5  200 fixed ambiguous test events per seed (unseen entities, contexts 0-11, actor by propensity, outcome + fog 0.6).
          For each member: ll from its beliefs and calibrated sig2; a = sign(f) v / fscale with v the summed evaluation of each
          of the two entities and f the training form with a common innate part (b = 0, w = 1: inn = -clamp(|db|-TOL)^2), as in
          sim4 lines 532-539; prior logit = kappa a with kappa = 5.
          r_V    = softmax(ll_ref + kappa a_own): ll_ref is the mean ll over all 4 roles x 4 members (beliefs made common), so
                   differences come only from the evaluation.   r_full = softmax(ll_own + kappa a_own).
          Each role's r = mean over its 4 members. H_r = mean |r(A) - r(B)| over events (first entity), S_r (A vs A_sgd), E_r
          (A vs A_env). shift = mean |r_V(A) - softmax(ll_ref)| (how much the evaluation moves attribution at all).
Writes p44_rung5_dev.json. Not a judgement."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim6, sim4, measure4 as m4
import p37_imprint as P
import p39_ladder_dev as L
import p41_ladder2_dev as Q
from p43_fog_calib import test_events
from cells import pers_list

sim6.NNPop = L.AdaPop
NOISE, KAPPA, COND, REF = 0.6, 5.0, "C-S+5", 0.5
ROLES4 = ("A", "B", "A_sgd", "A_env")


def rung5(vw, w, seed, K=4):
    e1, e2, c, act, db = test_events(w, seed, NOISE)
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    dbt = m4._t(db)
    inn = -(torch.clamp(dbt.abs() - sim4.TOL, min=0) ** 2)                      # common innate part (b = 0, w = 1)
    LL, A_ = {}, {}
    with torch.no_grad():
        for n in ROLES4:
            for m in range(K):
                v = vw[f"{n}#{m}"]
                pb = v.PB(S, Pp, C).reshape(-1, 2, sim4.KB); vv = v.V(S, Pp, C).reshape(-1, 2, sim4.KB)
                LL[(n, m)] = -((dbt[:, None, :] - pb) ** 2 / (2 * v.state["sig2"][None, None, :])).sum(-1)
                fs = max(v.state["fscale"], 1e-3)
                f = (inn + REF * (inn - vv.sum(1))).sum(1) / fs
                A_[(n, m)] = torch.sign(f)[:, None] * vv.sum(-1) / fs
        ll_ref = torch.stack(list(LL.values())).mean(0)
        rV = {n: torch.stack([torch.softmax(ll_ref + KAPPA * A_[(n, m)], 1) for m in range(K)]).mean(0)[:, 0].cpu().numpy() for n in ROLES4}
        rF = {n: torch.stack([torch.softmax(LL[(n, m)] + KAPPA * A_[(n, m)], 1) for m in range(K)]).mean(0)[:, 0].cpu().numpy() for n in ROLES4}
        r0 = torch.softmax(ll_ref, 1)[:, 0].cpu().numpy()
    out = {}
    for tag, R in (("V", rV), ("full", rF)):
        d = lambda a, b: float(np.abs(R[a] - R[b]).mean())
        out.update({f"H_r{tag}": d("A", "B"), f"S_r{tag}": d("A", "A_sgd"), f"E_r{tag}": d("A", "A_env")})
    out["shift"] = float(np.abs(rV["A"] - r0).mean())
    out["acc_full_A"] = float(np.where(act == e1, rF["A"], 1 - rF["A"]).mean())
    return out


def main(seeds=None):
    seeds = seeds or list(range(77380, 77390)); L.SEEDS = seeds; t0 = time.time()
    base = dict(P.BASE, noise=NOISE)
    worlds = {(s, "F1"): sim6.make_world(s, "F1") for s in seeds}
    L.ADAPTER["on"] = False
    pre_p = pers_list(seeds, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim6.run(pre_p, worlds, dict(learner="set", **base))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    L.ADAPTER["on"] = True
    pers = pers_list(seeds, ("F1",), L.roles(4), "", [COND])
    idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
    st = sim6.run(pers, worlds, dict(learner="nn0", partner_scale=0.0, **base), init_params={k: v[idx].clone() for k, v in shared.items()})
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        r = dict(seed=s, r2_un_A=m4.own_r2(vw["A#0"], "un")); r.update(Q.imprint(vw, "un")); r.update(rung5(vw, worlds[(s, f)], s)); rows.append(r)
    rows.sort(key=lambda r: r["seed"])
    med = lambda k: float(np.median([r[k] for r in rows]))
    cnt = lambda a, b: sum(r[a] > r[b] for r in rows)
    summ = {k: med(k) for k in rows[0] if k != "seed"}
    summ.update(n_H_gt_S_un=cnt("H_un", "S_un"), n_H_gt_E_un=cnt("H_un", "E_un"), n_HrV_gt_SrV=cnt("H_rV", "S_rV"), n_HrV_gt_ErV=cnt("H_rV", "E_rV"),
                n_Hrf_gt_Srf=cnt("H_rfull", "S_rfull"), n_Hrf_gt_Erf=cnt("H_rfull", "E_rfull"))
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    commit = __import__("subprocess").run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    out = dict(summary=summ, rows=rows, nonfinite=st["nonfinite"], env=env, commit=commit, secs=time.time() - t0, noise=NOISE, kappa=KAPPA)
    (HERE / "p44_rung5_dev.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=float)); print(env, round(out["secs"]), "s")


if __name__ == "__main__":
    main()
