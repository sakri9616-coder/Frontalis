"""P48: dev dry run of rung 6 (does interpretation change later learning; LADDER section 10, option B). Dev seeds 77380-77389.
The v9 world and architecture (sim7, amb 1.0, noise 0.1, v8 architecture, roles of p39 with 4 sets each).
All individuals live the same life up to step T1 = 4000 (after the early phase, n_early = 3200) under C-S+5 (kappa +5). There
the population is copied and continued to the end (step 8000) in two branches that differ ONLY in kappa:
  own      C-S+5: attribution = softmax(ll + 5 sign(f) v / fscale)   (the individual's own evaluation enters interpretation)
  neutral  C-S0 : attribution = softmax(ll)                           (no evaluation in interpretation)
Same experience streams, same SGD streams in both branches (the simulator seeds them by stream and chunk).
At the end, H, S, E of the evaluation on P_un (p41 imprint) in each branch. Amplification g_H = H_own - H_neu, path control
g_S = S_own - S_neu, sample control g_E = E_own - E_neu.
Writes p48_rung6_dev.json. Not a judgement."""
import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim7, sim4, measure4 as m4
import p37_imprint as P
import p39_ladder_dev as L
import p41_ladder2_dev as Q
import p46_amb_dev as A
from cells import pers_list

sim7.NNPop = L.AdaPop
AMB, T1 = 1.0, 4000


def run_branches(seeds):
    A.SEEDS = list(seeds); L.SEEDS = list(seeds); A.AMB["a"] = AMB
    base = dict(P.BASE, noise=0.1, amb=AMB)
    worlds = {(s, "F1"): sim7.make_world(s, "F1") for s in seeds}
    L.ADAPTER["on"] = False
    pre_p = pers_list(seeds, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim7.run(pre_p, worlds, dict(learner="set", **base))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    L.ADAPTER["on"] = True
    pers_own = pers_list(seeds, ("F1",), L.roles(4), "", ["C-S+5"])
    pers_neu = pers_list(seeds, ("F1",), L.roles(4), "", ["C-S0"])
    idx = torch.tensor([wi[q["world"][0]] for q in pers_own], device=sim4.DEV)
    cfg = dict(learner="nn0", partner_scale=0.0, **base)
    st = sim7.run(pers_own, worlds, dict(cfg, stop_at=T1), init_params={k: v[idx].clone() for k, v in shared.items()})
    assert st["t"] == T1, st["t"]
    st_neu = copy.deepcopy(st)
    st_own = sim7.run(pers_own, worlds, cfg, state=st)
    st_neu = sim7.run(pers_neu, worlds, cfg, state=st_neu)
    rows = []
    vo, vn = m4.views(st_own, pers_own, worlds), m4.views(st_neu, pers_neu, worlds)
    for (s, f, cond), vw in vo.items():
        wn = vn[(s, f, "C-S0")]
        io, ineu = Q.imprint(vw, "un"), Q.imprint(wn, "un")
        r = dict(seed=s, H_own=io["H_un"], S_own=io["S_un"], E_own=io["E_un"], H_neu=ineu["H_un"], S_neu=ineu["S_un"], E_neu=ineu["E_un"],
                 r2_own=A.r2_mixed(vw["A#0"], worlds[(s, f)]), r2_neu=A.r2_mixed(wn["A#0"], worlds[(s, f)]))
        r.update(g_H=r["H_own"] - r["H_neu"], g_S=r["S_own"] - r["S_neu"], g_E=r["E_own"] - r["E_neu"]); rows.append(r)
    nf = dict(own=st_own["nonfinite"], neu=st_neu["nonfinite"])
    return sorted(rows, key=lambda r: r["seed"]), nf


def main(seeds=None):
    seeds = seeds or list(range(77380, 77390)); t0 = time.time()
    rows, nf = run_branches(seeds)
    med = lambda k: float(np.median([r[k] for r in rows]))
    summ = {k: med(k) for k in rows[0] if k != "seed"}
    summ.update(n_gH_pos=sum(r["g_H"] > 0 for r in rows), n_gH_gt_gS=sum(r["g_H"] > r["g_S"] for r in rows),
                n_gH_gt_gE=sum(r["g_H"] > r["g_E"] for r in rows), floor_25pct_Hneu=0.25 * med("H_neu"))
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    commit = __import__("subprocess").run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    out = dict(summary=summ, rows=rows, nonfinite=nf, env=env, commit=commit, secs=time.time() - t0, T1=T1, amb=AMB)
    (HERE / "p48_rung6_dev.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=float)); print(env, round(out["secs"]), "s")


if __name__ == "__main__":
    main()
