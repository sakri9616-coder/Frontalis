"""P45: why does the evaluation move attribution so little in P44? Diagnosis only (no value is changed). Dev seeds 77380-77389,
the same training as P44 (same seeds, same code path, so the same networks).

On the 200 fog test events of each seed, for every member of role A, the attribution logit difference between the two
entities splits into
  belief part      dll  = ll[:,0] - ll[:,1]                 (from the outcome predictions and sig2)
  evaluation part  dpr  = kappa * (a[:,0] - a[:,1]),  a = sign(f) v / fscale,  v = sum over channels of the evaluation
Reported (medians over events, then over members and seeds):
  |dll|, |dpr|, share of events where |dpr| > |dll| (the evaluation could decide), fscale, |v| per entity, |v0 - v1| between the
  two entities, and the channel cancellation |sum_k V_k| / sum_k |V_k| (does summing the 5 channels erase the evaluation?).
Also the same split on the training-like stream of the last 2000 training steps is NOT available (targets only), so the test
events are the only source. Writes p45_rung5_diag.json."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim6, sim4, measure4 as m4
import p37_imprint as P
import p39_ladder_dev as L
import p44_rung5_dev as R
from p43_fog_calib import test_events
from cells import pers_list


def diag(vw, w, seed, K=4):
    e1, e2, c, act, db = test_events(w, seed, R.NOISE)
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    dbt = m4._t(db); inn = -(torch.clamp(dbt.abs() - sim4.TOL, min=0) ** 2)
    out = []
    with torch.no_grad():
        for m in range(K):
            v = vw[f"A#{m}"]
            pb = v.PB(S, Pp, C).reshape(-1, 2, sim4.KB); vv = v.V(S, Pp, C).reshape(-1, 2, sim4.KB)
            ll = -((dbt[:, None, :] - pb) ** 2 / (2 * v.state["sig2"][None, None, :])).sum(-1)
            fs = max(v.state["fscale"], 1e-3)
            f = (inn + R.REF * (inn - vv.sum(1))).sum(1) / fs
            vs = vv.sum(-1)                                    # (n, 2) summed evaluation of each entity
            a = torch.sign(f)[:, None] * vs / fs
            dll = (ll[:, 0] - ll[:, 1]).abs(); dpr = (R.KAPPA * (a[:, 0] - a[:, 1])).abs()
            canc = (vv.sum(-1).abs() / (vv.abs().sum(-1) + 1e-12)).flatten()
            out.append(dict(dll=float(dll.median()), dpr=float(dpr.median()), share_eval_decides=float((dpr > dll).float().mean()),
                            fscale=fs, v_abs=float(vs.abs().median()), v_gap=float((vs[:, 0] - vs[:, 1]).abs().median()),
                            V_channel_abs=float(vv.abs().median()), cancel=float(canc.median()), sig2=float(v.state["sig2"].mean())))
    return {k: float(np.median([o[k] for o in out])) for k in out[0]}


def main(seeds=None):
    seeds = seeds or list(range(77380, 77390)); L.SEEDS = seeds; t0 = time.time()
    base = dict(P.BASE, noise=R.NOISE)
    worlds = {(s, "F1"): sim6.make_world(s, "F1") for s in seeds}
    L.ADAPTER["on"] = False
    pre_p = pers_list(seeds, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim6.run(pre_p, worlds, dict(learner="set", **base))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    L.ADAPTER["on"] = True
    pers = pers_list(seeds, ("F1",), L.roles(4), "", [R.COND])
    idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
    st = sim6.run(pers, worlds, dict(learner="nn0", partner_scale=0.0, **base), init_params={k: v[idx].clone() for k, v in shared.items()})
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        r = dict(seed=s); r.update(diag(vw, worlds[(s, f)], s)); rows.append(r)
    rows.sort(key=lambda r: r["seed"])
    summ = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "seed"}
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    commit = __import__("subprocess").run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    out = dict(summary=summ, rows=rows, nonfinite=st["nonfinite"], env=env, commit=commit, secs=time.time() - t0)
    (HERE / "p45_rung5_diag.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=float)); print(env, round(out["secs"]), "s")


if __name__ == "__main__":
    main()
