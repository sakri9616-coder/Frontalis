"""P43: calibrate the fog (outcome noise) of the rung-5 world (LADDER section 8). Dev seeds 77380-77389 only.

The rule is written before running and looks ONLY at how certain attribution is, never at differences between individuals:
for each noise level, train the v8 architecture (shared encoder pre-trained in the same fog, C-off; one individual, role A,
one (adapter, head) set, C-S+5 = common fixed kappa 5, the rung-5 training condition), then on 200 fixed ambiguous test events per seed (unseen entities, contexts 0-11, the actor
drawn by the world's propensity, outcome + fog) compute the likelihood-only attribution r = softmax(ll) from the individual's
own beliefs and calibrated sig2. conf = max(r). Chosen noise = the grid level whose median conf is closest to 0.75 (halfway
between a coin flip and certainty). Accuracy (r on the true actor) and the learned kappa are reported.
Writes p43_fog_calib.json."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim6, sim4, measure4 as m4
import p37_imprint as P
import p39_ladder_dev as L
from cells import pers_list

sim6.NNPop = L.AdaPop
SEEDS = list(range(77380, 77390))
GRID = (0.1, 0.3, 0.6, 1.0)
TARGET = 0.75


def test_events(w, seed, noise, n=200):
    rng = np.random.default_rng([seed, 71])
    un = np.concatenate([w["un0"], w["un1"]])
    e1 = rng.choice(un, n); e2 = np.array([rng.choice(un[un != a]) for a in e1]); c = rng.integers(0, sim4.N_CTX_TR, n)
    act = np.where(rng.random(n) < 1 / (1 + np.exp(-(w["prop"][e1] - w["prop"][e2]))), e1, e2)
    db = sim4.outcome_np(w, act, np.where(act == e1, e2, e1), c) + noise * rng.normal(size=(n, sim4.KB))
    return e1, e2, c, act, db.astype(np.float32)


def attribution(v, ev):
    e1, e2, c, act, db = ev
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    with torch.no_grad():
        pb = v.PB(S, Pp, C).reshape(-1, 2, sim4.KB)
        ll = -((m4._t(db)[:, None, :] - pb) ** 2 / (2 * v.state["sig2"][None, None, :])).sum(-1)
        r = torch.softmax(ll, 1).cpu().numpy()
    return r, (act == e1)


def one_level(noise, worlds):
    base = dict(P.BASE, noise=noise)
    L.ADAPTER["on"] = False
    pre_p = pers_list(SEEDS, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim6.run(pre_p, worlds, dict(learner="set", **base))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    L.ADAPTER["on"] = True
    pers = pers_list(SEEDS, ("F1",), (("A#0", 0, 1, 3),), "", ["C-S+5"])
    idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
    st = sim6.run(pers, worlds, dict(learner="nn0", partner_scale=0.0, **base), init_params={k: v[idx].clone() for k, v in shared.items()})
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        v = vw["A#0"]; r, first = attribution(v, test_events(worlds[(s, f)], s, noise))
        conf = r.max(1); acc = np.where(first, r[:, 0], r[:, 1])
        rows.append(dict(seed=s, conf_median=float(np.median(conf)), acc_mean=float(acc.mean()), kappa=float(st["kappa"][rows.__len__()]),
                         r2_un=m4.own_r2(v, "un")))
    nf = st["nonfinite"]; del st; torch.cuda.empty_cache()
    return dict(noise=noise, conf_median=float(np.median([r["conf_median"] for r in rows])), acc_mean=float(np.mean([r["acc_mean"] for r in rows])),
                kappa_median=float(np.median([r["kappa"] for r in rows])), r2_un_median=float(np.median([r["r2_un"] for r in rows])), nonfinite=nf, rows=rows)


def main():
    t0 = time.time()
    worlds = {(s, "F1"): sim6.make_world(s, "F1") for s in SEEDS}
    levels = [one_level(nz, worlds) for nz in GRID]
    chosen = min(levels, key=lambda l: abs(l["conf_median"] - TARGET))["noise"]
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    commit = __import__("subprocess").run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    out = dict(levels=levels, chosen_noise=chosen, target=TARGET, env=env, commit=commit, secs=time.time() - t0)
    (HERE / "p43_fog_calib.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for l in levels:
        print({k: l[k] for k in ("noise", "conf_median", "acc_mean", "kappa_median", "r2_un_median", "nonfinite")})
    print("chosen noise", chosen, env, round(out["secs"]), "s")


if __name__ == "__main__":
    main()
