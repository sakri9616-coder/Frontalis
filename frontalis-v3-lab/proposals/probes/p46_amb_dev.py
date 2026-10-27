"""P46: the "either could have done it" fog (LADDER section 9). Dev seeds 77380-77389 only, mid scale. One job, two parts.

Part 1, calibration (rule fixed before running; looks only at attribution certainty and learning, never at differences between
individuals): for amb in GRID, train the v8 architecture in sim7 (noise 0.1 as in F1; condition C-S+5), one individual (A, one
set). On 200 fixed test events per seed in AMBIGUOUS contexts (even ids 0-10; unseen entities; actor by propensity; outcome mixed
as in sim7 + noise 0.1), conf = max softmax(ll). Learning check: R2 of the beliefs on P_un against the true expected outcome
of sim7 (the mixed one). Chosen amb = among levels with learning R2 >= 0.5, the one whose median conf is closest to 0.75.
If no level passes the learning check, part 2 is not run.
Part 2, rung 1 and rung 5 at the chosen amb, exactly as P44 (p44_rung5_dev.rung5) and P45 (p45_rung5_diag.diag), with the test
events replaced by the ambiguous-context events above. kappa +5 unchanged.
Writes p46_amb_dev.json. Not a judgement."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim7, sim4, measure4 as m4
import p37_imprint as P
import p39_ladder_dev as L
import p41_ladder2_dev as Q
import p44_rung5_dev as R
import p45_rung5_diag as D
from cells import pers_list

sim7.NNPop = L.AdaPop
SEEDS = list(range(77380, 77390))
GRID = (0.5, 0.75, 0.9, 1.0)
TARGET = 0.75
AMB = {"a": 0.0}


def mixed_outcome(w, s, p, c, amb):
    m = (np.asarray(c) % 2 == 0).astype(float)[:, None] * amb
    return (1 - m) * w["M"][s, c] + m * 0.5 * (w["M"][s, c] + w["M"][p, c])


def amb_events(w, seed, noise=0.1, n=200):
    rng = np.random.default_rng([seed, 73])
    un = np.concatenate([w["un0"], w["un1"]])
    e1 = rng.choice(un, n); e2 = np.array([rng.choice(un[un != a]) for a in e1]); c = 2 * rng.integers(0, sim4.N_CTX_TR // 2, n)
    act = np.where(rng.random(n) < 1 / (1 + np.exp(-(w["prop"][e1] - w["prop"][e2]))), e1, e2)
    oth = np.where(act == e1, e2, e1)
    db = mixed_outcome(w, act, oth, c, AMB["a"]) + noise * rng.normal(size=(n, sim4.KB))
    return e1, e2, c, act, db.astype(np.float32)


def r2_mixed(v, w):
    s, p, c = sim4.probe_set(w, "un")
    return m4.r2(v.get("un")["PB"], mixed_outcome(w, s, p, c, AMB["a"]))


def train(pers_roles, cond, amb, worlds):
    base = dict(P.BASE, noise=0.1, amb=amb)
    L.ADAPTER["on"] = False
    pre_p = pers_list(SEEDS, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim7.run(pre_p, worlds, dict(learner="set", **base))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    L.ADAPTER["on"] = True
    pers = pers_list(SEEDS, ("F1",), pers_roles, "", [cond])
    idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
    st = sim7.run(pers, worlds, dict(learner="nn0", partner_scale=0.0, **base), init_params={k: v[idx].clone() for k, v in shared.items()})
    return st, pers


def calib_level(amb, worlds):
    AMB["a"] = amb
    st, pers = train((("A#0", 0, 1, 3),), R.COND, amb, worlds)
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        v = vw["A#0"]; w = worlds[(s, f)]
        e1, e2, c, act, db = amb_events(w, s)
        S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
        with torch.no_grad():
            pb = v.PB(S, Pp, C).reshape(-1, 2, sim4.KB)
            ll = -((m4._t(db)[:, None, :] - pb) ** 2 / (2 * v.state["sig2"][None, None, :])).sum(-1)
            r = torch.softmax(ll, 1).cpu().numpy()
        rows.append(dict(seed=s, conf=float(np.median(r.max(1))), acc=float(np.where(act == e1, r[:, 0], r[:, 1]).mean()), r2_mixed=r2_mixed(v, w)))
    nf = st["nonfinite"]; del st; torch.cuda.empty_cache()
    med = lambda k: float(np.median([r[k] for r in rows]))
    return dict(amb=amb, conf_median=med("conf"), acc_mean=float(np.mean([r["acc"] for r in rows])), r2_mixed_median=med("r2_mixed"), nonfinite=nf, rows=rows)


def main():
    t0 = time.time()
    worlds = {(s, "F1"): sim7.make_world(s, "F1") for s in SEEDS}
    L.SEEDS = SEEDS
    levels = [calib_level(a, worlds) for a in GRID]
    ok = [l for l in levels if l["r2_mixed_median"] >= 0.5]
    out = dict(levels=levels, target=TARGET)
    if not ok:
        out["chosen_amb"] = None
    else:
        amb = min(ok, key=lambda l: abs(l["conf_median"] - TARGET))["amb"]; out["chosen_amb"] = amb; AMB["a"] = amb
        R.test_events = lambda w, seed, noise: amb_events(w, seed); D.test_events = R.test_events
        st, pers = train(L.roles(4), R.COND, amb, worlds)
        rows = []
        for (s, f, cond), vw in m4.views(st, pers, worlds).items():
            w = worlds[(s, f)]
            r = dict(seed=s, r2_mixed_A=r2_mixed(vw["A#0"], w)); r.update(Q.imprint(vw, "un")); r.update(R.rung5(vw, w, s))
            r.update({"diag_" + k: v for k, v in D.diag(vw, w, s).items()}); rows.append(r)
        rows.sort(key=lambda r: r["seed"])
        med = lambda k: float(np.median([r[k] for r in rows]))
        cnt = lambda a, b: sum(r[a] > r[b] for r in rows)
        summ = {k: med(k) for k in rows[0] if k != "seed"}
        summ.update(n_H_gt_S_un=cnt("H_un", "S_un"), n_H_gt_E_un=cnt("H_un", "E_un"), n_HrV_gt_SrV=cnt("H_rV", "S_rV"),
                    n_HrV_gt_ErV=cnt("H_rV", "E_rV"), n_Hrf_gt_Srf=cnt("H_rfull", "S_rfull"), n_Hrf_gt_Erf=cnt("H_rfull", "E_rfull"))
        out.update(summary=summ, rows=rows, nonfinite=st["nonfinite"])
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    commit = __import__("subprocess").run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    out.update(env=env, commit=commit, secs=time.time() - t0)
    (HERE / "p46_amb_dev.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for l in levels:
        print({k: l[k] for k in ("amb", "conf_median", "acc_mean", "r2_mixed_median", "nonfinite")})
    print("chosen amb", out["chosen_amb"])
    if out.get("summary"):
        print(json.dumps(out["summary"], indent=1, default=float))
    print(env, round(out["secs"]), "s")


if __name__ == "__main__":
    main()
