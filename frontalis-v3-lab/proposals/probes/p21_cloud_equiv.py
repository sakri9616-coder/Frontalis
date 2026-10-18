"""P21: local-vs-cloud equivalence check (proposals/offload_note.md section 4). Rule fixed BEFORE any cloud result, in the
same form as the bf16 rule of PREREG draft 3.1 section 10b. Rehearsal seeds 77340-77342 only (never evaluation seeds).
  run <label>          NN-SET, I-H, C-off and C-SLp, F1 F6m F5, roles A A2 B B2 A3 A4, rehearsal scale (2,000, S/N 1), lr 1e-3
  run <label> retest   the same, but every personality's env and sgd streams shifted by +1000 (a same-condition retest)
  compare <local> <cloud> <cloud_retest>
                       delta_m = |m_cloud - m_local| and nu_m = |m_cloud - m_cloud_retest| per world (seed x family x cycle)
                       E1 median(delta_m) <= 0.2 median(nu_m) for every m; E2 90th pct(delta_m) <= 0.5 median(nu_m);
                       E3 cloud non-finite count 0; E4 p13_ckpt_early on the cloud: 'identical' True for set and krr.
                       Level 1 (numerically equivalent) = E1 and E2 and E3 and E4.
                       Level 2 (retest-equivalent) = E5 and E3 and E4, E5: median(delta_m) <= 1.5 median(nu_m) for every m
                       (the cloud run differs from the local run no more than a same-condition retest does; expected when
                       floating-point differences grow along the SGD path, P18).
Writes p21_<label>.json, p21_compare.json."""
import json, sys
from pathlib import Path
import numpy as np
import sim4
import measure4 as m4
from cells import COND, ROLES, SCALE, pers_list

HERE = Path(__file__).parent
SEEDS = [77340, 77341, 77342]
FAMS = ("F1", "F6m", "F5")
CONDS = ["C-off", "C-SLp"]
ROLES6 = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))
M = ("D_AB_un", "D_AA_un", "Drho_AB_un", "Drho_AA_un", "Drho_AB_nc", "Drho_AA_nc", "swapd_AB", "swapd_AA",
     "xp_other_un", "xp_same_un", "xp_other_nc", "xp_same_nc", "r2_un", "r2_nc", "repro_un")


def run(label, retest=False):
    worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in FAMS}
    pers = pers_list(SEEDS, FAMS, ROLES6, conds=CONDS)
    if retest:
        for q in pers:
            q["env"] += 1000; q["sgd"] += 1000
    st = sim4.run(pers, worlds, dict(learner="set", innate="IH", **SCALE["rehearsal"], lr=1e-3))
    rows = []
    for (s, f, cond), v in m4.views(st, pers, worlds).items():
        r = m4.measure_group(v, worlds[(s, f)], s, "IH", COND[cond]["cycle"], das=False)
        r.update(seed=s, fam=f, cond=cond); rows.append(r)
    import torch
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               numpy=np.__version__, python=sys.version.split()[0])
    (HERE / f"p21_{label}.json").write_text(json.dumps(dict(rows=rows, nonfinite=st["nonfinite"], env=env, retest=retest,
                                                             secs=st["secs"]), indent=1, default=float), encoding="utf-8")


def compare(lo, cl, cr):
    L, C, R = (json.loads((HERE / f"p21_{x}.json").read_text(encoding="utf-8")) for x in (lo, cl, cr))
    key = lambda r: (r["seed"], r["fam"], r["cond"])
    li, ri = {key(r): r for r in L["rows"]}, {key(r): r for r in R["rows"]}
    out = {}
    for m in M:
        d = np.array([abs(r[m] - li[key(r)][m]) for r in C["rows"] if m in r])
        n = np.array([abs(r[m] - ri[key(r)][m]) for r in C["rows"] if m in r])
        if not len(d):
            continue
        out[m] = dict(delta_med=float(np.median(d)), delta_p90=float(np.percentile(d, 90)), nu_med=float(np.median(n)),
                      E1=bool(np.median(d) <= 0.2 * np.median(n)), E2=bool(np.percentile(d, 90) <= 0.5 * np.median(n)),
                      E5=bool(np.median(d) <= 1.5 * np.median(n)))
    ck = HERE / "p13_ckpt_early.json"
    e4 = all(v["identical"] for v in json.loads(ck.read_text(encoding="utf-8")).values()) if ck.exists() else None
    summ = dict(E1=all(v["E1"] for v in out.values()), E2=all(v["E2"] for v in out.values()), E5=all(v["E5"] for v in out.values()), E3=C["nonfinite"] == 0 and R["nonfinite"] == 0,
                E4=e4, env_local=L["env"], env_cloud=C["env"])
    summ["level1"] = bool(summ["E1"] and summ["E2"] and summ["E3"] and summ["E4"])
    summ["level2"] = bool(summ["E5"] and summ["E3"] and summ["E4"])
    (HERE / "p21_compare.json").write_text(json.dumps(dict(summary=summ, per_m=out), indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "retest")
    else:
        compare(*sys.argv[2:5])
