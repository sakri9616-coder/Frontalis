"""P33: the v5 evaluation (PREREG-v5-DRAFT.md). Idempotent jobs for sched.py, a merge check and the judgement.

  python -B p33_eval_v5.py constants              (before the freeze) from p32_setwg.json: J3v5 usability and m_abs by the
                                                  rule of PREREG-v5 section 6 -> p33_v5_constants.json
  python -B p33_eval_v5.py job <set|krr|nn0> <first seed>   one job: 5 seeds, C-off and C-SLp -> p33_eval_<learner>_<first>.json
  python -B p33_eval_v5.py check                  completeness and one-place report (no judgement)
  python -B p33_eval_v5.py judge                  check, then judge -> p33_judgment.json

Evaluation seeds 77300-77339 (first seeds 77300, 77305, ..., 77335). Dev seeds 77380 and 77385 are accepted as first seeds
for a dry run of the job (writes p33_dev_<learner>_<first>.json; no hash check).
- Jobs: set = NN-SET with wgate, roles A A' B B'; krr = KRR with krr_self_b (matched attribution), role A; nn0 = NN-0 with
  wgate, role A. Family F1, I-H, production scale.
- Idempotent: an existing result file means nothing to do. The result appears only when complete (.part + os.replace).
- An evaluation job checks the SHA-256 of every file listed in FREEZE-v5.md (LF) and refuses to run on a mismatch (exit 3).
- The merge refuses to judge unless all 24 files exist, each with its 5 seeds x 2 conditions, all on one machine type, and
  every commit the jobs ran holds byte-identical frozen files.
- Results are reported to the leader first and are not committed (probes/.gitignore)."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
FIRSTS = list(range(77300, 77340, 5))
DEV_FIRSTS = (77380, 77385)
LEARNERS = ("set", "krr", "nn0")
EXPECTED = [(L, s) for L in LEARNERS for s in FIRSTS]
CONDS = ["C-off", "C-SLp"]
FREEZE = HERE.parent / "FREEZE-v5.md"


def frozen_paths():
    block = FREEZE.read_text(encoding="utf-8").split("```")[1]
    return [(l.split(None, 1)[0], l.split(None, 1)[1].strip()) for l in block.strip().splitlines()]


def frozen_ok():
    if not FREEZE.exists():
        return ["FREEZE-v5.md missing"]
    return [p for sha, p in frozen_paths() if hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha]


def constants():
    """PREREG-v5 section 6: usable iff PL-state passes (8 of 10, floor = max(0.25 x median AA, m_abs)) and PL-state0 fails, in
    C-SLp; m_abs = max of the 10 PL-state0 J3v5 differences, rounded UP to 3 decimals (0.001 if not positive)."""
    import math
    import numpy as np
    from measure4 import judge
    cal = json.loads((HERE / "p32_setwg.json").read_text(encoding="utf-8"))["calibration"]
    null = cal["C-SLp|PL-state0"]["J3v5_diffs"]
    m_abs = max(0.001, math.ceil(max(null) * 1000) / 1000)
    out = dict(m_abs=m_abs, source="p32_setwg.json")
    for nm in ("PL-state", "PL-state0"):
        c = cal[f"C-SLp|{nm}"]
        floor = max(0.25 * float(np.median(c["J3v5_AA"])), m_abs)
        out[nm] = judge(c["J3v5_diffs"], floor, 0.8)
    out["J3v5_usable"] = bool(out["PL-state"]["passed"] and not out["PL-state0"]["passed"])
    (HERE / "p33_v5_constants.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


def job(learner, first):
    first = int(first)
    assert learner in LEARNERS and (first in FIRSTS or first in DEV_FIRSTS), "bad job"
    dev = first in DEV_FIRSTS
    out = HERE / (f"p33_dev_{learner}_{first}.json" if dev else f"p33_eval_{learner}_{first}.json")
    if out.exists():
        print("result exists; nothing to do:", out.name)
        return 0
    if not dev:
        bad = frozen_ok()
        if bad:
            print("FROZEN FILES DIFFER:", bad)
            return 3
    import numpy as np, torch
    import sim5, measure4 as m4
    from cells import ROLES, SCALE, COND, pers_list
    from d3_exp import j3v5
    from d2_diag import body_hinge_frac
    seeds = list(range(first, first + 5))
    lr = json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    ps = json.loads((HERE / "p16_pscale.json").read_text(encoding="utf-8"))
    roles = ROLES if learner == "set" else (("A", 0, 1, 3),)
    cfg = dict(learner=learner, innate="IH", **SCALE["prod"])
    cfg.update(dict(lr=lr, wgate=1) if learner == "set" else dict(lr=lr, wgate=1, partner_scale=ps.get("nn0", 0.0)) if learner == "nn0"
               else dict(partner_scale=ps.get("krr", 0.0), krr_self_b=1))
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in seeds}
    pers = pers_list(seeds, ("F1",), roles, "", CONDS)
    t0 = time.time()
    st = sim5.run(pers, worlds, cfg)
    idx = {(q["world"], q["role"], q["cond"]): i for i, q in enumerate(pers)}
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        A = vw["A"]
        r = dict(seed=s, cond=cond, r2_un=m4.own_r2(A, "un"), r2_nc=m4.own_r2(A, "nc"))
        if learner == "set":
            w = worlds[(s, f)]
            g = m4.measure_group(vw, w, s, "IH", COND[cond]["cycle"], das=False)
            r.update({k: g[k] for k in ("D_AB_un", "D_AA_un", "Drho_AB_un", "Drho_AA_un", "Drho_AB_nc", "Drho_AA_nc", "xp_other_un",
                                        "xp_same_un", "xp_other_nc", "xp_same_nc", "repro_un", "swapd_AB", "swapd_AA")})
            r["v5_swapd_AB"], r["v5_swapd_AA"] = j3v5(vw, w, s, True)
            tr = {ro: body_hinge_frac(st["buf"]["DB"][idx[((s, f), ro, cond)]].cpu().numpy())[2] for ro in ("A", "A2", "B")}
            r["w_L1_AB"] = {t: float(np.abs(tr["A"][t] - tr["B"][t]).sum()) for t in (4000, 12800, 20000, 32000)}
            r["w_L1_AA2"] = {t: float(np.abs(tr["A"][t] - tr["A2"][t]).sum()) for t in (4000, 12800, 20000, 32000)}
        rows.append(r)
    commit = subprocess.run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
               cuda=torch.version.cuda, numpy=np.__version__, python=sys.version.split()[0])
    res = dict(job=dict(learner=learner, seeds=seeds), env=env, commit=commit, frozen_check="dev" if dev else "ok",
               nonfinite=st["nonfinite"], wall_secs=time.time() - t0, rows=rows)
    part = out.with_suffix(".part")
    part.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    os.replace(part, out)
    print("wrote", out.name, env, round(res["wall_secs"]), "s")
    return 0


def same_code(commits):
    paths = ["frontalis-v3-lab/proposals/" + p for _, p in frozen_paths()]
    seen = set()
    for c in commits:
        h = []
        for p in paths:
            b = subprocess.run(["git", "show", f"{c}:{p}"], cwd=HERE, capture_output=True).stdout.replace(b"\r\n", b"\n")
            h.append(hashlib.sha256(b).hexdigest() if b else "MISSING")
        seen.add(tuple(h))
    return len(seen) == 1 and "MISSING" not in next(iter(seen), ("MISSING",))


def check():
    missing, files, envs, commits, problems = [], {}, set(), set(), []
    for L, s in EXPECTED:
        p = HERE / f"p33_eval_{L}_{s}.json"
        if not p.exists():
            missing.append(p.name); continue
        r = json.loads(p.read_text(encoding="utf-8")); files[(L, s)] = r
        envs.add(json.dumps(r["env"], sort_keys=True)); commits.add(r["commit"])
        if r.get("frozen_check") != "ok":
            problems.append(f"{p.name}: frozen check {r.get('frozen_check')}")
        got = sorted((row["seed"], row["cond"]) for row in r["rows"])
        if got != sorted((x, c) for x in range(s, s + 5) for c in CONDS):
            problems.append(f"{p.name}: seeds/conditions differ")
    rep = dict(n_expected=len(EXPECTED), n_found=len(files), missing=missing, machines=sorted(envs), commits=sorted(commits),
               problems=problems, complete=not missing and not problems, one_place=len(envs) == 1 and bool(commits) and same_code(commits))
    print(json.dumps(rep, indent=1))
    return rep, files


def judge_all():
    import numpy as np
    import judge4
    from measure4 import judge
    rep, files = check()
    if not (rep["complete"] and rep["one_place"]):
        print("NOT JUDGED: incomplete or mixed machines; report to the leader.")
        return 2
    K = json.loads((HERE / "p33_v5_constants.json").read_text(encoding="utf-8"))
    rows = {L: [r for (L2, _), f in files.items() if L2 == L for r in f["rows"]] for L in LEARNERS}
    out = dict(constants=K)
    for cond in CONDS:
        rs = sorted([r for r in rows["set"] if r["cond"] == cond], key=lambda r: r["seed"])
        d = [r["v5_swapd_AB"] - r["v5_swapd_AA"] for r in rs]
        floor = max(0.25 * float(np.median([r["v5_swapd_AA"] for r in rs])), K["m_abs"])
        o = dict(J3v5=judge(d, floor, 0.75))
        c = judge4.contrasts([dict(r, r2_nc=r["r2_nc"]) for r in rs])
        o["secondary"] = {j: c[j] for j in ("J1-a", "J2-a", "J2-b", "J4-a", "J4-b", "J5-a")}
        for ctl in ("krr", "nn0"):
            o[f"J5-c_{ctl}M" if ctl == "krr" else "J5-c_nn0"] = judge4.paired(rs, [r for r in rows[ctl] if r["cond"] == cond], "r2_nc", "r2_nc", 0.02)
        o["r2_un_median"] = {L: float(np.median([r["r2_un"] for r in rows[L] if r["cond"] == cond])) for L in LEARNERS}
        o["r2_nc_median"] = {L: float(np.median([r["r2_nc"] for r in rows[L] if r["cond"] == cond])) for L in LEARNERS}
        o["w_time_course_median"] = {k: {t: float(np.median([r[k][t] for r in rs])) for t in rs[0][k]} for k in ("w_L1_AB", "w_L1_AA2")}
        out[cond] = o
    slp, off = out["C-SLp"], out["C-off"]
    out["V1"] = bool(K["J3v5_usable"] and slp["J3v5"]["passed"] and not off["J3v5"]["passed"])
    out["V2"] = bool(slp["J5-c_krrM"]["passed"] and slp["J5-c_nn0"]["passed"])
    (HERE / "p33_judgment.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("V1", out["V1"], "V2", out["V2"])
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"constants": lambda: constants(), "job": lambda: job(a[1], a[2]), "check": lambda: (check(), 0)[1],
              "judge": lambda: judge_all()}[a[0]]())
