"""P33: the v5 evaluation (PREREG-v5-DRAFT.md). Idempotent jobs for sched.py, a merge check and the judgement.

  python -B p33_eval_v5.py constants              (before the freeze) from p32_setwg_modal.json (evaluation image): J3v5
                                                  usability and m_abs by the rule of PREREG-v5 section 6 -> p33_v5_constants.json
  python -B p33_eval_v5.py hashes                 print the SHA-256 block for FREEZE-v5.md (the freeze itself is a separate step)
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
- The cell list follows p33_v5_config.json ("v2": true -> the KRR and NN-0 cells for V2 are part of the evaluation).
- The judge re-derives the constants from the hashed p32_setwg_modal.json and refuses to judge if they differ from
  p33_v5_constants.json.
- Results are reported to the leader first and are not committed (local git exclude)."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
FIRSTS = list(range(77300, 77340, 5))
DEV_FIRSTS = (77380, 77385)
LEARNERS = ("set", "krr", "nn0")
CONFIG = json.loads((HERE / "p33_v5_config.json").read_text(encoding="utf-8"))
EVAL_LEARNERS = LEARNERS if CONFIG["v2"] else ("set",)
EXPECTED = [(L, s) for L in EVAL_LEARNERS for s in FIRSTS]
CONDS = ["C-off", "C-SLp"]
FREEZE = HERE.parent / "FREEZE-v5.md"


def frozen_paths():
    block = FREEZE.read_text(encoding="utf-8").split("```")[1]
    return [(l.split(None, 1)[0], l.split(None, 1)[1].strip()) for l in block.strip().splitlines()]


def frozen_ok():
    if not FREEZE.exists():
        return ["FREEZE-v5.md missing"]
    return [p for sha, p in frozen_paths() if hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha]


def derive_constants():
    """PREREG-v5 section 6: usable iff PL-state passes (8 of 10, floor = max(0.25 x median AA, m_abs)) and PL-state0 fails, in
    C-SLp; m_abs = max of the 10 PL-state0 J3v5 differences, rounded UP to 3 decimals (0.001 if not positive)."""
    import math
    import numpy as np
    from measure4 import judge
    src = json.loads((HERE / "p32_setwg_modal.json").read_text(encoding="utf-8"))
    cal = src["calibration"]
    null = cal["C-SLp|PL-state0"]["J3v5_diffs"]
    m_abs = max(0.001, math.ceil(max(null) * 1000) / 1000)
    out = dict(m_abs=m_abs, source="p32_setwg_modal.json", source_machine={k: src["meta"].get(k) for k in ("device", "torch", "commit")})
    for nm in ("PL-state", "PL-state0"):
        c = cal[f"C-SLp|{nm}"]
        floor = max(0.25 * float(np.median(c["J3v5_AA"])), m_abs)
        out[nm] = judge(c["J3v5_diffs"], floor, 0.8)
    out["J3v5_usable"] = bool(out["PL-state"]["passed"] and not out["PL-state0"]["passed"])
    return json.loads(json.dumps(out, default=float))


def constants():
    out = derive_constants()
    (HERE / "p33_v5_constants.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


def job(learner, first):
    first = int(first)
    assert learner in EVAL_LEARNERS and (first in FIRSTS or first in DEV_FIRSTS), "bad job (learner not in the frozen cell list?)"
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
    from measure5 import j3v5, w_course
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
            r["v5_swapd_AB"], r["v5_swapd_AA"] = j3v5(vw, w, s)
            if cond == "C-SLp":
                tr = {ro: w_course(st["buf"]["DB"][idx[((s, f), ro, cond)]].cpu().numpy()) for ro in ("A", "A2", "B")}
                r["w_L1_AB"] = {t: float(np.abs(tr["A"][t] - tr["B"][t]).sum()) for t in tr["A"]}
                r["w_L1_AA2"] = {t: float(np.abs(tr["A"][t] - tr["A2"][t]).sum()) for t in tr["A"]}
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
    if K != derive_constants():
        print("NOT JUDGED: p33_v5_constants.json differs from the constants re-derived from p32_setwg_modal.json")
        return 4
    rows = {L: [r for (L2, _), f in files.items() if L2 == L for r in f["rows"]] for L in EVAL_LEARNERS}
    out = dict(constants=K)
    for cond in CONDS:
        rs = sorted([r for r in rows["set"] if r["cond"] == cond], key=lambda r: r["seed"])
        d = [r["v5_swapd_AB"] - r["v5_swapd_AA"] for r in rs]
        floor = max(0.25 * float(np.median([r["v5_swapd_AA"] for r in rs])), K["m_abs"])
        o = dict(J3v5=judge(d, floor, 0.75))
        c = judge4.contrasts([dict(r, r2_nc=r["r2_nc"]) for r in rs])
        o["secondary"] = {j: c[j] for j in ("J1-a", "J2-a", "J2-b", "J4-a", "J4-b", "J5-a")}
        for ctl in [L for L in ("krr", "nn0") if L in EVAL_LEARNERS]:
            o[f"J5-c_{ctl}M" if ctl == "krr" else "J5-c_nn0"] = judge4.paired(rs, [r for r in rows[ctl] if r["cond"] == cond], "r2_nc", "r2_nc", 0.02)
        o["r2_un_median"] = {L: float(np.median([r["r2_un"] for r in rows[L] if r["cond"] == cond])) for L in EVAL_LEARNERS}
        o["r2_nc_median"] = {L: float(np.median([r["r2_nc"] for r in rows[L] if r["cond"] == cond])) for L in EVAL_LEARNERS}
        if cond == "C-SLp":
            o["w_time_course_median"] = {k: {t: float(np.median([r[k][t] for r in rs])) for t in rs[0][k]} for k in ("w_L1_AB", "w_L1_AA2")}
        out[cond] = o
    slp, off = out["C-SLp"], out["C-off"]
    # C-off is a sanity row, not evidence: w = 1 for every C-off personality, so the J3v5 difference is -swapd_AA <= 0 by
    # construction. If it passes anyway, the computation is broken (reported as a failure of the run, not as a result).
    out["C-off_sanity_ok"] = not off["J3v5"]["passed"]
    out["V1"] = bool(K["J3v5_usable"] and slp["J3v5"]["passed"]) if out["C-off_sanity_ok"] else None
    out["V2"] = bool(slp["J5-c_krrM"]["passed"] and slp["J5-c_nn0"]["passed"]) if CONFIG["v2"] else "not in the frozen cell list"
    (HERE / "p33_judgment.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("V1", out["V1"], "V2", out["V2"])
    return 0


HASHED = ["PREREG-v5-DRAFT.md", "probes/sim4.py", "probes/measure4.py", "probes/cells.py", "probes/judge4.py", "probes/measure5.py",
          "probes/diag5/make_sim5.py", "probes/diag5/sim5.py", "probes/p32_v5_dev.py", "probes/p33_eval_v5.py", "probes/p32_setwg_modal.json",
          "probes/p33_v5_constants.json", "probes/p33_v5_config.json", "probes/p15_lr.json", "probes/p16_pscale.json"]


def hashes():
    for p in HASHED:
        print(hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), p)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"constants": lambda: constants(), "job": lambda: job(a[1], a[2]), "check": lambda: (check(), 0)[1],
              "judge": lambda: judge_all(), "hashes": hashes}[a[0]]())
