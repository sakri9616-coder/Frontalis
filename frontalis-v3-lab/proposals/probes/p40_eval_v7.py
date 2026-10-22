"""P40: the v7 evaluation (PREREG-v7-DRAFT.md, LADDER.md). Idempotent jobs, a completeness check and the judgement.

  python -B p40_eval_v7.py hashes            print the SHA-256 block for FREEZE-v7.md
  python -B p40_eval_v7.py job <first seed>   one job: 10 seeds, adapter arm + o3 arm -> p40_eval_<first>.json
  python -B p40_eval_v7.py devcheck           run the same code on the P39 dev seeds and compare with p39_ladder_dev.json (no hash check)
  python -B p40_eval_v7.py check              completeness and one-place report
  python -B p40_eval_v7.py judge              check, then judge the ladder in order -> p40_judgment.json

Evaluation seeds 77500-77539 (first seeds 77500, 77510, 77520, 77530). Measurement code is p39_ladder_dev.py (unchanged);
this file only runs it on other seeds and judges. An evaluation job refuses to run unless FREEZE-v7.md exists, lists exactly
HASHED, and every listed file has its frozen SHA-256 (LF); exit 3 otherwise. Results go to the leader first."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
FIRSTS = list(range(77500, 77540, 10))
FREEZE = HERE.parent / "FREEZE-v7.md"
HASHED = ["PREREG-v7-DRAFT.md", "LADDER.md", "probes/sim4.py", "probes/measure4.py", "probes/cells.py", "probes/diag5/make_sim5.py",
          "probes/diag5/sim5.py", "probes/p37_imprint.py", "probes/p39_ladder_dev.py", "probes/p40_eval_v7.py", "probes/p15_lr.json"]
RESID_MAX = 0.10   # rung 4 is "established" only if the median value-matching residual is at most this (PREREG-v7 section 3)


def frozen_paths():
    block = FREEZE.read_text(encoding="utf-8").split("```")[1]
    return [(l.split(None, 1)[0], l.split(None, 1)[1].strip()) for l in block.strip().splitlines()]


def frozen_ok():
    if not FREEZE.exists():
        return ["FREEZE-v7.md missing"]
    try:
        fp = frozen_paths()
    except Exception as e:
        return [f"FREEZE-v7.md unreadable: {e!r}"]
    if sorted(p for _, p in fp) != sorted(HASHED):
        return ["FREEZE-v7.md block does not list exactly the HASHED files"]
    bad = []
    for sha, p in fp:
        try:
            ok = hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == sha
        except OSError:
            ok = False
        if not ok:
            bad.append(p)
    return bad


def run_seeds(seeds):
    import torch
    import sim5
    import p37_imprint as P
    import p39_ladder_dev as L
    from cells import pers_list
    L.SEEDS = list(seeds)
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in seeds}
    L.ADAPTER["on"] = False
    pre_p = pers_list(seeds, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim5.run(pre_p, worlds, dict(learner="set", **P.BASE))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    nf = dict(pretrain=stp["nonfinite"]); del stp; torch.cuda.empty_cache()
    ada, nf["ada"] = L.run_arm(True, worlds, shared, wi)
    o3, nf["o3"] = L.run_arm(False, worlds, shared, wi)
    return dict(ada=[ada[s] for s in seeds], o3=[o3[s] for s in seeds]), nf


def job(first):
    first = int(first)
    assert first in FIRSTS, "bad first seed"
    out = HERE / f"p40_eval_{first}.json"
    bad = frozen_ok()
    if bad:
        print("FROZEN FILES DIFFER:", bad)
        return 3
    if out.exists():
        print("result exists; nothing to do:", out.name)
        return 0
    import numpy as np, torch
    seeds = list(range(first, first + 10)); t0 = time.time()
    arms, nf = run_seeds(seeds)
    commit = subprocess.run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
               cuda=torch.version.cuda, numpy=np.__version__, python=sys.version.split()[0])
    res = dict(job=dict(first=first, seeds=seeds), env=env, commit=commit, frozen_check="ok", nonfinite=nf, wall_secs=time.time() - t0, arms=arms)
    part = out.with_suffix(".part")
    part.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    os.replace(part, out)
    print("wrote", out.name, env, round(res["wall_secs"]), "s")
    return 0


def devcheck():
    """Review C2: p40 must reproduce P39 on the dev seeds (same machine type). Writes p40_devcheck.json."""
    import numpy as np
    ref = json.loads((HERE / "p39_ladder_dev.json").read_text(encoding="utf-8"))
    arms, nf = run_seeds(list(range(77380, 77390)))
    diff = {arm: max(abs(a[k] - b[k]) for a, b in zip(arms[arm], ref[arm]) for k in a if k != "seed") for arm in ("ada", "o3")}
    out = dict(max_abs_diff=diff, nonfinite=nf, ref_commit=ref["commit"])
    (HERE / "p40_devcheck.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


def same_code(commits):
    fp = frozen_paths()
    for c in commits:
        for sha, p in fp:
            b = subprocess.run(["git", "show", f"{c}:frontalis-v3-lab/proposals/{p}"], cwd=HERE, capture_output=True).stdout.replace(b"\r\n", b"\n")
            if not b or hashlib.sha256(b).hexdigest() != sha:
                return False
    return bool(commits)


def check():
    missing, files, envs, commits, problems = [], {}, set(), set(), []
    for s in FIRSTS:
        p = HERE / f"p40_eval_{s}.json"
        if not p.exists():
            missing.append(p.name); continue
        r = json.loads(p.read_text(encoding="utf-8")); files[s] = r
        envs.add(json.dumps(r["env"], sort_keys=True)); commits.add(r["commit"])
        if r.get("frozen_check") != "ok":
            problems.append(f"{p.name}: frozen check {r.get('frozen_check')}")
        for arm in ("ada", "o3"):
            if sorted(x["seed"] for x in r["arms"].get(arm, [])) != list(range(s, s + 10)):
                problems.append(f"{p.name}: seeds of arm {arm} differ")
        if r["env"].get("gpu", "cpu") == "cpu":
            problems.append(f"{p.name}: ran on CPU")
        if any(v != 0 for v in r["nonfinite"].values()):
            problems.append(f"{p.name}: non-finite values")
    rep = dict(n_expected=len(FIRSTS), n_found=len(files), missing=missing, machines=sorted(envs), commits=sorted(commits),
               problems=problems, complete=not missing and not problems, one_place=len(envs) == 1 and bool(commits) and same_code(commits))
    print(json.dumps(rep, indent=1))
    return rep, files


def judge_all():
    import numpy as np
    if frozen_ok():
        print("NOT JUDGED: frozen files differ now:", frozen_ok())
        return 2
    rep, files = check()
    if not (rep["complete"] and rep["one_place"]):
        print("NOT JUDGED: incomplete or mixed machines; report to the leader.")
        return 2
    A = sorted([x for f in files.values() for x in f["arms"]["ada"]], key=lambda x: x["seed"])
    O = {x["seed"]: x for f in files.values() for x in f["arms"]["o3"]}
    n = len(A); need = 0.75 * n
    med = lambda rows, k: float(np.median([r[k] for r in rows]))
    ratio_rung = lambda h, s_, e: dict(H=med(A, h), S=med(A, s_), E=med(A, e), H_over_S=med(A, h) / max(med(A, s_), 1e-12),
                                      n_H_gt_S=sum(r[h] > r[s_] for r in A), n_H_gt_E=sum(r[h] > r[e] for r in A))
    r1 = ratio_rung("H", "S", "E"); r1["r2_un_A"] = med(A, "r2_un_A")
    r1["passed"] = bool(r1["H"] >= 2 * r1["S"] and r1["n_H_gt_S"] >= need and r1["n_H_gt_E"] >= need and r1["r2_un_A"] >= 0.5)
    r2 = ratio_rung("Hf", "Sf", "Ef")
    r2["passed"] = bool(r2["H"] >= 2 * r2["S"] and r2["n_H_gt_S"] >= need and r2["n_H_gt_E"] >= need)
    d3 = [r["r2_nc_A"] - O[r["seed"]]["r2_nc_A"] for r in A]
    r3 = dict(median=float(np.median(d3)), n_pos=sum(x > 0 for x in d3), passed=bool(sum(x > 0 for x in d3) >= need and np.median(d3) >= 0.02))
    d4 = [r["swapd_AB"] - r["swapd_AA"] for r in A]
    floor = 0.25 * med(A, "swapd_AA")
    r4 = dict(median=float(np.median(d4)), n_pos=sum(x > 0 for x in d4), floor=floor, vmatch_resid=med(A, "vmatch_resid"))
    r4["pattern_met"] = bool(r4["n_pos"] >= need and r4["median"] >= floor)
    r4["passed"] = bool(r4["pattern_met"] and r4["vmatch_resid"] <= RESID_MAX)
    out = dict(n=n, rung1=r1, rung2=r2, rung3=r3, rung4=r4,
               o3_rung1=dict(H=float(np.median([o["H"] for o in O.values()])), S=float(np.median([o["S"] for o in O.values()])),
                             E=float(np.median([o["E"] for o in O.values()]))))
    # judged in order: a rung counts only if every earlier rung passed (all are reported)
    ok, judged = True, {}
    for k in ("rung1", "rung2", "rung3", "rung4"):
        judged[k] = out[k]["passed"] if ok else "not judged (an earlier rung failed)"
        ok = ok and out[k]["passed"]
    out["ladder"] = judged
    out["frozen_now"] = frozen_ok()
    (HERE / "p40_judgment.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


def hashes():
    for p in HASHED:
        print(hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), p)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"job": lambda: job(a[1]), "check": lambda: (check(), 0)[1], "judge": judge_all, "hashes": hashes, "devcheck": devcheck}[a[0]]())
