"""P42: the v8 evaluation (PREREG-v8-DRAFT.md, LADDER.md sections 1-7). Idempotent jobs, a completeness check and the judgement.

  python -B p42_eval_v8.py hashes            print the SHA-256 block for FREEZE-v8.md
  python -B p42_eval_v8.py devcheck          run on the P41 dev seeds and compare with p41_ladder2_dev.json (no hash check)
  python -B p42_eval_v8.py job <first seed>   one job: 10 seeds -> p42_eval_<first>.json
  python -B p42_eval_v8.py check              completeness and one-place report
  python -B p42_eval_v8.py judge              check, then judge rungs 1-4 in order -> p42_judgment.json

Evaluation seeds 77540-77579 (first seeds 77540, 77550, 77560, 77570). Measurements: p41_ladder2_dev.imprint (rungs 1, 2 on
P_un and rung 3 on P_nc) and p41_ladder2_dev.rung4b (rung 4), on the P39 architecture (p39_ladder_dev). An evaluation job refuses
to run unless FREEZE-v8.md lists exactly HASHED and every file matches (exit 3). Results go to the leader first."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
FIRSTS = list(range(77540, 77580, 10))
FREEZE = HERE.parent / "FREEZE-v8.md"
HASHED = ["PREREG-v8-DRAFT.md", "LADDER.md", "probes/sim4.py", "probes/measure4.py", "probes/cells.py", "probes/diag5/make_sim5.py",
          "probes/diag5/sim5.py", "probes/p37_imprint.py", "probes/p39_ladder_dev.py", "probes/p41_ladder2_dev.py", "probes/p42_eval_v8.py",
          "probes/p15_lr.json"]
RESID_X_MAX = 0.10


def frozen_paths():
    block = FREEZE.read_text(encoding="utf-8").split("```")[1]
    return [(l.split(None, 1)[0], l.split(None, 1)[1].strip()) for l in block.strip().splitlines()]


def frozen_ok():
    if not FREEZE.exists():
        return ["FREEZE-v8.md missing"]
    try:
        fp = frozen_paths()
    except Exception as e:
        return [f"FREEZE-v8.md unreadable: {e!r}"]
    if sorted(p for _, p in fp) != sorted(HASHED):
        return ["FREEZE-v8.md block does not list exactly the HASHED files"]
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
    import sim5, sim4, measure4 as m4
    import p37_imprint as P
    import p39_ladder_dev as L
    import p41_ladder2_dev as Q
    from cells import pers_list
    L.SEEDS = list(seeds)
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in seeds}
    L.ADAPTER["on"] = False
    pre_p = pers_list(seeds, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim5.run(pre_p, worlds, dict(learner="set", **P.BASE))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    nf = dict(pretrain=stp["nonfinite"]); del stp; torch.cuda.empty_cache()
    L.ADAPTER["on"] = True
    pers = pers_list(seeds, ("F1",), L.roles(4), "", ["C-SLp"])
    idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
    st = sim5.run(pers, worlds, dict(learner="nn0", partner_scale=0.0, **P.BASE), init_params={k: v[idx].clone() for k, v in shared.items()})
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        r = dict(seed=s, r2_un_A=m4.own_r2(vw["A#0"], "un"))
        r.update(Q.imprint(vw, "un")); r.update(Q.imprint(vw, "nc")); r.update(Q.rung4b(vw, worlds[(s, f)], s))
        rows.append(r)
    nf["ada"] = st["nonfinite"]; del st; torch.cuda.empty_cache()
    return sorted(rows, key=lambda r: r["seed"]), nf


def devcheck():
    ref = json.loads((HERE / "p41_ladder2_dev.json").read_text(encoding="utf-8"))
    rows, nf = run_seeds(list(range(77380, 77390)))
    R = {r["seed"]: r for r in ref["rows"]}
    diff = max(abs(r[k] - R[r["seed"]][k]) for r in rows for k in R[r["seed"]] if k != "seed")
    out = dict(max_abs_diff=diff, nonfinite=nf, ref_commit=ref["commit"])
    (HERE / "p42_devcheck.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


def job(first):
    first = int(first)
    assert first in FIRSTS, "bad first seed"
    out = HERE / f"p42_eval_{first}.json"
    bad = frozen_ok()
    if bad:
        print("FROZEN FILES DIFFER:", bad)
        return 3
    if out.exists():
        print("result exists; nothing to do:", out.name)
        return 0
    import numpy as np, torch
    seeds = list(range(first, first + 10)); t0 = time.time()
    rows, nf = run_seeds(seeds)
    commit = subprocess.run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
               cuda=torch.version.cuda, numpy=np.__version__, python=sys.version.split()[0])
    res = dict(job=dict(first=first, seeds=seeds), env=env, commit=commit, frozen_check="ok", nonfinite=nf, wall_secs=time.time() - t0, rows=rows)
    part = out.with_suffix(".part")
    part.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    os.replace(part, out)
    print("wrote", out.name, env, round(res["wall_secs"]), "s")
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
        p = HERE / f"p42_eval_{s}.json"
        if not p.exists():
            missing.append(p.name); continue
        r = json.loads(p.read_text(encoding="utf-8")); files[s] = r
        envs.add(json.dumps(r["env"], sort_keys=True)); commits.add(r["commit"])
        if r.get("frozen_check") != "ok":
            problems.append(f"{p.name}: frozen check {r.get('frozen_check')}")
        if sorted(x["seed"] for x in r["rows"]) != list(range(s, s + 10)):
            problems.append(f"{p.name}: seeds differ")
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
    A = sorted([x for f in files.values() for x in f["rows"]], key=lambda x: x["seed"])
    n = len(A); need = 0.75 * n
    med = lambda k: float(np.median([r[k] for r in A]))

    def pattern(h, s_, e):
        o = dict(H=med(h), S=med(s_), E=med(e), H_over_S=med(h) / max(med(s_), 1e-12), n_H_gt_S=sum(r[h] > r[s_] for r in A),
                 n_H_gt_E=sum(r[h] > r[e] for r in A))
        o["met"] = bool(o["H"] >= 2 * o["S"] and o["n_H_gt_S"] >= need and o["n_H_gt_E"] >= need)
        return o
    r1 = pattern("H_un", "S_un", "E_un"); r1["r2_un_A"] = med("r2_un_A"); r1["passed"] = bool(r1["met"] and r1["r2_un_A"] >= 0.5)
    r2 = pattern("Hf_un", "Sf_un", "Ef_un"); r2["passed"] = r2["met"]
    r3 = dict(evaluation=pattern("H_nc", "S_nc", "E_nc"), features=pattern("Hf_nc", "Sf_nc", "Ef_nc"))
    r3["passed"] = bool(r3["evaluation"]["met"] and r3["features"]["met"])          # both (user decision 2026-10-22)
    d4 = [r["swapd_AB"] - r["swapd_AA"] for r in A]
    r4 = dict(median=float(np.median(d4)), n_pos=sum(x > 0 for x in d4), floor=0.25 * med("swapd_AA"), resid_X=med("resid_X"), resid_all=med("resid_all"))
    r4["pattern_met"] = bool(r4["n_pos"] >= need and r4["median"] >= r4["floor"])
    r4["passed"] = bool(r4["pattern_met"] and r4["resid_X"] <= RESID_X_MAX)
    out = dict(n=n, rung1=r1, rung2=r2, rung3=r3, rung4=r4)
    ok, judged = True, {}
    for k in ("rung1", "rung2", "rung3", "rung4"):
        judged[k] = out[k]["passed"] if ok else "not judged (an earlier rung failed)"
        ok = ok and out[k]["passed"]
    out["ladder"] = judged
    out["frozen_now"] = frozen_ok()
    (HERE / "p42_judgment.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


def hashes():
    for p in HASHED:
        print(hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), p)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"job": lambda: job(a[1]), "check": lambda: (check(), 0)[1], "judge": judge_all, "hashes": hashes, "devcheck": devcheck}[a[0]]())
