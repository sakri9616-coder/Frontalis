"""P47: the v9 evaluation (PREREG-v9-DRAFT.md; LADDER sections 8-9). Idempotent jobs, a completeness check and the judgement.

  python -B p47_eval_v9.py hashes            print the SHA-256 block for FREEZE-v9.md
  python -B p47_eval_v9.py devcheck          run on the P46 dev seeds and compare with p46_amb_dev.json part 2 (no hash check)
  python -B p47_eval_v9.py job <first seed>   one job: 10 seeds -> p47_eval_<first>.json
  python -B p47_eval_v9.py check              completeness and one-place report
  python -B p47_eval_v9.py judge              check, then judge rung 1 then rung 5 -> p47_judgment.json

Evaluation seeds 77580-77619 (first seeds 77580, 77590, 77600, 77610). World sim7 with amb = 1.0 (P46 rule), noise 0.1,
condition C-S+5 (kappa +5), the v8 architecture (p39_ladder_dev), roles of p39 (4 sets each). Measurements: p41 imprint (rung 1),
p44 rung5 and p45 diag with the ambiguous-context events of p46. An evaluation job refuses to run unless FREEZE-v9.md lists
exactly HASHED and every file matches (exit 3). Results go to the leader first."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
FIRSTS = list(range(77580, 77620, 10))
AMB = 1.0
FREEZE = HERE.parent / "FREEZE-v9.md"
HASHED = ["PREREG-v9-DRAFT.md", "LADDER.md", "probes/sim4.py", "probes/measure4.py", "probes/cells.py", "probes/diag5/make_sim5.py",
          "probes/diag5/sim5.py", "probes/diag5/make_sim6.py", "probes/diag5/sim6.py", "probes/diag5/make_sim7.py", "probes/diag5/sim7.py",
          "probes/p37_imprint.py", "probes/p39_ladder_dev.py", "probes/p41_ladder2_dev.py", "probes/p43_fog_calib.py", "probes/p44_rung5_dev.py",
          "probes/p45_rung5_diag.py", "probes/p46_amb_dev.py", "probes/p46_amb_dev.json", "probes/p47_eval_v9.py", "probes/p15_lr.json"]


def frozen_paths():
    block = FREEZE.read_text(encoding="utf-8").split("```")[1]
    return [(l.split(None, 1)[0], l.split(None, 1)[1].strip()) for l in block.strip().splitlines()]


def frozen_ok():
    if not FREEZE.exists():
        return ["FREEZE-v9.md missing"]
    try:
        fp = frozen_paths()
    except Exception as e:
        return [f"FREEZE-v9.md unreadable: {e!r}"]
    if sorted(p for _, p in fp) != sorted(HASHED):
        return ["FREEZE-v9.md block does not list exactly the HASHED files"]
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
    import sim7, measure4 as m4
    import p39_ladder_dev as L
    import p41_ladder2_dev as Q
    import p44_rung5_dev as R
    import p45_rung5_diag as D
    import p46_amb_dev as A
    A.SEEDS = list(seeds); L.SEEDS = list(seeds); A.AMB["a"] = AMB
    R.test_events = lambda w, seed, noise: A.amb_events(w, seed); D.test_events = R.test_events
    worlds = {(s, "F1"): sim7.make_world(s, "F1") for s in seeds}
    st, pers = A.train(L.roles(4), R.COND, AMB, worlds)
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        w = worlds[(s, f)]
        r = dict(seed=s, r2_mixed_A=A.r2_mixed(vw["A#0"], w)); r.update(Q.imprint(vw, "un")); r.update(R.rung5(vw, w, s))
        r.update({"diag_" + k: v for k, v in D.diag(vw, w, s).items()}); rows.append(r)
    nf = dict(ada=st["nonfinite"]); del st; torch.cuda.empty_cache()
    return sorted(rows, key=lambda r: r["seed"]), nf


def devcheck():
    ref = json.loads((HERE / "p46_amb_dev.json").read_text(encoding="utf-8"))
    rows, nf = run_seeds(list(range(77380, 77390)))
    Rr = {r["seed"]: r for r in ref["rows"]}
    diff = max(abs(r[k] - Rr[r["seed"]][k]) for r in rows for k in Rr[r["seed"]] if k != "seed")
    out = dict(max_abs_diff=diff, nonfinite=nf, ref_commit=ref["commit"])
    (HERE / "p47_devcheck.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0 if diff <= 1e-6 else 1


def job(first):
    first = int(first)
    assert first in FIRSTS, "bad first seed"
    out = HERE / f"p47_eval_{first}.json"
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
        p = HERE / f"p47_eval_{s}.json"
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
        o = dict(H=med(h), S=med(s_), E=med(e), H_over_S=med(h) / max(med(s_), 1e-12), H_over_E=med(h) / max(med(e), 1e-12),
                 n_H_gt_S=sum(r[h] > r[s_] for r in A), n_H_gt_E=sum(r[h] > r[e] for r in A))
        o["met"] = bool(o["H"] >= 2 * o["S"] and o["n_H_gt_S"] >= need and o["n_H_gt_E"] >= need)
        return o
    r1 = pattern("H_un", "S_un", "E_un"); r1["r2_mixed_A"] = med("r2_mixed_A"); r1["passed"] = bool(r1["met"] and r1["r2_mixed_A"] >= 0.5)
    r5 = pattern("H_rV", "S_rV", "E_rV"); r5["passed"] = r5["met"] if r1["passed"] else "not judged (rung 1 failed)"   # review C1
    out = dict(n=n, rung1=r1, rung5=r5, report_only=dict(rung5_full=pattern("H_rfull", "S_rfull", "E_rfull"), features_un=pattern("Hf_un", "Sf_un", "Ef_un"),
               shift=med("shift"), diag={k[5:]: med(k) for k in A[0] if k.startswith("diag_")}))
    out["ladder"] = dict(rung1=r1["passed"], rung5=r5["passed"])
    out["frozen_now"] = frozen_ok()
    (HERE / "p47_judgment.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


def hashes():
    for p in HASHED:
        print(hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), p)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"job": lambda: job(a[1]), "check": lambda: (check(), 0)[1], "judge": judge_all, "hashes": hashes, "devcheck": devcheck}[a[0]]())
