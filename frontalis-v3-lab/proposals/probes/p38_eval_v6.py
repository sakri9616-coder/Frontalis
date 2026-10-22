"""P38: the v6 evaluation (PREREG-v6-DRAFT.md). Idempotent jobs, a completeness check and the judgement.

  python -B p38_eval_v6.py hashes            print the SHA-256 block for FREEZE-v6.md (the freeze itself is a separate step)
  python -B p38_eval_v6.py job <first seed>   one job: 10 seeds, four arms -> p38_eval_<first>.json
  python -B p38_eval_v6.py check              completeness and one-place report (no judgement)
  python -B p38_eval_v6.py judge              check, then judge -> p38_judgment.json

Evaluation seeds 77440-77479 (first seeds 77440, 77450, 77460, 77470). The dev seeds 77380 is accepted as a first seed for a dry
run (writes p38_dev_<first>.json; no hash check). An evaluation job refuses to run unless FREEZE-v6.md exists and every file in
its block has the frozen SHA-256 (LF); exit 3 otherwise. Results are reported to the leader first and are not committed.

Arms (all F1, I-H, mid scale, C-SLp, no choice scenes; code shared with p37_imprint.py):
  o3     PRIMARY. Per-world encoder pre-trained on a common stream (no individual history), frozen; each individual learns only the
         head, 4 heads per individual (same history, 4 SGD streams), evaluation = mean of the 4.
  o2     report only. The same shared encoder, one head (K = 1).
  nn0k4  report only. Frozen RANDOM encoder, 4 heads averaged (separates "learned shared encoder" from "averaging heads").
  base   report only. NN-SET, K = 1.
Roles per arm: A, B, A_sgd, A_env (p37_imprint.QUAD). H = D(A,B), S = D(A,A_sgd), E = D(A,A_env)."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
FIRSTS = list(range(77440, 77480, 10))
DEV_FIRSTS = (77380,)
ARMS = ("o3", "o2", "nn0k4", "base")
FREEZE = HERE.parent / "FREEZE-v6.md"
HASHED = ["PREREG-v6-DRAFT.md", "probes/sim4.py", "probes/measure4.py", "probes/cells.py", "probes/diag5/make_sim5.py",
          "probes/diag5/sim5.py", "probes/p37_imprint.py", "probes/p38_eval_v6.py", "probes/p15_lr.json", "probes/sched/modal_runner.py"]


def frozen_paths():
    block = FREEZE.read_text(encoding="utf-8").split("```")[1]
    return [(l.split(None, 1)[0], l.split(None, 1)[1].strip()) for l in block.strip().splitlines()]


def frozen_ok():
    if not FREEZE.exists():
        return ["FREEZE-v6.md missing"]
    return [p for sha, p in frozen_paths() if hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha]


def run_arms(seeds):
    import torch
    import sim5, sim4
    import p37_imprint as P
    from cells import pers_list
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in seeds}
    nonfinite, res = {}, {}

    def go(pers, learner, init=None):
        st = sim5.run(pers, worlds, dict(learner=learner, **({} if learner == "set" else dict(partner_scale=0.0)), **P.BASE), **({"init_params": init} if init is not None else {}))
        return st

    pre_p = pers_list(seeds, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = go(pre_p, "set")
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    nonfinite["pretrain"] = stp["nonfinite"]
    del stp; torch.cuda.empty_cache()
    for arm in ARMS:
        K = 4 if arm in ("o3", "nn0k4") else 1
        pers = pers_list(seeds, ("F1",), P.roles_for(K), "", ["C-SLp"])
        init = None
        if arm in ("o3", "o2"):
            idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
            init = {k: v[idx].clone() for k, v in shared.items()}
        st = go(pers, "set" if arm == "base" else "nn0", init)
        E = P.evals(st, pers, worlds)
        res[arm] = P.table(E, K)
        for r in res[arm]:
            r["r2_un_min"] = min(E[r["seed"]][f"{n}#0"]["r2_un"] for n, _, _ in P.QUAD)
        nonfinite[arm] = st["nonfinite"]
        del st; torch.cuda.empty_cache()
    return res, nonfinite


def job(first):
    first = int(first)
    assert first in FIRSTS or first in DEV_FIRSTS, "bad first seed"
    dev = first in DEV_FIRSTS
    out = HERE / (f"p38_dev_{first}.json" if dev else f"p38_eval_{first}.json")
    if not dev:
        bad = frozen_ok()
        if bad:
            print("FROZEN FILES DIFFER:", bad)
            return 3
    if out.exists():
        print("result exists; nothing to do:", out.name)
        return 0
    import numpy as np, torch
    seeds = list(range(first, first + 10))
    t0 = time.time()
    rows, nonfinite = run_arms(seeds)
    commit = subprocess.run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
               cuda=torch.version.cuda, numpy=np.__version__, python=sys.version.split()[0])
    res = dict(job=dict(first=first, seeds=seeds), env=env, commit=commit, frozen_check="dev" if dev else "ok",
               nonfinite=nonfinite, wall_secs=time.time() - t0, arms=rows)
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
    for s in FIRSTS:
        p = HERE / f"p38_eval_{s}.json"
        if not p.exists():
            missing.append(p.name); continue
        r = json.loads(p.read_text(encoding="utf-8")); files[s] = r
        envs.add(json.dumps(r["env"], sort_keys=True)); commits.add(r["commit"])
        if r.get("frozen_check") != "ok":
            problems.append(f"{p.name}: frozen check {r.get('frozen_check')}")
        for arm in ARMS:
            if sorted(x["seed"] for x in r["arms"].get(arm, [])) != list(range(s, s + 10)):
                problems.append(f"{p.name}: seeds of arm {arm} differ")
        if any(v != 0 for v in r["nonfinite"].values() if isinstance(v, (int, float))):
            problems.append(f"{p.name}: non-finite values")
    rep = dict(n_expected=len(FIRSTS), n_found=len(files), missing=missing, machines=sorted(envs), commits=sorted(commits),
               problems=problems, complete=not missing and not problems, one_place=len(envs) == 1 and bool(commits) and same_code(commits))
    print(json.dumps(rep, indent=1))
    return rep, files


def verdict(rows):
    """Frozen pattern of PREREG-v6 section 5 for 40 seeds (p37_imprint.verdict is hard-wired to the 10-seed thresholds)."""
    import numpy as np
    med = lambda k: float(np.median([r[k] for r in rows]))
    nHS, nHE = sum(r["H"] > r["S"] for r in rows), sum(r["H"] > r["E"] for r in rows)
    need = 0.75 * len(rows)
    return dict(n=len(rows), H=med("H"), S=med("S"), E=med("E"), H_over_S=med("H") / max(med("S"), 1e-12), H_over_E=med("H") / max(med("E"), 1e-12),
                n_H_gt_S=nHS, n_H_gt_E=nHE, r2_un_A=med("r2_un_A"), r2_un_min_roles=med("r2_un_min"),
                success=bool(med("H") >= 2 * med("S") and nHS >= need and nHE >= need and med("r2_un_A") >= 0.5))


def judge_all():
    import numpy as np
    rep, files = check()
    if frozen_ok():
        print("NOT JUDGED: frozen files differ now:", frozen_ok())
        return 2
    if not (rep["complete"] and rep["one_place"]):
        print("NOT JUDGED: incomplete or mixed machines; report to the leader.")
        return 2
    rows = {a: sorted([x for f in files.values() for x in f["arms"][a]], key=lambda x: x["seed"]) for a in ARMS}
    out = {a: verdict(rows[a]) for a in ARMS}
    ratio = lambda a: {r["seed"]: r["H"] / max(r["S"], 1e-12) for r in rows[a]}
    ro3, rn0 = ratio("o3"), ratio("nn0k4")
    out["report_only"] = dict(n_seeds_o3_ratio_gt_nn0k4_ratio=sum(ro3[s] > rn0[s] for s in ro3), n_seeds=len(ro3),
                              median_H_over_E_o3=float(np.median([r["H"] / max(r["E"], 1e-12) for r in rows["o3"]])))
    # the frozen success pattern (PREREG-v6 section 5) is applied to o3 only, with 40 seeds
    out["V3_primary"] = out["o3"]["success"]
    out["V3_enc_report_only"] = dict(o3_beats_nn0k4_in=out["report_only"]["n_seeds_o3_ratio_gt_nn0k4_ratio"], of=len(ro3), met=out["report_only"]["n_seeds_o3_ratio_gt_nn0k4_ratio"] >= 30)
    # re-hash now, not only at job time
    out["frozen_now"] = frozen_ok()
    (HERE / "p38_judgment.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))
    return 0


def hashes():
    for p in HASHED:
        print(hashlib.sha256((HERE.parent / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), p)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit({"job": lambda: job(a[1]), "check": lambda: (check(), 0)[1], "judge": judge_all, "hashes": hashes}[a[0]]())
