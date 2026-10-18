"""P31: one independent, idempotent job of the frozen evaluation (FREEZE.md; PREREG draft 4 rev. b2, section 10d), for the
scheduler (sched/sched.py, with --allow-eval and --only modal). It runs exactly the cell that p30_eval_main.py train runs,
for 5 consecutive evaluation seeds, and writes ONE result file:  p31_eval_<learner>_<family>_<first seed>.json

  python -B p31_eval_job.py <set|krr|nn0> <F1|F6m> <first seed>      first seed in 77100, 77105, ..., 77135 (F6m: set only)

- Idempotent: if the result file already exists, the job does nothing (exit 0). A failed or interrupted job leaves no result
  file and is simply rerun. A finished result is final: it is never recomputed or replaced (no second draw of a seed).
- Before running, it checks the SHA-256 of every frozen file against FREEZE.md (line endings normalised to LF, as committed);
  a mismatch aborts the job (exit 3) without running anything.
- The cell, roles, scale, learning rate, partner scales, measurement and path-individuality quantities are those of
  p30_eval_main.py (imported, not copied). The only difference from `p30_eval_main.py train` is the batching: 5 seeds per run
  instead of 40 (the personalities' random streams are per personality, so the same computation is done; floating-point
  summation over a different batch size can differ in the last bits, which P18/P21 show grows to retest size at most).
- The result file records the machine (GPU, torch, CUDA, numpy, Python), the git commit and the hash check, so that the merge
  (p31_eval_merge.py) can refuse a mixture of machines."""
import hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).parent
FIRSTS = list(range(77100, 77140, 5))


def frozen_ok():
    text = (HERE.parent / "FREEZE.md").read_text(encoding="utf-8")
    block = text.split("```")[1]
    bad = []
    for line in block.strip().splitlines():
        sha, path = line.split(None, 1)
        b = (HERE.parent / path.strip()).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(b).hexdigest() != sha:
            bad.append(path.strip())
    return bad


def main(learner, fam, first):
    first = int(first)
    assert learner in ("set", "krr", "nn0") and fam in ("F1", "F6m") and first in FIRSTS, "bad job"
    assert not (fam == "F6m" and learner != "set"), "F6m is NN-SET only (PREREG 10d)"
    out = HERE / f"p31_eval_{learner}_{fam}_{first}.json"
    if out.exists():
        print("result exists; nothing to do:", out.name)
        return 0
    bad = frozen_ok()
    if bad:
        print("FROZEN FILES DIFFER:", bad)
        return 3
    import torch, numpy as np
    import p30_eval_main as P
    from cells import run_cell, ROLES
    seeds = list(range(first, first + 5))
    lr = json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    ps = json.loads((HERE / "p16_pscale.json").read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp()) / "cells.json"
    t0 = time.time()
    if learner == "set":
        run_cell("set", "IH", P.CONDS, (fam,), seeds, P.ROLES_SET, "prod", tmp, tag=f"_eval_{fam}_{first}", extra_cfg=dict(lr=lr),
                 log=sys.stdout, extra_fn=P.path_individuality)
    else:
        run_cell(learner, "IH", P.CONDS, ("F1",), seeds, ROLES, "prod", tmp, tag=f"_eval_{first}",
                 extra_cfg=dict(lr=lr, partner_scale=ps.get(learner, 0.0)), log=sys.stdout)
    cells = json.loads(tmp.read_text(encoding="utf-8"))
    commit = subprocess.run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip()
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
               cuda=torch.version.cuda, numpy=np.__version__, python=sys.version.split()[0])
    res = dict(job=dict(learner=learner, fam=fam, seeds=seeds), env=env, commit=commit, frozen_check="ok",
               wall_secs=time.time() - t0, cells=cells)
    part = out.with_suffix(".part")
    part.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    os.replace(part, out)             # the result file appears only when complete
    print("wrote", out.name, env, round(res["wall_secs"]), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
