"""P31 merge: combine the per-job result files of the frozen evaluation into p30_cells.json and judge with the frozen
p30_eval_main.py judge. Runs locally on the returned files (no GPU).

  python -B p31_eval_merge.py check     completeness and machine report only (no judgement)
  python -B p31_eval_merge.py judge     check, then write p30_cells.json and run the frozen judgement -> p30_judgment.json

Rules (fixed with the job split, before any evaluation job ran):
- Complete = every expected job (set x {F1, F6m} x 8 seed groups, krr x F1 x 8, nn0 x F1 x 8) has exactly one result file, and
  each file holds exactly its 5 seeds x 2 cycle conditions. Otherwise: no judgement.
- One place = all files report the same GPU name, torch, CUDA, numpy and Python version and the same git commit, and every
  file passed the frozen-hash check. Otherwise: no judgement; report to the leader (offload_note.md sections 1 and 4).
- Results are reported to the leader first and are not committed or pushed."""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
FIRSTS = list(range(77100, 77140, 5))
EXPECTED = [("set", f, s) for f in ("F1", "F6m") for s in FIRSTS] + [(L, "F1", s) for L in ("krr", "nn0") for s in FIRSTS]


def check():
    missing, files, envs, commits, problems = [], {}, set(), set(), []
    for L, f, s in EXPECTED:
        p = HERE / f"p31_eval_{L}_{f}_{s}.json"
        if not p.exists():
            missing.append(p.name); continue
        r = json.loads(p.read_text(encoding="utf-8"))
        files[p.name] = r
        envs.add(json.dumps(r["env"], sort_keys=True)); commits.add(r["commit"])
        if r.get("frozen_check") != "ok":
            problems.append(f"{p.name}: frozen check {r.get('frozen_check')}")
        got = sorted({(row["seed"], row["cond"]) for c in r["cells"].values() for row in c["rows"]})
        want = sorted((x, c) for x in range(s, s + 5) for c in ("C-off", "C-SLp"))
        if got != want:
            problems.append(f"{p.name}: seeds/conditions {got} != {want}")
    extra = sorted(q.name for q in HERE.glob("p31_eval_*.json") if q.name not in {f"p31_eval_{L}_{f}_{s}.json" for L, f, s in EXPECTED})
    rep = dict(n_expected=len(EXPECTED), n_found=len(files), missing=missing, unexpected=extra, machines=sorted(envs),
               commits=sorted(commits), problems=problems,
               complete=not missing and not extra and not problems, one_place=len(envs) == 1 and len(commits) == 1)
    print(json.dumps(rep, indent=1))
    return rep, files


def judge():
    rep, files = check()
    if not (rep["complete"] and rep["one_place"]):
        print("NOT JUDGED: incomplete or mixed machines; report to the leader.")
        return 2
    merged = {}
    for r in files.values():
        for k, v in r["cells"].items():
            assert k not in merged, k
            merged[k] = v
    (HERE / "p30_cells.json").write_text(json.dumps(merged, indent=1, default=float), encoding="utf-8")
    return subprocess.run([sys.executable, "-B", "p30_eval_main.py", "judge"], cwd=HERE).returncode


if __name__ == "__main__":
    sys.exit(0 if sys.argv[1] == "check" and check() else judge())
