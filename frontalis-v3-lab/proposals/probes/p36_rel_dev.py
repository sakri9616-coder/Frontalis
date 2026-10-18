"""P36: NN-REL calibration and reduced production pass (PREREG draft 4 section 10c 6(b)), dev seeds 77380-77389 only.
Idempotent jobs for sched.py (one result file each; an existing file means nothing to do):
  python -B p36_rel_dev.py calib <first>        first in 77380, 77385: NN-REL, I-H, F1, C-off and C-S0, roles A A2 B B2 A3 A4,
                                                5 seeds, production scale, all planted calibration nets -> p36_rel_calib_<first>.json
  python -B p36_rel_dev.py pass <fam> <first>   fam F1 or F6m: NN-REL, I-H, C-off and C-SLp, roles A A2 B B2, 5 seeds
                                                -> p36_rel_pass_<fam>_<first>.json
  python -B p36_rel_dev.py report               (local, no GPU) merge all six files, apply judge4 (contrasts per cell and the
                                                calibration expectations), put the NN-SET dev numbers next to them -> p36_rel_report.json
Learning rate from p16_rel_lr.json (1e-3, P16). Cells, measurement and calibration are those of cells.run_cell (draft-4 code)."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRSTS = (77380, 77385)
ROLES6 = (("A", 0, 1, 3), ("A2", 0, 2, 4), ("B", 1, 1, 3), ("B2", 1, 2, 4), ("A3", 0, 5, 6), ("A4", 0, 7, 8))
JOBS = [("calib", None, f) for f in FIRSTS] + [("pass", fam, f) for fam in ("F1", "F6m") for f in FIRSTS]


SMOKE = bool(os.environ.get("P36_SMOKE"))   # local mechanics check only: 200 experiences, file names end in _smoke


def out_path(kind, fam, first):
    sfx = "_smoke" if SMOKE else ""
    return HERE / (f"p36_rel_calib_{first}{sfx}.json" if kind == "calib" else f"p36_rel_pass_{fam}_{first}{sfx}.json")


def calib15(v):
    """J2 calibration at amplitude 1.5 (PREREG draft 4 rev. b2, section 10d): PL-struct and PL-rand, no DAS (as p25)."""
    import measure4 as m4
    from cells import planted_views
    A = v["A"]
    if "A3" not in v or A.w["fam"] != "F1":
        return {}
    cyc = int(float(A.st["cyc_vec"][A.i]))
    pv = planted_views(v, A.w, A.w["seed"], mult=1.5)
    return {"calib15": {nm: m4.measure_group(pv[nm], A.w, A.w["seed"], "IH", cyc, das=False) for nm in ("PL-struct", "PL-rand")}}


def job(kind, fam, first):
    first = int(first)
    assert first in FIRSTS and kind in ("calib", "pass") and (kind == "calib" or fam in ("F1", "F6m"))
    out = out_path(kind, fam, first)
    if out.exists():
        print("result exists; nothing to do:", out.name)
        return 0
    import numpy as np, torch
    import cells
    from cells import run_cell, ROLES
    if SMOKE:
        cells.SCALE["prod"] = dict(n_steps=200, n_early=80, spn=1)
    seeds = list(range(first, first + 5))
    lr = json.loads((HERE / "p16_rel_lr.json").read_text(encoding="utf-8"))["chosen"]
    tmp = Path(tempfile.mkdtemp()) / "cells.json"
    t0 = time.time()
    if kind == "calib":
        run_cell("rel", "IH", ["C-off", "C-S0"], ("F1",), seeds, ROLES6, "prod", tmp, calib_conds=("C-off", "C-S0"),
                 tag=f"_p36calib_{first}", extra_cfg=dict(lr=lr), log=sys.stdout, extra_fn=calib15)
    else:
        run_cell("rel", "IH", ["C-off", "C-SLp"], (fam,), seeds, ROLES, "prod", tmp, tag=f"_p36pass_{fam}_{first}",
                 extra_cfg=dict(lr=lr), log=sys.stdout)
    cells = json.loads(tmp.read_text(encoding="utf-8"))
    env = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__, numpy=np.__version__,
               commit=subprocess.run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip())
    res = dict(job=dict(kind=kind, fam=fam, seeds=seeds), env=env, wall_secs=time.time() - t0, cells=cells)
    part = out.with_suffix(".part")
    part.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    os.replace(part, out)
    print("wrote", out.name, env, round(res["wall_secs"]), "s")
    return 0


def report():
    import judge4
    files = {out_path(*j).name: out_path(*j) for j in JOBS}
    missing = [n for n, p in files.items() if not p.exists()]
    if missing:
        print("missing:", missing)
        return 2
    merged = {}
    meta = {}
    for n, p in files.items():
        r = json.loads(p.read_text(encoding="utf-8"))
        meta[n] = dict(env=r["env"], wall_secs=r["wall_secs"], nonfinite=[c["meta"]["nonfinite"] for c in r["cells"].values()])
        for k, v in r["cells"].items():
            merged[k] = v
    rep_rel = judge4.report(merged)
    from collections import defaultdict
    c15 = defaultdict(list)
    for v in merged.values():
        for r in v["rows"]:
            for nm, cr in r.get("calib15", {}).items():
                c15[(r["cond"], nm)].append(cr)
    for (cond, nm), rows in c15.items():
        c = judge4.contrasts(rows)
        rep_rel["_calibration"][f"rel|IH|{cond}|F1|prod|{nm}@1.5"] = {j: dict(passed=c[j]["passed"], expected=True, ok=c[j]["passed"], n_pos=c[j]["n_pos"],
                                                                         median=c[j]["median"], floor=c[j]["floor"]) for j in ("J2-a", "J2-b")}
    set_dev = json.loads((HERE / "p15_cells_set_IH.json").read_text(encoding="utf-8"))
    rep_set = judge4.report(set_dev)
    keep = ("J1-a", "J1-b", "J2-a", "J2-b", "J3", "J4-a", "J4-b", "J5-a", "repro")
    def brief(c):
        return {k: (c[k]["n_pos"], c[k]["n"], round(c[k]["median"], 4), round(c[k]["floor"], 4), c[k]["passed"]) for k in keep if k in c and "n_pos" in c[k]}
    out = dict(jobs=meta, rel={k: brief(v) for k, v in rep_rel.items() if not k.startswith("_")},
               rel_calibration=rep_rel["_calibration"],
               set_dev={k: brief(v) for k, v in rep_set.items() if not k.startswith("_") and ("|F1|" in k or "|F6m|" in k)},
               desc={k: {d: v["desc"].get(d) for d in ("r2_un", "r2_nc", "D_AB_un", "D_AA_un")} for k, v in rep_rel.items() if not k.startswith("_")})
    (HERE / "p36_rel_report.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float)[:6000])
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "report":
        sys.exit(report())
    sys.exit(job("calib", None, a[1]) if a[0] == "calib" else job("pass", a[1], a[2]))
