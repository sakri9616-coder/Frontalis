"""P15: production-scale dry run on DEV seeds 77380-77389 (PREREG draft 4 section 9) + calibration (section 6).
Step 1 (lr): NN-SET, I-H, F1, C-off, role A only; lr in {1e-4, 3e-4, 1e-3}; choose by median own-prediction R2 on P_un
         (no judgement quantity is used).
Step 2 (cells): learner x innate x condition, families F1 F2 F6m F5, roles A A2 B B2 (+ A3 A4 in F1 for calibration and the
         noise roles), N = 32,000, S/N = 4 for NN; ridge learners see each experience once (exact fit on all history).
Run: python -B p15_prod.py lr | cells <learner> <innate> <cond,...> | all"""
import json, sys
from pathlib import Path
import numpy as np
import sim4
import measure4 as m4
from cells import COND, ROLES, ROLES_NOISE, run_cell, pers_list, SCALE

HERE = Path(__file__).parent
SEEDS = list(range(77380, 77390))
LOG = open(HERE / "p15_stderr.txt", "a", encoding="utf-8")


def step_lr(lr):
    out_p = HERE / f"p15_lr_{lr:g}.json"
    if out_p.exists():
        return
    worlds = {(s, "F1"): sim4.make_world(s, "F1") for s in SEEDS}
    pers = pers_list(SEEDS, ("F1",), (("A", 0, 1, 3),))
    st = sim4.run(pers, worlds, dict(learner="set", innate="IH", **COND["C-off"], **SCALE["prod"], lr=lr))
    vw = m4.views(st, pers, worlds)
    r2 = [m4.own_r2(v["A"], "un") for v in vw.values()]
    res = dict(lr=lr, r2_un_median=float(np.median(r2)), r2_un=r2, r2_nc=[m4.own_r2(v["A"], "nc") for v in vw.values()],
               secs=st["secs"], nonfinite=st["nonfinite"])
    out_p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("lr", lr, res["r2_un_median"], f"{st['secs']:.0f}s", file=LOG); LOG.flush()


def choose_lr():
    res = [json.loads((HERE / f"p15_lr_{lr:g}.json").read_text(encoding="utf-8")) for lr in (1e-4, 3e-4, 1e-3)]
    best = max(res, key=lambda r: r["r2_un_median"])["lr"]
    (HERE / "p15_lr.json").write_text(json.dumps(dict(chosen=best, runs=res), indent=1), encoding="utf-8")
    return best


def chosen_lr():
    p = HERE / "p15_lr.json"
    return json.loads(p.read_text(encoding="utf-8"))["chosen"] if p.exists() else 3e-4


def step_cells(learner, innate, conds, fams=("F1", "F2", "F6m", "F5"), calib_conds=("C-off", "C-S0")):
    out_p = HERE / "p15_cells.json"
    roles = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))
    run_cell(learner, innate, conds, fams, SEEDS, roles, "prod", out_p, calib_conds=calib_conds if learner in ("set", "rel") else (),
             extra_cfg=dict(lr=chosen_lr()), log=LOG)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "lr":
        step_lr(float(a[1]))
    elif a[0] == "choose":
        print(choose_lr())
    elif a[0] == "cells":
        step_cells(a[1], a[2], a[3].split(","))
