"""P30: the ONE evaluation of the frozen minimum main path (PREREG draft 4 section 10d, rev. b2). Evaluation seeds
77100-77139, used once. Run only after the freeze (FREEZE.md) and the leader's go-ahead; results are reported to the leader
first and not published.

  python -B p30_eval_main.py train     runs the cells (resumable from checkpoints), writes p30_cells.json
  python -B p30_eval_main.py judge     applies the frozen rules, writes p30_judgment.json

Cells (all I-H, production scale N = 32,000, S/N = 4, lr from p15_lr.json, partner scales from p16_pscale.json):
  NN-SET : F1 and F6m x C-off and C-SLp x roles A A2 B B2 A_env A_sgd x 40 seeds (run per family: 2 runs of 480)
  KRR    : F1 x C-off and C-SLp x roles A A2 B B2 x 40 seeds
  NN-0   : the same as KRR
Judgement (judge4 contrasts; 30 of 40 seeds in the predicted direction and median >= floor):
  usable(J): every calibration expectation (PREREG section 6) that involves J is met on the dev seeds, in C-off (P15 cell,
             m = 1; for J2 the PL-struct / PL-rand expectations at m = 1.5 from P19) and in C-SLp (p25_calib_slp.json, J2 at
             m = 1.5). A contrast that is not usable is not used, and every claim that requires it is not made (no re-measuring
             after the freeze).
  L2 (F1, C-SLp): J1-a, J1-b, J2-a, J2-b, J3, J4-a, J4-b, J5-a, J5-c_krr, J5-c_nn0 all passed and all usable.
  L3: L2 and J4-d_F6m passed.
  Null conditions on the evaluation seeds: J3 passing in C-off, or J4 passing for NN-0, makes that contrast unusable.
  Reported, not judged: path individuality D(A,A'), D(A,A_env), D(A,A_sgd), D(A,B)/D(A,A_sgd) per group (P_un, as J1-a)."""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import sim4, measure4 as m4, judge4
from cells import ROLES, run_cell, pers_list, SCALE, COND

HERE = Path(__file__).parent
SEEDS = list(range(77100, 77140))
CONDS = ["C-off", "C-SLp"]
ROLES_SET = ROLES + (("A_env", 0, 2, 3), ("A_sgd", 0, 1, 4))
OUT = HERE / "p30_cells.json"
CK = os.path.join(tempfile.gettempdir(), "v4d_ckpt_p30")
LOG = open(HERE / "p30_stderr.txt", "a", encoding="utf-8")
L2_REQ = ("J1-a", "J1-b", "J2-a", "J2-b", "J3", "J4-a", "J4-b", "J5-a", "J5-c_krr", "J5-c_nn0")


def path_individuality(v):
    """Reported, not judged: distances from A to its retests on P_un (the J1-a distance)."""
    V = {r: v[r].get("un")["V"] for r in ("A", "A2", "B", "B2", "A_env", "A_sgd")}
    sc = np.mean([np.abs(V[r]).mean() for r in ("A", "A2", "B", "B2")]) * m4.KB + 1e-12
    d = {x: m4.dist(V["A"], V[x], sc) for x in ("A2", "B", "A_env", "A_sgd")}
    return dict(pi_D_A_A2=d["A2"], pi_D_A_B=d["B"], pi_D_A_Aenv=d["A_env"], pi_D_A_Asgd=d["A_sgd"],
                pi_ratio_B_over_sgd=d["B"] / max(d["A_sgd"], 1e-12))


def train():
    lr = json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    ps = json.loads((HERE / "p16_pscale.json").read_text(encoding="utf-8"))
    for fam in ("F1", "F6m"):
        run_cell("set", "IH", CONDS, (fam,), SEEDS, ROLES_SET, "prod", OUT, tag=f"_eval_{fam}", extra_cfg=dict(lr=lr), log=LOG,
                 ckpt_dir=CK, extra_fn=path_individuality)
    for L in ("krr", "nn0"):
        run_cell(L, "IH", CONDS, ("F1",), SEEDS, ROLES, "prod", OUT, tag="_eval", extra_cfg=dict(lr=lr, partner_scale=ps.get(L, 0.0)),
                 log=LOG, ckpt_dir=CK)


def usable():
    cal = json.loads((HERE / "p30_calib_coff.json").read_text(encoding="utf-8"))   # frozen copy of the P15 C-off calibration
    p19 = json.loads((HERE / "p19_results.json").read_text(encoding="utf-8"))["1.5"]
    p25 = json.loads((HERE / "p25_calib_slp.json").read_text(encoding="utf-8"))
    ok = {}
    def mark(j, good):
        ok[j] = ok.get(j, True) and bool(good)
    for name in ("PL-struct", "PL-rand", "PL-global", "PL-noise", "PL-state", "PL-state0"):
        for j, x in cal.get(f"set|IH|C-off|F1|prod|{name}", {}).items():
            if name in ("PL-struct", "PL-rand") and j.startswith("J2"):
                mark(j, p19[name][j]["passed"] is True)          # expected pass at m = 1.5
            else:
                mark(j, x["ok"])
        for j, x in p25.get(name, {}).items():
            if not (name in ("PL-struct", "PL-rand") and j.startswith("J2")):
                mark(j, x["ok"])
    for name in ("PL-struct", "PL-rand"):
        for j, x in p25.get(name + "@1.5", {}).items():
            if j.startswith("J2"):
                mark(j, x["ok"])
    return ok


def judge():
    res = json.loads(OUT.read_text(encoding="utf-8"))
    rep = judge4.report(res)
    use = usable()
    key = "set|IH|C-SLp|F1|prod"
    cell = rep.get(key, {})
    l2 = {j: dict(passed=bool(cell.get(j, {}).get("passed", False)), usable=use.get(j.split("_")[0], True) if not j.startswith("J5-c") else True)
          for j in L2_REQ}
    # null conditions on the evaluation seeds (PREREG section 6): a contrast that passes where it must fail is not used
    if rep.get("set|IH|C-off|F1|prod", {}).get("J3", {}).get("passed"):
        l2["J3"]["usable"] = False
    for j in ("J4-a", "J4-b"):
        if rep.get("nn0|IH|C-SLp|F1|prod", {}).get(j, {}).get("passed") or rep.get("nn0|IH|C-off|F1|prod", {}).get(j, {}).get("passed"):
            l2[j]["usable"] = False
    L2 = all(v["passed"] and v["usable"] for v in l2.values())
    L3 = L2 and bool(cell.get("J4-d_F6m", {}).get("passed", False))
    out = dict(L2=L2, L3=L3, L2_requirements=l2, usable=use, cells=rep)
    (HERE / "p30_judgment.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(dict(L2=L2, L3=L3, requirements=l2), indent=1))


if __name__ == "__main__":
    {"train": train, "judge": judge}[sys.argv[1]]()
