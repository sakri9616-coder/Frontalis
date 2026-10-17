"""P34: local dev-seed work during the v5 evaluation (dev seeds 77380-77389 only; never 77100-77139 or 77300-77339).
  j3v5base   J3v5 (frozen measure5) on the saved NN-SET / I-H / C-SLp production state WITHOUT wgate (P15 cell resumed to
             32,000, hash 967538036a35f9e8): how much of a J3v5 effect exists without the mechanism change. Needed to read V1.
             Arg: path of the saved state (diag5/d1_finish.py output). Writes p34_j3v5base.json.
  relgate    NN-REL learnability gate of PREREG draft 4 section 10c 6(a): dev seeds 10, production scale, F1 F6m F2, C-off,
             role A, I-H, lr 1e-3 (p16_rel_lr.json). Pass iff the median own-prediction R2 on P_un in F1 >= 0.5.
             Checkpoints every 2,000 experiences in %TEMP%/v4d_ckpt_p34 and resumes from them (bit-identical, P13).
             Writes p34_relgate.json.
Run: python -B p34_local_dev.py j3v5base <state.pt> | relgate"""
import json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim4, measure4 as m4
from cells import ROLES, SCALE, pers_list, planted_views

SEEDS = list(range(77380, 77390))


def j3v5base(path):
    from measure5 import j3v5
    from d1_finish import setup
    worlds, pers, cfg = setup()
    st = sim4.load_state(path, pers, worlds, cfg)
    st["cfg"] = {**sim4.DEFAULT, **cfg}; st["wkeys"] = sorted({q["world"] for q in pers}); st["wid"] = None
    h = sim4.state_hash(st); assert h == "967538036a35f9e8", h
    rows = []
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        if f != "F1":
            continue
        w = worlds[(s, f)]; pv = planted_views(vw, w, s)
        r = dict(seed=s)
        for nm, v in (("real", vw), ("PL-state", pv["PL-state"]), ("PL-state0", pv["PL-state0"])):
            ab, aa = j3v5(v, w, s)
            r[nm] = dict(AB=ab, AA=aa, diff=ab - aa)
        rows.append(r)
    summ = {nm: dict(n_pos=int(sum(r[nm]["diff"] > 0 for r in rows)), median=float(np.median([r[nm]["diff"] for r in rows])),
                     AA_median=float(np.median([r[nm]["AA"] for r in rows])), max=float(max(r[nm]["diff"] for r in rows)))
            for nm in ("real", "PL-state", "PL-state0")}
    (HERE / "p34_j3v5base.json").write_text(json.dumps(dict(state_hash=h, summary=summ, rows=rows), indent=1, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def relgate():
    lr = json.loads((HERE / "p16_rel_lr.json").read_text(encoding="utf-8"))["chosen"]
    fams = ("F1", "F6m", "F2")
    worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in fams}
    pers = pers_list(SEEDS, fams, (ROLES[0],), "", ["C-off"])
    ck_dir = Path(tempfile.gettempdir()) / "v4d_ckpt_p34"; ck_dir.mkdir(exist_ok=True)
    ck = ck_dir / "rel_gate_IH_C-off_prod.pt"
    cfg = dict(learner="rel", innate="IH", **SCALE["prod"], lr=lr, ckpt_every=2000, ckpt_path=str(ck))
    state, resumed = None, False
    if ck.exists():
        state = sim4.load_state(str(ck), pers, worlds, cfg); state["phase"] = "main"; resumed = True
        print("resuming at", state["t"], flush=True)
    t0 = time.time()
    st = sim4.run(pers, worlds, cfg, state=state)
    rows = [dict(seed=s, fam=f, r2_un=m4.own_r2(vw["A"], "un"), r2_nc=m4.own_r2(vw["A"], "nc")) for (s, f, c), vw in m4.views(st, pers, worlds).items()]
    med = {f: float(np.median([r["r2_un"] for r in rows if r["fam"] == f])) for f in fams}
    out = dict(rule="median R2_un(F1) >= 0.5 (PREREG draft 4 section 10c 6(a))", passed=med["F1"] >= 0.5, median_r2_un=med,
               median_r2_nc={f: float(np.median([r["r2_nc"] for r in rows if r["fam"] == f])) for f in fams},
               lr=lr, secs_this_run=time.time() - t0, resumed=resumed, nonfinite=st["nonfinite"],
               device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__, rows=rows)
    (HERE / "p34_relgate.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("passed", "median_r2_un", "median_r2_nc", "secs_this_run")}), flush=True)


if __name__ == "__main__":
    {"j3v5base": lambda: j3v5base(sys.argv[2]), "relgate": relgate}[sys.argv[1]]()
