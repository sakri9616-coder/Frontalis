"""P25: the dev runs that the MINIMUM main path needs before freezing (DESIGN draft 4 section 14, PREREG 10d). Dev seeds
77380-77389 only.
  calib_slp   calibration (all planted nets, incl. PL-state for J3) in C-SLp, the cycle condition of the minimum path, at
              production scale: resumes NN-SET / I-H / C-SLp from the 28,000-experience checkpoint of the P15 cell (resume is
              bit-identical), measures F1 with planted_views, judges against EXPECT (PL-state: J3 expected to pass in C-SLp).
              Writes p25_calib_slp.json.
  nn0         NN-0 (frozen encoder, same architecture as NN-SET) at production scale, I-H, C-off and C-SLp, F1 only, roles
              A A2 B B2 A3 A4: the matched control for "the features were acquired" and the J4 null. Writes
              p15_cells_nn0_IH.json (picked up by p15_summarize.py / judge4 J5-c_nn0).
Run: python -B p25_min_dev.py <calib_slp|nn0>"""
import glob, json, os, sys, tempfile
from pathlib import Path
import sim4, measure4 as m4, judge4
from cells import COND, ROLES, SCALE, pers_list, planted_views, run_cell

HERE = Path(__file__).parent
SEEDS = list(range(77380, 77390))
ROLES6 = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))
LR = lambda: json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]


def calib_slp():
    fams = ("F1", "F2", "F6m", "F5")          # the P15 cell's personality list (needed to resume its checkpoint)
    worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in fams}
    pers = pers_list(SEEDS, fams, ROLES6, "", ["C-SLp"])
    cfg = dict(learner="set", innate="IH", **SCALE["prod"], lr=LR())
    ck = max(glob.glob(os.path.join(tempfile.gettempdir(), "v4d_ckpt_p15", "set_IH_C-SLp_prod*.pt")), key=os.path.getmtime)
    st = sim4.load_state(ck, pers, worlds, cfg); st["phase"] = "main"
    st = sim4.run(pers, worlds, cfg, state=st)
    vw = m4.views(st, pers, worlds)
    rows = {}
    for (s, f, cond), v in vw.items():
        if f != "F1":
            continue
        for name, pv in planted_views(v, worlds[(s, f)], s).items():
            rows.setdefault(name, []).append(m4.measure_group(pv, worlds[(s, f)], s, "IH", COND[cond]["cycle"], das=False))
    out = {}
    for name, rs in rows.items():
        c = judge4.contrasts(rs)
        exp = judge4.EXPECT.get(name, {})
        out[name] = {j: dict(passed=c[j]["passed"], expected=e, ok=c[j]["passed"] == e, n_pos=c[j]["n_pos"], median=c[j]["median"],
                             floor=c[j]["floor"]) for j, e in exp.items() if j in c}
    out["_hash"] = sim4.state_hash(st); out["_ckpt"] = os.path.basename(ck)
    (HERE / "p25_calib_slp.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for n, d in out.items():
        if not n.startswith("_"):
            print(n, {j: (x["n_pos"], round(x["median"], 3), round(x["floor"], 3), "ok" if x["ok"] else "NG") for j, x in d.items()})


def nn0():
    ps = json.loads((HERE / "p16_pscale.json").read_text(encoding="utf-8"))
    run_cell("nn0", "IH", ["C-off", "C-SLp"], ("F1",), SEEDS, ROLES6, "prod", HERE / "p15_cells_nn0_IH.json",
             extra_cfg=dict(lr=LR(), partner_scale=ps.get("nn0", 0.0)), log=sys.stderr,
             ckpt_dir=os.path.join(tempfile.gettempdir(), "v4d_ckpt_p15"))


if __name__ == "__main__":
    {"calib_slp": calib_slp, "nn0": nn0}[sys.argv[1]]()
