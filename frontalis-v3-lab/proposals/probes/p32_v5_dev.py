"""P32: dev-seed runs for the v5 preregistration draft (dev seeds 77380-77389 only; production scale; family F1; I-H).
  setwg    NN-SET with the candidate mechanism wgate (sim5), roles A A2 B B2 A3 A4, C-off and C-SLp. Measures every draft-4
           quantity (no DAS), the corrected J3 (J3v5: evaluative state only, no value-net carry-over, donor w also gates the
           receiver's plasticity), all planted calibration nets (m = 1 and, for PL-struct / PL-rand, m = 1.5), and the time
           course of the evaluative state at the fixed points 4,000 / 12,800 / 20,000 / 32,000 experiences.
  krrself  KRR with the matched attribution rule (krr_self_b: its own likelihood posterior in every condition), role A,
           C-off and C-SLp: the extrapolation control of v5.
Writes p32_<job>.json in this directory (the cloud runners copy back top-level files only).
Run: python -B p32_v5_dev.py <setwg|krrself>"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim5, sim4, measure4 as m4, judge4
from cells import ROLES, SCALE, COND, pers_list, planted_views
from d3_exp import j3v5
from d2_diag import body_hinge_frac

SC = os.environ.get("P32_SCALE", "prod")            # smoke tests only: P32_SCALE=rehearsal P32_NSEEDS=2
SEEDS = list(range(77380, 77390))[:int(os.environ.get("P32_NSEEDS", "10"))]
TPTS = tuple(t for t in (4000, 12800, 20000, 32000) if t <= SCALE[SC]["n_steps"]) or (1000,)
KEYS = ("D_AB_un", "D_AA_un", "D_AB_nc", "D_AA_nc", "Drho_AB_un", "Drho_AA_un", "Drho_AB_nc", "Drho_AA_nc", "swapd_AB", "swapd_AA",
        "xp_other_un", "xp_same_un", "xp_other_nc", "xp_same_nc", "repro_un", "r2_un", "r2_nc")


def setwg():
    lr = json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    R = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in SEEDS}
    pers = pers_list(SEEDS, ("F1",), R, "", ["C-off", "C-SLp"])
    cfg = dict(learner="set", innate="IH", **SCALE[SC], lr=lr, wgate=1)
    t0 = time.time()
    st = sim5.run(pers, worlds, cfg)
    t_train = time.time() - t0
    idx = {(q["world"], q["role"], q["cond"]): i for i, q in enumerate(pers)}
    rows, calib = [], {}
    for (s, f, cond), vw in m4.views(st, pers, worlds).items():
        w = worlds[(s, f)]; cyc = COND[cond]["cycle"]
        g = m4.measure_group(vw, w, s, "IH", cyc, das=False)
        r = dict(seed=s, cond=cond, **{k: g[k] for k in KEYS})
        r["v5_swapd_AB"], r["v5_swapd_AA"] = j3v5(vw, w, s, True)
        tr = {}
        for ro in ("A", "A2", "B"):
            tr[ro] = body_hinge_frac(st["buf"]["DB"][idx[((s, f), ro, cond)]].cpu().numpy())[2]
        r["w_L1_AB"] = {t: float(np.abs(tr["A"][t] - tr["B"][t]).sum()) for t in TPTS}
        r["w_L1_AA2"] = {t: float(np.abs(tr["A"][t] - tr["A2"][t]).sum()) for t in TPTS}
        rows.append(r)
        pvs = {**planted_views(vw, w, s), **{k + "@1.5": v for k, v in planted_views(vw, w, s, mult=1.5).items() if k in ("PL-struct", "PL-rand")}}
        for nm, pv in pvs.items():
            gp = m4.measure_group(pv, w, s, "IH", cyc, das=False)
            cr = dict(seed=s, **{k: gp[k] for k in KEYS})
            if nm.startswith("PL-state"):
                cr["v5_swapd_AB"], cr["v5_swapd_AA"] = j3v5(pv, w, s, True)
            calib.setdefault(f"{cond}|{nm}", []).append(cr)
    summ = {}
    for cond in ("C-off", "C-SLp"):
        rs = [r for r in rows if r["cond"] == cond]
        c = judge4.contrasts(rs)
        summ[cond] = {j: dict(n_pos=c[j]["n_pos"], median=c[j]["median"], floor=c[j]["floor"], passed=c[j]["passed"]) for j in ("J1-a", "J2-a", "J2-b", "J3", "J4-a", "J4-b", "J5-a")}
        summ[cond]["r2_nc_median"] = float(np.median([r["r2_nc"] for r in rs]))
        summ[cond]["J3v5_diffs"] = [r["v5_swapd_AB"] - r["v5_swapd_AA"] for r in rs]
        summ[cond]["J3v5_AA"] = [r["v5_swapd_AA"] for r in rs]
        summ[cond]["w_L1_AB_median"] = {t: float(np.median([r["w_L1_AB"][t] for r in rs])) for t in TPTS}
        summ[cond]["w_L1_AA2_median"] = {t: float(np.median([r["w_L1_AA2"][t] for r in rs])) for t in TPTS}
    cal = {}
    for k, rs in calib.items():
        c = judge4.contrasts(rs)
        cal[k] = {j: dict(n_pos=c[j]["n_pos"], median=c[j]["median"], floor=c[j]["floor"], passed=c[j]["passed"]) for j in ("J1-a", "J2-a", "J2-b", "J3", "J4-a", "J4-b", "repro")}
        if "v5_swapd_AB" in rs[0]:
            cal[k]["J3v5_diffs"] = [r["v5_swapd_AB"] - r["v5_swapd_AA"] for r in rs]
            cal[k]["J3v5_AA"] = [r["v5_swapd_AA"] for r in rs]
    meta = dict(train_secs=t_train, total_secs=time.time() - t0, nonfinite=st["nonfinite"], hash=sim4.state_hash(st),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    (HERE / f"p32_setwg{'' if SC == 'prod' else '_' + SC}.json").write_text(json.dumps(dict(meta=meta, summary=summ, calibration=cal, rows=rows, calib_rows=calib), indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta), flush=True)


def krrself():
    ps = json.loads((HERE / "p16_pscale.json").read_text(encoding="utf-8"))
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in SEEDS}
    pers = pers_list(SEEDS, ("F1",), (("A", 0, 1, 3),), "", ["C-off", "C-SLp"])
    cfg = dict(learner="krr", innate="IH", **SCALE[SC], partner_scale=ps.get("krr", 1.0), krr_self_b=1)
    t0 = time.time()
    st = sim5.run(pers, worlds, cfg)
    rows = [dict(seed=s, cond=cond, r2_tr=m4.own_r2(vw["A"], "tr"), r2_un=m4.own_r2(vw["A"], "un"), r2_nc=m4.own_r2(vw["A"], "nc"))
            for (s, f, cond), vw in m4.views(st, pers, worlds).items()]
    meta = dict(secs=time.time() - t0, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__)
    (HERE / f"p32_krrself{'' if SC == 'prod' else '_' + SC}.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    {"setwg": setwg, "krrself": krrself}[sys.argv[1]]()
