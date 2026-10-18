"""P32: dev-seed runs for the v5 preregistration (dev seeds 77380-77389 only; production scale; family F1; I-H).
  setwg <tag>    NN-SET with wgate (sim5), roles A A2 B B2 A3 A4, C-off and C-SLp. Every draft-4 quantity (no DAS), J3v5
                 (measure5: donor w only, body state carried), all planted calibration nets (m = 1; PL-struct / PL-rand also
                 m = 1.5), and the C-SLp time course of w at 4,000 / 12,800 / 20,000 / 32,000 experiences.
  krrself <tag>  KRR with krr_self_b (the NN-SET soft-min attribution rule, in every condition), role A, C-off and C-SLp.
  krrcheck       platform check of the frozen draft-4 KRR path (two dev seeds).
Writes p32_<job>_<tag>.json here (the cloud runners copy back top-level files only), with machine and git commit.
<tag> names the machine (modal = the evaluation image; kaggle = cross-check).
2026-10-18 rev. (freeze-gate reviews A2-A4, C2): J3v5 carries the donor's w only and carries the receiver's body state;
krr_self_b uses the NN's rule; results record the commit. Earlier outputs p32_setwg.json / p32_krrself.json used the first
version (kept as history)."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim5, sim4, measure4 as m4, judge4
from measure5 import j3v5, w_course, TPTS
from cells import ROLES, SCALE, COND, pers_list, planted_views

SC = os.environ.get("P32_SCALE", "prod")            # smoke tests only: P32_SCALE=rehearsal P32_NSEEDS=2
SEEDS = list(range(77380, 77390))[:int(os.environ.get("P32_NSEEDS", "10"))]
KEYS = ("D_AB_un", "D_AA_un", "D_AB_nc", "D_AA_nc", "Drho_AB_un", "Drho_AA_un", "Drho_AB_nc", "Drho_AA_nc", "swapd_AB", "swapd_AA",
        "xp_other_un", "xp_same_un", "xp_other_nc", "xp_same_nc", "repro_un", "r2_un", "r2_nc")


def env():
    return dict(device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
                cuda=torch.version.cuda, numpy=np.__version__, python=sys.version.split()[0], scale=SC,
                commit=subprocess.run("git rev-parse --short HEAD", shell=True, cwd=HERE, capture_output=True, text=True).stdout.strip())


def out_name(job, tag):
    return HERE / f"p32_{job}_{tag}{'' if SC == 'prod' else '_' + SC}.json"


def setwg(tag):
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
        r["v5_swapd_AB"], r["v5_swapd_AA"] = j3v5(vw, w, s)
        if cond == "C-SLp":
            tr = {ro: w_course(st["buf"]["DB"][idx[((s, f), ro, cond)]].cpu().numpy()) for ro in ("A", "A2", "B")}
            r["w_L1_AB"] = {t: float(np.abs(tr["A"][t] - tr["B"][t]).sum()) for t in tr["A"]}
            r["w_L1_AA2"] = {t: float(np.abs(tr["A"][t] - tr["A2"][t]).sum()) for t in tr["A"]}
        rows.append(r)
        pvs = {**planted_views(vw, w, s), **{k + "@1.5": v for k, v in planted_views(vw, w, s, mult=1.5).items() if k in ("PL-struct", "PL-rand")}}
        for nm, pv in pvs.items():
            gp = m4.measure_group(pv, w, s, "IH", cyc, das=False)
            cr = dict(seed=s, **{k: gp[k] for k in KEYS})
            if nm.startswith("PL-state"):
                cr["v5_swapd_AB"], cr["v5_swapd_AA"] = j3v5(pv, w, s)
            calib.setdefault(f"{cond}|{nm}", []).append(cr)
    summ = {}
    for cond in ("C-off", "C-SLp"):
        rs = [r for r in rows if r["cond"] == cond]
        c = judge4.contrasts(rs)
        summ[cond] = {j: dict(n_pos=c[j]["n_pos"], median=c[j]["median"], floor=c[j]["floor"], passed=c[j]["passed"]) for j in ("J1-a", "J2-a", "J2-b", "J3", "J4-a", "J4-b", "J5-a")}
        summ[cond]["r2_nc_median"] = float(np.median([r["r2_nc"] for r in rs]))
        summ[cond]["J3v5_diffs"] = [r["v5_swapd_AB"] - r["v5_swapd_AA"] for r in rs]
        summ[cond]["J3v5_AA"] = [r["v5_swapd_AA"] for r in rs]
        if cond == "C-SLp":
            summ[cond]["w_L1_AB_median"] = {t: float(np.median([r["w_L1_AB"][t] for r in rs])) for t in rs[0]["w_L1_AB"]}
            summ[cond]["w_L1_AA2_median"] = {t: float(np.median([r["w_L1_AA2"][t] for r in rs])) for t in rs[0]["w_L1_AA2"]}
    cal = {}
    for k, rs in calib.items():
        c = judge4.contrasts(rs)
        cal[k] = {j: dict(n_pos=c[j]["n_pos"], median=c[j]["median"], floor=c[j]["floor"], passed=c[j]["passed"]) for j in ("J1-a", "J2-a", "J2-b", "J3", "J4-a", "J4-b", "repro")}
        if "v5_swapd_AB" in rs[0]:
            cal[k]["J3v5_diffs"] = [r["v5_swapd_AB"] - r["v5_swapd_AA"] for r in rs]
            cal[k]["J3v5_AA"] = [r["v5_swapd_AA"] for r in rs]
    meta = dict(train_secs=t_train, total_secs=time.time() - t0, nonfinite=st["nonfinite"], hash=sim4.state_hash(st), **env())
    out_name("setwg", tag).write_text(json.dumps(dict(meta=meta, summary=summ, calibration=cal, rows=rows, calib_rows=calib), indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta), flush=True)


def krrself(tag):
    ps = json.loads((HERE / "p16_pscale.json").read_text(encoding="utf-8"))
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in SEEDS}
    pers = pers_list(SEEDS, ("F1",), (("A", 0, 1, 3),), "", ["C-off", "C-SLp"])
    cfg = dict(learner="krr", innate="IH", **SCALE[SC], partner_scale=ps.get("krr", 1.0), krr_self_b=1)
    t0 = time.time()
    st = sim5.run(pers, worlds, cfg)
    rows = [dict(seed=s, cond=cond, r2_tr=m4.own_r2(vw["A"], "tr"), r2_un=m4.own_r2(vw["A"], "un"), r2_nc=m4.own_r2(vw["A"], "nc"))
            for (s, f, cond), vw in m4.views(st, pers, worlds).items()]
    meta = dict(secs=time.time() - t0, **env())
    out_name("krrself", tag).write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta), flush=True)


def krrcheck():
    """Platform check, not a v5 quantity: the frozen draft-4 KRR path (sim4, partner scale 0, C-SLp, role A) on two dev
    seeds at production scale (Colab's torch cu130 build failed in magma_spotrs_batched on 2026-10-18)."""
    ps = json.loads((HERE / "p16_pscale.json").read_text(encoding="utf-8"))
    worlds = {(s, "F1"): sim4.make_world(s, "F1") for s in SEEDS[:2]}
    pers = pers_list(SEEDS[:2], ("F1",), (("A", 0, 1, 3),), "", ["C-SLp"])
    t0 = time.time()
    st = sim4.run(pers, worlds, dict(learner="krr", innate="IH", **SCALE[SC], partner_scale=ps.get("krr", 1.0)))
    rows = [dict(seed=s, r2_nc=m4.own_r2(vw["A"], "nc")) for (s, f, cond), vw in m4.views(st, pers, worlds).items()]
    meta = dict(secs=time.time() - t0, **env())
    (HERE / "p32_krrcheck.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta), rows, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"setwg": lambda: setwg(a[1]), "krrself": lambda: krrself(a[1]), "krrcheck": krrcheck}[a[0]]()
