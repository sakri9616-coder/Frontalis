"""Summarize the production-scale dev cells (p15_cells_*.json) with judge4; writes p15_judge.json. Prints per cell the
judgement counts (n_pos of 10, pass) and key descriptives."""
import glob, json
from pathlib import Path
import judge4
HERE = Path(__file__).parent
res = {}
for p in sorted(glob.glob(str(HERE / "p15_cells_*.json"))):
    res.update(json.loads(Path(p).read_text(encoding="utf-8")))
rep = judge4.report(res)
(HERE / "p15_judge.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
J = ["J1-a", "J1-b", "J2-a", "J2-b", "J3", "J4-a", "J4-b", "J5-a", "J5-c_krr", "J5-c_hb", "J5-c_rf", "J4-d_F6m", "J4-d_F5", "repro"]
for k, v in rep.items():
    if k.startswith("_"):
        continue
    d = v["desc"]
    print(k.replace("|prod", "").ljust(22), " ".join(f"{j}:{v[j]['n_pos']}{'+' if v[j]['passed'] else '-'}" for j in J if j in v),
          "| D_AB %.3f D_AA %.3f sw %.3f/%.3f r2nc %.2f r2rm %.2f" % (d["D_AB_un"], d["D_AA_un"], d["swapd_AB"], d["swapd_AA"], d["r2_nc"], d["r2_rm"]))
for k, v in rep["_calibration"].items():
    print(k.replace("|prod", ""), {j: (x["n_pos"], round(x["median"], 3), "ok" if x["ok"] else "NG") for j, x in v.items()})
for k, d in res.items():
    m = d["meta"]; print(k, "P", m["P"], "train", round(m["train_secs"]), "meas", round(m["meas_secs"]), "nonfinite", m["nonfinite"], "pers-steps/s", round(m["pers_steps_per_sec"]))
