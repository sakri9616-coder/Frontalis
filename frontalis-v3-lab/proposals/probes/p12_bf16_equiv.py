"""P12: dry run of the bf16 equivalence procedure (PREREG s10b, fixed in commit dbf1b93 BEFORE this run) on the
draft-3 learner (MLP, sim.py), dev seeds 77380-77389, families F1/F5/F6, cycle off and kappa = +5.
This is a PROCEDURE check: the binding test is on the NN-REL learner in Stage 1. H2' quantities are not implemented yet;
the metrics available in measure.py stand in for them (listed in METRICS).
Runs: fp32, bf16 (amp=1, training only; measurement fp32), fp32' (env and sgd seeds of every role shifted: same-condition replicate)."""
import json, sys, time
from pathlib import Path
import numpy as np
import sim
from measure import evaluate
import p8_das as p8

METRICS = ["D_AB", "D_both", "own_r2", "inter_AB", "inter_Aboth", "nl_cos_AB", "nl_cos_Aboth",
           "das_close_AB", "das_close_Aboth", "das_close_shuf", "close_AB_k4", "close_Aboth_k4"]
CONFIGS = {"kappa5_on": p8.CONFIGS["D_MCA_on"], "off": p8.CONFIGS["D_MC_off"]}


def shifted(pers):
    return [dict(q, env=q["env"] + 500000, sgd=q["sgd"] + 500000) for q in pers]


def main():
    worlds = {(s, f): sim.make_world(s, f) for s in p8.SEEDS for f in p8.FAMS}
    pers = p8.personalities()
    out = {}
    for cname, cfg in CONFIGS.items():
        rows = {}
        for run, (pp, amp) in {"fp32": (pers, 0), "bf16": (pers, 1), "fp32p": (shifted(pers), 0)}.items():
            res = sim.run(pp, worlds, {**cfg, "amp": amp})
            rows[run] = {tuple(r["world"]): r for r in evaluate(res, pp, worlds, das=True)}
            rows[run]["_meta"] = dict(secs=res["secs"], nonfinite=res["nonfinite"])
            print(cname, run, f"{res['secs']:.0f}s nonfinite {res['nonfinite']}", file=sys.stderr)
        wk = [k for k in rows["fp32"] if k != "_meta"]
        res_m = {}
        for m in METRICS:
            d = np.array([abs(rows["bf16"][k][m] - rows["fp32"][k][m]) for k in wk])
            nu = np.array([abs(rows["fp32p"][k][m] - rows["fp32"][k][m]) for k in wk])
            md, p90, mn = float(np.nanmedian(d)), float(np.nanpercentile(d, 90)), float(np.nanmedian(nu))
            res_m[m] = dict(delta_median=md, delta_p90=p90, nu_median=mn, E1=md <= 0.2 * mn, E2=p90 <= 0.5 * mn)
        e3 = rows["bf16"]["_meta"]["nonfinite"] == 0
        e4 = float(max(rows["bf16"][k]["D_exact"] for k in wk))
        out[cname] = dict(metrics=res_m, E3_nonfinite_zero=e3, E4_bf16_exact_rerun_maxD=e4,
                          PASS=bool(all(v["E1"] and v["E2"] for v in res_m.values()) and e3 and e4 == 0.0),
                          secs={r: rows[r]["_meta"]["secs"] for r in rows})
        print(json.dumps({cname: out[cname]}, indent=1))
        sys.stdout.flush()
    Path(__file__).with_name("p12_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
