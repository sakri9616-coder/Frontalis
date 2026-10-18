"""P3: cause analysis of the P2 failures (dev seeds 77380-77385 only). Each config is one batched GPU run.

Roles per world (seed x family): A, A_exact (bit-identical rerun: determinism), A_env (only environment noise differs),
A_sgd (only minibatch order differs), A_both (both differ), B (other history class, same env/sgd streams as A).
Configs vary ONE mechanism at a time relative to the P2 baseline. Judgement criteria (PREREG draft 2) are not touched.
Run: PYTHONUTF8=1 python -B p3_diagnose.py [config names...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import sim
from measure import evaluate, summarize

SEEDS = list(range(77380, 77386))
FAMS = ["F1", "F6", "F5"]
CONFIGS = {
    "base_on": dict(cycle=1),                                  # = P2 (spn 1, ema 0.01, no norm, shared encoder)
    "base_off": dict(cycle=0),
    "norm_on": dict(cycle=1, norm_targets=1, clip=1.0),        # fix of the non-finite failure
    "norm_off": dict(cycle=0, norm_targets=1, clip=1.0),
    "spn4_on": dict(cycle=1, norm_targets=1, clip=1.0, spn=4),
    "spn4_off": dict(cycle=0, norm_targets=1, clip=1.0, spn=4),
    "slow_on": dict(cycle=1, norm_targets=1, clip=1.0, w_rate=0.001),
    "two_on": dict(cycle=1, norm_targets=1, clip=1.0, w_mode="two"),
    "cum_on": dict(cycle=1, norm_targets=1, clip=1.0, w_mode="cum"),
    "sep_on": dict(cycle=1, norm_targets=1, clip=1.0, separate=1),
    "sep_off": dict(cycle=0, norm_targets=1, clip=1.0, separate=1),
}
KEYS = ["D_AB", "D_exact", "D_env", "D_sgd", "D_both", "own_r2", "gap_from_repr", "diff_pr", "diff_top4", "repr_pr_B",
        "diff_on_readout", "close_AB_k4", "close_Aboth_k4", "close_rand_k4", "close_AB_k16", "close_Aboth_k16", "close_rand_k16",
        "inter_AB", "inter_Aboth", "nl_cos_AB", "nl_cos_Aboth", "nl_finite"]


def personalities():
    pers = []
    for s in SEEDS:
        for f in FAMS:
            base = dict(world=(s, f), init=s * 7 + 1)
            E0, E1, S0, S1 = s * 10 + 1, s * 10 + 2, s * 10 + 3, s * 10 + 4
            pers += [dict(base, role="A", early=0, env=E0, sgd=S0), dict(base, role="A_exact", early=0, env=E0, sgd=S0),
                     dict(base, role="A_env", early=0, env=E1, sgd=S0), dict(base, role="A_sgd", early=0, env=E0, sgd=S1),
                     dict(base, role="A_both", early=0, env=E1, sgd=S1), dict(base, role="B", early=1, env=E0, sgd=S0)]
    return pers


def main(names):
    worlds = {(s, f): sim.make_world(s, f) for s in SEEDS for f in FAMS}
    pers = personalities()
    out_path = Path(__file__).resolve().parent / "p3_results.json"
    allres = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for name in names:
        t0 = time.time()
        res = sim.run(pers, worlds, CONFIGS[name])
        rows = evaluate(res, pers, worlds)
        summ = summarize(rows, KEYS)
        allres[name] = dict(cfg=CONFIGS[name], train_secs=res["secs"], nonfinite_steps=res["nonfinite"], P=len(pers),
                            summary=summ, rows=[{**r, "world": list(r["world"])} for r in rows])
        out_path.write_text(json.dumps(allres, indent=1), encoding="utf-8")
        print(name, f"train {res['secs']:.0f}s total {time.time()-t0:.0f}s nonfinite {res['nonfinite']}", file=sys.stderr)
        for f, d in summ.items():
            print(json.dumps({"cfg": name, "fam": f, **d}))
        sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:] or list(CONFIGS))
