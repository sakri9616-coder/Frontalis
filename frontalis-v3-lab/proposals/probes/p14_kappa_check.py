"""P14: does the kappa-L criterion decide the sign? (PREREG draft 4 section 7; dev seeds 77380-77389.)
F0a (outcome independent of the actor: attribution has no correct answer) and F1; 4 roles x 10 seeds; NN-SET; I-H and I0;
criteria Lp (predictive likelihood), Lh (evolution strategy on the innate signal, self choice), Ln (random walk);
Lp also with the uncalibrated fixed temperature 0.2 (draft 3). N = 32,000 experiences, S/N = 1 (kappa updates are per
experience; S/N only changes the number of SGD steps)."""
import json, sys
from pathlib import Path
import numpy as np
import sim4
from cells import COND, ROLES, pers_list

OUT = Path(__file__).with_name("p14_results.json")
SEEDS = list(range(77380, 77390))
RUNS = {"main_IH": dict(temp="cal"), "main_I0": dict(temp="cal"), "temp0.2": dict(temp=0.2)}


def main(n_steps=32000):
    fams = ("F0a", "F1")
    worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in fams}
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for run, extra in RUNS.items():
        if run in res:
            continue
        combos = [(run[-2:], c) for c in ("C-SLp", "C-SLh", "C-SLn")] if run.startswith("main") else [("IH", "C-SLp")]
        pers = []
        for innate, cond in combos:
            for q in pers_list(SEEDS, fams, ROLES, conds=[cond]):
                q["innate"] = innate; q["group"] = q["group"] + (innate,); pers.append(q)
        st = sim4.run(pers, worlds, dict(learner="set", innate="IH", n_steps=n_steps, n_early=int(0.4 * n_steps), spn=1, **extra))
        k = st["kappa"].cpu().numpy()
        out = {}
        for innate, cond in combos:
            for f in fams:
                sel = np.array([q["world"][1] == f and q["innate"] == innate and q["cond"] == cond for q in pers])
                d = dict(frac_pos=float((k[sel] > 0).mean()), median=float(np.median(k[sel])), q10=float(np.quantile(k[sel], 0.1)), q90=float(np.quantile(k[sel], 0.9)))
                for cls in (0, 1):
                    sc = sel & np.array([q["early"] == cls for q in pers])
                    d[f"class{cls}_median"] = float(np.median(k[sc])); d[f"class{cls}_frac_pos"] = float((k[sc] > 0).mean())
                out[f"{innate}|{cond}|{f}"] = d
            out[f"{innate}|{cond}|sign_decided_by_criterion"] = bool(not (0.2 <= out[f"{innate}|{cond}|F0a"]["frac_pos"] <= 0.8))
        out["secs"] = st["secs"]; out["nonfinite"] = st["nonfinite"]; out["P"] = len(pers)
        res[run] = out
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(run, json.dumps(out), file=sys.stderr); sys.stderr.flush()
        del st


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32000)
