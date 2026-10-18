"""P19: power curve of the distance contrasts at production scale (DESIGN draft 4 section 8 stage 0; diagnostic).
Motivation: at production scale (P15, set IH C-off) the planted nets at signal-to-noise 1 sit at the effect-size floor
(PL-struct J2-a median 0.146 vs floor 0.151), while at rehearsal scale they were well above it. This measures, on the SAME
trained nets (dev seeds 77380-77389, resumed from the 28,000-experience checkpoint of p15 'set IH C-off'; resume is
bit-identical), how the pass/fail of J1-a, J2-a, J2-b, J4-a, J4-b depends on the planted amplitude
(alpha multiplier m in {0.5, 1, 1.5, 2, 3}). No criterion, floor or expectation is changed by this probe.
Run: python -B p19_power.py"""
import json, os, tempfile
from pathlib import Path
import numpy as np
import sim4, measure4 as m4, judge4
from cells import COND, ROLES, SCALE, pers_list, planted_views

HERE = Path(__file__).parent
SEEDS = list(range(77380, 77390))
FAMS = ("F1", "F2", "F6m", "F5")
MULTS = (0.5, 1.0, 1.5, 2.0, 3.0)


def main():
    roles = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))
    lr = json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in FAMS}
    pers = pers_list(SEEDS, FAMS, roles, "", ["C-off"])
    cfg = dict(learner="set", innate="IH", **SCALE["prod"], lr=lr)
    import glob   # the first run used the unhashed name; later runs (cells.run_cell with resume) add a config hash
    ck = max(glob.glob(os.path.join(tempfile.gettempdir(), "v4d_ckpt_p15", "set_IH_C-off_prod*.pt")), key=os.path.getmtime)
    st = sim4.load_state(ck, pers, worlds, cfg); st["phase"] = "main"
    st = sim4.run(pers, worlds, cfg, state=st)
    vw = m4.views(st, pers, worlds)
    out = {}
    for m in MULTS:
        rows = {}
        for gk, v in vw.items():
            s, f, cond = gk
            if f != "F1":
                continue
            for name, pv in planted_views(v, worlds[(s, f)], s, mult=m).items():
                if name.startswith("PL-state"):
                    continue
                rows.setdefault(name, []).append(m4.measure_group(pv, worlds[(s, f)], s, "IH", COND[cond]["cycle"], das=False))
        out[str(m)] = {name: {j: dict(n_pos=c["n_pos"], median=c["median"], floor=c["floor"], passed=c["passed"])
                              for j, c in judge4.contrasts(rs).items() if j in ("J1-a", "J2-a", "J2-b", "J4-a", "J4-b", "repro")}
                       for name, rs in rows.items()}
        print(m, {n: {j: (x["n_pos"], round(x["median"], 3), round(x["floor"], 3)) for j, x in d.items()} for n, d in out[str(m)].items()}, flush=True)
    out["_hash"] = sim4.state_hash(st)
    (HERE / "p19_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
