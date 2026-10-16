"""P16: why does NN-SET not learn F1 at production scale (P15 lr step: median R2 on P_un < 0 for lr 1e-4, 3e-4, 1e-3)?
DESIGN draft 4 section 8 stage 2 (can the world be learned?). Criterion: own-prediction R2 only (no judgement quantity).
Dev seeds 77380-77389, F1 (and F6m), C-off, role A. Variants are run one per process: python -B p16_nn_learnability.py <name>"""
import json, sys
from pathlib import Path
import numpy as np
import sim4, measure4 as m4
from cells import COND, pers_list

HERE = Path(__file__).parent
SEEDS = list(range(77380, 77390))
VARIANTS = {
    "base_mid": dict(learner="set", n_steps=8000, n_early=0, spn=4),
    "nopartner_mid": dict(learner="set", n_steps=8000, n_early=0, spn=4, mask_partner=1),
    "wd1e-2_mid": dict(learner="set", n_steps=8000, n_early=0, spn=4, wd=1e-2),
    "nent320_mid": dict(learner="set", n_steps=8000, n_early=0, spn=4, n_tr=320),
    "nent1280_mid": dict(learner="set", n_steps=8000, n_early=0, spn=4, n_tr=1280),
    "krr_mid": dict(learner="krr", n_steps=8000, n_early=0, spn=1),
    "rf_mid": dict(learner="rf", n_steps=8000, n_early=0, spn=4),
    "rel_mid": dict(learner="rel", n_steps=8000, n_early=0, spn=4),
    # rev. a (after the partner finding): partner gate initialised at 0 in NN learners; asymmetric F2
    "gate_mid": dict(learner="set", n_steps=8000, n_early=0, spn=4),
    "gate_rel_mid": dict(learner="rel", n_steps=8000, n_early=0, spn=4),
    "krr_p0_mid": dict(learner="krr", n_steps=8000, n_early=0, spn=1, partner_scale=0.0),
    "krr_p05_mid": dict(learner="krr", n_steps=8000, n_early=0, spn=1, partner_scale=0.5),
    "rf_p0_mid": dict(learner="rf", n_steps=8000, n_early=0, spn=4, partner_scale=0.0),
}


def main(name):
    v = dict(VARIANTS[name]); n_tr = v.pop("n_tr", 80)
    fams = ("F1", "F6m", "F2") if name in ("gate_mid", "gate_rel_mid", "krr_p0_mid", "krr_p05_mid", "rf_p0_mid") else ("F1", "F6m")
    worlds = {(s, f): sim4.make_world(s, f, n_tr) for s in SEEDS for f in fams}
    pers = pers_list(SEEDS, fams, (("A", -1, 1, 3),))
    traj = []
    st = None
    chunks = [v["n_steps"] // 4 * (i + 1) for i in range(4)]
    for stop in chunks:   # R2 along training (same run, resumed)
        st = sim4.run(pers, worlds, dict(innate="IH", **COND["C-off"], **v, stop_at=stop), state=st)
        vw = m4.views(st, pers, worlds)
        r = {f: float(np.median([m4.own_r2(g["A"], "un") for k, g in vw.items() if k[1] == f])) for f in fams}
        rtr = {f: float(np.median([m4.own_r2(g["A"], "tr") for k, g in vw.items() if k[1] == f])) for f in fams}
        traj.append(dict(step=stop, r2_un=r, r2_tr=rtr))
        print(name, stop, r, rtr, file=sys.stderr, flush=True)
    out = HERE / "p16_results.json"
    d = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    d[name] = dict(cfg=VARIANTS[name], traj=traj, secs=st["secs"])
    out.write_text(json.dumps(d, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
