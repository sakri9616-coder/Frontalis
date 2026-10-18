"""P18: where does the retest distance D(A, A') come from? (DESIGN draft 4 section 8 stage 3; diagnostic, no criterion uses it)
Rehearsal seeds 77340-77342. Roles: A (env 1, sgd 3), A2 = A' (env 2, sgd 4), B (other history class, env 1, sgd 3),
A_exact (= A, determinism check), A_env (env 2, sgd 3: only the experience sampling differs), A_sgd (env 1, sgd 4: only
the minibatch order differs), A3 (env 5, sgd 6). Init is shared by all roles of a seed (pers_list: init = seed*7+1).
Distances: the same per-channel L1 distance on P_un as J1-a (measure4.dist with the group scale), uncentred.
Run: python -B p18_noise_decomp.py <learner> <scale>"""
import json, sys
from pathlib import Path
import numpy as np
import sim4, measure4 as m4
from cells import COND, ROLES_NOISE, pers_list, SCALE

HERE = Path(__file__).parent
SEEDS = [77340, 77341, 77342]
FAMS = ("F1", "F6m")
CONDS = ["C-off", "C-SLp"]


def main(learner, scale):
    ps = json.loads((HERE / "p16_pscale.json").read_text(encoding="utf-8"))
    lr = json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    extra = dict(lr=lr) if learner in ("set", "rel", "pre") else dict(lr=lr, partner_scale=ps.get(learner, 1.0))
    worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in FAMS}
    pers = pers_list(SEEDS, FAMS, ROLES_NOISE, conds=CONDS)
    st = sim4.run(pers, worlds, dict(learner=learner, innate="IH", **SCALE[scale], **extra))
    vw = m4.views(st, pers, worlds)
    rows = []
    for gk, v in vw.items():
        s, f, cond = gk
        V = {r: v[r].get("un")["V"] for r in v}
        scale_ = np.mean([np.abs(V[r]).mean() for r in ("A", "A2", "B", "B2")]) * m4.KB + 1e-12
        r = dict(seed=s, fam=f, cond=cond, **{f"D_A_{x}": m4.dist(V["A"], V[x], scale_) for x in ("A2", "B", "A_exact", "A_env", "A_sgd", "A3")})
        r["r2_un_A"] = m4.own_r2(v["A"], "un")
        rows.append(r)
    summ = {}
    for f in FAMS:
        for c in CONDS:
            sub = [r for r in rows if r["fam"] == f and r["cond"] == c]
            summ[f"{f}|{c}"] = {k: float(np.median([r[k] for r in sub])) for k in sub[0] if k.startswith(("D_", "r2"))}
    out = HERE / "p18_results.json"
    d = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    d[f"{learner}_{scale}"] = dict(rows=rows, median=summ, secs=st["secs"], nonfinite=st["nonfinite"], P=len(pers))
    out.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    print(learner, scale, json.dumps(summ, indent=0, default=lambda x: round(x, 3)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
