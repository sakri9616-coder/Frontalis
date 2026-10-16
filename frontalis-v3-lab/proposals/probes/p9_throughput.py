"""P9: throughput of the batched simulator (personality-steps per second) at several population sizes. Dev seeds only.
Config = the P5/P8 learner (separate encoders, multi-channel head, feature-only, normalised targets, self-confirming attribution)."""
import json, sys, time
from pathlib import Path
import sim, p3_diagnose as p3
cfg = dict(norm_targets=1, clip=1.0, separate=1, lr=3e-4, mask_id=1, mc=1, cycle=1, w_mode="cum", aff=5.0, n_steps=300, n_early=120)
worlds = {(s, f): sim.make_world(s, f) for s in p3.SEEDS + list(range(77386, 77390)) for f in p3.FAMS}
base = []
for s in list(range(77380, 77390)):
    for f in p3.FAMS:
        b = dict(world=(s, f), init=s * 7 + 1)
        base += [dict(b, role=r, early=e, env=s * 10 + en, sgd=s * 10 + sg) for r, e, en, sg in
                 (("A", 0, 1, 3), ("A_exact", 0, 1, 3), ("A_env", 0, 2, 3), ("A_sgd", 0, 1, 4), ("A_both", 0, 2, 4), ("B", 1, 1, 3))]
out = {}
for mult in (1, 2, 4, 8):
    pers = [dict(q, init=q["init"] + 1000 * m, sgd=q["sgd"] + 1000 * m) for m in range(mult) for q in base]
    res = sim.run(pers, worlds, cfg)
    out[len(pers)] = dict(secs=res["secs"], steps=300, pers_steps_per_sec=len(pers) * 300 / res["secs"])
    print(len(pers), out[len(pers)], file=sys.stderr)
    del res
Path(__file__).with_name("p9_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out))
