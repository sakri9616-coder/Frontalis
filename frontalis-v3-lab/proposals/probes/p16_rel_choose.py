"""Choose NN-REL's learning rate by the PREREG draft 4 section 9 rule (median own-prediction R2 on P_un in F1, C-off, role A;
no judgement quantity), from the 8,000-experience runs gate_rel_lr1e-4_mid, gate_rel_mid (3e-4), gate_rel_lr1e-3_mid.
Deviation recorded in LOG-dev: chosen at 8,000 experiences (S/N 4), not at production scale (32,000), because one NN-REL
production-scale run per learning rate takes about 10 h on the shared GPU. Writes p16_rel_lr.json."""
import json
from pathlib import Path

HERE = Path(__file__).parent
RUNS = {1e-4: "gate_rel_lr1e-4_mid", 3e-4: "gate_rel_mid", 1e-3: "gate_rel_lr1e-3_mid"}
res = {}
for lr, name in RUNS.items():
    p = HERE / f"p16_results_{name}.json"
    if p.exists():
        res[lr] = json.loads(p.read_text(encoding="utf-8"))[name]["traj"][-1]["r2_un"]["F1"]
best = max(res, key=res.get)
(HERE / "p16_rel_lr.json").write_text(json.dumps(dict(chosen=best, r2_un_F1_at_8000=res, scale="mid (8000, S/N 4)"), indent=1), encoding="utf-8")
print(best, res)
