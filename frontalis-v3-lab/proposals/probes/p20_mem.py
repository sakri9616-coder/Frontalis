"""P20: GPU memory per personality at the production configuration (N = 32,000, S/N = 4), for the offload plan
(proposals/saturn_offload.md). Two small populations (P = 24 and 48: dev seeds 77380-77381, F1 F6m, 6 roles x 1-2 conds), a
few steps each; peak allocated memory -> fixed part + per-personality slope. Dev seeds only. Writes p20_mem.json.
The GPU is shared while this runs, so only this process's own allocations (torch.cuda.max_memory_allocated) are reported."""
import json, sys
from pathlib import Path
import torch
import sim4
from cells import ROLES, SCALE, pers_list

HERE = Path(__file__).parent
ROLES6 = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))


def peak(learner, conds, steps):
    worlds = {(s, f): sim4.make_world(s, f) for s in (77380, 77381) for f in ("F1", "F6m")}
    pers = pers_list([77380, 77381], ("F1", "F6m"), ROLES6, conds=conds)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    st = sim4.run(pers, worlds, dict(learner=learner, innate="IH", **SCALE["prod"], lr=1e-3, partner_scale=0.0, stop_at=steps))
    pk = torch.cuda.max_memory_allocated() - base
    del st; torch.cuda.empty_cache()
    return len(pers), pk / 2**20


res = {}
for learner in sys.argv[1:] or ("set", "rel", "krr", "rf", "hb"):
    a = peak(learner, ["C-off"], 60); b = peak(learner, ["C-off", "C-SLp"], 60)
    slope = (b[1] - a[1]) / (b[0] - a[0]); fixed = a[1] - slope * a[0]
    res[learner] = dict(P_small=a[0], MiB_small=a[1], P_large=b[0], MiB_large=b[1], MiB_per_pers=slope, MiB_fixed=fixed)
    print(learner, res[learner], flush=True)
(HERE / "p20_mem.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
