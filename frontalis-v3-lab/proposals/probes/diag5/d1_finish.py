"""Cause investigation (dev seeds 77380-77389 only): resume the NN-SET / I-H / C-SLp production cell of P15 from its
28,000-experience checkpoint to 32,000 (the same state p25 calib_slp measured) and save it to a scratch path, so that the
diagnostics can be iterated without retraining. Frozen code is imported, not modified."""
import glob, json, os, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import sim4
from cells import ROLES, SCALE, pers_list

SEEDS = list(range(77380, 77390))
ROLES6 = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))
FAMS = ("F1", "F2", "F6m", "F5")


def setup(cond="C-SLp"):
    worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in FAMS}
    pers = pers_list(SEEDS, FAMS, ROLES6, "", [cond])
    lr = json.loads((HERE.parent / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    cfg = dict(learner="set", innate="IH", **SCALE["prod"], lr=lr)
    return worlds, pers, cfg


if __name__ == "__main__":
    out = sys.argv[1]
    worlds, pers, cfg = setup()
    ck = os.path.join(tempfile.gettempdir(), "v4d_ckpt_p15", "set_IH_C-SLp_prod.pt")
    st = sim4.load_state(ck, pers, worlds, cfg); st["phase"] = "main"
    st = sim4.run(pers, worlds, cfg, state=st)
    print("hash", sim4.state_hash(st), "t", st["t"], "secs", st["secs"])
    sim4.save_state(st, out)
