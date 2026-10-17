"""Checkpoint/resume inside the early (history) phase: stop at 300 of n_early 800, resume via cells.run_cell's path
(load_state + phase 'main'), compare the state hash with an uninterrupted run. Rehearsal seed 77340 only. Writes p13_ckpt_early.json."""
import json, os, tempfile
from pathlib import Path
import sim4
from cells import COND, ROLES, pers_list
HERE = Path(__file__).parent
res = {}
for learner in ("set", "krr"):
    worlds = {(77340, "F1"): sim4.make_world(77340, "F1")}
    pers = pers_list([77340], ("F1",), ROLES, conds=["C-off", "C-SLp"])
    cfg = dict(learner=learner, innate="IH", n_steps=1000, n_early=800, spn=2, lr=1e-3, partner_scale=0.0)
    h_full = sim4.state_hash(sim4.run(pers, worlds, cfg))
    tmp = os.path.join(tempfile.gettempdir(), f"p13e_{learner}.pt")
    sim4.save_state(sim4.run(pers, worlds, dict(cfg, stop_at=300)), tmp)
    st = sim4.load_state(tmp, pers, worlds, cfg); st["phase"] = "main"
    h_res = sim4.state_hash(sim4.run(pers, worlds, cfg, state=st)); os.remove(tmp)
    res[learner] = dict(identical=h_full == h_res, full=h_full, resumed=h_res)
    print(learner, res[learner])
(HERE / "p13_ckpt_early.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
