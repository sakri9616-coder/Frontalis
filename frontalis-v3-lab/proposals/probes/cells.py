"""Shared definitions for draft-4 runs: cycle conditions, roles, personality lists, cell runner (train -> measure),
calibration (planted views). Dev seeds / rehearsal seeds only until the protocol is frozen."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import sim4
import measure4 as m4

COND = sim4.COND_CODES
MAIN_COND = ["C-off", "C-S0", "C-H0", "C-SLp", "C-SLh", "C-SLn"]
ROLES = (("A", 0, 1, 3), ("A2", 0, 2, 4), ("B", 1, 1, 3), ("B2", 1, 2, 4))
ROLES_NOISE = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8), ("A_exact", 0, 1, 3), ("A_env", 0, 2, 3), ("A_sgd", 0, 1, 4))
SCALE = {"dev": dict(n_steps=3000, n_early=1200, spn=1), "prod": dict(n_steps=32000, n_early=12800, spn=4),
         "mid": dict(n_steps=8000, n_early=3200, spn=1), "rehearsal": dict(n_steps=2000, n_early=800, spn=1)}


def pers_list(seeds, fams, roles, tag="", conds=None):
    out = []
    for cond in (conds or [None]):
        for s in seeds:
            for f in fams:
                for r, e, en, sg in roles:
                    q = dict(world=(s, f), group=(s, f, cond or tag), role=r, early=e, env=s * 10 + en, sgd=s * 10 + sg, init=s * 7 + 1)
                    if cond:
                        q["cond"] = cond
                    out.append(q)
    return out


def _unit(rng, n):
    v = rng.normal(size=n); return v / np.linalg.norm(v)


def planted_views(vw, w, seed, mult=1.0):
    """Calibration views (PREREG draft 4 section 6). vw has roles A, A2, A3, A4 (real NN views; A3, A4 are further
    independent retests of A). Planted B := A3 + alpha*u*phi, B' := A4 + alpha*u*phi (an independent retest PLUS the planted
    history effect), so that the A-B contrast contains retest noise like a real B (first version used B := A + plant, which
    makes D(A,B) equal D(A,A') by construction; corrected before any calibration result at dev/production scale)."""
    A, A2 = vw["A"], vw["A2"]
    g = _unit(np.random.default_rng([seed, 71]), sim4.KB); g2 = _unit(np.random.default_rng([seed, 72]), sim4.KB)
    Wv = A.p["h.Wv"][0]
    u = Wv @ torch.tensor(g2, dtype=torch.float32, device=sim4.DEV); u = u / u.norm()
    eff = (u @ Wv).abs().sum().item()
    s_tr, p_tr, c_tr = sim4.probe_set(w, "tr")
    om = np.random.default_rng([seed, 73]).normal(size=sim4.D_IN) * 2 / np.sqrt(sim4.D_IN); om[sim4.ID_DIMS] = 0

    def stdz(fn):
        z = fn(s_tr, p_tr, c_tr); mu, sd = z.mean(), z.std() + 1e-9
        return lambda s, p, c: ((fn(s, p, c) - mu) / sd).astype(np.float32)
    phis = {"PL-struct": stdz(lambda s, p, c: sim4.outcome_np(w, s, p, c) @ g),
            "PL-rand": stdz(lambda s, p, c: np.tanh(sim4.rows_np(w, s, p, c) @ om)),
            "PL-global": lambda s, p, c: np.ones(len(s), np.float32)}
    noise = np.abs(m4.cent(A.get("un")["V"]) - m4.cent(A2.get("un")["V"])).sum(1).mean()
    out = {}
    s_un, p_un, c_un = sim4.probe_set(w, "un")
    for name, fn in phis.items():
        ph = fn(s_un, p_un, c_un)
        mag = (np.abs(ph - ph.mean()).mean() if name != "PL-global" else 1.0) * eff
        alpha = float(mult * noise / (mag + 1e-12))   # mult: power curve only (P19); the calibration of PREREG section 6 uses 1
        out[name] = {"A": A, "A2": A2, "B": m4.Planted(vw["A3"], u, fn, alpha), "B2": m4.Planted(vw["A4"], u, fn, alpha)}
    out["PL-noise"] = {"A": A, "A2": A2, "B": vw["A3"], "B2": vw["A4"]}
    # evaluative-state calibration for J3: donor B = A with permuted channel weights (PL-state) or identical (PL-state0)
    # J3: donor B = an independent retest A3 whose channel weights are permuted (a systematic evaluative-state difference on
    # top of retest noise; first version used A itself, which removes the retest noise; corrected after the 2-seed smoke test)
    perm = np.random.default_rng([seed, 74]).permutation(sim4.KB)
    A3 = vw["A3"]
    Bs = m4.View.__new__(m4.View); Bs.__dict__.update({k: v for k, v in A3.__dict__.items() if k != "cache"}); Bs.cache = {}
    Bs.state = dict(A3.state); Bs.state["w"] = A3.state["w"][torch.tensor(perm, device=sim4.DEV)]
    out["PL-state"] = {"A": A, "A2": A2, "B": Bs, "B2": vw["B2"]}
    out["PL-state0"] = {"A": A, "A2": A2, "B": A3, "B2": vw["B2"]}
    return out


def run_cell(learner, innate, conds, fams, seeds, roles, scale, out_path, calib_conds=(), tag="", extra_cfg=None, remap=True,
             ckpt_dir=None, log=sys.stderr, resume=True, extra_fn=None):
    """One batched run: one learner and innate condition, several cycle conditions (per personality), families, seeds."""
    worlds = {(s, f): sim4.make_world(s, f) for s in seeds for f in fams}
    pers = pers_list(seeds, fams, roles, tag, conds)
    cfg = dict(learner=learner, innate=innate, **SCALE[scale], **(extra_cfg or {}))
    name = f"{learner}_{innate}_{'+'.join(conds)}_{scale}{tag}"
    resumed = False
    if ckpt_dir:
        # the checkpoint file name carries a hash of the configuration and the personality list, so that a stale file of
        # another configuration is never resumed; if a file exists (an interrupted run), resume from it (bit-identical: P13)
        import hashlib
        key = hashlib.sha1(json.dumps(dict(cfg=cfg, pers=pers), sort_keys=True, default=str).encode()).hexdigest()[:10]
        os.makedirs(ckpt_dir, exist_ok=True)
        cfg.update(ckpt_every=4000, ckpt_path=os.path.join(ckpt_dir, f"{name}_{key}.pt"))
    t0 = time.time()
    state = None
    if ckpt_dir and resume and os.path.exists(cfg["ckpt_path"]):
        state = sim4.load_state(cfg["ckpt_path"], pers, worlds, cfg); state["phase"] = "main"; resumed = True
        print(f"{name}: resuming from {cfg['ckpt_path']}", file=log); log.flush()
    st = sim4.run(pers, worlds, cfg, state=state)
    t_train = time.time() - t0
    vw = m4.views(st, pers, worlds)
    rows = []
    for gk, v in vw.items():
        s, f, cond = gk
        cyc = COND[cond]["cycle"]
        r = m4.measure_group(v, worlds[(s, f)], s, innate, cyc)
        r.update(learner=learner, innate=innate, cond=cond, seed=s, fam=f, scale=scale, tag=tag)
        if extra_fn is not None:   # extra descriptive quantities per group (P30: path individuality)
            r.update(extra_fn(v))
        if cond in calib_conds and f == "F1" and v["A"].nn and "A3" in v:
            r["calib"] = {name_: m4.measure_group(pv, worlds[(s, f)], s, innate, cyc) for name_, pv in planted_views(v, worlds[(s, f)], s).items()}
        rows.append(r)
    t_meas = time.time() - t0 - t_train
    if remap:   # J5-b (reported): 400 more experiences in the channel-remapped world; a fresh learner on the same 400
        rcfg = dict(cfg, phase="remap", n_steps=400, n_early=0, ckpt_every=0)
        st2 = sim4.run(pers, worlds, rcfg, state=st)
        vw2 = m4.views(st2, pers, worlds)
        fresh = sim4.run(pers, worlds, rcfg)
        vwf = m4.views(fresh, pers, worlds)
        for r, gk in zip(rows, vw.keys()):
            r["r2_rm"] = m4.own_r2(vw2[gk]["A"], "un", remap=True); r["r2_rm_fresh"] = m4.own_r2(vwf[gk]["A"], "un", remap=True)
        del st2, fresh
    meta = dict(learner=learner, innate=innate, conds=list(conds), scale=scale, P=len(pers), train_secs=t_train, meas_secs=t_meas, resumed=resumed,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__, host=os.environ.get("COMPUTERNAME") or os.uname().nodename,
                nonfinite=st["nonfinite"], pers_steps_per_sec=len(pers) * cfg["n_steps"] / max(st["secs"], 1e-9),
                kappa_final=st["kappa"].cpu().numpy().round(4).tolist(), pers=[dict(role=q["role"], cond=q.get("cond"), seed=q["world"][0], fam=q["world"][1]) for q in pers])
    d = json.loads(Path(out_path).read_text(encoding="utf-8")) if Path(out_path).exists() else {}
    d[name] = dict(meta=meta, rows=rows)
    Path(out_path).write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    print(f"{name} P={len(pers)} train {t_train:.0f}s meas {t_meas:.0f}s nonfinite {st['nonfinite']}", file=log)
    log.flush()
    del st
    torch.cuda.empty_cache()
    return rows
