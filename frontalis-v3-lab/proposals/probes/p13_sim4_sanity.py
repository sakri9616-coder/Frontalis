"""P13: sanity of the draft-4 simulator (dev seeds 77380-77389 only).
(1) world statistics per family (effect spread, hinge firing rate on a random sequence),
(2) learnability by learner and family (dev scale: 3000 experiences, S/N 1, C-off, I-H, role A): R2 on P_un and P_nc,
(3) determinism (A vs A_exact) and checkpoint/resume bit-identity,
(4) throughput per learner.
Run: python -B p13_sim4_sanity.py [parts...]"""
import json, sys, time, os, tempfile
from pathlib import Path
import numpy as np
import torch
import sim4
import measure4 as m4

OUT = Path(__file__).with_name("p13_results.json")
SEEDS = list(range(77380, 77390))


def save(key, val):
    d = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1), encoding="utf-8")


def part_world():
    res = {}
    for f in ("F1", "F2", "F6m", "F6", "F5", "F0a"):
        sd, fr = [], []
        for s in SEEDS:
            w = sim4.make_world(s, f)
            rng = np.random.default_rng([s, 99])
            pool = np.concatenate([w["tr0"], w["tr1"]])
            e1 = rng.choice(pool, 3000); e2 = rng.choice(pool, 3000); c = rng.integers(0, 12, 3000)
            o = sim4.outcome_np(w, e1, e2, c) + 0.1 * rng.normal(size=(3000, sim4.KB))
            b = np.zeros(sim4.KB); hits = 0
            for t in range(3000):
                b = sim4.DECAY * b + o[t]; hits += (np.abs(b) > sim4.TOL).mean()
            sd.append(o.std(0).mean()); fr.append(hits / 3000)
        res[f] = dict(outcome_sd=float(np.mean(sd)), hinge_rate=float(np.mean(fr)))
        print(f, res[f], file=sys.stderr)
    save("world", res)


def pers_list(seeds, fams, roles):
    out = []
    for s in seeds:
        for f in fams:
            for r, e, en, sg in roles:
                out.append(dict(world=(s, f), group=(s, f), role=r, early=e, env=s * 10 + en, sgd=s * 10 + sg, init=s * 7 + 1))
    return out


def part_learn():
    fams = ("F1", "F2", "F6m", "F5")
    worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS[:4] for f in fams}
    pers = pers_list(SEEDS[:4], fams, (("A", -1, 1, 3),))
    res = {}
    for L in ("set", "rel", "rf", "nn0", "krr", "hb", "fixk", "lin"):
        st = sim4.run(pers, worlds, dict(learner=L, cycle=0, innate="IH", choice="none", n_steps=3000, n_early=0))
        vw = m4.views(st, pers, worlds)
        rr = {}
        for g, v in vw.items():
            rr.setdefault(g[1], []).append((m4.own_r2(v["A"], "un"), m4.own_r2(v["A"], "nc")))
        res[L] = {f: dict(r2_un=float(np.median([a for a, _ in x])), r2_nc=float(np.median([b for _, b in x]))) for f, x in rr.items()}
        res[L]["secs"] = st["secs"]; res[L]["nonfinite"] = st["nonfinite"]
        print(L, json.dumps(res[L]), file=sys.stderr)
        del st
        torch.cuda.empty_cache()
    save("learn_dev_scale", res)


def part_ckpt():
    worlds = {(s, "F1"): sim4.make_world(s, "F1") for s in SEEDS[:2]}
    roles = (("A", 0, 1, 3), ("A_exact", 0, 1, 3), ("A2", 0, 2, 4), ("B", 1, 1, 3), ("B2", 1, 2, 4))
    pers = pers_list(SEEDS[:2], ("F1",), roles)
    res = {}
    for L in ("set", "rel", "krr"):
        cfg = dict(learner=L, innate="IH", pw="S", kmode="Lp", n_steps=800, n_early=320)
        full = sim4.run(pers, worlds, cfg)
        h_full = sim4.state_hash(full)
        ex = [float((full["net"].value_vec if hasattr(full["net"], "p") else None) is None)]
        tmp = os.path.join(tempfile.gettempdir(), f"p13_ckpt_{L}.pt")
        part = sim4.run(pers, worlds, dict(cfg, stop_at=400))
        sim4.save_state(part, tmp)
        del part
        st = sim4.load_state(tmp, pers, worlds, cfg)
        st["phase"] = "main"
        resumed = sim4.run(pers, worlds, cfg, state=st)
        h_res = sim4.state_hash(resumed)
        vw = m4.views(full, pers, worlds)
        dA = [float(np.abs(v["A"].get("un")["V"] - v["A_exact"].get("un")["V"]).max()) for v in vw.values()]
        res[L] = dict(hash_full=h_full, hash_resumed=h_res, identical=h_full == h_res, A_vs_Aexact_maxdiff=max(dA))
        print(L, res[L], file=sys.stderr)
        os.remove(tmp)
    save("ckpt_determinism", res)


def part_speed():
    res = {}
    for L in ("set", "rel", "krr", "hb", "rf"):
        for nseed in (2, 8):
            fams = ("F1", "F2", "F6m", "F5")
            worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS[:nseed] for f in fams}
            roles = tuple((r, e, en, sg) for r, e, en, sg in (("A", 0, 1, 3), ("A2", 0, 2, 4), ("B", 1, 1, 3), ("B2", 1, 2, 4))) * 5
            roles = tuple((f"{r}{k}", e, en + 10 * k, sg + 10 * k) for k, (r, e, en, sg) in enumerate(roles))
            pers = pers_list(SEEDS[:nseed], fams, roles)
            st = sim4.run(pers, worlds, dict(learner=L, innate="IH", pw="S", kmode="Lp", n_steps=200, n_early=80))
            res[f"{L}@{len(pers)}"] = dict(P=len(pers), pers_steps_per_sec=len(pers) * 200 / st["secs"])
            print(L, res[f"{L}@{len(pers)}"], file=sys.stderr)
            del st
            torch.cuda.empty_cache()
    save("speed", res)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["world", "learn", "ckpt", "speed"]
    for p in parts:
        globals()[f"part_{p}"]()
