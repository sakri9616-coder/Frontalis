"""P35: own-state intervention (dev seeds 77380-77389 only; mid scale). Pre-note in LOG-dev.md (P35), written before running.
NN-SET, I-H, C-SLp, F1, roles A and A' (retest). For wgate in (0, 1): train to t0 = 3,200 (end of the early phase), save, then
continue three ways with identical history and minibatch order: orig, perm (A's cumulative state cum permuted across channels by
a fixed derangement per seed), swap (A's cum replaced by A''s cum). Final evaluation of A on P_un is compared (centered L1,
the draft-4 dist scale). Writes p35_intervene.json.
Run: python -B p35_intervene.py   (also: P35_TIMING=1 measures NN-REL throughput for the cost table instead)"""
import json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim5, sim4, measure4 as m4
from cells import SCALE, pers_list

SEEDS = list(range(77380, 77390))
ROLES2 = (("A", 0, 1, 3), ("A2", 0, 2, 4))
T0 = SCALE["mid"]["n_early"]
NOCHOICE = bool(os.environ.get("P35_NOCHOICE"))   # 2nd pass: no choice scenes, so the experience sequence after t0 is identical
TAG = "_nochoice" if NOCHOICE else ""


def derangement(seed):
    rng = np.random.default_rng([seed, 81])
    while True:
        p = rng.permutation(sim4.KB)
        if not np.any(p == np.arange(sim4.KB)):
            return p


def finals(st, pers, worlds):
    out = {}
    for (s, f, c), vw in m4.views(st, pers, worlds).items():
        out[s] = {r: dict(V=vw[r].get("un")["V"], w=vw[r].state["w"].cpu().numpy()) for r in ("A", "A2")}
    return out


def main():
    lr = json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in SEEDS}
    pers = pers_list(SEEDS, ("F1",), ROLES2, "", ["C-SLp"])
    iA = {q["world"][0]: i for i, q in enumerate(pers) if q["role"] == "A"}
    iA2 = {q["world"][0]: i for i, q in enumerate(pers) if q["role"] == "A2"}
    tmp = Path(tempfile.gettempdir()) / "v4d_ckpt_p35"; tmp.mkdir(exist_ok=True)
    res = {}
    t_start = time.time()
    for wg in (0, 1):
        cfg = dict(learner="set", innate="IH", **SCALE["mid"], lr=lr, wgate=wg, **({"choice": "none"} if NOCHOICE else {}))
        ck = tmp / f"t0_wg{wg}{TAG}.pt"
        st = sim5.run(pers, worlds, dict(cfg, stop_at=T0))
        assert st["t"] == T0
        sim5.save_state(st, str(ck)); del st; torch.cuda.empty_cache()
        F = {}
        for var in ("orig", "perm", "swap"):
            st = sim5.load_state(str(ck), pers, worlds, cfg); st["phase"] = "main"
            with torch.no_grad():
                for s in SEEDS:
                    a, a2 = iA[s], iA2[s]
                    if var == "perm":
                        st["cum"][a] = st["cum"][a][torch.tensor(derangement(s), device=sim4.DEV)]
                    elif var == "swap":
                        st["cum"][a] = st["cum"][a2].clone()
            st = sim5.run(pers, worlds, cfg, state=st)
            F[var] = finals(st, pers, worlds)
            del st; torch.cuda.empty_cache()
        rows = []
        for s in SEEDS:
            Vo, Vo2 = F["orig"][s]["A"]["V"], F["orig"][s]["A2"]["V"]
            scale = np.mean([np.abs(Vo).mean(), np.abs(Vo2).mean()]) * sim4.KB + 1e-12
            D = lambda a, b: m4.dist(m4.cent(a), m4.cent(b), scale)
            rows.append(dict(seed=s, d_perm=D(F["perm"][s]["A"]["V"], Vo), d_swap=D(F["swap"][s]["A"]["V"], Vo), d_rt=D(Vo, Vo2),
                             w_end_L1_perm=float(np.abs(F["perm"][s]["A"]["w"] - F["orig"][s]["A"]["w"]).sum()),
                             w_end_L1_swap=float(np.abs(F["swap"][s]["A"]["w"] - F["orig"][s]["A"]["w"]).sum())))
        res[f"wgate{wg}"] = rows
    med = lambda rs, k: float(np.median([r[k] for r in rs]))
    out = dict(rows=res, secs=time.time() - t_start)
    for wg in ("wgate0", "wgate1"):
        rs = res[wg]
        n = sum(r["d_perm"] > r["d_swap"] for r in rs)
        out[f"Q1_{wg}"] = dict(n_perm_gt_swap=n, median_d_perm=med(rs, "d_perm"), median_d_swap=med(rs, "d_swap"), median_d_rt=med(rs, "d_rt"),
                               yes=bool(n >= 8 and med(rs, "d_perm") >= 0.25 * med(rs, "d_rt")),
                               median_w_end_L1_perm=med(rs, "w_end_L1_perm"), median_w_end_L1_swap=med(rs, "w_end_L1_swap"))
    diff = [a["d_perm"] - b["d_perm"] for a, b in zip(res["wgate1"], res["wgate0"])]
    out["Q2"] = dict(n_pos=int(sum(d > 0 for d in diff)), median=float(np.median(diff)), floor=0.25 * med(res["wgate1"], "d_rt"),
                     yes=bool(sum(d > 0 for d in diff) >= 8 and np.median(diff) >= 0.25 * med(res["wgate1"], "d_rt")))
    (HERE / f"p35_intervene{TAG}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))


def timing():
    """NN-REL throughput on this GPU at production settings (dev seeds; 300 steps; nothing kept)."""
    lr = json.loads((HERE / "p16_rel_lr.json").read_text(encoding="utf-8"))["chosen"]
    out = {}
    for P_seeds in (5, 10):
        seeds = SEEDS[:P_seeds]
        worlds = {(s, f): sim4.make_world(s, f) for s in seeds for f in ("F1", "F6m")}
        R = (("A", 0, 1, 3), ("A2", 0, 2, 4), ("B", 1, 1, 3), ("B2", 1, 2, 4), ("A3", 0, 5, 6), ("A4", 0, 7, 8))
        pers = pers_list(seeds, ("F1", "F6m"), R, "", ["C-off"])
        cfg = dict(learner="rel", innate="IH", **SCALE["prod"], lr=lr, stop_at=300)
        torch.cuda.synchronize(); t = time.time()
        st = sim4.run(pers, worlds, cfg); torch.cuda.synchronize()
        out[len(pers)] = dict(pers_exp_per_s=len(pers) * 300 / (time.time() - t), max_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
        del st; torch.cuda.empty_cache()
    (HERE / "p35_rel_timing.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    timing() if os.environ.get("P35_TIMING") else main()
