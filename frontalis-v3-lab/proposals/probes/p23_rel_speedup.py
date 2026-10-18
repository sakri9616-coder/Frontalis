"""P23: measured speedups for NN-REL and their equivalence to the baseline implementation (DESIGN draft 4 section 14;
compute-reduction plan). Dev seeds 77380-77389 only. The adoption rule below was committed BEFORE any equivalence run.

  speed                 throughput (personality-steps / s) at the production configuration (N = 32,000, S/N = 4), timed over
                        steps 50..150 after a warm-up (shortened from 50..350 before any timing was recorded), two repetitions in alternating order (the GPU is shared; the ratio to
                        the baseline measured in the same window is the reported quantity). Variants: base, sdpa, fused_opt,
                        exact (= sdpa + fused_opt), tf32, amp (bf16; speed only), d32 (model change; speed only), and the
                        population size P in {64, 128, 256} for base and exact. NN-SET base at P = 128 for reference.
                        torch.compile is not available on this machine (no Triton on Windows; checked 2026-10-17).
  equiv <name>          rehearsal scale (2,000 experiences, S/N 1, lr 1e-3), NN-REL, I-H, dev seeds 77380-77384,
                        F1 F6m F5, C-off and C-SLp, roles A A2 B B2 (P = 120): 30 worlds. Names: base, retest (base with every
                        env/sgd stream +1000), exact, tf32, batch2 (base, but run inside a P = 240 population that also
                        contains C-S0 and C-H0; only the C-off / C-SLp worlds are compared).
  compare <name>        per world and quantity m (the judgement quantities, as in p21):
                          delta = m_variant - m_base,  nu = |m_retest - m_base| (same-condition retest = the noise floor)
                        E1 median|delta| <= 0.2 median(nu); E2 90th pct |delta| <= 0.5 median(nu);
                        E3 non-finite 0; E4 the variant is deterministic (two short identical runs have the same state hash);
                        E5 median|delta| <= 1.5 median(nu); E6 |median(delta)| <= 0.5 median(nu) (no systematic shift).
  ADOPTION RULE (fixed before running):
    - exact-arithmetic reorganisations (sdpa, fused_opt, exact, batch2: the same operations in a different summation order)
      are adopted for the evaluation iff E3, E4, E5 and E6 hold for every m. Reason: P18 showed that the learner's
      trajectory is chaotic in the minibatch order, so any change in the last bits grows to retest size; E1/E2 cannot be met
      by any change, and E5/E6 ask that the variant be another draw of the same process without a systematic shift.
    - precision reductions (tf32) follow the draft-3.1 bf16 rule (E1, E2, E3, E4), unchanged. bf16 itself already failed that
      rule once (P12) and is not retested (the rule allows one test); its speed is reported for information only.
    - model changes (d32) are not implementation speedups and are not adopted through this check.
Writes p23_speed.json, p23_equiv_<name>.json, p23_compare_<name>.json."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import sim4
import measure4 as m4
from cells import COND, ROLES, SCALE, pers_list

HERE = Path(__file__).parent
M = ("D_AB_un", "D_AA_un", "Drho_AB_un", "Drho_AA_un", "Drho_AB_nc", "Drho_AA_nc", "swapd_AB", "swapd_AA",
     "xp_other_un", "xp_same_un", "xp_other_nc", "xp_same_nc", "r2_un", "r2_nc", "repro_un")
VARIANTS = {"base": {}, "sdpa": dict(sdpa=1), "fused_opt": dict(fused_opt=1), "exact": dict(sdpa=1, fused_opt=1),
            "tf32": dict(tf32=1), "amp": dict(amp=1), "d32": dict(d=32)}


def timed(learner, var, P_conds):
    seeds = [77380, 77381, 77382, 77383]; fams = ("F1", "F6m", "F2", "F5")
    worlds = {(s, f): sim4.make_world(s, f) for s in seeds for f in fams}
    pers = pers_list(seeds, fams, ROLES, conds=P_conds)
    cfg = dict(learner=learner, innate="IH", **SCALE["prod"], lr=1e-3, **VARIANTS[var])
    st = sim4.run(pers, worlds, dict(cfg, stop_at=50)); torch.cuda.synchronize()
    s0 = st["secs"]; st = sim4.run(pers, worlds, dict(cfg, stop_at=150), state=st); torch.cuda.synchronize()
    sec = st["secs"] - s0
    out = dict(learner=learner, var=var, P=len(pers), secs=sec, pers_steps_per_s=len(pers) * 100 / sec,
               peak_MiB=torch.cuda.max_memory_allocated() / 2**20)
    del st; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return out


def speed():
    conds = {64: ["C-off"], 128: ["C-off", "C-SLp"], 256: ["C-off", "C-SLp", "C-S0", "C-H0"]}
    plan = [("rel", v, 128) for v in ("base", "sdpa", "fused_opt", "exact", "tf32", "amp", "d32")] + \
           [("rel", "base", 64), ("rel", "base", 256), ("rel", "exact", 256), ("set", "base", 128)]
    res = []
    for rep in range(2):
        for item in (plan if rep == 0 else plan[::-1]):
            r = timed(item[0], item[1], conds[item[2]]); r["rep"] = rep; res.append(r)
            print(json.dumps(r), flush=True)
            (HERE / "p23_speed.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def equiv(name):
    seeds = [77380, 77381, 77382, 77383, 77384]; fams = ("F1", "F6m", "F5")
    conds = ["C-off", "C-SLp"] + (["C-S0", "C-H0"] if name == "batch2" else [])
    worlds = {(s, f): sim4.make_world(s, f) for s in seeds for f in fams}
    pers = pers_list(seeds, fams, ROLES, conds=conds)
    if name == "retest":
        for q in pers:
            q["env"] += 1000; q["sgd"] += 1000
    var = {"base": {}, "retest": {}, "batch2": {}}.get(name, VARIANTS.get(name, {}))
    cfg = dict(learner="rel", innate="IH", **SCALE["rehearsal"], lr=1e-3, **var)
    st = sim4.run(pers, worlds, cfg)
    rows = []
    for (s, f, cond), v in m4.views(st, pers, worlds).items():
        if cond not in ("C-off", "C-SLp"):
            continue
        r = m4.measure_group(v, worlds[(s, f)], s, "IH", COND[cond]["cycle"], das=False)
        r.update(seed=s, fam=f, cond=cond); rows.append(r)
    nonfinite, secs = st["nonfinite"], st["secs"]
    del st; torch.cuda.empty_cache()
    # E4: determinism of this variant (two short identical runs, P = 24)
    small = pers_list([77380], ("F1",), ROLES, conds=["C-off", "C-SLp"]) + pers_list([77381], ("F6m",), ROLES, conds=["C-off", "C-SLp"])
    if name == "retest":
        for q in small:
            q["env"] += 1000; q["sgd"] += 1000
    w2 = {k: worlds[k] for k in {q["world"] for q in small}}
    h = [sim4.state_hash(sim4.run(small, w2, dict(cfg, n_steps=300, n_early=120))) for _ in range(2)]
    (HERE / f"p23_equiv_{name}.json").write_text(json.dumps(dict(rows=rows, nonfinite=nonfinite, secs=secs, P=len(pers),
                                                                  deterministic=h[0] == h[1], hashes=h), indent=1, default=float), encoding="utf-8")
    print(name, "P", len(pers), "secs", round(secs), "nonfinite", nonfinite, "deterministic", h[0] == h[1])


def compare(name):
    ld = lambda n: json.loads((HERE / f"p23_equiv_{n}.json").read_text(encoding="utf-8"))
    B, R, V = ld("base"), ld("retest"), ld(name)
    key = lambda r: (r["seed"], r["fam"], r["cond"])
    bi, ri = {key(r): r for r in B["rows"]}, {key(r): r for r in R["rows"]}
    per = {}
    for m in M:
        ks = [key(r) for r in V["rows"] if m in r and np.isfinite(r[m]) and np.isfinite(bi[key(r)][m]) and np.isfinite(ri[key(r)][m])]
        vi = {key(r): r for r in V["rows"]}
        d = np.array([vi[k][m] - bi[k][m] for k in ks]); nu = np.array([abs(ri[k][m] - bi[k][m]) for k in ks])
        if not len(d):
            continue
        nm = float(np.median(nu))
        per[m] = dict(n=len(d), abs_delta_med=float(np.median(np.abs(d))), abs_delta_p90=float(np.percentile(np.abs(d), 90)),
                      signed_delta_med=float(np.median(d)), nu_med=nm,
                      E1=bool(np.median(np.abs(d)) <= 0.2 * nm), E2=bool(np.percentile(np.abs(d), 90) <= 0.5 * nm),
                      E5=bool(np.median(np.abs(d)) <= 1.5 * nm), E6=bool(abs(np.median(d)) <= 0.5 * nm))
    allm = lambda e: all(v[e] for v in per.values())
    summ = dict(E1=allm("E1"), E2=allm("E2"), E5=allm("E5"), E6=allm("E6"), E3=V["nonfinite"] == 0, E4=V["deterministic"],
                secs_variant=V["secs"], secs_base=B["secs"])
    cls = "precision" if name in ("tf32",) else "exact"
    summ["class"] = cls
    summ["adopted"] = bool(summ["E1"] and summ["E2"] and summ["E3"] and summ["E4"]) if cls == "precision" else \
        bool(summ["E3"] and summ["E4"] and summ["E5"] and summ["E6"])
    (HERE / f"p23_compare_{name}.json").write_text(json.dumps(dict(summary=summ, per_m=per), indent=1), encoding="utf-8")
    print(name, json.dumps(summ))


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "speed":
        speed()
    elif a == "equiv":
        equiv(sys.argv[2])
    else:
        compare(sys.argv[2])
