"""P24: GPU-hour estimate per stage on the local RTX 3050 (shared with another agent), from measured throughputs only
(DESIGN draft 4 section 14). No run here; reads the timing metadata of earlier runs.
Per-personality cost (seconds, training + measurement) at the production configuration:
  set, krr, rf, hb : measured production cells (p15_cells_*.json), mean over cells
  nn0, pre         : rehearsal timing (p17_cells.json) x set's production/rehearsal ratio (SGD learners, S/N 4)
  fixk, lin        : rehearsal timing x KRR's production/rehearsal ratio (ridge learners, one pass)
  rel              : set's production training cost x the measured REL/SET throughput ratio (p23_speed.json, same time
                     window, P = 128; the adopted speed variant if p23_compare_<v>.json says adopted) + measurement scaled from
                     the REL rehearsal (p17_cells_lr.json)
Workload in personalities per learner and innate condition (roles A A' B B' = 4 per world unless stated).
Writes p24_results.json and prints the tables."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
ld = lambda n: json.loads((HERE / n).read_text(encoding="utf-8")) if (HERE / n).exists() else {}


def per_pers(d, pred=lambda k: True):
    v = [(m["meta"]["train_secs"] + m["meta"].get("meas_secs", 0.0)) / m["meta"]["P"] for k, m in d.items() if pred(k)]
    t = [m["meta"]["train_secs"] / m["meta"]["P"] for k, m in d.items() if pred(k)]
    s = [m["meta"].get("meas_secs", 0.0) / m["meta"]["P"] for k, m in d.items() if pred(k)]
    return (float(np.mean(v)), float(np.mean(t)), float(np.mean(s))) if v else None


def main():
    cost, src = {}, {}
    for L in ("set", "krr", "rf", "hb"):
        d = {}
        for inn in ("IH", "I0"):
            d.update(ld(f"p15_cells_{L}_{inn}.json"))
        r = per_pers(d)
        if r:
            cost[L] = r; src[L] = f"production cells ({len(d)})"
    reh = ld("p17_cells.json")
    rp = {k.split("_")[0]: per_pers({k: v}) for k, v in reh.items() if "meta" in v and "P" in v["meta"]}
    set_ratio_t = cost["set"][1] / rp["set"][1]; set_ratio_m = cost["set"][2] / max(rp["set"][2], 1e-9)
    krr_ratio_t = cost["krr"][1] / rp["krr"][1]
    for L, ratio in (("nn0", set_ratio_t), ("pre", set_ratio_t), ("fixk", krr_ratio_t), ("lin", krr_ratio_t), ("rf", set_ratio_t), ("hb", krr_ratio_t)):
        if L not in cost and L in rp:
            t = rp[L][1] * ratio; m = rp[L][2] * set_ratio_m
            cost[L] = (t + m, t, m); src[L] = "rehearsal x ratio (estimate)"
    # NN-REL
    sp = ld("p23_speed.json")
    rel_ratio, var = None, "base"
    if sp:
        def thr(learner, v, P):
            x = [r["pers_steps_per_s"] for r in sp if r["learner"] == learner and r["var"] == v and r["P"] == P]
            return float(np.median(x)) if x else None
        base_set = thr("set", "base", 128)
        for v in ("exact", "base"):
            if v == "exact" and not ld("p23_compare_exact.json").get("summary", {}).get("adopted"):
                continue
            if base_set and thr("rel", v, 128):
                rel_ratio, var = base_set / thr("rel", v, 128), v; break
    if rel_ratio is None:
        rel_ratio = 16.5; var = "P10/P13 estimate"
    rl = ld("p17_cells_lr.json")
    relm = [per_pers({k: v})[2] for k, v in rl.items() if k.startswith("rel_")]
    rel_meas = (float(np.mean(relm)) * set_ratio_m) if relm else cost["set"][2] * 2
    cost["rel"] = (cost["set"][1] * rel_ratio + rel_meas, cost["set"][1] * rel_ratio, rel_meas)
    src["rel"] = f"set x {rel_ratio:.1f} ({var})"
    H = lambda L, n: n * cost[L][0] / 3600.0

    controls = ("krr", "hb", "rf", "fixk", "lin", "nn0", "pre")
    # full PREREG as written (main innates IH, I0)
    full = {"main 4 fams x 40 seeds x 6 conds x 4 roles": 3840, "noise roles (10 seeds x 3)": 720, "F6 (report)": 960, "F7 (40 worlds)": 960}
    sec = {"secondary conds (5) x 4 fams x 40 x 4 (NN only)": 3200, "I-HE (NN-SET only), main": 3840}
    out = {"cost_s_per_pers": {L: dict(total=c[0], train=c[1], meas=c[2], source=src.get(L)) for L, c in cost.items()}}
    tab = {}
    for L in ("set", "rel") + controls:
        n = sum(full.values()) * 2
        tab[L] = H(L, n)
    tab["secondary (set, rel)"] = H("set", 3200 * 2) + H("rel", 3200 * 2) + H("set", 3840)
    out["full_prereg_gpu_h"] = tab
    # staged plan (DESIGN section 14): looks at 10, 20, 40 seeds with a futility rule; controls KRR and HB always, the other
    # controls only for blocks with a non-futile NN claim; NN-REL behind a learnability gate and staged by family
    W = lambda seeds, fams=4, conds=6, roles=4: seeds * fams * conds * roles
    S1a = W(10) + 5 * 4 * 6 * 3 + W(10, 1)        # seeds 1-10 (+ noise roles in 5 seeds, + F7 worlds 1-10)
    S1b = W(10) + W(10, 1)                        # seeds 11-20
    S2 = W(20) + W(20, 1)                         # seeds 21-40
    other = ("rf", "fixk", "lin", "nn0", "pre")
    st = {}
    st["S0 dev: NN-REL learnability gate (F1 F6m F2, C-off, role A, 10 dev seeds)"] = 30 * 32000 / 78.0 / 3600.0
    st["S0 dev: NN-REL calibration (F1, C-off/C-S0, 6 roles) + reduced sec. 9 pass (I-H, F1 F6m, C-off C-SLp), if gate passes"] = H("rel", 10 * 2 * 6 + 10 * 2 * 2 * 4)
    st["S0 dev: remaining production cells (queue A)"] = H("set", 3 * 240) + H("krr", 240) + H("rf", 480) + H("hb", 480)
    st["S1a eval seeds 1-10: NN-SET + KRR + HB, 2 innates"] = 2 * (H("set", S1a) + H("krr", S1a) + H("hb", S1a))
    st["S1a' other controls (RF NN-0 FIX-K LIN PRE), only non-futile blocks"] = 2 * sum(H(L, S1a) for L in other)
    st["S1b eval seeds 11-20: NN-SET + KRR + HB (non-futile blocks)"] = 2 * (H("set", S1b) + H("krr", S1b) + H("hb", S1b))
    st["S1b' other controls seeds 11-20 (non-futile)"] = 2 * sum(H(L, S1b) for L in other)
    st["S2 eval seeds 21-40: NN-SET + all controls (non-futile)"] = 2 * (H("set", S2) + sum(H(L, S2) for L in ("krr", "hb") + other))
    st["R1a NN-REL F1 F2 seeds 1-10, 2 innates"] = 2 * H("rel", W(10, 2))
    st["R1b NN-REL F1 F2 seeds 11-20 (non-futile)"] = 2 * H("rel", W(10, 2))
    st["R2 NN-REL F6m F5 seeds 1-20 + F7 1-20 (only if R1 not futile)"] = 2 * H("rel", W(20, 2) + W(20, 1))
    st["R3 NN-REL seeds 21-40, all families + F7 (non-futile)"] = 2 * H("rel", W(20) + W(20, 1))
    st["S4 optional by budget rule: F6, secondary conds, I-HE (set + rel)"] = 2 * H("set", 960 + 3200) + 2 * H("rel", 960 + 3200) + H("set", 3840)
    out["staged_gpu_h"] = st
    g = lambda *ks: sum(v for k, v in st.items() if any(k.startswith(x) for x in ks))
    A = g("S0 dev: NN-REL learnability", "S0 dev: remaining", "S1a eval")
    out["scenarios"] = {
        "A: REL gate fails; every NN-SET block futile at 10 seeds": A,
        "B: REL gate passes; everything futile at the first look": A + g("S0 dev: NN-REL calibration", "R1a"),
        "C: REL gate passes; nothing futile (all main stages, no S4)": sum(v for k, v in st.items() if not k.startswith("S4")),
        "D: C + S4": sum(st.values()),
    }
    # MINIMUM main path (PREREG 10d, DESIGN 14, after the user's correction): NN-SET only; I-H; F1; C-off and C-SLp;
    # roles A A' B B'; 20 evaluation seeds (77100-77119; 77120-77139 stay unused); controls KRR and NN-0 in the same cells
    cell = 20 * 1 * 2 * 4
    mn = {}
    mn["dev before freeze: C-SLp calibration (resume P15 cell) + NN-0 dev cell + KRR C-SLp dev cell"] = H("set", 240) * 4000 / 32000 + H("set", 60) * 0 + H("nn0", 120) + H("krr", 240)
    mn["EVAL main: NN-SET + KRR + NN-0, I-H, F1, C-off/C-SLp, 20 seeds"] = H("set", cell) + H("krr", cell) + H("nn0", cell)
    opt = {}
    opt["O-L3 structure: F6m and F5 for NN-SET + KRR (trigger: L2 signal in F1)"] = 2 * (H("set", cell) + H("krr", cell))
    opt["O-cycle: C-S0 C-H0 C-SLh C-SLn, F1, NN-SET + KRR + NN-0 (trigger: L1 or L2 signal in C-SLp)"] = 2 * (H("set", cell) + H("krr", cell) + H("nn0", cell))
    opt["O-I0: value type, same cells with I0 + dev calibration (trigger: user decision)"] = 1.25 * (H("set", cell) + H("krr", cell) + H("nn0", cell))
    opt["O-controls: RF HB PRE-NN FIX-K LIN in the main cells (trigger: J5-c passes vs KRR and NN-0)"] = sum(H(L, cell) for L in ("rf", "hb", "pre", "fixk", "lin"))
    opt["O-REL: gate + dev calibration + F1 F2 C-off/C-SLp 20 seeds (trigger: L2 signal AND relational reason)"] = 3.4 + H("rel", 120) + H("rel", 2 * cell)
    out["minimum"] = dict(main=mn, optional=opt, main_total=sum(mn.values()))
    (HERE / "p24_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["cost_s_per_pers"].items():
        print(f"{k:5s} {v['total']:7.1f} s/pers (train {v['train']:.1f}, meas {v['meas']:.1f})  {v['source']}")
    print("full PREREG (main, 2 innates), GPU-h:", {k: round(v) for k, v in tab.items()}, "total", round(sum(tab.values())))
    for k, v in st.items():
        print(f"  {v:7.1f} h  {k}")
    for k, v in out["scenarios"].items():
        print(f"  {v:7.1f} h  ({v/24:.1f} days)  {k}")
    print("MINIMUM main path:")
    for k, v in out["minimum"]["main"].items():
        print(f"  {v:6.2f} h  {k}")
    print(f"  {out['minimum']['main_total']:6.2f} h  total")
    print("optional stages (each only by its trigger):")
    for k, v in out["minimum"]["optional"].items():
        print(f"  {v:6.2f} h  {k}")


if __name__ == "__main__":
    main()
