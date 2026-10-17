"""P17: full-pipeline rehearsal (PREREG draft 4 section 10; CRITIQUE-2 D6). Rehearsal seeds 77340-77342 only.
All learners x I-H and I0 (I-HE for NN-SET) x the 6 main cycle conditions x F1 F2 F6m F5, roles A A2 B B2 A3 A4, at the
reduced scale (2,000 experiences, S/N 1); generation -> learning -> checkpoint -> interruption and resume -> measurement ->
judgement summary -> report skeleton. One production-scale slice (NN-SET, I-H, C-SLp, F1, 3 seeds).
Run: python -B p17_rehearsal.py <learner|ckpt|prodslice|report>"""
import json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
import sim4, measure4 as m4, judge4
from cells import MAIN_COND, ROLES, run_cell, pers_list, SCALE, COND

HERE = Path(__file__).parent
SEEDS = [77340, 77341, 77342]
FAMS = ("F1", "F2", "F6m", "F5")
TAG = sys.argv[2] if len(sys.argv) > 2 else ""   # "lr": rerun with the production learning rates (DESIGN 13 Q-g); old results kept
OUT = HERE / (f"p17_cells_{TAG}.json" if TAG else "p17_cells.json")
LOG = open(HERE / "p17_stderr.txt", "a", encoding="utf-8")
ROLES6 = ROLES + (("A3", 0, 5, 6), ("A4", 0, 7, 8))
PSCALE = {"krr": 0.0, "rf": 0.0, "hb": 1.0, "fixk": 1.0, "lin": 1.0, "nn0": 1.0}   # overwritten by p17_pscale.json if present


def pscale(learner):
    p = HERE / "p16_pscale.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get(learner, PSCALE.get(learner, 1.0))
    return PSCALE.get(learner, 1.0)


def lr_for(learner):
    """Without TAG: the sim4 default 3e-4 (the first rehearsal). With TAG: NN-SET's production choice (p15_lr.json) for all
    learners except NN-REL, which uses its own choice (p16_rel_lr.json) if present."""
    if not TAG:
        return {}
    if learner == "rel" and (HERE / "p16_rel_lr.json").exists():
        return dict(lr=json.loads((HERE / "p16_rel_lr.json").read_text(encoding="utf-8"))["chosen"])
    return dict(lr=json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"])


def pretrain(seeds, fams, n=2000):
    """PRE-NN: NN-SET trained on other worlds of the same family (seed + 100000), mixed history, C-off; returns params per
    personality in the order of pers_list(seeds, fams, ROLES6, conds=MAIN_COND)."""
    worlds = {(s + 100000, f): sim4.make_world(s + 100000, f) for s in seeds for f in fams}
    pp = pers_list([s + 100000 for s in seeds], fams, (("A", -1, 1, 3),))
    st = sim4.run(pp, worlds, dict(learner="set", innate="IH", **COND["C-off"], n_steps=n, n_early=0, spn=1, **lr_for("set")))
    by = {q["world"]: i for i, q in enumerate(pp)}
    target = pers_list(seeds, fams, ROLES6, conds=MAIN_COND)
    idx = torch.tensor([by[(q["world"][0] + 100000, q["world"][1])] for q in target], device=sim4.DEV)
    return {k: v.detach()[idx] for k, v in st["net"].p.items()}


def learner_cells(learner):
    innates = ("IH", "I0", "IHE") if learner == "set" else ("IH", "I0")
    for innate in innates:
        extra = dict(partner_scale=pscale(learner)) if learner not in ("set", "rel", "pre") else {}
        extra.update(lr_for(learner))
        if learner == "pre":
            init = pretrain(SEEDS, FAMS)
            # run_cell builds the same personality order; pass the init through extra_cfg is not possible -> run here
            worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in FAMS}
            pers = pers_list(SEEDS, FAMS, ROLES6, conds=MAIN_COND)
            st = sim4.run(pers, worlds, dict(learner="pre", innate=innate, **SCALE["rehearsal"], **lr_for("pre")), init_params=init)
            vw = m4.views(st, pers, worlds)
            rows = []
            for gk, v in vw.items():
                s, f, cond = gk
                r = m4.measure_group(v, worlds[(s, f)], s, innate, COND[cond]["cycle"], das=False)
                r.update(learner="pre", innate=innate, cond=cond, seed=s, fam=f, scale="rehearsal", tag="")
                rows.append(r)
            d = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
            d[f"pre_{innate}"] = dict(meta=dict(P=len(pers), train_secs=st["secs"], nonfinite=st["nonfinite"]), rows=rows)
            OUT.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
            print(f"pre {innate} P={len(pers)} {st['secs']:.0f}s", file=LOG); LOG.flush()
            continue
        run_cell(learner, innate, MAIN_COND, FAMS, SEEDS, ROLES6, "rehearsal", OUT,
                 calib_conds=("C-off", "C-S0") if learner in ("set", "rel") and innate == "IH" else (), extra_cfg=extra,
                 ckpt_dir=os.path.join(tempfile.gettempdir(), "p17_ckpt"), log=LOG)


def ckpt_test():
    """Interrupt and resume a full-condition run; compare with the uninterrupted run (bit identity)."""
    res = {}
    for learner in ("set", "rel", "krr"):
        worlds = {(s, f): sim4.make_world(s, f) for s in SEEDS for f in FAMS}
        pers = pers_list(SEEDS, FAMS, ROLES6, conds=MAIN_COND)
        cfg = dict(learner=learner, innate="IH", n_steps=600, n_early=240, spn=1, partner_scale=pscale(learner))
        full = sim4.run(pers, worlds, cfg); h_full = sim4.state_hash(full); del full
        tmp = os.path.join(tempfile.gettempdir(), f"p17_int_{learner}.pt")
        part = sim4.run(pers, worlds, dict(cfg, stop_at=300)); sim4.save_state(part, tmp); del part
        st = sim4.load_state(tmp, pers, worlds, cfg); st["phase"] = "main"
        res_ = sim4.run(pers, worlds, cfg, state=st); h_res = sim4.state_hash(res_); del res_
        os.remove(tmp)
        res[learner] = dict(P=len(pers), identical=h_full == h_res, hash_full=h_full, hash_resumed=h_res)
        print("ckpt", learner, res[learner], file=LOG); LOG.flush()
        torch.cuda.empty_cache()
    (HERE / "p17_ckpt.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def prodslice():
    run_cell("set", "IH", ["C-SLp"], ("F1",), SEEDS, ROLES6, "prod", HERE / "p17_prodslice.json", calib_conds=(), log=LOG,
             ckpt_dir=os.path.join(tempfile.gettempdir(), "p17_ckpt"))


def report():
    res = json.loads(OUT.read_text(encoding="utf-8"))
    rep = judge4.report(res)
    (HERE / (f"p17_judge_{TAG}.json" if TAG else "p17_judge.json")).write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    # report skeleton: one line per cell, the L1/L2/L3 verdict by the PREREG section 8 rule (3 seeds: the 30/40 rule
    # cannot be met; the skeleton shows the format only)
    lines = ["| 学習器 | 生得 | 循環 | 族 | J1-a | J2-a | J2-b | J3 | J4-a | J4-b | J5-a | J5-c(KRR) | 層 |", "|" + "---|" * 13]
    for k, v in rep.items():
        if k.startswith("_"):
            continue
        L, I, C, Fm, S = k.split("|")
        p = lambda j: ("○" if v.get(j, {}).get("passed") else "×") if j in v else "–"
        l2 = all(v.get(j, {}).get("passed") for j in ("J1-a", "J2-a", "J2-b", "J3", "J4-a", "J4-b", "J5-a")) and v.get("J5-c_krr", {}).get("passed", False)
        lay = "L2" if l2 else ("L1" if v.get("J1-a", {}).get("passed") else "—")
        lines.append(f"| {L} | {I} | {C} | {Fm} | {p('J1-a')} | {p('J2-a')} | {p('J2-b')} | {p('J3')} | {p('J4-a')} | {p('J4-b')} | {p('J5-a')} | {p('J5-c_krr')} | {lay} |")
    (HERE / (f"p17_report_skeleton_{TAG}.md" if TAG else "p17_report_skeleton.md")).write_text("# リハーサルの報告の雛形（種 77340〜77342、縮めた規模。合否は 3 種なので形式の確認だけ）\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "ckpt":
        ckpt_test()
    elif a == "prodslice":
        prodslice()
    elif a == "report":
        report()
    else:
        learner_cells(a)
