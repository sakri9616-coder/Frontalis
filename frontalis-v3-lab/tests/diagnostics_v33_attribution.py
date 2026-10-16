"""診断（テストではない）: v3.3 で合否が変わった既存の判定（T4、U3 / W5、U7）の原因の切り分け。結果は tests/results/diag_v33_attribution.json。

v3.3 の追加は 2 つ: (A) 類を除いたヘッド（leave-class-out、SPEC-v3.3 1.2）、(B) 潜在原因による条件づけ（4.2）。
(B) は single_cause = True で、(A) は head_kind = "none" で外せる（head_kind = "none" + single_cause = True は v3.2 の head_kind = "none" と完全に同じ数値。
test_v32_mechanisms.py V32Equivalence）。この 4 通りと既定で、判定の統計を測る。

v3.2 の数値（v3.2 のコードで測った、リポジトリには残っていない値。CHANGELOG の v3.2 節と同じ）:
  T4 可塑性 m = 8 / 32   : 既定 1.045 / 1.008、head_kind = "none" 0.699 / 0.644
  U3 順位の一致率        : 既定 0.90、head_kind = "none" 0.85
実行: python3 tests/diagnostics_v33_attribution.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import KINDS, SEEDS, align_shell, dump, feed, load_json, load_jsonl, make, mh, pmap  # noqa: E402
import test_t04_continual as t4  # noqa: E402
from npnv3 import innate  # noqa: E402

V32 = {"T4_plasticity": {"default": [1.045, 1.008], "head_none": [0.699, 0.644]},
       "U3_rank_agreement": {"default": 0.90, "head_none": 0.85}}
# default = 既定（lc_design = "revised"。設計の改訂後）。spec_literal = SPEC-v3.3 4.2 のとおりの条件づけ（改訂前の既定。CHANGELOG の B.2 の数値はこの行）
CONFIGS = {"default": {}, "spec_literal": {"lc_design": "spec_literal"}, "single_cause": {"single_cause": True}, "head_none": {"head_kind": "none"},
           "head_none+single_cause": {"head_kind": "none", "single_cause": True}}
NEW = load_json("new_event.json")
COLLAB = {"entity": "u7_new_collab", "class": "collaborator", "context": "music_session"}
BAND = {"entity": "u7_new_band", "class": "band_mate", "context": "music_session"}


def t4_job(a):
    name, seed, m = a
    return t4.run(seed, m, **CONFIGS[name])


def u3_job(a):
    name, seed = a
    kw = CONFIGS[name]
    pert = innate.perturb_table(np.array(innate.M), np.random.default_rng([seed, 31]), 0.3)
    out = {}
    for tag, table in (("o", None), ("p", pert)):
        f = {}
        for k in KINDS:
            n = make(seed, effect_table=table, **kw)
            for e in mh.generate_history(k, seed, 36):
                n.experience(e)
            f[k] = align_shell(n).appraise(NEW)["f"]
        out[tag] = tuple(sorted(KINDS, key=lambda k: f[k]))
        out["f_" + tag] = f
    return out


def u7_job(a):
    kind, steps, seed = a
    n = make(seed, head_kind=kind, head_fit_steps=steps)
    feed(n, load_jsonl("T8_train.jsonl"))
    pc, pb = n.probe(COLLAB)["p_person"]["credit_taken"], n.probe(BAND)["p_person"]["credit_taken"]
    return abs(pc - pb), pc, pb


def main():
    res = {"v3.2_reference": V32}
    res["T4"] = {}
    for name in CONFIGS:
        res["T4"][name] = {}
        for m in (8, 32):
            rows = pmap(t4_job, [(name, s, m) for s in SEEDS])
            res["T4"][name][f"m{m}"] = {k: float(np.mean([r[k] for r in rows])) for k in
                                        ("persistence", "plasticity", "N_before", "N_after", "truth", "d0")}
        print("T4", name, {m: round(v["plasticity"], 3) for m, v in res["T4"][name].items()})
    res["U3"] = {}
    for name in CONFIGS:
        rows = pmap(u3_job, [(name, s) for s in SEEDS])
        agree = float(np.mean([r["o"] == r["p"] for r in rows]))
        gap_kx = float(np.mean([abs(r["f_o"]["K"] - r["f_o"]["X"]) for r in rows]))
        res["U3"][name] = {"rank_agreement": agree, "mean_abs_f_gap_K_vs_X": gap_kx,
                           "seeds_K_below_X": int(sum(r["f_o"]["K"] < r["f_o"]["X"] for r in rows))}
        print("U3", name, res["U3"][name])
    res["U7"] = {}
    none_gap = 0.2014796069848809            # v3.2 / v3.3 とも同じ（no_head の T8 の誤差。U07.json）
    for kind in ("mlp", "linear", "knn1"):
        for steps in (100, 300, 1000, 3000):
            rows = pmap(u7_job, [(kind, steps, s) for s in SEEDS])
            res["U7"][f"{kind}|{steps}"] = {"gap": float(np.mean([r[0] for r in rows])), "collab": float(np.mean([r[1] for r in rows])),
                                            "band": float(np.mean([r[2] for r in rows])),
                                            "ratio_to_none": float(np.mean([r[0] for r in rows]) / none_gap)}
        print("U7", kind, {k.split('|')[1]: round(v["ratio_to_none"], 3) for k, v in res["U7"].items() if k.startswith(kind)})
    dump("diag_v33_attribution", res)


if __name__ == "__main__":
    main()
