"""診断（テストではない）: 設計の改訂後（lc_design = "revised"。既定）の E 系の結果の原因の切り分け。判定には使わず、**採用もしていない**。結果は tests/results/diag_e_revised.json。

背景: v3.3 第 1 部の設計の改訂（オーケストレーター承認。結果を見た後。固定した値: d = 1、α = 0.5、β_first = 5、β_later = 0.2、σ_r = 0.15、K = 5）を実装し、E 系を走らせると、
E6 は通ったが、E1・E3・E4・E5・E7・E9 と E8(b)（値を持つ別の手がかり）が通らなかった（E01_E09.json）。次の 3 つで原因を調べる。
  A. 感度: 固定した値を 1 つずつ動かしたときの E 系の指標（α、d、σ_r、β、K、時間の単位）。**どれも既定には入れていない**。次の改訂の材料。
  B. 解剖: seed 0 の獲得直後と消去直後の原因の状態（作られた順序・β_k・質量・試行数）。
  C. 算術: E1 が d = 1・α = 0.5 のもとで原理的に満たせるか（質量と α の比）。
実行: python3 tests/diagnostics_e_revised.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e_protocol import (CTX_A, CTX_B, CTX_C, CUE_NAME, Counter, EXT_CONTEXTS_3, run_variant, trial)  # noqa: E402
from helpers import SEEDS, dump, make, pmap  # noqa: E402
from npnv3.value import Conditioning  # noqa: E402


class HalfTime(Conditioning):
    """時間の単位を 2 経験に（1 試行 = 2 経験のプロトコルでは「試行」単位）。質量 Σ ((t − t_i)/2 + 1)^(−d)。"""
    def _mass_row(self, i, t):
        n = self.hist_n[i]
        if n == 0:
            return np.zeros(self.K)
        dt = np.maximum(t - self.hist_t[i][:n], 0.0) / 2.0 + 1.0
        return np.bincount(self.hist_k[i][:n], weights=dt ** (-self.cfg.lc_decay_d), minlength=self.K)[: self.K]


def use_half_time(npn):
    npn.cond.__class__ = HalfTime


VARIANTS = {
    "default (revised)": ({}, None),
    "alpha 0.1": ({"lc_alpha": 0.1}, None),
    "alpha 0.05": ({"lc_alpha": 0.05}, None),
    "alpha 0.01": ({"lc_alpha": 0.01}, None),
    "d 0.5": ({"lc_decay_d": 0.5}, None),
    "d 2.0": ({"lc_decay_d": 2.0}, None),
    "sigma_r 0.10": ({"lc_sigma_r": 0.10}, None),
    "sigma_r 0.05": ({"lc_sigma_r": 0.05}, None),
    "beta 5 / 5 (no ordering)": ({"lc_beta_first": 5.0, "lc_beta_later": 5.0}, None),
    "beta 0.2 / 0.2": ({"lc_beta_first": 0.2, "lc_beta_later": 0.2}, None),
    "K 10": ({"lc_max_causes": 10}, None),
    "K 3": ({"lc_max_causes": 3}, None),
    "time unit = 2 experiences": ({}, use_half_time),
    "alpha 0.05 + sigma_r 0.05": ({"lc_alpha": 0.05, "lc_sigma_r": 0.05}, None),
    "alpha 0.05 + sigma_r 0.05 + time unit 2": ({"lc_alpha": 0.05, "lc_sigma_r": 0.05}, use_half_time),
}


def job(args):
    name, seed = args
    kw, post = VARIANTS[name]
    main = run_variant(seed, **kw)
    # run_variant の post_make を使う変種は保存・復元（E8c）が意味を持たないので full=False（E1 などを測るため full の部分だけ別に実行）
    if post is not None:
        main = _full_without_restore(seed, kw, post)
    one = run_variant(seed, full=False, post_make=post, **kw)["ABC"]
    three = run_variant(seed, full=False, ext_contexts=EXT_CONTEXTS_3, post_make=post, **kw)["ABC"]
    return {"main": main, "e9_diff": one - three}


def _full_without_restore(seed, kw, post):
    """run_variant の full 部分から保存・復元だけを除いた版（tests/diagnostics_e.py の run_variant_full_no_restore と同じ手順）。"""
    import copy
    from e_protocol import (N_ACQ, N_EXT, N_REINSTATE, N_UNRELATED, RE_ACQ_MAX, first_reaching, pretrain_controls, read, unrelated, lone_harm)
    rng = np.random.default_rng([seed, 303])
    c = Counter(f"E{seed}")
    npn = make(seed, **kw)
    post(npn)
    out = {}
    pretrain_controls(npn, c, rng)
    acq = []
    for _ in range(N_ACQ):
        trial(npn, c, rng, "acq", CTX_A)
        acq.append(read(npn, CTX_A))
    c_acq = acq[-1]
    m1 = copy.deepcopy(npn)
    unrelated(m1, c, np.random.default_rng([seed, 304]), N_UNRELATED)
    out["E1"] = read(m1, CTX_A) / c_acq
    for _ in range(N_EXT):
        trial(npn, c, rng, "ext", CTX_B)
    out["E2_B"] = read(npn, CTX_B) / c_acq
    out["ABA"] = read(npn, CTX_A) / c_acq
    out["ABC"] = read(npn, CTX_C) / c_acq
    m5 = copy.deepcopy(npn)
    unrelated(m5, c, np.random.default_rng([seed, 306]), N_UNRELATED)
    out["E5_after"] = read(m5, CTX_B) / c_acq
    m6 = copy.deepcopy(npn)
    lone_harm(m6, c, np.random.default_rng([seed, 307]), CTX_B, N_REINSTATE)
    out["E6_reinstated"] = read(m6, CTX_B) / c_acq
    m6c = copy.deepcopy(npn)
    unrelated(m6c, c, np.random.default_rng([seed, 308]), N_REINSTATE)
    out["E6_control"] = read(m6c, CTX_B) / c_acq
    m7 = copy.deepcopy(npn)
    re = []
    for _ in range(RE_ACQ_MAX):
        trial(m7, c, rng, "acq", CTX_B)
        re.append(read(m7, CTX_B))
    out["E7_first"] = first_reaching(acq, 0.7, c_acq)
    out["E7_re"] = first_reaching(re, 0.7, c_acq)
    return out


def summarize(rows):
    m = lambda k: float(np.mean([r["main"][k] for r in rows]))
    return {"E1": m("E1"), "E2_B": m("E2_B"), "E3_ABA": m("ABA"), "E4_ABC": m("ABC"),
            "E5_increase": float(np.mean([r["main"]["E5_after"] - r["main"]["E2_B"] for r in rows])),
            "E6_difference": float(np.mean([r["main"]["E6_reinstated"] - r["main"]["E6_control"] for r in rows])),
            "E7_ratio": m("E7_re") / m("E7_first"), "E9_difference": float(np.mean([r["e9_diff"] for r in rows]))}


def anatomy(seed=0):
    rng = np.random.default_rng([seed, 303])
    c = Counter("D")
    npn = make(seed)
    snap = {}

    def dump_state(tag):
        cd = npn.cond
        i = cd.idx[CUE_NAME]
        t = float(npn._clock())
        mass = cd._mass_row(i, t)
        rows = {"t": t, "causes": [], "response": {ctx: cd.value_of(CUE_NAME, ctx, t) for ctx in (CTX_A, CTX_B, CTX_C)}}
        for k in range(cd.K):
            if cd.AL[i, k] > 0.5:
                rows["causes"].append({"k": k, "C": float(cd.CK[i, k]), "beta": float(cd.BK[i, k]), "mass": float(mass[k]),
                                       "trials": int((cd.hist_k[i][:cd.hist_n[i]] == k).sum()),
                                       "n_by_context": {nm: float(cd.CNT[i, k, j]) for nm, j in cd.ctx_idx.items()}})
        rows["causes_created_so_far"] = int(cd.CRE[i])
        snap[tag] = rows

    first_trials = []
    for k in range(12):
        trial(npn, c, rng, "acq", CTX_A)
        d = npn.cond.last_detail.get(CUE_NAME)
        if d and k < 4:
            first_trials.append({"acq_trial": k, "target_y": d["y_x"], "post": d["post"], "post_new": d["post_new"], "map": d["map"], "created": d["created"]})
    snap["first_acquisition_trials"] = first_trials
    dump_state("after_acquisition")
    for k in range(20):
        trial(npn, c, rng, "ext", CTX_B)
    dump_state("after_extinction")
    return snap


def arithmetic():
    """E1: 獲得直後（t = 24、質量 M0）と 50 経験後（t = 74、質量 M1）の応答の比 = [M1/(M1 + α)] / [M0/(M0 + α)]（原因が 1 つで文脈の項が 1 のとき）。"""
    out = {}
    times = [2 * i for i in range(12)]
    for d in (0.5, 1.0, 2.0):
        m0 = sum((24 - t + 1.0) ** -d for t in times)
        m1 = sum((74 - t + 1.0) ** -d for t in times)
        row = {}
        for alpha in (0.5, 0.1, 0.05, 0.01):
            row[f"alpha={alpha}"] = (m1 / (m1 + alpha)) / (m0 / (m0 + alpha))
        out[f"d={d}"] = {"M0": m0, "M1": m1, "E1_ratio_if_single_cause_with_12_trials": row}
    return out


def main():
    res = {"anatomy_seed0": anatomy(0), "E1_arithmetic": arithmetic()}
    table = {}
    for name in VARIANTS:
        rows = pmap(job, [(name, s) for s in SEEDS])
        table[name] = summarize(rows)
        print(f"{name:42s} " + " ".join(f"{k}={v:+.3f}" for k, v in table[name].items()))
    res["variants"] = table
    res["thresholds (SPEC-v3.3 4.4)"] = {"E1": ">= 0.7", "E2_B": "<= 0.3", "E3_ABA": ">= 0.5", "E4_ABC": ">= 0.4", "E5_increase": ">= 0.15",
                                       "E6_difference": ">= 0.15", "E7_ratio": "<= 0.6", "E9_difference": ">= 0.1"}
    dump("diag_e_revised", res)
    a = res["anatomy_seed0"]
    for tag in ("first_acquisition_trials", "after_acquisition", "after_extinction"):
        print(tag, str(a[tag])[:1400])
    print("E1 arithmetic:", res["E1_arithmetic"])


if __name__ == "__main__":
    main()
