"""診断（テストではない）: E 系の不合格の原因の切り分け。判定には使わない。結果は tests/results/diag_e.json。

E 系（SPEC-v3.3 4.4）を既定（SPEC どおりの式）で走らせた後に足した事後の診断で、次を調べる。
  1. 原因の内訳: 消去の後の原因の状態（価値・計数・時刻）と、文脈 A・B・C での事前の重み（なぜ ABA / ABC が小さいか）
  2. E7 の single_cause が「節約なし」にならない原因: 関心の重み a の感作（no_concern = True で切り分け）
  3. σ_r への感度（0.15 は SPEC の値。この実装の C の尺度（獲得直後の |C| ≈ 0.2）に対して大きい: d' = |C|/σ_r ≈ 1.3）
  4. 構造の変種（ライブラリには入れていない。診断のためのサブクラス）:
       power  : 時間の重みを冪乗則 1/(1 + Δ/τ) にする（Gershman et al. 2010 の ddCRP は冪乗則。指数関数では原因どうしの比が時間で変わらない）
       ctx    : 復位で、t_k の更新に加えて、その文脈の計数を 1 足す（手がかりなしの嫌悪の結果を、嫌悪の原因に帰属する）
       all    : power + ctx + σ_r = 0.05
**これは改訂前（lc_design = "spec_literal"）の診断**。設計の改訂後は、改訂後の診断が diagnostics_e_revised.py。
実行: python3 tests/diagnostics_e.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e_protocol import (CTX_A, CTX_B, CTX_C, CUE_NAME, Counter, EXT_CONTEXTS_3, lone_harm, run_variant, trial)  # noqa: E402
from helpers import SEEDS, dump, make, pmap  # noqa: E402
from npnv3.value import Conditioning  # noqa: E402


class PowerKernel(Conditioning):
    def _recency(self, dt):
        return 1.0 / (1.0 + dt / self.cfg.lc_tau)


def use_power(npn):
    npn.cond.__class__ = PowerKernel


def lone_harm_ctx(npn, c, rng, ctx, n):
    """復位の変種: t_k の更新に加えて、更新した原因の文脈の計数を 1 足す。"""
    for _ in range(n):
        lone_harm(npn, c, rng, ctx, 1)
        for cue, k in npn.ledger.entries[-1]["td"]["reinstated"]:
            i = npn.cond.idx[cue]
            j = npn.cond._ctx_col(ctx, True)
            npn.cond.CNT[i, k, j] += 1.0


def summarize(rows, key, which="latent"):
    return float(np.mean([r[key] for r in rows]))


def measures(rows, full=True):
    m = {"E2_B": summarize(rows, "E2_B"), "E3_ABA": summarize(rows, "ABA"), "E4_ABC": summarize(rows, "ABC")}
    if full:
        m["E1"] = summarize(rows, "E1")
        m["E5_increase"] = float(np.mean([r["E5_after"] - r["E2_B"] for r in rows]))
        m["E6_difference"] = float(np.mean([r["E6_reinstated"] - r["E6_control"] for r in rows]))
        m["E7_ratio"] = float(np.mean([r["E7_re"] for r in rows]) / np.mean([r["E7_first"] for r in rows]))
    return m


def job(args):
    name, seed = args
    kw = {"lc_design": "spec_literal"}        # 改訂前（SPEC-v3.3 4.2 のとおり）の診断。設計の改訂後も、改訂前の数値を再現するために spec_literal を指定する
    post = None
    lone = None
    single = False
    if name == "default":
        pass
    elif name == "single":
        single = True
    elif name.startswith("sigma") and "+" not in name:
        kw["lc_sigma_r"] = float(name[5:])
    elif name == "no_concern":
        kw["no_concern"] = True
    elif name == "single_no_concern":
        single = True; kw["no_concern"] = True
    elif name == "power":
        post = use_power
    elif name == "ctx":
        lone = lone_harm_ctx
    elif name == "power+ctx":
        post = use_power; lone = lone_harm_ctx
    elif name == "all":
        post = use_power; lone = lone_harm_ctx; kw["lc_sigma_r"] = 0.05
    elif name == "sigma0.05+power":
        post = use_power; kw["lc_sigma_r"] = 0.05
    if post is None:
        r = {"seed": seed, "main": run_variant(seed, single_cause=single, lone_harm_fn=lone, **kw)}
    else:
        r = {"seed": seed, "main": run_variant_full_no_restore(seed, single, post, lone, kw)}   # 保存・復元（E8c）は変種では意味がないので飛ばす
    r["e9_one"] = run_variant(seed, single_cause=single, full=False, post_make=post, **kw)["ABC"]
    r["e9_three"] = run_variant(seed, single_cause=single, ext_contexts=EXT_CONTEXTS_3, full=False, post_make=post, **kw)["ABC"]
    return r


def run_variant_full_no_restore(seed, single, post, lone, kw):
    """変種用: E8(c)（保存・復元）を飛ばして E1〜E7 を測る。run_variant の full 部分から保存・復元だけを除いた版。"""
    import copy
    from e_protocol import (N_ACQ, N_EXT, N_REINSTATE, N_UNRELATED, RE_ACQ_MAX, first_reaching, pretrain_controls, read, unrelated)
    rng = np.random.default_rng([seed, 303])
    c = Counter(f"E{seed}")
    npn = make(seed, single_cause=single, **kw)
    if post is not None:
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
    (lone or lone_harm)(m6, c, np.random.default_rng([seed, 307]), CTX_B, N_REINSTATE)
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


def anatomy(seed=0):
    """消去の後の原因の状態と、文脈 A・B・C での事前の重み。"""
    rng = np.random.default_rng([seed, 303]); c = Counter("D")
    npn = make(seed, lc_design="spec_literal")
    for _ in range(12):
        trial(npn, c, rng, "acq", CTX_A)
    snap = {}
    def dump_state(tag):
        cd = npn.cond
        i = cd.idx[CUE_NAME]
        t = float(npn._clock())
        rows = {"t": t, "causes": []}
        w = {ctx: cd._prior_weights(np.array([i]), ctx, t)[0] for ctx in (CTX_A, CTX_B, CTX_C)}
        for k in range(cd.K):
            if cd.AL[i, k] > 0.5:
                rows["causes"].append({"k": k, "C": float(cd.CK[i, k]), "N": float(cd.CNT[i, k].sum()),
                                       "n_by_context": {nm: float(cd.CNT[i, k, j]) for nm, j in cd.ctx_idx.items()},
                                       "last_used": float(cd.TK[i, k]),
                                       "prior_weight": {ctx: float(w[ctx][k]) for ctx in w}})
        rows["alpha"] = cd.cfg.lc_alpha
        rows["response"] = {ctx: cd.value_of(CUE_NAME, ctx, t) for ctx in w}
        for ctx in w:
            tot = float(w[ctx].sum()) + cd.cfg.lc_alpha
            rows[f"share_{ctx}"] = [float(x / tot) for x in w[ctx]] + [float(cd.cfg.lc_alpha / tot)]
        snap[tag] = rows
    dump_state("after_acquisition")
    first = []
    for k in range(20):
        trial(npn, c, rng, "ext", CTX_B)
        d = npn.cond.last_detail.get(CUE_NAME)
        if d and k < 4:
            first.append({"trial": k, "posterior": d["post"], "posterior_new": d["post_new"], "created": d["created"]})
    snap["first_extinction_trials"] = first
    dump_state("after_extinction")
    return snap


def main():
    res = {"anatomy_seed0": anatomy(0)}
    names = ["default", "single", "no_concern", "single_no_concern", "sigma0.10", "sigma0.05", "sigma0.03",
             "power", "ctx", "power+ctx", "sigma0.05+power", "all"]
    table = {}
    for name in names:
        rows = pmap(job, [(name, s) for s in SEEDS])
        main_rows = [r["main"] for r in rows]
        m = measures(main_rows)
        m["E9_one_minus_three"] = float(np.mean([r["e9_one"] - r["e9_three"] for r in rows]))
        table[name] = m
        print(f"{name:18s} " + " ".join(f"{k}={v:+.3f}" for k, v in m.items()))
    res["variants"] = table
    res["thresholds (SPEC-v3.3 4.4, for reading the table)"] = {
        "E1": ">= 0.7", "E2_B": "<= 0.3", "E3_ABA": ">= 0.5 (single <= 0.3)", "E4_ABC": ">= 0.4 (single <= 0.3)",
        "E5_increase": ">= 0.15 (single < 0.05)", "E6_difference": ">= 0.15 (single < 0.05)",
        "E7_ratio": "<= 0.6 (single >= 0.9)", "E9_one_minus_three": ">= 0.1"}
    dump("diag_e", res)
    a = res["anatomy_seed0"]
    for tag in ("after_acquisition", "after_extinction"):
        print(tag, json.dumps(a[tag], ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
