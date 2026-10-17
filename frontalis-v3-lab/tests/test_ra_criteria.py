"""v3.4 Stage 5: SPEC-v3.4 第 8 節の判定基準 RA0〜RA14 と RA-Y4 の評価（比較対象としての Y 系。ユーザーの 2026-10-17 の再定義の下では、NPN 研究の達成ではない）。

使い方:
  python -B tests/test_ra_criteria.py evaluate --seeds-dir <per-seed JSON のあるディレクトリ> --out <RA*.json と stage5_summary.md を書くディレクトリ> [--phase stage5]
  python -B tests/test_ra_criteria.py            （単体テストだけ。評価の種も予行の種も実行しない。合成データで判定の論理を確かめる）

入力は tests/stage5_run.py が書く stage5_seed_<seed>.json（種ごと）。ここでは計算を走らせず、結果の集計と判定だけを行う。統計の単位は種。
判定の定義は SPEC のまま。曖昧な点は CHANGELOG-v3.4.md の S5-n に先に記録し、判定を易しくしない読みを選ぶ。2 通りの読みがある場合は、
**両方の読みが合格のときだけ合格、片方だけなら「部分的な合格」**として書く（dual 規則。S5-1）。「人格が形成された」「循環が成立した」とは書かない。
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_histories_v34 as H  # noqa: E402
import ra_protocol as RP  # noqa: E402

LABEL = "比較対象（Y 系）。ユーザーの 2026-10-17 の再定義の下では、本来の NPN 研究の達成ではない。"
CELLS = [(e, b, s) for e in H.ERRS for b in H.BGS for s in H.SRCS]
BG_PAIRS = (("trust", "betray"), ("trust", "neutral"), ("betray", "neutral"))
KEY_TAU = "collaborator|project_meeting"
TOL = 1e-9
FORBIDDEN_PHRASES = ("人格が形成された", "循環が成立した", "personality formed", "loop established")


# ======================================================================
# 小さな道具
# ======================================================================
def ck(cell: Tuple[str, str, str]) -> str:
    return "|".join(cell)


def combine(a: Optional[bool], b: Optional[bool]) -> str:
    """dual 規則（S5-1）: 2 通りの読みの両方が真なら pass、片方だけなら partial、どちらも偽なら fail。判定できなければ undetermined。"""
    if a is None or b is None:
        return "undetermined"
    return "pass" if (a and b) else ("partial" if (a or b) else "fail")


def per_err_verdict(oks: Dict[str, Optional[bool]]) -> str:
    """ERR ごとの合否から全体の判定。3 つすべてで合格なら pass、一部なら partial、なければ fail（SPEC RA1・RA2。K1: 3 つすべて）。"""
    vals = list(oks.values())
    if len(vals) != 3 or any(v is None for v in vals):
        return "undetermined"
    n = sum(bool(v) for v in vals)
    return "pass" if n == 3 else ("partial" if n >= 1 else "fail")


def ci_scalar(x: Sequence[float], level: float = 0.95) -> Dict[str, float]:
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m, lo, hi = RP.mean_ci(x.reshape(-1, 1), level)
    return {"mean": float(m[0]), "lo": float(lo[0]), "hi": float(hi[0]), "n": int(x.size)}


def boot_ratio(num: Sequence[float], den: Sequence[float], level: float = 0.95) -> Dict[str, float]:
    """平均どうしの比 mean(num)/mean(den) の、種を単位にしたブートストラップのパーセンタイル区間。"""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    n = len(num)
    if n == 0 or abs(den.mean()) < 1e-15:
        return {"ratio": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": n}
    idx = RP._boot_idx(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"ratio": float(num.mean() / den.mean()), "lo": float(np.nanpercentile(r, lo)), "hi": float(np.nanpercentile(r, hi)), "n": n}


def frac_report(k: int, n: int, level: float = 0.95) -> Dict[str, float]:
    p, lo, hi = RP.wilson(k, n, level)
    return {"k": int(k), "n": int(n), "p": p, "wilson_lo": lo, "wilson_hi": hi}


def finite(x: Any) -> float:
    return float(x) if x is not None and np.isfinite(x) else float("nan")


# ======================================================================
# データ
# ======================================================================
class Data:
    def __init__(self, per_seed: Dict[int, Dict[str, Any]]):
        self.d = per_seed
        self.seeds = sorted(per_seed)
        self.valid: Dict[Tuple[str, str, str], Dict[int, bool]] = {}
        self._scale: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def run(self, s: int, cell: Tuple[str, str, str]) -> Dict[str, Any]:
        return self.d[s]["main"][ck(cell)]["run"]

    def cond(self, s: int, cell, c: str) -> Dict[str, Any]:
        return self.run(s, cell)["cond"][c]

    def R(self, s: int, cell, c: str, which: str) -> np.ndarray:
        return np.asarray(self.cond(s, cell, c)[which], dtype=np.float64)

    def ctrl(self, s: int, cell, c: str) -> Dict[str, Any]:
        key = json.dumps(RP.get_condition(c).prefix_overrides, default=str)
        return self.run(s, cell)["ctrl"][key]

    def checks(self, s: int, cell) -> Dict[str, Any]:
        return self.d[s]["checks"][ck(cell)]

    @property
    def scale(self) -> Tuple[np.ndarray, np.ndarray]:
        """正規化の尺度（SPEC 第 7.3 節）: 主要因計画の全個体の COR の直前の R（C-v34）の標準偏差。"""
        if self._scale is None:
            X = np.array([self.R(s, c, "C-v34", "R_pre") for s in self.seeds for c in CELLS])
            self._scale = RP.scale_of(X)
        return self._scale

    def norm(self, X: np.ndarray, cols: Optional[Sequence[int]] = None) -> np.ndarray:
        sc, mk = self.scale
        Y = RP.normalize(X, sc, mk)
        return Y[..., list(cols)] if cols is not None else Y

    def cols(self, which: str) -> List[int]:
        """マスクを通した後の成分の位置のうち、価値の部分 V または予測と受け止め方の部分 P に当たるもの。"""
        _, mk = self.scale
        kept = list(np.where(mk)[0])
        idx = RP.IDX_V if which == "V" else RP.IDX_P
        return [i for i, j in enumerate(kept) if j in idx]

    def ok_seeds(self, cells: Sequence[Tuple[str, str, str]]) -> List[int]:
        return [s for s in self.seeds if all(self.valid.get(c, {}).get(s, False) for c in cells)]


def pair_test(X1: Dict[int, np.ndarray], X2: Dict[int, np.ndarray], data: Data, cols: Optional[Sequence[int]] = None, level: float = RP.LEVEL_MAIN) -> Optional[Dict[str, Any]]:
    """2 つの条件（種 → 生の CE ベクトル）の向きが違うか。両方に値のある種だけを使う（RA0 で除いた個体を除く）。"""
    common = sorted(set(X1) & set(X2))
    if len(common) < 3:
        return None
    A = data.norm(np.array([X1[s] for s in common]), cols)
    B = data.norm(np.array([X2[s] for s in common]), cols)
    r = RP.directions_differ(A, B, level)
    r["n_seeds"] = len(common)
    r["mean1"] = [float(v) for v in A.mean(axis=0)]
    r["mean2"] = [float(v) for v in B.mean(axis=0)]
    return r


# ======================================================================
# 判定
# ======================================================================
def ra0(data: Data) -> Dict[str, Any]:
    cells_out: Dict[str, Any] = {}
    excluded: List[Dict[str, Any]] = []
    ok_all = True
    for cell in CELLS:
        v: Dict[int, bool] = {}
        for s in data.seeds:
            a, nc = data.R(s, cell, "C-v34", "R_pre"), data.R(s, cell, "C-nocorr-v34", "R_cor")
            dpre = abs(a[0] - data.R(s, cell, "C-v34", "R_ctrl_pre")[0])
            imm = float(np.max(np.abs(data.R(s, cell, "C-v34", "R_cor") - nc))) > 1e-12
            v[s] = bool(dpre > 1e-6 and imm)
            if not v[s]:
                excluded.append({"seed": s, "cell": ck(cell), "abs_d_pre": float(dpre), "CE_imm_nonzero": bool(imm)})
        data.valid[cell] = v
        k = sum(v.values())
        fr = frac_report(k, len(v))
        cells_out[ck(cell)] = {**fr, "ok": bool(fr["p"] >= 0.95)}
        ok_all &= bool(fr["p"] >= 0.95)
    return {"id": "RA0", "verdict": "pass" if ok_all else "fail", "label": LABEL, "cells": cells_out, "n_excluded_individuals": len(excluded), "excluded": excluded[:200],
            "rule": "各セルで 95% 以上の種が |d_pre| > 1e-6 かつ CE_imm ≠ 0（|差| > 1e-12）を満たす。満たさない個体は以降の判定から除く（pairwise: 比べる 2 つの両方に値のある種だけを使う）",
            "scale_components_dropped": [int(i) for i in np.where(~data.scale[1])[0]]}


def ce_cell(data: Data, cell, ca: str, cb: str, which: str = "R_fol") -> Dict[int, np.ndarray]:
    return {s: data.R(s, cell, ca, which) - data.R(s, cell, cb, which) for s in data.seeds if data.valid[cell].get(s)}


def ra1_for(data: Data, ca: str, cb: str) -> Dict[str, Any]:
    """SRC-none の各 ERR で、BG の 3 組の CE の向きを比べる。V・P の部分と、自己強化の点検も添える。"""
    out: Dict[str, Any] = {"cond": ca, "baseline": cb, "by_err": {}}
    for err in H.ERRS:
        ce = {bg: ce_cell(data, (err, bg, "none"), ca, cb) for bg in H.BGS}
        pairs: Dict[str, Any] = {}
        n_diff = 0
        n_det = 0
        for a, b in BG_PAIRS:
            r = pair_test(ce[a], ce[b], data)
            if r is None:
                pairs[f"{a}~{b}"] = None
                continue
            n_det += 1
            n_diff += int(r["differ"])
            rv, rp = pair_test(ce[a], ce[b], data, data.cols("V")), pair_test(ce[a], ce[b], data, data.cols("P"))
            pairs[f"{a}~{b}"] = {"differ": r["differ"], "a_opposite_sign": r["a_opposite_sign"], "opposite_components": r["opposite_components"], "b_cos_upper_lt_0.5": r["b_cos_upper_lt_0.5"],
                                 "cos": r["cos"], "cos_lo": r["cos_lo"], "cos_hi": r["cos_hi"], "n_seeds": r["n_seeds"],
                                 "V_part": None if rv is None else {"differ": rv["differ"], "cos": rv["cos"], "cos_hi": rv["cos_hi"]},
                                 "P_part": None if rp is None else {"differ": rp["differ"], "cos": rp["cos"], "cos_hi": rp["cos_hi"]}}
        # (ii) 自己強化だけかの点検: cos(CE_b, R_前,b − R_前,対照)。条件 ca の自分の R_pre・R_ctrl_pre を使う
        selfc: Dict[str, Any] = {}
        for bg in H.BGS:
            cell = (err, bg, "none")
            seeds = sorted(ce[bg])
            if not seeds:
                continue
            m_ce = data.norm(np.array([ce[bg][s] for s in seeds])).mean(axis=0)
            m_pre = data.norm(np.array([data.R(s, cell, ca, "R_pre") - data.R(s, cell, ca, "R_ctrl_pre") for s in seeds])).mean(axis=0)
            selfc[bg] = RP.cosine(m_ce, m_pre)
        all_hi = bool(selfc) and all(abs(c) > 0.95 for c in selfc.values() if np.isfinite(c)) and len(selfc) == 3
        out["by_err"][err] = {"pairs": pairs, "n_pairs_differ": n_diff, "n_pairs_determined": n_det, "ok": (n_diff >= 2) if n_det == 3 else None,
                              "self_reinforcement_cos": selfc, "ce_only_scaling_of_existing_evaluation": all_hi}
    oks = {e: v["ok"] for e, v in out["by_err"].items()}
    out["ok_by_err"] = oks
    out["verdict"] = per_err_verdict(oks)
    return out


def ra1(data: Data) -> Dict[str, Any]:
    main = ra1_for(data, "C-v34", "C-nocorr-v34")
    v33 = ra1_for(data, "C-v33", "C-nocorr")
    toff = ra1_for(data, "A-τ-off", "C-nocorr-v34")
    ok34 = main["verdict"] in ("pass", "partial")
    note = []
    if main["verdict"] == "fail":
        note.append("RA1 不合格: 「同じ訂正で人格ごとに向きが違う学習」は、この機構と経験史の範囲では示せなかった。閾値も経験史も変えない。")
    if main["verdict"] in ("pass", "partial"):
        for nm, r in (("C-v33", v33), ("A-τ-off", toff)):
            if r["verdict"] in ("pass", "partial"):
                note.append(f"RA1 は {nm} でも（{r['verdict']}。ERR ごと: {r['ok_by_err']}）合格側: その向きの違いは M3 の寄与ではない（既存の階層カルマンの Σ、または M1 の寄与）。")
    return {"id": "RA1", "verdict": main["verdict"], "label": LABEL, "ok_by_err": main["ok_by_err"], "c_v34": main, "c_v33": v33, "a_tau_off": toff, "conclusion_notes": note,
            "rule": "SRC-none の各 ERR で BG の 3 組中 2 組以上の向きが違えばその ERR で合格。3 つの ERR すべてで合格なら pass、一部なら partial（合格した ERR を明記）、なければ fail。98.33% 区間（Bonferroni）"}


def ra2(data: Data) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_err": {}}
    oks: Dict[str, Optional[bool]] = {}
    oks_hm: Dict[str, Optional[bool]] = {}
    for err in H.ERRS:
        sw = {s: data.d[s]["swap"][err]["ev"] for s in data.seeds if err in data.d[s]["swap"]}
        n_excl = sum(len(v["excluded"]) for v in sw.values())
        directed: Dict[str, Any] = {}
        nd, npair = 0, 0
        for bt in H.BGS:
            for bs in H.BGS:
                if bt == bs:
                    continue
                key = f"{bt}<-{bs}"
                have = [s for s in sw if key in sw[s]["val"]]
                if len(have) < max(3, len(sw) // 2 + 1):
                    directed[key] = None
                    continue
                own = {s: np.array(sw[s]["CE"][bt]) for s in have}
                swp = {s: np.array(sw[s]["val"][key]) for s in have}
                tgt = {s: np.array(sw[s]["CE"][bs]) for s in have}
                r = pair_test(own, swp, data)
                c_own = RP.cosine(data.norm(np.array([own[s] for s in have])).mean(0), data.norm(np.array([tgt[s] for s in have])).mean(0))
                c_swp = RP.cosine(data.norm(np.array([swp[s] for s in have])).mean(0), data.norm(np.array([tgt[s] for s in have])).mean(0))
                directed[key] = {"differ": r["differ"], "a_opposite_sign": r["a_opposite_sign"], "b_cos_upper_lt_0.5": r["b_cos_upper_lt_0.5"], "cos_own_vs_swapped": r["cos"], "cos_hi": r["cos_hi"],
                                 "n_seeds": r["n_seeds"], "cos_CE_b_vs_CE_bprime": c_own, "cos_CE_swapped_vs_CE_bprime": c_swp, "swapped_moved_closer_to_bprime": bool(c_swp > c_own)}
                npair += 1
                nd += int(r["differ"])
        ok = (nd >= 4) if npair == 6 else None
        oks[err] = ok
        hm: Dict[str, Any] = {}
        okp = 0
        n_hm = 0
        for bt, bs in BG_PAIRS:
            key = f"{bt}<-{bs}"
            rho = []
            for s in sw:
                if key not in sw[s]["hm"]:
                    continue
                ceb = data.norm(np.array(sw[s]["CE"][bt]))
                cebp = data.norm(np.array(sw[s]["CE"][bs]))
                gap = float(np.linalg.norm(ceb - cebp))
                if gap > 1e-12:
                    rho.append(float(np.linalg.norm(data.norm(np.array(sw[s]["hm"][key])) - cebp) / gap))
            if rho:
                n_hm += 1
                med = float(np.median(rho))
                hm[key] = {"median_rho": med, "n_seeds": len(rho), "ok": bool(med >= 0.5)}
                okp += int(med >= 0.5)
            else:
                hm[key] = None
        oks_hm[err] = (okp >= 2) if n_hm == 3 else None
        out["by_err"][err] = {"directed": directed, "n_directed_differ": nd, "n_directed_determined": npair, "ok": ok, "n_excluded_swaps": n_excl,
                              "hm_eq": hm, "hm_ok": oks_hm[err]}
    out["ok_by_err"] = oks
    out["verdict"] = per_err_verdict(oks)
    out["RA2_hm"] = {"ok_by_err": oks_hm, "verdict": per_err_verdict(oks_hm), "note": "hm-eq は C-v34 で欲求状態 h と気分 m だけを揃える。ρ = ‖CE^{hm←b′} − CE_b′‖ / ‖CE_b − CE_b′‖ の種ごとの中央値 ≥ 0.5 が各 ERR で 3 組中 2 組以上"}
    out["shapley"] = shapley_report(data)
    al: Dict[str, Any] = {}
    for err in H.ERRS:
        k1 = k2 = n = 0
        for s in data.seeds:
            for key, v in data.d[s].get("align", {}).get(err, {}).items():
                n += 1
                k1 += int(v["aligned"])
                k2 += int(v["aligned_literal"])
        al[err] = {"n_directed_pairs": n, "aligned_reading_L1": k1, "aligned_literal_reading(sensitivity)": k2, "excluded_under_L1": n - k1, "excluded_under_literal": n - k2}
    out["alignment_counts"] = al
    out["alignment_note"] = "入れ替えは aligned()（リードの判断 L1: 価値ネットの特徴の名前・並びと経験記録の id の並びの一致。C は丸ごと入れ替え）を使う。文面どおり（C の手がかりの並びも一致）では、上の除かれる数のとおりになる"
    notes = []
    out["rule"] = "SRC-none の各 ERR で 6 通り（両方向）中 4 通り以上の向きが違えばその ERR で合格。3 つの ERR すべてで pass、一部で partial、なければ fail。98.33% 区間"
    out["id"] = "RA2"
    out["label"] = LABEL
    out["conclusion_notes"] = notes
    return out


def shapley_report(data: Data) -> Dict[str, Any]:
    """RA2(b) の報告: Shapley 分担（種 3400〜3419、7 群）。速さに当たる群（G3、G5 の κ = G5k）と中身に当たる群（G1、G2、G4、G5 の τ = G5t）を分ける。G6（h・m）は別に書く。"""
    out: Dict[str, Any] = {"players": list(RP.SHAPLEY_PLAYERS_SPLIT_G5), "by_err": {}}
    for err in H.ERRS:
        out["by_err"][err] = {}
        for bt, bs in BG_PAIRS:
            key = f"{err}|{bt}<-{bs}"
            rows = []
            for s in data.seeds:
                run = data.d[s]["shapley"].get(key)
                if run is None:
                    continue
                r = RP.shapley_from_run(run, *data.scale)
                if r["phi"] is not None:
                    rows.append(r)
            if not rows:
                out["by_err"][err][f"{bt}<-{bs}"] = None
                continue
            phi = {g: [r["phi"][g] for r in rows] for g in RP.SHAPLEY_PLAYERS_SPLIT_G5}
            speed = [r["phi"]["G3"] + r["phi"]["G5k"] for r in rows]
            content = [r["phi"]["G1"] + r["phi"]["G2"] + r["phi"]["G4"] + r["phi"]["G5t"] for r in rows]
            out["by_err"][err][f"{bt}<-{bs}"] = {"n_seeds": len(rows), "n_gap_zero_excluded": sum(1 for s in data.seeds if data.d[s]["shapley"].get(key) is not None) - len(rows),
                                                 "phi": {g: ci_scalar(v) for g, v in phi.items()}, "speed_G3_G5k": ci_scalar(speed), "content_G1_G2_G4_G5t": ci_scalar(content),
                                                 "mixed_G6_h_m": ci_scalar(phi["G6"]), "v_all_mean": float(np.mean([r["v_all"] for r in rows]))}
    return out


def tau_final(data: Data, s: int, cell) -> float:
    lt = data.checks(s, cell)["ev"]["v34"]["log_tau"]
    return float(lt.get(KEY_TAU, 0.0))


def ra3(data: Data) -> Dict[str, Any]:
    err = "interp"
    dtau: Dict[Tuple[str, str], Dict[int, np.ndarray]] = {}
    for bg in H.BGS:
        for src in H.SRCS:
            cell = (err, bg, src)
            d = {}
            for s in data.seeds:
                if not data.valid[cell].get(s):
                    continue
                a = data.R(s, cell, "C-v34", "R_fol") - data.R(s, cell, "C-v34", "R_cor")
                b = data.R(s, cell, "clamp-τ", "R_fol") - data.R(s, cell, "clamp-τ", "R_cor")
                d[s] = a - b
            dtau[(bg, src)] = d
    # (1) 非ゼロ
    one: Dict[str, Any] = {}
    one_ok_literal = True
    for bg in H.BGS:
        seeds = sorted(set.intersection(*[set(dtau[(bg, s)]) for s in H.SRCS]))
        if len(seeds) < 3:
            one[bg] = None
            one_ok_literal = False
            continue
        pooled = data.norm(np.array([np.mean([dtau[(bg, sr)][s] for sr in H.SRCS], axis=0) for s in seeds]))
        m, lo, hi = RP.mean_ci(pooled, RP.LEVEL_MAIN)
        nz = [int(i) for i in np.where((lo > 0) | (hi < 0))[0]]
        one[bg] = {"n_seeds": len(seeds), "nonzero_components": nz, "ok": bool(nz), "mean": [float(v) for v in m], "lo": [float(v) for v in lo], "hi": [float(v) for v in hi]}
        one_ok_literal &= bool(nz)
    one_strict: Dict[str, Any] = {}
    one_ok_strict = True
    for bg in H.BGS:
        for src in H.SRCS:
            d = dtau[(bg, src)]
            seeds = sorted(d)
            if len(seeds) < 3:
                one_strict[f"{bg}|{src}"] = None
                one_ok_strict = False
                continue
            m, lo, hi = RP.mean_ci(data.norm(np.array([d[s] for s in seeds])), RP.LEVEL_MAIN)
            nz = [int(i) for i in np.where((lo > 0) | (hi < 0))[0]]
            one_strict[f"{bg}|{src}"] = {"nonzero_components": nz, "ok": bool(nz), "max_abs_mean": float(np.max(np.abs(m)))}
            one_ok_strict &= bool(nz)
    # (2) 交互作用（SRC ごとに BG 3 組の向き）
    two: Dict[str, Any] = {}
    n_src_ok = 0
    for src in H.SRCS:
        nd, nde = 0, 0
        prs = {}
        for a, b in BG_PAIRS:
            r = pair_test(dtau[(a, src)], dtau[(b, src)], data)
            if r is None:
                prs[f"{a}~{b}"] = None
                continue
            nde += 1
            nd += int(r["differ"])
            prs[f"{a}~{b}"] = {"differ": r["differ"], "a_opposite_sign": r["a_opposite_sign"], "opposite_components": r["opposite_components"], "cos": r["cos"], "cos_hi": r["cos_hi"], "n_seeds": r["n_seeds"]}
        ok = (nd >= 2) if nde == 3 else None
        two[src] = {"pairs": prs, "n_pairs_differ": nd, "ok": ok}
        n_src_ok += int(bool(ok))
    two_ok = n_src_ok >= 2
    # (3)(4)(5)
    inv = merged_checks(data)
    i3, i4, i5 = inv["RA3_3"], inv["RA3_4"], inv["RA3_5"]
    broad5_ok, broad5_n = True, 0
    for e in ("report", "report-x"):
        for bg in H.BGS:
            for src in H.SRCS:
                cell = (e, bg, src)
                for s in data.seeds:
                    broad5_n += 1
                    same = (np.array_equal(data.R(s, cell, "C-v34", "R_fol"), data.R(s, cell, "clamp-τ", "R_fol")) and np.array_equal(data.R(s, cell, "C-v34", "R_cor"), data.R(s, cell, "clamp-τ", "R_cor")))
                    no_tau = data.checks(s, cell)["ev"]["v34"]["log_tau"] == {}
                    broad5_ok &= bool(same and no_tau)
    # τ の符号の分布
    sign: Dict[str, Any] = {}
    for bg in H.BGS:
        for src in H.SRCS:
            vals = [tau_final(data, s, (err, bg, src)) for s in data.seeds]
            sign[f"{bg}|{src}"] = {"down": sum(v < -1e-12 for v in vals) / len(vals), "up": sum(v > 1e-12 for v in vals) / len(vals), "zero": sum(abs(v) <= 1e-12 for v in vals) / len(vals),
                                   "mean_log_tau": float(np.mean(vals)), "max_abs_log_tau": float(np.max(np.abs(vals)))}
    # τ-yoke（報告）
    yoke: Dict[str, Any] = {}
    for src in H.SRCS:
        yoke[src] = {}
        for t in H.BGS:
            rows = []
            for s in data.seeds:
                if not data.valid[(err, t, src)].get(s):
                    continue
                lyoke = np.mean([np.array(data.d[s]["yoke"][f"{src}|{t}<-{dn}"]["R_fol"]) - np.array(data.d[s]["yoke"][f"{src}|{t}<-{dn}"]["R_cor"]) for dn in H.BGS if dn != t], axis=0)
                lv34 = data.R(s, (err, t, src), "C-v34", "R_fol") - data.R(s, (err, t, src), "C-v34", "R_cor")
                rows.append(lv34 - lyoke)
            if len(rows) >= 3:
                m, lo, hi = RP.mean_ci(data.norm(np.array(rows)), RP.LEVEL_MAIN)
                yoke[src][t] = {"mean": [float(v) for v in m], "nonzero_components": [int(i) for i in np.where((lo > 0) | (hi < 0))[0]], "n_seeds": len(rows)}
        yp = {}
        for a, b in BG_PAIRS:
            xa, xb = {}, {}
            for s in data.seeds:
                for t, store in ((a, xa), (b, xb)):
                    if not data.valid[(err, t, src)].get(s):
                        continue
                    ly = np.mean([np.array(data.d[s]["yoke"][f"{src}|{t}<-{dn}"]["R_fol"]) - np.array(data.d[s]["yoke"][f"{src}|{t}<-{dn}"]["R_cor"]) for dn in H.BGS if dn != t], axis=0)
                    store[s] = (data.R(s, (err, t, src), "C-v34", "R_fol") - data.R(s, (err, t, src), "C-v34", "R_cor")) - ly
            r = pair_test(xa, xb, data)
            yp[f"{a}~{b}"] = None if r is None else {"differ": r["differ"], "cos": r["cos"], "cos_hi": r["cos_hi"]}
        yoke[src]["pairs_differ"] = yp
    inv_ok = bool(i3["ok"] and i4["ok"] and i5["ok"] and broad5_ok)
    pass_literal = bool(one_ok_literal and two_ok and inv_ok)
    pass_strict = bool(one_ok_strict and two_ok and inv_ok)
    verdict = "fail" if not inv_ok else combine(pass_literal, pass_strict)
    notes = []
    if not inv_ok:
        notes.append("RA3(3)(4)(5) の少なくとも 1 つが破れた: F2（学習規則が自分の定義どおりに動いていない）。")
    if verdict in ("fail",) and inv_ok:
        notes.append("RA3 不合格: 「主観的評価 → 次の学習」の経路は、構造としてはあるが、人格ごとに向きを変えるほどには働いていない。")
    return {"id": "RA3", "verdict": verdict, "label": LABEL, "one_nonzero_by_bg_pooled_over_src(literal)": one, "one_ok_literal": one_ok_literal,
            "one_nonzero_by_bg_and_src(strict)": one_strict, "one_ok_strict": one_ok_strict, "two_interaction_by_src": two, "two_n_src_ok": n_src_ok, "two_ok": two_ok,
            "three_isolation_clamp_tau": i3, "four_mood_negative_control": i4, "five_origin_negative_control": i5, "five_broad_main_design": {"ok": broad5_ok, "n_individuals": broad5_n},
            "tau_sign_distribution": sign, "tau_yoke_L_fol_v34_minus_yoke": yoke, "conclusion_notes": notes,
            "reading_note": "(1) は SPEC の「各 BG で Δτ_b のある成分の区間が 0 を含まない」を、SRC を種ごとに平均して BG ごとに（literal）と、BG × SRC の 9 セルすべてで（strict）の両方で判定し、両方合格のときだけ pass、片方だけなら partial（S5-2）",
            "rule": "(1) が全 BG で、かつ (2) が 3 つの SRC のうち 2 つ以上で（BG 3 組中 2 組以上の向きが違う）、かつ (3)(4)(5) が全個体で満たされれば合格"}


def ra4(data: Data) -> Dict[str, Any]:
    def r_ind(cell, s, c) -> float:
        d_pre = data.R(s, cell, c, "R_pre")[0] - data.R(s, cell, c, "R_ctrl_pre")[0]
        d_post = data.R(s, cell, c, "R_cor")[0] - data.R(s, cell, c, "R_ctrl_pre")[0]
        return float(d_post / d_pre) if abs(d_pre) > 1e-12 else float("nan")

    def cell_r(c: str) -> Dict[Tuple, Dict[str, Any]]:
        out = {}
        for cell in CELLS:
            ss = [s for s in data.seeds if data.valid[cell].get(s)]
            rr = np.array([r_ind(cell, s, c) for s in ss])
            dpre = np.array([data.R(s, cell, c, "R_pre")[0] - data.R(s, cell, c, "R_ctrl_pre")[0] for s in ss])
            dpost = np.array([data.R(s, cell, c, "R_cor")[0] - data.R(s, cell, c, "R_ctrl_pre")[0] for s in ss])
            out[cell] = {"seeds": ss, "r": rr, "mean_r": float(np.nanmean(rr)) if len(rr) else float("nan"), "median_r": float(np.nanmedian(rr)) if len(rr) else float("nan"),
                         "sd_r": float(np.nanstd(rr, ddof=1)) if len(rr) > 1 else float("nan"), "ratio_of_means": float(dpost.mean() / dpre.mean()) if len(rr) and abs(dpre.mean()) > 1e-15 else float("nan")}
        return out
    cv34 = cell_r("C-v34")
    means = np.array([cv34[c]["mean_r"] for c in CELLS])
    rom = np.array([cv34[c]["ratio_of_means"] for c in CELLS])
    a_ind = float(np.nanmax(means) - np.nanmin(means))
    a_rom = float(np.nanmax(rom) - np.nanmin(rom))
    a_ok_ind, a_ok_rom = bool(a_ind >= 0.15), bool(a_rom >= 0.15)
    # (b) reliable − unreliable（種ごとに BG・ERR で平均した後）
    diffs = []
    for s in data.seeds:
        def mean_src(src):
            v = [r_ind(c, s, "C-v34") for c in CELLS if c[2] == src and data.valid[c].get(s)]
            return float(np.nanmean(v)) if v else float("nan")
        diffs.append(mean_src("reliable") - mean_src("unreliable"))
    cb = ci_scalar(diffs, RP.LEVEL_OTHER)
    b_ok = bool(cb["n"] >= 3 and (cb["lo"] > 0 or cb["hi"] < 0))
    # (c) セル内の種間の標準偏差が 0.0134 を超えるセルが 27 中 18 以上
    over = [ck(c) for c in CELLS if np.isfinite(cv34[c]["sd_r"]) and cv34[c]["sd_r"] > 0.0134]
    c_ok = bool(len(over) >= 18)
    sub = {}
    for nm in ("C-v33", "A-g-const", "A-κ-fixed", "A-τ-off", "A-clock-off", "C-OR", "local-reattr"):
        cr = cell_r(nm)
        sub[nm] = {"mean_r_by_cell": {ck(c): cr[c]["mean_r"] for c in CELLS}, "ratio_of_means_by_cell": {ck(c): cr[c]["ratio_of_means"] for c in CELLS},
                   "range_mean_r": float(np.nanmax([cr[c]["mean_r"] for c in CELLS]) - np.nanmin([cr[c]["mean_r"] for c in CELLS])),
                   "n_cells_sd_over_0.0134": int(sum(np.isfinite(cr[c]["sd_r"]) and cr[c]["sd_r"] > 0.0134 for c in CELLS))}
    verdict_a = combine(a_ok_ind, a_ok_rom)
    parts = [verdict_a == "pass", b_ok, c_ok]
    verdict = "pass" if all(parts) else ("fail" if not any(p for p in parts) and verdict_a == "fail" else "partial")
    cls = []
    if verdict_a != "pass":
        cls.append("(a) 27 セルの平均の範囲が 0.15 未満（または片方の読みだけ満たす）: 設計仮説と矛盾する性質（再評価の強さが経験史でほとんど変わらない）として記録する。")
    if not c_ok:
        cls.append("(c) 種の情報量の基準を満たさない: 標本数に基づく主張をすべて取り下げる（SPEC）。")
    return {"id": "RA4", "verdict": verdict, "label": LABEL,
            "a": {"range_of_cell_means_individual_r": a_ind, "range_of_cell_ratio_of_means": a_rom, "ok_individual_mean": a_ok_ind, "ok_ratio_of_means": a_ok_rom, "verdict_dual": verdict_a,
                  "cell_mean_r": {ck(c): cv34[c]["mean_r"] for c in CELLS}, "cell_median_r": {ck(c): cv34[c]["median_r"] for c in CELLS}, "cell_ratio_of_means": {ck(c): cv34[c]["ratio_of_means"] for c in CELLS}},
            "b": {"mean_diff_reliable_minus_unreliable": cb, "ok": b_ok, "predicted_sign": "reliable の方が r が小さい（差 < 0）", "sign_matches_prediction": bool(cb["mean"] < 0) if cb["n"] else None},
            "c": {"cells_over_0.0134": len(over), "ok": c_ok, "sd_by_cell": {ck(c): cv34[c]["sd_r"] for c in CELLS}},
            "ablations_and_c_v33": sub, "classification_notes": cls,
            "reading_note": "r は個体ごとの d_post/d_pre（SPEC）。(a) は個体 r の平均と、平均どうしの比（Stage 1 の定義）の両方で判定し、両方が 0.15 以上のときだけ満たす（S5-3）。(b)(c) は個体 r の種間の変動を使う",
            "rule": "(a) 範囲 ≥ 0.15、(b) reliable − unreliable の 95% 区間が 0 を含まない、(c) セル内の種間の標準偏差 > 0.0134 のセルが 27 中 18 以上。全部で pass、一部で partial"}


# ----------------------------------------------------------------------
# 故障の検査の集計
# ----------------------------------------------------------------------
def merged_checks(data: Data) -> Dict[str, Dict[str, Any]]:
    """cell_task / seed_task の inv と、観察者の inv を、検査 ID ごとにまとめる。1 つでも破れたら破れ。"""
    out: Dict[str, Dict[str, Any]] = {}

    def add(name: str, v: Dict[str, Any], meta: Any) -> None:
        e = out.setdefault(name, {"ok": True, "n_run": 0, "n_fail": 0, "first_fails": []})
        e["n_run"] += 1
        if not v["ok"]:
            e["ok"] = False
            e["n_fail"] += 1
            if len(e["first_fails"]) < 3:
                e["first_fails"].append({"meta": meta, "fails": v["fails"][:1]})
    for s in data.seeds:
        d = data.d[s]
        for key, r in d["checks"].items():
            for name, v in r["inv"].items():
                add(name, v, r["meta"])
        if d.get("seed_task"):
            for name, v in d["seed_task"]["inv"].items():
                add(name, v, d["seed_task"]["meta"])
        for key, r in d["main"].items():
            for name, v in r["observer"]["inv"].items():
                add("OBS_" + name, v, {"seed": s, "cell": key})
    for k in ("RA3_3", "RA3_4", "RA3_5", "RA5_range", "RA5_formula", "RA5_pred", "RA5_sigma", "RA5_remainder", "RA5_reconstruct", "RA6", "RA7a", "RA8a", "RA8b", "RA8c", "RA8d",
              "H1", "H2", "H3", "H4", "H5", "H7", "H8", "H9", "RA12"):
        out.setdefault(k, {"ok": True, "n_run": 0, "n_fail": 0, "first_fails": []})
    return out


def sweep_records(data: Data):
    for s in data.seeds:
        for key, t in data.d[s]["sweep"].items():
            for rec in t["records"]:
                yield s, t["bg"], t["err"], rec


def sweep_r(rec: Dict[str, Any], c: str) -> Tuple[float, float, float]:
    x = rec["run"]["cond"][c]
    d_pre = x["R_pre"][0] - x["R_ctrl_pre"][0]
    d_post = x["R_cor"][0] - x["R_ctrl_pre"][0]
    return d_pre, d_post, (d_post / d_pre if abs(d_pre) > 1e-6 else float("nan"))


def r_of_n(data: Data, c: str, filt=None) -> Dict[str, Any]:
    """N ごとに、全セル（M・BG・ERR）をまとめた平均どうしの比 r(N)（Stage 1 と同じ定義）。種ごとに平均してから比をとり、種を単位に区間を作る。"""
    idx = sweep_index(data)
    out = {}
    for N in (2, 4, 8, 12, 16):
        num, den = [], []
        for s in data.seeds:
            dp, dq = [], []
            for bg, err, rec in idx.get((s, N), []):
                if filt and not filt(bg, err, rec):
                    continue
                d_pre, d_post, _ = sweep_r(rec, c)
                dp.append(d_pre)
                dq.append(d_post)
            if dp:
                den.append(float(np.mean(dp)))
                num.append(float(np.mean(dq)))
        out[N] = boot_ratio(num, den)
    return out


_SW_INDEX: Dict[int, Dict] = {}


def sweep_index(data: Data) -> Dict:
    k = id(data)
    if k not in _SW_INDEX:
        d: Dict = {}
        for s, bg, err, rec in sweep_records_cached(data):
            d.setdefault((s, rec["N"]), []).append((bg, err, rec))
        _SW_INDEX[k] = d
    return _SW_INDEX[k]


_SW_CACHE: Dict[int, List] = {}


def sweep_records_cached(data: Data):
    k = id(data)
    if k not in _SW_CACHE:
        _SW_CACHE[k] = list(sweep_records(data))
    return _SW_CACHE[k]


def overshoot_counts(data: Data) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in ("C-v34", "C-OR", "C-v33", "C-retro"):
        by_n: Dict[int, List[int]] = {}
        for s, bg, err, rec in sweep_records_cached(data):
            d_pre, d_post, r = sweep_r(rec, c)
            e = by_n.setdefault(rec["N"], [0, 0, 0])
            if np.isfinite(r):
                e[1] += 1
                e[0] += int(r < -0.05)
            else:
                e[2] += 1
        out[c] = {str(N): {"overshoot": v[0], "n_with_defined_r": v[1], "n_excluded_small_d_pre": v[2], "fraction": v[0] / v[1] if v[1] else float("nan")} for N, v in sorted(by_n.items())}
    return out


def ra5(data: Data) -> Dict[str, Any]:
    inv = merged_checks(data)
    names = ["RA5_range", "RA5_formula", "RA5_pred", "RA5_sigma", "RA5_remainder", "RA5_reconstruct"]
    ok_inv = all(inv[n]["ok"] for n in names)
    n_run_ok = all(inv[n]["n_run"] > 0 for n in names)
    recon = []
    n_over_expected = 0
    for s, bg, err, rec in sweep_records_cached(data):
        for c in ("C-v34", "C-OR"):
            d_pre, d_post, r = sweep_r(rec, c)
            if np.isfinite(r) and r < -0.05:
                n_over_expected += 1
                if c not in rec["recon"]:
                    recon.append({"seed": s, "bg": bg, "err": err, "N": rec["N"], "M": rec["M"], "cond": c, "missing": True})
        for c, x in rec["recon"].items():
            recon.append({"seed": s, "bg": bg, "err": err, "N": rec["N"], "M": rec["M"], "cond": c, "err_dW": x["err_dW"], "err_gD": x["err_gD"], "ok": bool(x["err_dW"] <= TOL and x["err_gD"] <= TOL)})
    recon_ok = all(x.get("ok", False) for x in recon)
    # 主要因計画（N = 8）の反対側へ行った個体
    main_over: Dict[str, Any] = {}
    for c in ("C-v34", "C-OR", "C-v33"):
        k = n = 0
        for cell in CELLS:
            for s in data.seeds:
                if not data.valid[cell].get(s):
                    continue
                d_pre = data.R(s, cell, c, "R_pre")[0] - data.R(s, cell, c, "R_ctrl_pre")[0]
                r = (data.R(s, cell, c, "R_cor")[0] - data.R(s, cell, c, "R_ctrl_pre")[0]) / d_pre
                n += 1
                k += int(r < -0.05)
        main_over[c] = frac_report(k, n)
    curves = {c: r_of_n(data, c) for c in ("C-v34", "C-OR", "C-v33", "C-retro")}
    verdict = "pass" if (ok_inv and recon_ok and n_run_ok) else "fail"
    notes = [] if verdict == "pass" else ["RA5 の不変量または再構成が破れた: F2/F3。"]
    return {"id": "RA5", "verdict": verdict, "label": LABEL, "invariants": {n: inv[n] for n in names}, "sweep_overshoot_reconstruction": {"n_expected_c_v34_c_or": n_over_expected, "n_reconstructed": sum(1 for x in recon if not x.get("missing")),
            "n_missing": sum(1 for x in recon if x.get("missing")), "all_within_1e-9": recon_ok, "max_err_dW": max([x["err_dW"] for x in recon if "err_dW" in x] or [0.0]),
            "max_err_gD": max([x["err_gD"] for x in recon if "err_gD" in x] or [0.0])}, "r_of_N_pooled_all_cells(report)": curves, "overshoot_counts_sweep(report)": overshoot_counts(data),
            "overshoot_main_design_N8(report)": main_over, "conclusion_notes": notes,
            "rule": "各訂正で g ∈ (0, 1)（C-v34）、C-OR の g^OR の式、吸収の後の予測の変化 = g D（1e-9）、Σ 不変、反対側へ行った個体は台帳から 1e-9 以内で再構成。破れたら F2/F3。r(N) と個数は報告"}


def ra5_rc(data: Data) -> Dict[str, Any]:
    def by(filt):
        return r_of_n(data, "C-v33", filt)
    allc = by(None)
    m0 = by(lambda bg, err, rec: rec["M"] == 0)
    m4 = by(lambda bg, err, rec: rec["M"] == 4)
    per_err = {e: by(lambda bg, err, rec, e=e: err == e) for e in H.ERRS}
    # RC2/RC3 の素材（補助。判定は Stage 1 の診断用の種で済み）
    S, absorbed = [], []
    for N in (2, 4, 8, 12, 16):
        vals = [rec["run"]["cond"]["C-v33"]["gain"]["S_all"] for s, bg, err, rec in sweep_records_cached(data) if rec["N"] == N]
        S.append(float(np.mean(vals)) if vals else float("nan"))
        absorbed.append(1.0 - allc[N]["ratio"])
    pe = float(np.corrcoef(S, absorbed)[0, 1]) if np.all(np.isfinite(S)) and np.std(S) > 0 and np.std(absorbed) > 0 else float("nan")
    ind = sh = 0.0
    a2 = a1 = 0.0
    n_a2_moved = n = 0
    for s, bg, err, rec in sweep_records_cached(data):
        x = rec["run"]["cond"]["C-v33"]
        sp = x["layer_split"]["A"]
        ind += sp["ind"]
        sh += sp["people"] + sp["cls"] + sp["emb"]
        d1, d2 = x["R_cor"][0] - x["R_pre"][0], x["R_cor"][1] - x["R_pre"][1]
        a1 += d1
        a2 += d2
        n += 1
        n_a2_moved += int(abs(d2) > 1e-6)
    return {"id": "RA5-RC", "verdict": "report-only", "label": LABEL, "r_of_N_C_v33_all": allc, "M0": m0, "M4": m4, "by_err": per_err,
            "RC2_material": {"S_all_by_N": S, "one_minus_r_by_N": absorbed, "pearson_5points": pe},
            "RC3_material": {"shared_fraction_signed": float(sh / (sh + ind)) if (sh + ind) != 0 else float("nan"), "mean_dlike_A": a1 / n if n else float("nan"), "mean_dlike_A2": a2 / n if n else float("nan"),
                             "fraction_A2_moved": n_a2_moved / n if n else float("nan")},
            "note": "RC1 は diag-q0（再評価のときだけ q = 0）を要し、評価の種の掃引の条件（C-v34・C-OR・C-v33・C-retro・C-nocorr）には含まれないので、ここでは判定しない。RC1〜RC3 の判定は Stage 1 の診断用の種（stage1_rc.json: RC1 当たり、RC2 当たり、RC3 外れ）。"
                    "ここは C-v33 の N の掃引を評価の種で報告するもの（SPEC RA5-RC）"}


def ra6(data: Data) -> Dict[str, Any]:
    inv = merged_checks(data)
    ok = inv["RA6"]["ok"] and inv["OBS_RA6"]["ok"]
    ce_ok = True
    bad = []
    for s in data.seeds:
        for key, r in data.d[s]["main"].items():
            for label, counts in r["observer"]["ce_counts"].items():
                mode, noreap, retro = label.split("|")[:3]
                expected = 8 if (mode.startswith("v34") and noreap == "False" and retro == "False") else 0
                if counts != [expected]:
                    ce_ok = False
                    bad.append({"seed": s, "cell": key, "label": label, "counts": counts, "expected": expected})
    verdict = "pass" if (ok and ce_ok and inv["RA6"]["n_run"] > 0) else "fail"
    return {"id": "RA6", "verdict": verdict, "label": LABEL, "cell_task_C_v34_and_C_OR": inv["RA6"], "observer_all_conditions": inv["OBS_RA6"],
            "correction_experience_count": {"ok": ce_ok, "expected": "M3 を持つ v34 系の条件はちょうど N 件、v33 系・対照（C-none・C-retro）は 0 件", "violations": bad[:20]},
            "conclusion_notes": [] if verdict == "pass" else ["RA6 が破れた: F4（当時の主観的な記録が書き換わった）。"], "rule": "全条件・全個体で当時の台帳と original_outcomes が完全一致、訂正の経験の追記はちょうど N 件"}


def closeness(P_mis: np.ndarray, P_ctrl: np.ndarray, P_post: np.ndarray) -> Dict[str, float]:
    return boot_ratio(P_mis - P_post, P_mis - P_ctrl)


def ra7(data: Data) -> Dict[str, Any]:
    inv = merged_checks(data)
    a_ok = bool(inv["RA7a"]["ok"] and inv["OBS_RA7a"]["ok"] and inv["RA7a"]["n_run"] > 0)

    def pc(cell_list, c: str, ctrl_which: str = "R_ctrl_pre") -> Dict[str, float]:
        """近づき (P_mis − P_post)/(P_mis − P_ctrl)。種ごとにセルを平均してから、平均どうしの比を種を単位に区間にする。"""
        Pm, Pc, Pp = [], [], []
        for s in data.seeds:
            m_, c_, p_ = [], [], []
            for cell in cell_list:
                if not data.valid[cell].get(s):
                    continue
                m_.append(data.R(s, cell, c, "R_pre")[3])
                c_.append(data.R(s, cell, c, ctrl_which)[3])
                p_.append(data.R(s, cell, c, "R_cor")[3])
            if m_:
                Pm.append(float(np.mean(m_)))
                Pc.append(float(np.mean(c_)))
                Pp.append(float(np.mean(p_)))
        Pm, Pc, Pp = (np.array(x) for x in (Pm, Pc, Pp))
        return closeness(Pm, Pc, Pp)
    table: Dict[str, Any] = {}
    for c in ("C-v34", "C-v33"):
        table[c] = {}
        for err in H.ERRS:
            table[c][err] = {bg: {"pooled_src": pc([(err, bg, s) for s in H.SRCS], c), "src_none": pc([(err, bg, "none")], c)} for bg in H.BGS}
    lr: Dict[str, Any] = {}
    for c in ("local-reattr", "local-reattr-v33"):
        lr[c] = {}
        for err in H.ERRS:
            lr[c][err] = {bg: pc([(err, bg, s) for s in H.SRCS], c, "R_ctrl_cor") for bg in H.BGS}
    # (d) 近づきの不足 1 − 近づき の BG 3 セルの間の範囲（C-v34）。両方の読み（SRC をまとめる / SRC-none）で 0.1 以上のときだけ「経験史に依存する性質」
    d_out: Dict[str, Any] = {}
    all_dep = True
    for c in ("C-v34", "C-v33"):
        d_out[c] = {}
        for err in H.ERRS:
            sh_pool = [1.0 - table[c][err][bg]["pooled_src"]["ratio"] for bg in H.BGS]
            sh_none = [1.0 - table[c][err][bg]["src_none"]["ratio"] for bg in H.BGS]
            rp, rn = float(np.nanmax(sh_pool) - np.nanmin(sh_pool)), float(np.nanmax(sh_none) - np.nanmin(sh_none))
            dep = bool(rp >= 0.1 and rn >= 0.1)
            d_out[c][err] = {"range_pooled_src": rp, "range_src_none": rn, "shortfall_pooled_src": sh_pool, "shortfall_src_none": sh_none, "class": "経験史に依存する性質" if dep else "設計仮説と矛盾する性質（逐次の φ の推定の副産物）"}
            if c == "C-v34":
                all_dep &= dep
    old = old_protocol(data)
    verdict = "pass" if a_ok else "fail"
    return {"id": "RA7", "verdict": verdict, "label": LABEL, "a_counts_match": {"cell_task": inv["RA7a"], "observer_all_conditions": inv["OBS_RA7a"]},
            "b_Y1_closeness_p_person(report-only)": table, "b_old_protocol_Y1_on_evaluation_seeds(C-v33, report-only)": old.get("Y1") if old else None,
            "c_local_reattr_closeness_to_treated_control(report-only)": lr, "d_shortfall_range_over_bg": d_out, "d_classification_C_v34_all_err": "経験史に依存する性質" if all_dep else "（ERR によっては）設計仮説と矛盾する性質",
            "conclusion_notes": [] if a_ok else ["RA7(a) が破れた: F2。"],
            "rule": "(a) 訂正後の信念の計数が観測の記録から作り直した値に一致（F2）。(b)(c) は報告のみ（J5）。(d) 範囲 ≥ 0.1 なら経験史に依存する性質、でなければ設計仮説と矛盾する性質。"
                    "(d) は SRC をまとめた読みと SRC-none の読みの両方で ≥ 0.1 のときだけ前者（S5-4）。verdict は (a) だけで決める"}


def old_protocol(data: Data) -> Optional[Dict[str, Any]]:
    """v3.3 の Y1〜Y7 のプロトコルそのまま（C-v33。評価の種）。tests/test_y_correction.stats() を使う。"""
    rows = [data.d[s]["old_y"] for s in data.seeds if data.d[s].get("old_y")]
    if not rows:
        return None
    import test_y_correction as T
    T._ROWS = rows
    try:
        st = T.stats()
    finally:
        T._ROWS = None
    return json.loads(json.dumps(st, default=float))


def ra8(data: Data) -> Dict[str, Any]:
    inv = merged_checks(data)
    a_ok, b_ok = inv["RA8a"]["ok"], inv["RA8b"]["ok"]
    c_rule_ok, d_ok = inv["RA8c"]["ok"], inv["RA8d"]["ok"]
    dis = int(sum(data.checks(s, c)["ev"]["v34"]["ra8a_ledger_total_sign_disagreements"] for s in data.seeds for c in CELLS))
    n_ind = len(data.seeds) * len(CELLS)
    tb: Dict[str, Any] = {}
    for cell in CELLS:
        k = sum(int(any(data.checks(s, cell)["ev"]["v34"]["tau_boundary"].values())) for s in data.seeds)
        fr = frac_report(k, len(data.seeds))
        tb[ck(cell)] = {**fr, "boundary_dominates(>50%)": bool(fr["p"] > 0.5)}
    dom = [k for k, v in tb.items() if v["boundary_dominates(>50%)"]]
    kap: Dict[str, Any] = {}
    sums: Dict[str, List[float]] = {}
    for s in data.seeds:
        for cell in CELLS:
            for src, v in data.checks(s, cell)["ev"]["v34"]["alpha_beta_sum"].items():
                sums.setdefault(f"{ck(cell)}|{src}", []).append(v)
    for k, v in sums.items():
        kap[k] = {"mean_alpha_plus_beta": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
    moves = [data.checks(s, c)["ev"]["v34"]["kappa_move"] for s in data.seeds for c in CELLS]
    old = old_protocol(data)
    verdict = "pass" if (a_ok and b_ok and c_rule_ok and d_ok and inv["RA8a"]["n_run"] > 0) else "fail"
    return {"id": "RA8", "verdict": verdict, "label": LABEL,
            "a_sign_of_own_observation_update(corrected reading S4-8)": inv["RA8a"], "a_ledger_sum_sign_disagreements(separate number)": {"count": dis, "n_individuals": n_ind, "fraction": dis / n_ind if n_ind else float("nan")},
            "b_gain_floor": inv["RA8b"], "c_tau_rule_and_boundary_push": inv["RA8c"], "c_tau_at_boundary_fraction_by_cell": tb, "c_cells_where_boundary_dominates": dom,
            "d_kappa_moves_under_opposite_verification": inv["RA8d"], "d_kappa_alpha_plus_beta_at_end": kap, "d_kappa_move_down_up_extremes": {"min_down": float(np.min([m[0] for m in moves])), "max_up": float(np.max([m[1] for m in moves]))},
            "v33_Y3_ratio_old_protocol_on_evaluation_seeds(report)": (old or {}).get("Y3"),
            "conclusion_notes": ([] if verdict == "pass" else ["RA8 が破れた: F2/F6（学習の飽和: 経験や訂正が与えられても、規則が状態を動かさない）。"])
            + (["τ が境界にある個体がセルの 50% を超えたセル: 設計仮説と矛盾する性質（境界が挙動を決めている）: " + ", ".join(dom)] if dom else []),
            "rule": "(a)(b) と (c)(d) の「反対向きに動くか」は全個体で満たす（F2・F6）。持続の強さと差が縮む回数は合否にしない"}


def ra9(data: Data) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in ("C-nocorr", "C-nocorr-v34"):
        out[c] = {}
        for comp, nm in ((0, "r1_like_A(value)"), (3, "r4_P_taken_A(belief)"), (5, "r6_w_taken_A(interpretation)")):
            rows = {}
            for err in H.ERRS:
                num, den = [], []
                for s in data.seeds:
                    dn, dd = [], []
                    for bg in H.BGS:
                        for src in H.SRCS:
                            cell = (err, bg, src)
                            if not data.valid[cell].get(s):
                                continue
                            ctrl_fol = data.ctrl(s, cell, c)["R_fol"][comp]
                            dd.append(data.R(s, cell, c, "R_pre")[comp] - data.ctrl(s, cell, c)["R_pre"][comp])
                            dn.append(data.R(s, cell, c, "R_fol")[comp] - ctrl_fol)
                    if dd:
                        num.append(float(np.mean(dn)))
                        den.append(float(np.mean(dd)))
                rows[err] = boot_ratio(num, den)
            out[c][nm] = rows
    old = old_protocol(data)
    return {"id": "RA9", "verdict": "report-only", "label": LABEL, "remaining_after_FOL_over_before_FOL(nocorr − ctrl)": out, "v33_Y7_old_protocol(report)": (old or {}).get("Y7"),
            "rule": "C-nocorr で、信念と価値の対照との差が FOL の前後でどれだけ残るか（種ごとに全セルを平均してから比。報告のみ。v3.3 の Y7 の継続）"}


def ra10(data: Data, h6: Dict[str, Any]) -> Dict[str, Any]:
    inv = merged_checks(data)
    h7_all = True
    n7 = 0
    for cell in CELLS:
        for s in data.seeds:
            a, b = data.R(s, cell, "C-v34", "R_cor"), data.R(s, cell, "C-none-v34", "R_cor")
            n7 += 1
            h7_all &= bool(a[3] == b[3] and a[4] == b[4])
    items = {"H1": inv["H1"], "H1_all_conditions_observer": inv["OBS_H1"], "H2": inv["H2"], "H2_all_conditions_observer_W_kappa": inv["OBS_H2_W_kappa"], "H2_all_conditions_observer_log_tau": inv["OBS_H2_logtau"],
             "H3": inv["H3"], "H4": inv["H4"], "H5": inv["H5"], "H7_cell_task": inv["H7"], "H7_all_27_cells": {"ok": h7_all, "n_individuals": n7}, "H8": inv["H8"], "H9": inv["H9"]}
    ok = all(v["ok"] for v in items.values()) and all(items[k]["n_run"] > 0 for k in ("H1", "H2", "H3", "H4", "H5", "H8", "H9"))
    h6_ok = h6.get("status") == "pass"
    verdict = ("pass" if h6_ok else "partial") if ok else "fail"
    return {"id": "RA10", "verdict": verdict, "label": LABEL, "items": items, "H6": h6,
            "conclusion_notes": ([] if ok else ["RA10 の H1〜H9 のいずれかが破れた: F1〜F3。"]) + ([] if h6_ok else ["H6 は Stage 5 で再実行していない（または結果が pass でない）ので partial: " + str(h6.get("note", ""))]),
            "rule": "H1〜H9 を全条件で満たす。H6 は作業用の写しで v3.3 の全テストを実行する別の手順"}


def ra11() -> Dict[str, Any]:
    f = ROOT / "tests" / "results" / "v34" / "stage4_mutation.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    s = d["initial_set"]
    return {"id": "RA11", "verdict": "pass" if s["passed"] else "fail", "label": LABEL, "source": "tests/results/v34/stage4_mutation.json（変異の種 3450〜3459。Stage 4 で実行済み。ここでは再実行しない）",
            "n_detected": s["n_detected"], "n_mutants": len(s["summary"]), "must_detect_MU2_MU3_MU4_MU5_MU7": s["must_detect"], "per_mutant": {m: {"detected": v["detected"], "kind": v["kind"]} for m, v in s["summary"].items()},
            "baseline_invariant_failures": s["baseline_invariant_failures"], "rule": "MU2〜MU5 と MU7 がすべて検出され、8 体中 7 体以上が検出されれば合格"}


def ra12(data: Data) -> Dict[str, Any]:
    inv = merged_checks(data)
    collapsed = inv["RA12"]["n_fail"]
    tr: Dict[str, Any] = {}
    for err in H.ERRS:
        tr[err] = {}
        for ent in ["B", "A2", "A", "P_mean_of_12"]:
            dcf, dcf_nc = [], []
            for s in data.seeds:
                dd, dn = [], []
                for bg in H.BGS:
                    for src in H.SRCS:
                        cell = (err, bg, src)
                        if not data.valid[cell].get(s):
                            continue
                        e = data.d[s]["extras"][ck(cell)]
                        f = (lambda dct: float(np.mean([dct[f"P{k:02d}"] for k in range(1, 13)]))) if ent == "P_mean_of_12" else (lambda dct, ent=ent: float(dct[ent]))
                        dd.append(f(e["fol"]) - f(e["pre"]))
                        dn.append(f(e["fol"]) - f(e["nocorr_fol"]))
                if dd:
                    dcf.append(float(np.mean(dd)))
                    dcf_nc.append(float(np.mean(dn)))
            tr[err][ent] = {"like_change_pre_to_after_FOL": ci_scalar(dcf), "like_difference_vs_nocorr_after_FOL": ci_scalar(dcf_nc)}
    verdict = "pass" if (collapsed == 0 and inv["RA12"]["n_run"] > 0) else "fail"
    return {"id": "RA12", "verdict": verdict, "label": LABEL, "collapsed_individuals": {"count": collapsed, "n_checked": inv["RA12"]["n_run"], "first": inv["RA12"]["first_fails"]},
            "transfer_report_not_in_u(B, A2, A, P01-P12)": tr, "conclusion_notes": [] if verdict == "pass" else ["RA12 が破れた: F5（無関係な価値の崩壊）。"],
            "rule": "u = (Q1〜Q4 の好悪、概念 music と solo_work の好悪)。ノルムが直前の 50% 未満、または |成分| > 0.02 の成分の符号が 2 つ以上反転する個体が 0 なら合格（C-v34 と、ERR-interp では C-OR）"}


def spread(vecs_by_bg: Dict[str, np.ndarray]) -> float:
    ms = [v.mean(axis=0) for v in vecs_by_bg.values()]
    return float(np.mean([np.linalg.norm(ms[i] - ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]))


def ra13(data: Data) -> Dict[str, Any]:
    cV = data.cols("V")
    out: Dict[str, Any] = {"by_err": {}}
    for err in H.ERRS:
        e_out: Dict[str, Any] = {}
        for br_name, c_main, c_nc in (("C-v34", "C-v34", "C-nocorr-v34"),):
            sp_pre, sp_cor, sp_fol, sp_cor_nc, sp_fol_nc = [], [], [], [], []
            ce_norm, ce_imm_norm, rep_main, rep_nc, rep_ctrl = [], [], [], [], []
            for src in H.SRCS:
                seeds = data.ok_seeds([(err, bg, src) for bg in H.BGS])
                if len(seeds) < 3:
                    continue
                get = lambda c, w: {bg: data.norm(np.array([data.R(s, (err, bg, src), c, w) for s in seeds]), cV) for bg in H.BGS}
                sp_pre.append(spread(get(c_main, "R_pre")))
                sp_cor.append(spread(get(c_main, "R_cor")))
                sp_fol.append(spread(get(c_main, "R_fol")))
                sp_cor_nc.append(spread(get(c_nc, "R_cor")))
                sp_fol_nc.append(spread(get(c_nc, "R_fol")))
                for bg in H.BGS:
                    cell = (err, bg, src)
                    blank = {s: np.array(data.d[s]["extras"][ck(cell)]["R_blank_fol"]) for s in seeds}
                    ce_norm.append(float(np.mean([np.linalg.norm(data.norm(data.R(s, cell, c_main, "R_fol") - data.R(s, cell, c_nc, "R_fol"))) for s in seeds])))
                    ce_imm_norm.append(float(np.mean([np.linalg.norm(data.norm(data.R(s, cell, c_main, "R_cor") - data.R(s, cell, c_nc, "R_cor"))) for s in seeds])))
                    rep_main.append(float(np.mean([np.linalg.norm(data.norm(data.R(s, cell, c_main, "R_fol") - blank[s])) for s in seeds])))
                    rep_nc.append(float(np.mean([np.linalg.norm(data.norm(data.R(s, cell, c_nc, "R_fol") - blank[s])) for s in seeds])))
                    rep_ctrl.append(float(np.mean([np.linalg.norm(data.norm(np.array(data.ctrl(s, cell, c_main)["R_fol"]) - blank[s])) for s in seeds])))
            if sp_pre:
                m = lambda x: float(np.mean(x))
                e_out = {"continuity_V_BG_spread": {"pre_COR": m(sp_pre), "after_COR_C_v34": m(sp_cor), "after_FOL_C_v34": m(sp_fol), "after_COR_C_nocorr": m(sp_cor_nc), "after_FOL_C_nocorr": m(sp_fol_nc),
                                                    "retained_C_v34": m(sp_fol) / m(sp_pre) if m(sp_pre) > 0 else float("nan"), "retained_C_nocorr": m(sp_fol_nc) / m(sp_pre) if m(sp_pre) > 0 else float("nan")},
                         "plasticity_norm_of_normalized_CE": {"CE_after_FOL(C_v34 - C_nocorr)": m(ce_norm), "CE_imm": m(ce_imm_norm)},
                         "replacement_distance_to_blank_persona_after_FOL": {"C_v34": m(rep_main), "C_nocorr": m(rep_nc), "control(no misattribution)": m(rep_ctrl)}}
        out["by_err"][err] = e_out
    return {"id": "RA13", "verdict": "report-only", "label": LABEL, **out,
            "definitions": "連続性: BG 間の V の平均ベクトルの対ごとの距離の平均（正規化した V 成分）を、COR の直前と FOL の後で比べる。可塑性: 正規化した CE のノルム。置き換え: FOL の後の R と、FOL だけを与えた白紙の v34 人格の R の正規化した距離。"
                           "どの量も、時間だけを進めた C-nocorr-v34 の同じ量を並べる。SRC ごとに計算して平均",
            "rule": "報告のみ（J4）"}


def ra14(data: Data) -> Dict[str, Any]:
    def r_ind(cell, s) -> float:
        d_pre = data.R(s, cell, "C-OR", "R_pre")[0] - data.R(s, cell, "C-OR", "R_ctrl_pre")[0]
        return float((data.R(s, cell, "C-OR", "R_cor")[0] - data.R(s, cell, "C-OR", "R_ctrl_pre")[0]) / d_pre)
    cellv: Dict[Tuple, Dict[str, float]] = {}
    for cell in CELLS:
        ss = [s for s in data.seeds if data.valid[cell].get(s)]
        rr = np.array([r_ind(cell, s) for s in ss])
        dpre = np.array([data.R(s, cell, "C-OR", "R_pre")[0] - data.R(s, cell, "C-OR", "R_ctrl_pre")[0] for s in ss])
        dpost = np.array([data.R(s, cell, "C-OR", "R_cor")[0] - data.R(s, cell, "C-OR", "R_ctrl_pre")[0] for s in ss])
        cellv[cell] = {"frac_r_lt_-0.05": float(np.mean(rr < -0.05)) if len(rr) else float("nan"), "mean_excess_individual": float(np.mean(np.maximum(0.0, -rr))) if len(rr) else float("nan"),
                       "excess_ratio_of_means": float(-dpost.mean() / dpre.mean()) if len(rr) and abs(dpre.mean()) > 1e-15 else float("nan"), "k": int(np.sum(rr < -0.05)), "n": len(rr)}
    cls: Dict[str, Any] = {}
    for err in H.ERRS:
        ranges: Dict[str, float] = {}
        for meas in ("mean_excess_individual", "excess_ratio_of_means"):
            bg_r = float(np.mean([np.ptp([cellv[(err, bg, src)][meas] for bg in H.BGS]) for src in H.SRCS]))
            src_r = float(np.mean([np.ptp([cellv[(err, bg, src)][meas] for src in H.SRCS]) for bg in H.BGS]))
            ranges[meas] = max(bg_r, src_r)
            ranges[meas + "_bg_range"], ranges[meas + "_src_range"] = bg_r, src_r
        dep = bool(ranges["mean_excess_individual"] >= 0.1 and ranges["excess_ratio_of_means"] >= 0.1)
        cls[err] = {**ranges, "class": "経験史に依存する性質" if dep else "設計仮説と矛盾する性質"}
    # N の掃引（SRC-none）
    sw: Dict[str, Any] = {}
    for err in H.ERRS:
        sw[err] = {}
        for N in (2, 4, 8, 12, 16):
            k = n = 0
            ex = []
            for s, bg, e, rec in sweep_records_cached(data):
                if e != err or rec["N"] != N:
                    continue
                d_pre, d_post, r = sweep_r(rec, "C-OR")
                if np.isfinite(r):
                    n += 1
                    k += int(r < -0.05)
                    ex.append(max(0.0, -r))
            sw[err][str(N)] = {"fraction_overshoot": k / n if n else float("nan"), "mean_excess": float(np.mean(ex)) if ex else float("nan"), "k": k, "n": n}
    return {"id": "RA14", "verdict": "report-only", "label": LABEL, "main_design_N8_cells": {ck(c): v for c, v in cellv.items()}, "classification_by_err": cls, "sweep_C_OR_srcnone": sw,
            "reading_note": "超えた量 −r は、個体ごとの max(0, −r) の平均と、平均どうしの比 −mean(d_post)/mean(d_pre) の 2 通りで計算し、BG（または SRC）の間の範囲がどちらも 0.1 以上のときだけ「経験史に依存する性質」（S5-5）。故障は RA5・RA10 でだけ判定する",
            "rule": "報告と分類。故障の判定はしない"}


def ra_y4(data: Data) -> Dict[str, Any]:
    vals = {}
    for cell in CELLS:
        ss = [s for s in data.seeds if data.valid[cell].get(s)]
        dpre = np.array([data.R(s, cell, "C-retro", "R_pre")[0] - data.R(s, cell, "C-retro", "R_ctrl_pre")[0] for s in ss])
        dpost = np.array([data.R(s, cell, "C-retro", "R_cor")[0] - data.R(s, cell, "C-retro", "R_ctrl_pre")[0] for s in ss])
        vals[cell] = (float(dpre.mean()), float(dpost.mean()))
    gm_pre = float(np.mean([v[0] for v in vals.values()]))
    gm_post = float(np.mean([v[1] for v in vals.values()]))
    ratio_means = abs(gm_post) / abs(gm_pre) if abs(gm_pre) > 1e-15 else float("nan")
    per_cell = {ck(c): (abs(v[1]) / abs(v[0]) if abs(v[0]) > 1e-15 else float("nan")) for c, v in vals.items()}
    mean_of_ratios = float(np.nanmean(list(per_cell.values())))
    per_err = {e: float(np.nanmean([per_cell[ck((e, bg, s))] for bg in H.BGS for s in H.SRCS])) for e in H.ERRS}
    verdict = combine(bool(ratio_means <= 0.1), bool(mean_of_ratios <= 0.1))
    return {"id": "RA-Y4", "verdict": verdict, "label": LABEL, "ratio_of_grand_means(literal)": ratio_means, "mean_of_cell_ratios(strict)": mean_of_ratios, "by_err_mean_of_cell_ratios": per_err, "cell_ratio": per_cell,
            "grand_mean_d_pre": gm_pre, "grand_mean_d_post": gm_post,
            "reading_note": "「全セルの平均で満たす」を、全セルの平均どうしの比（literal）と、セルごとの比の平均（strict）の両方で判定し、両方 ≤ 0.1 のときだけ pass（S5-6）。d_pre が 0 に近いセル（ERR-interp）では比が不安定で、対照として判定できない",
            "rule": "C-retro で COR の直後の |d| ≤ 0.1 |d_pre|（主要因計画の全セルの平均）"}


FAULT_MAP = {"F1": ["H1"], "F2": ["RA3_3", "RA3_4", "RA3_5", "RA5_range", "RA5_formula", "RA5_pred", "RA5_sigma", "RA5_remainder", "RA7a", "RA8a", "RA8b", "RA8c", "RA8d", "H3", "H4", "H5", "H7", "H8", "H9"],
             "F3": ["RA5_reconstruct", "H2"], "F4": ["RA6"], "F5": ["RA12"], "F6": ["RA8a", "RA8b", "RA8c", "RA8d"]}


def faults(data: Data, res: Dict[str, Any]) -> Dict[str, Any]:
    inv = merged_checks(data)
    out = {}
    for f, names in FAULT_MAP.items():
        bad = [n for n in names if not inv[n]["ok"]] + [("OBS_" + n) for n in names if ("OBS_" + n) in inv and not inv["OBS_" + n]["ok"]]
        out[f] = {"found": bool(bad), "failing_checks": bad}
    return out


def conclusion_rules(res: Dict[str, Any]) -> List[str]:
    r1, r2, r3 = res["RA1"], res["RA2"], res["RA3"]
    notes: List[str] = []
    if r1["verdict"] == "fail":
        notes.append("RA1 不合格: 「同じ訂正で人格ごとに向きが違う学習」は、この機構と経験史の範囲では示せなかった。")
    if r1["verdict"] in ("pass", "partial") and r2["verdict"] == "fail":
        notes.append("RA1 が合格（または部分的な合格）で RA2 が不合格: 向きの違いは、価値の評価状態ではなく、信念・経験記録（何を事実と受け止めたか）から来る。")
    if r1["verdict"] == "partial" or r2["verdict"] == "partial":
        notes.append(f"RA1 / RA2 は ERR ごとに報告する（RA1 合格 ERR: {[e for e, v in r1['ok_by_err'].items() if v]}、RA2 合格 ERR: {[e for e, v in r2['ok_by_err'].items() if v]}）。")
    if r3["verdict"] == "fail":
        notes.append("RA3 不合格: 「主観的評価 → 次の学習」の経路は、構造としてはあるが、人格ごとに向きを変えるほどには働いていない。")
    for nm, key in (("C-v33", "c_v33"), ("A-τ-off", "a_tau_off")):
        if r1["verdict"] in ("pass", "partial") and r1[key]["verdict"] in ("pass", "partial"):
            notes.append(f"RA1 が {nm} でも合格側: その向きの違いは M3 の寄与ではない（既存の階層カルマンの Σ、または M1 の寄与）。")
    notes.append("どの判定でも、各判定が示した範囲より先のことは書かない。")
    return notes


def evaluate(per_seed: Dict[int, Dict[str, Any]], h6: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = Data(per_seed)
    h6 = h6 or {"status": "not re-run", "note": "Stage 5 では v3.3 の全テストを再実行していない（Stage 3 の結果: stage3_h6.json。Stage 4 は npnv3 に既定で働かないオプションを足した）"}
    res: Dict[str, Any] = {}
    res["RA0"] = ra0(data)
    for nm, fn in (("RA1", ra1), ("RA2", ra2), ("RA3", ra3), ("RA4", ra4), ("RA5", ra5), ("RA5-RC", ra5_rc), ("RA6", ra6), ("RA7", ra7), ("RA8", ra8), ("RA9", ra9), ("RA12", ra12), ("RA13", ra13), ("RA14", ra14), ("RA-Y4", ra_y4)):
        res[nm] = fn(data)
    res["RA10"] = ra10(data, h6)
    res["RA11"] = ra11()
    res["_conclusion_rules"] = conclusion_rules(res)
    res["_faults"] = faults(data, res)
    res["_old_protocol_C_v33_evaluation_seeds"] = old_protocol(data)
    res["_meta"] = {"seeds": data.seeds, "n_seeds": len(data.seeds), "label": LABEL, "scale_dropped_components": [int(i) for i in np.where(~data.scale[1])[0]]}
    return res


# ======================================================================
# 出力
# ======================================================================
def jsonable(x: Any) -> Any:
    from stage5_tasks import to_jsonable
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return to_jsonable(x)


def f3(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) and np.isfinite(x) else str(x)


def summary_md(res: Dict[str, Any]) -> str:
    L: List[str] = []
    m = res["_meta"]
    L.append("# Stage 5 評価の要約（v3.4 Y 系）")
    L.append("")
    L.append(f"**{LABEL}** 種 {m['seeds'][0]}〜{m['seeds'][-1]}（n = {m['n_seeds']}）。統計の単位は種。判定の定義は SPEC-v3.4 第 8 節のまま。")
    L.append("")
    L.append("## 判定の一覧")
    L.append("")
    L.append("| ID | 判定 | 主な数値 |")
    L.append("| --- | --- | --- |")
    r = res
    L.append(f"| RA0 前提 | {r['RA0']['verdict']} | 除かれた個体 {r['RA0']['n_excluded_individuals']}（27 セル × {m['n_seeds']} 種） |")
    L.append(f"| RA1 同じ訂正、違う向き（主） | {r['RA1']['verdict']} | ERR ごとの合格 {r['RA1']['c_v34']['ok_by_err']}。C-v33: {r['RA1']['c_v33']['verdict']}、A-τ-off: {r['RA1']['a_tau_off']['verdict']} |")
    L.append(f"| RA2 評価状態が向きを変える（主） | {r['RA2']['verdict']} | ERR ごとの合格 {r['RA2']['ok_by_err']}。RA2-hm: {r['RA2']['RA2_hm']['verdict']} {r['RA2']['RA2_hm']['ok_by_err']} |")
    L.append(f"| RA3 主観的評価 → 次の学習の向き（主） | {r['RA3']['verdict']} | (1) literal {r['RA3']['one_ok_literal']} / strict {r['RA3']['one_ok_strict']}、(2) SRC 合格 {r['RA3']['two_n_src_ok']}/3、(3) {r['RA3']['three_isolation_clamp_tau']['ok']}、(4) {r['RA3']['four_mood_negative_control']['ok']}、(5) {r['RA3']['five_origin_negative_control']['ok']} |")
    a = r["RA4"]
    L.append(f"| RA4 再評価の強さの形成（副） | {a['verdict']} | (a) 範囲 {f3(a['a']['range_of_cell_means_individual_r'])}（平均どうしの比 {f3(a['a']['range_of_cell_ratio_of_means'])}）、(b) 差 {f3(a['b']['mean_diff_reliable_minus_unreliable']['mean'])}（95% {f3(a['b']['mean_diff_reliable_minus_unreliable']['lo'])}〜{f3(a['b']['mean_diff_reliable_minus_unreliable']['hi'])}）、(c) 基準超えのセル {a['c']['cells_over_0.0134']}/27 |")
    L.append(f"| RA5 吸収の不変量と N の掃引 | {r['RA5']['verdict']} | 再構成 {r['RA5']['sweep_overshoot_reconstruction']['n_reconstructed']} 件、最大誤差 {r['RA5']['sweep_overshoot_reconstruction']['max_err_gD']:.2e} |")
    L.append(f"| RA5-RC | {r['RA5-RC']['verdict']} | C-v33 の r(N) は下の表 |")
    L.append(f"| RA6 当時の記録の保存 | {r['RA6']['verdict']} | 破れ {r['RA6']['cell_task_C_v34_and_C_OR']['n_fail']} + {r['RA6']['observer_all_conditions']['n_fail']} |")
    L.append(f"| RA7 事実の訂正 | {r['RA7']['verdict']}（(a) のみ。(b)(c) は報告） | (d) {r['RA7']['d_classification_C_v34_all_err']} |")
    L.append(f"| RA8 飽和と張り付き | {r['RA8']['verdict']} | 台帳の合計の符号の不一致 {r['RA8']['a_ledger_sum_sign_disagreements(separate number)']['count']}（別の数）。τ が境界で支配的なセル {len(r['RA8']['c_cells_where_boundary_dominates'])} |")
    L.append(f"| RA9 訂正のない誤解 | report-only | 下に |")
    L.append(f"| RA10 数値的な健全性 | {r['RA10']['verdict']} | H6: {r['RA10']['H6'].get('status')} |")
    L.append(f"| RA11 検出力 | {r['RA11']['verdict']} | {r['RA11']['n_detected']}/{r['RA11']['n_mutants']}（Stage 4 の結果。再実行なし） |")
    L.append(f"| RA12 無関係な価値の崩壊 | {r['RA12']['verdict']} | 崩壊した個体 {r['RA12']['collapsed_individuals']['count']} |")
    L.append(f"| RA13 連続性と可塑性 | report-only | 下に |")
    L.append(f"| RA14 過剰な反省の性質 | report-only | ERR ごとの分類 {({e: v['class'] for e, v in r['RA14']['classification_by_err'].items()})} |")
    L.append(f"| RA-Y4 自動消去の対照 | {r['RA-Y4']['verdict']} | 平均どうしの比 {f3(r['RA-Y4']['ratio_of_grand_means(literal)'])}、セルごとの比の平均 {f3(r['RA-Y4']['mean_of_cell_ratios(strict)'])} |")
    L.append("")
    L.append("## 結論の規則（事前に固定）の適用")
    for n in r["_conclusion_rules"]:
        L.append(f"- {n}")
    L.append("")
    L.append("## 故障（F1〜F6）")
    for f, v in r["_faults"].items():
        L.append(f"- {f}: {'見つかった ' + str(v['failing_checks']) if v['found'] else '見つからなかった'}")
    L.append("")
    L.append("## C-v33（旧設計）との比較")
    old = r.get("_old_protocol_C_v33_evaluation_seeds")
    L.append(f"- RA1 の同じ計算（C-v33）: {r['RA1']['c_v33']['verdict']}、ERR ごと {r['RA1']['c_v33']['ok_by_err']}。A-τ-off: {r['RA1']['a_tau_off']['verdict']} {r['RA1']['a_tau_off']['ok_by_err']}。")
    L.append(f"- RA4: C-v33 の 27 セルの平均 r の範囲 {f3(r['RA4']['ablations_and_c_v33']['C-v33']['range_mean_r'])}、セル内の標準偏差が基準を超えるセル {r['RA4']['ablations_and_c_v33']['C-v33']['n_cells_sd_over_0.0134']}/27。")
    if old:
        L.append(f"- 旧プロトコル（v3.3 の Y1〜Y7、評価の種）: Y1 近づき {f3(old['Y1']['closeness'])}、Y2 残る割合 {f3(old['Y2']['remaining_fraction'])}、Y3 の比 {f3(old['Y3']['ratio'])}、Y4 {f3(old['Y4']['ratio_to_pre'])}、Y6 {f3(old['Y6']['shrink'])}、Y7 {f3(old['Y7']['belief_remaining'])}/{f3(old['Y7']['value_remaining'])}。")
    L.append("- RA5-RC: C-v33 の r(N)（全セルをまとめた平均どうしの比）: " + ", ".join(f"N={N}: {f3(v['ratio'])}" for N, v in r["RA5-RC"]["r_of_N_C_v33_all"].items()))
    L.append("- RA7(b): 近づき（p_person）は RA7.json の表。RA-Y4 は C-retro（v3.3 型の対照）。")
    L.append("")
    L.append("")
    L.append("## 報告の数値（合否にしないもの）")
    L.append("- RA5 r(N)（全セルをまとめた平均どうしの比、N = 2/4/8/12/16）:")
    for c, cur in r["RA5"]["r_of_N_pooled_all_cells(report)"].items():
        L.append(f"  - {c}: " + ", ".join(f"{f3(v['ratio'])}" for v in cur.values()))
    L.append("- RA5 反対側へ行った個体（N の掃引、割合）: " + "; ".join(f"{c}: " + ", ".join(f"N={n} {f3(v['fraction'])}" for n, v in d.items()) for c, d in r["RA5"]["overshoot_counts_sweep(report)"].items() if c in ("C-v34", "C-OR", "C-v33")))
    L.append("- RA7(b) Y1 の近づき（p_person、SRC をまとめた値。C-v34 / C-v33）:")
    for err in H.ERRS:
        L.append(f"  - {err}: " + ", ".join(f"{bg} {f3(r['RA7']['b_Y1_closeness_p_person(report-only)']['C-v34'][err][bg]['pooled_src']['ratio'])}/{f3(r['RA7']['b_Y1_closeness_p_person(report-only)']['C-v33'][err][bg]['pooled_src']['ratio'])}" for bg in H.BGS))
    L.append("- RA7(c) local-reattr（処理を受けた対照への近づき。C-v34 / C-v33）: " + "; ".join(f"{err}: " + ", ".join(f"{f3(r['RA7']['c_local_reattr_closeness_to_treated_control(report-only)']['local-reattr'][err][bg]['ratio'])}/{f3(r['RA7']['c_local_reattr_closeness_to_treated_control(report-only)']['local-reattr-v33'][err][bg]['ratio'])}" for bg in H.BGS) for err in H.ERRS))
    L.append("- RA7(d) 近づきの不足の BG 間の範囲（C-v34）: " + ", ".join(f"{e}: {f3(v['range_pooled_src'])}/{f3(v['range_src_none'])} → {v['class']}" for e, v in r["RA7"]["d_shortfall_range_over_bg"]["C-v34"].items()))
    L.append("- RA9 訂正のない誤解（FOL の後に残る割合、r1 価値 / r4 信念、C-nocorr-v34 / C-nocorr）: " + "; ".join(f"{e}: {f3(r['RA9']['remaining_after_FOL_over_before_FOL(nocorr − ctrl)']['C-nocorr-v34']['r1_like_A(value)'][e]['ratio'])}/{f3(r['RA9']['remaining_after_FOL_over_before_FOL(nocorr − ctrl)']['C-nocorr-v34']['r4_P_taken_A(belief)'][e]['ratio'])}、{f3(r['RA9']['remaining_after_FOL_over_before_FOL(nocorr − ctrl)']['C-nocorr']['r1_like_A(value)'][e]['ratio'])}/{f3(r['RA9']['remaining_after_FOL_over_before_FOL(nocorr − ctrl)']['C-nocorr']['r4_P_taken_A(belief)'][e]['ratio'])}" for e in H.ERRS))
    L.append("- RA13（連続性 = BG 間の V の広がりが FOL の後に何倍残るか。C-v34 / C-nocorr-v34）: " + "; ".join(f"{e}: {f3(v['continuity_V_BG_spread']['retained_C_v34'])}/{f3(v['continuity_V_BG_spread']['retained_C_nocorr'])}" for e, v in r["RA13"]["by_err"].items() if v))
    L.append("- RA2 の Shapley（種 3400〜3419。速さ G3+G5κ / 中身 G1+G2+G4+G5τ / G6、ERR-interp trust←betray）: " + (lambda d: "n/a" if not d else f"{f3(d['speed_G3_G5k']['mean'])} / {f3(d['content_G1_G2_G4_G5t']['mean'])} / {f3(d['mixed_G6_h_m']['mean'])}")(r["RA2"]["shapley"]["by_err"]["interp"].get("trust<-betray")))
    L.append("- RA2 入れ替えの並び（有向の組の除かれる数、L1 の読み / 文面どおりの読み）: " + ", ".join(f"{e}: {v['excluded_under_L1']}/{v['excluded_under_literal']}（{v['n_directed_pairs']} 組）" for e, v in r["RA2"]["alignment_counts"].items()))
    L.append("")
    L.append("## 注意")
    L.append("- 2 通りの読みがある判定（S5-1〜S5-6）は、両方の読みで合格のときだけ pass、片方だけなら partial と書いた。理由は CHANGELOG-v3.4.md の S5 節。")
    L.append("- 固執、嫌悪の持続、反証への抵抗は故障ではない。故障は F1〜F6 だけ。")
    return "\n".join(L) + "\n"


def write_outputs(res: Dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        if k.startswith("_"):
            continue
        with open(out / f"{k}.json", "w", encoding="utf-8") as f:
            json.dump(jsonable(v), f, ensure_ascii=False, indent=1)
    with open(out / "stage5_extra.json", "w", encoding="utf-8") as f:
        json.dump(jsonable({k: v for k, v in res.items() if k.startswith("_")}), f, ensure_ascii=False, indent=1)
    (out / "stage5_summary.md").write_text(summary_md(res), encoding="utf-8")


def load_seed_files(d: Path, phase: str) -> Dict[int, Dict[str, Any]]:
    seeds = list(RP.PHASES[phase])
    RP.guard_seeds(seeds, phase)
    out = {}
    for s in seeds:
        f = d / f"stage5_seed_{s}.json"
        if f.exists():
            out[s] = json.loads(f.read_text(encoding="utf-8"))
    return out


# ======================================================================
# 単体テスト（合成データ。評価の種も予行の種も実行しない）
# ======================================================================
class CriteriaLogic(unittest.TestCase):
    def test_level_and_wilson_follow_the_spec(self):
        self.assertAlmostEqual(RP.LEVEL_MAIN, 1 - 0.05 / 3, places=12)
        p, lo, hi = RP.wilson(38, 40, 0.95)
        self.assertAlmostEqual(p, 0.95)
        self.assertTrue(lo < p < hi)
        self.assertLess(RP.wilson(40, 40, 0.95)[1], 0.95)          # n = 40 では全員でも Wilson の下限は 0.95 に届かない（RA0 は点推定 ≥ 0.95 で判定する）

    def test_dual_rule_never_makes_a_criterion_easier(self):
        self.assertEqual(combine(True, True), "pass")
        self.assertEqual(combine(True, False), "partial")
        self.assertEqual(combine(False, True), "partial")
        self.assertEqual(combine(False, False), "fail")
        self.assertEqual(combine(None, True), "undetermined")

    def test_k1_needs_all_three_err(self):
        self.assertEqual(per_err_verdict({"interp": True, "report": True, "report-x": True}), "pass")
        self.assertEqual(per_err_verdict({"interp": True, "report": True, "report-x": False}), "partial")
        self.assertEqual(per_err_verdict({"interp": False, "report": False, "report-x": False}), "fail")
        self.assertEqual(per_err_verdict({"interp": True, "report": None, "report-x": True}), "undetermined")

    def test_direction_difference_rule(self):
        rng = np.random.default_rng(1)
        X1 = rng.normal(1.0, 0.1, size=(40, 3))
        X2 = rng.normal(1.0, 0.1, size=(40, 3))
        self.assertFalse(RP.directions_differ(X1, X2, RP.LEVEL_MAIN)["differ"])
        X3 = X1.copy()
        X3[:, 0] *= -1                                      # 成分 0 の符号が逆（区間は 0 を含まない）
        self.assertTrue(RP.directions_differ(X1, X3, RP.LEVEL_MAIN)["a_opposite_sign"])
        X4 = np.stack([np.ones(40), -np.ones(40), np.zeros(40) + 1e-3], axis=1) + rng.normal(0, 0.01, size=(40, 3))
        X5 = np.stack([np.ones(40), np.ones(40), np.zeros(40) + 1e-3], axis=1) + rng.normal(0, 0.01, size=(40, 3))
        self.assertTrue(RP.directions_differ(X4, X5, RP.LEVEL_MAIN)["differ"])

    def test_boot_ratio_is_deterministic_and_unit_is_the_seed(self):
        a = boot_ratio(np.arange(1.0, 41.0), np.ones(40) * 2.0)
        b = boot_ratio(np.arange(1.0, 41.0), np.ones(40) * 2.0)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["ratio"], 10.25)
        self.assertEqual(a["n"], 40)

    def test_conclusion_rules_text(self):
        base = {"RA1": {"verdict": "fail", "ok_by_err": {}, "c_v33": {"verdict": "fail"}, "a_tau_off": {"verdict": "fail"}}, "RA2": {"verdict": "fail", "ok_by_err": {}}, "RA3": {"verdict": "fail"}}
        n = conclusion_rules(base)
        self.assertTrue(any("RA1 不合格" in x for x in n))
        self.assertTrue(any("RA3 不合格" in x for x in n))
        base["RA1"].update(verdict="pass", c_v33={"verdict": "pass"})
        n = conclusion_rules(base)
        self.assertTrue(any("信念・経験記録" in x for x in n))
        self.assertTrue(any("C-v33 でも合格側" in x for x in n))
        for x in n:
            for ph in FORBIDDEN_PHRASES:
                self.assertNotIn(ph, x)

    def test_summary_template_never_says_personality_formed(self):
        src = Path(__file__).read_text(encoding="utf-8")
        body = src.split("def summary_md", 1)[1].split("def write_outputs", 1)[0]
        for ph in FORBIDDEN_PHRASES:
            self.assertNotIn(ph, body)

    def test_stage5_files_are_not_run_by_import(self):
        # このモジュールを import しても評価の種は実行されない（計算は stage5_run.py だけ）
        self.assertEqual(RP.PHASES["stage5"], RP.SEEDS_EVAL)
        with self.assertRaises(PermissionError):
            RP.guard_seeds([3400], "dev")


def main_cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evaluate"])
    ap.add_argument("--seeds-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="stage5", choices=["dev", "stage5"])
    ap.add_argument("--h6", default=None, help="H6 の結果の JSON（status, note）")
    a = ap.parse_args()
    per = load_seed_files(Path(a.seeds_dir), a.phase)
    if not per:
        sys.exit("per-seed JSON がない")
    h6 = json.loads(Path(a.h6).read_text(encoding="utf-8")) if a.h6 else None
    res = evaluate(per, h6)
    write_outputs(res, Path(a.out))
    print("done:", {k: v["verdict"] for k, v in res.items() if not k.startswith("_")})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        main_cli()
    else:
        unittest.main()
