"""v3.4 Stage 1 の診断: 旧設計（C-v33）の再現と、原因仮説 RC1〜RC3（SPEC-v3.4 第 3 節）の検証。新しい機構は使わない。

実行（診断用の種 3460〜3479。評価・変異・予約の種は guard が拒否する）:
    python -B tests/diagnostics_v34_rc.py --phase diag            # 全部
    python -B tests/diagnostics_v34_rc.py --phase dev --grid smoke --out <dir>   # 開発用の種 3490〜3499 での動作確認（結果は判定に使わない）

やること:
  (1) 旧 Y1〜Y7: tests/yz_protocol.py の y_worker（v3.3 のプロトコルそのまま）を診断用の種で実行し、tests/test_y_correction.py の stats() で集計する。
  (2) 旧プロトコルの N の掃引（批評の検算 B の再現の試み。C-v33 と diag-q0）。
  (3) 新しい経験史（data/make_histories_v34.py）での N の掃引: N ∈ {2, 4, 8, 12, 16} × M ∈ {0, 4} × BG 3 × ERR 3 × SRC-none × 種 20 本を、
      C-v33、diag-q0、C-retro、C-nocorr で実行する。
  (4) RC1〜RC3 の判定（下の定義は、診断用の種の結果を見る前に固定した。CHANGELOG-v3.4.md の 2026-10-16 の項にも同じ定義を書いた）。

RC の判定の定義（事前に固定。結果を見て変えない）:
  r(N) の集計: N ごとに、全セル（M・BG・ERR）と全種をまとめた「平均どうしの比」r = mean(d_post) / mean(d_pre)。d_pre = like(A)_誤帰属 − like(A)_対照（COR の直前）、
      d_post = like(A)_COR の直後 − like(A)_対照（同じ対照）。比は個体ごとの r の平均ではない（d_pre が 0 に近い個体で比が発散するため）。
  RC1: (i) 5 つの N での r(N) の最小二乗の傾き（N に対して）が、diag-q0 では C-v33 の傾きの絶対値の 50% 以下、かつ (ii) diag-q0 の r(16) ≥ 0。両方で「当たり」。
       追加で M = 0 のみ、ERR ごとの値も報告（判定は全セルをまとめたもの）。
  RC2: 個体ごとの訂正の利得 gain_i = Σ_e Σ_k |Δ_ek| g_ek / Σ_e Σ_k |Δ_ek|（Δ = 再評価で求めた含意された変化 shift、g = 更新の直前の予測空間の利得 zᵀΣ_k z / (zᵀΣ_k z + R/(a_k w_reapp))）。
       C-v33 の個体ごとの S = Σ_i gain_i。N ごとに S の平均と 1 − r(N) の Pearson 相関（5 点）が 0.9 以上で「当たり」。
       追加で 90 セル（N × M × BG × ERR）での相関も報告（判定は 5 点の相関）。
  RC3: 個体ごとに、C-v33 の訂正の前後の like(A) の変化 Δlike_A を、特徴の層（ind = ind:A、shared = people + cls + emb）ごとに分ける（合計が Δlike_A に一致）。
       共有の割合 = Σ_個体 shared / Σ_個体 (shared + ind)（符号つき、個体をまとめた比）が 0.30 以上、かつ
       A2（履歴に現れない同じ類の相手）の好悪の変化 Δlike_A2 の平均が Δlike_A の平均と同じ符号で、90% 以上の個体で |Δlike_A2| > 1e-6 のとき「当たり」。
       追加で |shared| / (|shared| + |ind|) の平均、ERR・N ごとの値も報告。
  RC-Y1・RC-Y2・RC-loop は数値の予測を持たない（構造と原因の記述）。報告は、Y1 の近づき（p_person と p_ind）、BG の間の r の範囲だけ。
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ra_protocol as P  # noqa: E402
import make_histories_v34 as H  # noqa: E402

OUT_DEFAULT = HERE / "results" / "v34"
NS = (2, 4, 8, 12, 16)
MS = (0, 4)
CONDS = ("C-v33", "diag-q0", "C-retro", "C-nocorr")


def _r(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [float(f"{v:.12g}") for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{x:.12g}")
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    return x


# ----------------------------------------------------------------------
# (1)(2) 旧プロトコル
# ----------------------------------------------------------------------
def old_y_worker(args) -> Dict[str, Any]:
    import yz_protocol as Y
    seed, phase = args
    P.guard_seeds([seed], phase)
    return Y.y_worker(seed)


def old_n_worker(args) -> Dict[str, Any]:
    """旧プロトコル（誤帰属 N 経験 + B との挟み込み、訂正 N 件）の N の掃引。C-v33 と diag-q0。"""
    import yz_protocol as Y
    seed, n, phase = args
    P.guard_seeds([seed], phase)
    Y.N_J = n
    mis, ctrl = Y.make(seed), Y.make(seed)
    Y.feed(mis, Y.y_history(seed, True))
    Y.feed(ctrl, Y.y_history(seed, False))
    l_mis, l_ctrl = Y.like_A(mis), Y.like_A(ctrl)
    out = {"seed": seed, "n": n, "like_mis": l_mis, "like_ctrl": l_ctrl}
    for name, q0 in (("v33", False), ("q0", True)):
        c = copy.deepcopy(mis)
        with P.reappraise_probe([], q0=q0):
            Y.feed(c, Y.y_corrections(seed))
        out[f"like_{name}"] = Y.like_A(c)
    return out


# ----------------------------------------------------------------------
# (3) 新しい経験史での N の掃引
# ----------------------------------------------------------------------
def sweep_worker(args) -> Dict[str, Any]:
    seed, bg, err, n, m, phase = args
    r = P.run_individual(seed, bg, "none", err, n, m, CONDS, phase)
    ctrl = next(iter(r["ctrl"].values()))
    rec: Dict[str, Any] = {"seed": seed, "bg": bg, "err": err, "n": n, "m": m, "ctrl_pre": ctrl["R_pre"], "ctrl_fol": ctrl["R_fol"]}
    for c in CONDS:
        x = r["cond"][c]
        rec[c] = {"R_pre": x["R_pre"], "R_cor": x["R_cor"], "R_fol": x["R_fol"], "gain": x["gain"], "split": x["layer_split"]}
    return _r(rec)


def grid(seeds, smoke: bool):
    ns = (2, 8, 16) if smoke else NS
    ms = (0,) if smoke else MS
    return [(s, bg, err, n, m) for n in ns for m in ms for bg in H.BGS for err in H.ERRS for s in seeds]


# ----------------------------------------------------------------------
# 集計
# ----------------------------------------------------------------------
def ratio(recs: List[Dict[str, Any]], cond: str) -> float:
    dpre = np.mean([x[cond]["R_pre"][0] - x["ctrl_pre"][0] for x in recs])
    dpost = np.mean([x[cond]["R_cor"][0] - x["ctrl_pre"][0] for x in recs])
    return float(dpost / dpre) if abs(dpre) > 1e-12 else float("nan")


def slope(xs, ys) -> float:
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    return float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze(raw: List[Dict[str, Any]], ns) -> Dict[str, Any]:
    def sel(**kw):
        return [x for x in raw if all(x[k] == v for k, v in kw.items())]

    out: Dict[str, Any] = {"n_individuals": len(raw)}
    # r(N)
    def rcurve(cond, **kw):
        return {int(n): ratio(sel(n=n, **kw), cond) for n in ns}
    curves = {c: {"all": rcurve(c), "M0": rcurve(c, m=0), **{f"ERR={e}": rcurve(c, err=e) for e in H.ERRS}} for c in ("C-v33", "diag-q0", "C-retro")}
    out["r_of_N"] = curves
    # RC1
    nn = list(ns)
    s33 = slope(nn, [curves["C-v33"]["all"][n] for n in nn])
    sq0 = slope(nn, [curves["diag-q0"]["all"][n] for n in nn])
    rq16 = curves["diag-q0"]["all"][max(nn)]
    out["RC1"] = {"slope_v33": s33, "slope_q0": sq0, "slope_ratio": abs(sq0) / abs(s33) if s33 != 0 else float("nan"),
                  "r_q0_at_max_N": rq16, "criterion_i": bool(abs(sq0) <= 0.5 * abs(s33)), "criterion_ii": bool(rq16 >= 0.0)}
    out["RC1"]["held"] = out["RC1"]["criterion_i"] and out["RC1"]["criterion_ii"]
    out["RC1"]["slopes_by_subset"] = {k: {"v33": slope(nn, [curves["C-v33"][k][n] for n in nn]), "q0": slope(nn, [curves["diag-q0"][k][n] for n in nn]),
                                           "r_q0_at_max_N": curves["diag-q0"][k][max(nn)]} for k in curves["C-v33"]}
    # RC2
    S_by_n = [float(np.mean([x["C-v33"]["gain"]["S_all"] for x in sel(n=n)])) for n in nn]
    SA_by_n = [float(np.mean([x["C-v33"]["gain"]["S_A"] for x in sel(n=n)])) for n in nn]
    absorbed = [1.0 - curves["C-v33"]["all"][n] for n in nn]
    cells = [(n, m, bg, e) for n in nn for m in sorted({x["m"] for x in raw}) for bg in H.BGS for e in H.ERRS]
    cs, ca = [], []
    for (n, m, bg, e) in cells:
        rr = sel(n=n, m=m, bg=bg, err=e)
        cs.append(float(np.mean([x["C-v33"]["gain"]["S_all"] for x in rr])))
        ca.append(1.0 - ratio(rr, "C-v33"))
    out["RC2"] = {"S_all_by_N": dict(zip(nn, S_by_n)), "S_A_by_N": dict(zip(nn, SA_by_n)), "absorbed_1_minus_r_by_N": dict(zip(nn, absorbed)),
                  "pearson_5points": pearson(S_by_n, absorbed), "pearson_5points_S_A": pearson(SA_by_n, absorbed), "pearson_cells": pearson(cs, ca),
                  "n_cells": len(cells)}
    out["RC2"]["held"] = bool(out["RC2"]["pearson_5points"] >= 0.9)
    # RC3
    def split_sum(rows, who="A"):
        ind = np.array([x["C-v33"]["split"][who]["ind"] for x in rows])
        sh = np.array([x["C-v33"]["split"][who]["people"] + x["C-v33"]["split"][who]["cls"] + x["C-v33"]["split"][who]["emb"] for x in rows])
        return ind, sh
    ind, sh = split_sum(raw)
    d_a = np.array([x["C-v33"]["R_cor"][0] - x["C-v33"]["R_pre"][0] for x in raw])
    d_a2 = np.array([x["C-v33"]["R_cor"][1] - x["C-v33"]["R_pre"][1] for x in raw])
    frac = float(sh.sum() / (sh + ind).sum()) if (sh + ind).sum() != 0 else float("nan")
    frac_abs = float(np.mean(np.abs(sh) / np.maximum(np.abs(sh) + np.abs(ind), 1e-12)))
    same_sign = bool(np.sign(d_a2.mean()) == np.sign(d_a.mean()) and d_a2.mean() != 0)
    moves = float(np.mean(np.abs(d_a2) > 1e-6))
    out["RC3"] = {"shared_fraction_signed_ratio_of_sums": frac, "shared_fraction_abs_mean": frac_abs, "mean_dlike_A": float(d_a.mean()), "mean_dlike_A2": float(d_a2.mean()),
                  "ratio_A2_to_A": float(d_a2.mean() / d_a.mean()) if d_a.mean() != 0 else float("nan"),
                  "check_split_sum_equals_dlike_A_max_abs_err": float(np.max(np.abs((ind + sh) - d_a))),
                  "A2_same_sign_as_A": same_sign, "fraction_individuals_with_A2_moved": moves,
                  "by_ERR": {e: _rc3_sub(sel(err=e)) for e in H.ERRS}, "by_N": {int(n): _rc3_sub(sel(n=n)) for n in nn}}
    out["RC3"]["held"] = bool(frac >= 0.30 and same_sign and moves >= 0.90)
    # RA0・RA7(b)・RA-Y4・BG の間の r の範囲（N = 8, M = 0）
    main = sel(n=8, m=0) if 8 in nn else []
    if main:
        out["main_N8_M0"] = _main_cell_report(main)
    return out


def _rc3_sub(rows):
    if not rows:
        return None
    ind = np.array([x["C-v33"]["split"]["A"]["ind"] for x in rows])
    sh = np.array([x["C-v33"]["split"]["A"]["people"] + x["C-v33"]["split"]["A"]["cls"] + x["C-v33"]["split"]["A"]["emb"] for x in rows])
    d_a = np.array([x["C-v33"]["R_cor"][0] - x["C-v33"]["R_pre"][0] for x in rows])
    d_a2 = np.array([x["C-v33"]["R_cor"][1] - x["C-v33"]["R_pre"][1] for x in rows])
    return {"shared_fraction": float(sh.sum() / (sh + ind).sum()) if (sh + ind).sum() != 0 else float("nan"), "mean_dlike_A": float(d_a.mean()),
            "mean_dlike_A2": float(d_a2.mean()), "frac_A2_moved": float(np.mean(np.abs(d_a2) > 1e-6))}


def _main_cell_report(rows) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": len(rows)}
    d_pre = np.array([x["C-v33"]["R_pre"][0] - x["ctrl_pre"][0] for x in rows])
    d_cor = np.array([x["C-v33"]["R_cor"][0] - x["C-v33"]["R_pre"][0] for x in rows])
    out["RA0"] = {"frac_abs_d_pre_gt_1e-6": float(np.mean(np.abs(d_pre) > 1e-6)), "frac_CE_imm_nonzero": float(np.mean(np.abs(d_cor) > 0))}
    by: Dict[str, Any] = {}
    for e in H.ERRS:
        for bg in H.BGS:
            rr = [x for x in rows if x["err"] == e and x["bg"] == bg]
            if not rr:
                continue
            P_mis = np.mean([x["C-v33"]["R_pre"][3] for x in rr])
            P_ctrl = np.mean([x["ctrl_pre"][3] for x in rr])
            P_post = np.mean([x["C-v33"]["R_cor"][3] for x in rr])
            by[f"{e}|{bg}"] = {"r_v33": ratio(rr, "C-v33"), "r_q0": ratio(rr, "diag-q0"), "r_retro": ratio(rr, "C-retro"),
                               "mean_d_pre": float(np.mean([x["C-v33"]["R_pre"][0] - x["ctrl_pre"][0] for x in rr])),
                               "Y1_closeness_p_person": float((P_mis - P_post) / (P_mis - P_ctrl)) if abs(P_mis - P_ctrl) > 1e-12 else float("nan"),
                               "std_r_across_seeds_individual": float(np.nanstd([(x["C-v33"]["R_cor"][0] - x["ctrl_pre"][0]) / (x["C-v33"]["R_pre"][0] - x["ctrl_pre"][0])
                                                                               for x in rr if abs(x["C-v33"]["R_pre"][0] - x["ctrl_pre"][0]) > 1e-6]))}
    out["cells"] = by
    for e in H.ERRS:
        rr = [x for x in rows if x["err"] == e]
        rs = [by[f"{e}|{bg}"]["r_v33"] for bg in H.BGS if f"{e}|{bg}" in by]
        out[f"BG_range_of_r_v33|{e}"] = float(max(rs) - min(rs)) if rs else None
        d_pre_m = np.mean([x["C-v33"]["R_pre"][0] - x["ctrl_pre"][0] for x in rr])
        d_retro = np.mean([x["C-retro"]["R_cor"][0] - x["ctrl_pre"][0] for x in rr])
        out[f"RA-Y4_retro_abs_d_over_abs_d_pre|{e}"] = float(abs(d_retro) / abs(d_pre_m)) if abs(d_pre_m) > 1e-12 else None
    return out


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("dev", "diag"), default="diag")
    ap.add_argument("--grid", choices=("full", "smoke"), default="full")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--parts", default="oldY,oldN,sweep")
    a = ap.parse_args()
    seeds = P.SEEDS_DIAG if a.phase == "diag" else P.SEEDS_DEV
    P.guard_seeds(seeds, a.phase)
    if a.phase == "dev" and a.out is None:
        sys.exit("--phase dev には --out（リポジトリ外のディレクトリ）が要る")
    out = Path(a.out) if a.out else OUT_DEFAULT
    out.mkdir(parents=True, exist_ok=True)
    tag = "stage1" if a.phase == "diag" else "dev"
    parts = a.parts.split(",")
    smoke = a.grid == "smoke"
    t0 = time.time()
    if "oldY" in parts:
        import test_y_correction as T
        rows = P.pmap(old_y_worker, [(s, a.phase) for s in seeds], a.workers)
        T._ROWS = rows
        st = T.stats()
        (out / f"{tag}_oldY.json").write_text(json.dumps(_r({"seeds": list(seeds), "stats": st, "note": "v3.3 の yz_protocol をそのまま、診断用の種で実行"}), ensure_ascii=False, indent=1), encoding="utf-8")
        print("oldY done", time.time() - t0, flush=True)
    if "oldN" in parts:
        ns = (2, 8, 16) if smoke else NS
        res = P.pmap(old_n_worker, [(s, n, a.phase) for n in ns for s in seeds], a.workers)
        curve = {}
        for n in ns:
            rr = [x for x in res if x["n"] == n]
            dpre = np.mean([x["like_mis"] - x["like_ctrl"] for x in rr])
            curve[int(n)] = {k: float(np.mean([x[f"like_{k}"] - x["like_ctrl"] for x in rr]) / dpre) for k in ("v33", "q0")}
        (out / f"{tag}_oldN.json").write_text(json.dumps(_r({"seeds": list(seeds), "r_of_N": curve, "critique_check_B_values": {"2": 0.60, "4": 0.52, "8": 0.30, "12": -0.01, "16": -0.34},
                                                              "slope_v33": slope(list(curve), [curve[n]["v33"] for n in curve]), "slope_q0": slope(list(curve), [curve[n]["q0"] for n in curve]),
                                                              "raw": res}), ensure_ascii=False, indent=1), encoding="utf-8")
        print("oldN done", time.time() - t0, flush=True)
    if "sweep" in parts:
        items = [(s, bg, err, n, m, a.phase) for (s, bg, err, n, m) in grid(seeds, smoke)]
        raw = P.pmap(sweep_worker, items, a.workers)
        ns = (2, 8, 16) if smoke else NS
        (out / f"{tag}_sweep_raw.json").write_text(json.dumps({"seeds": list(seeds), "grid": a.grid, "conds": list(CONDS), "r_names": P.R_NAMES, "records": raw}), encoding="utf-8")
        res = analyze(raw, ns)
        (out / f"{tag}_rc.json").write_text(json.dumps(_r(res), ensure_ascii=False, indent=1), encoding="utf-8")
        print("sweep done", time.time() - t0, flush=True)
        print(json.dumps(_r({k: res[k] for k in ("RC1", "RC2")}), indent=1))
        print(json.dumps(_r({k: v for k, v in res["RC3"].items() if k not in ("by_ERR", "by_N")}), indent=1))


if __name__ == "__main__":
    main()
