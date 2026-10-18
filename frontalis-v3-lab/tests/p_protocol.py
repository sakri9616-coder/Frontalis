"""P 系（Z-persist: 強制的な減衰を持たない正当化。SPEC-v3.3 第 8.3〜8.4 節）の共通プロトコル。tests/test_p_persist.py と診断が使う。

SPEC の P1〜P7 の判定基準は実装前に固定されている（閾値は一度も変えていない）。ここに書くのは、SPEC に細部がなかったプロトコルの決め
（**数値を見る前に決めた**。試走で見た動作と、そのあとに決めた点は CHANGELOG の「v3.3 第2部」に書く）。乱数種 0〜19。条件は justification_mode = "persist"（既定の設計 spec_literal）。

【共通の履歴】
  warm_persona（ally = 類 friend に connection と support、foe = 類 rival に harm と betrayal を 6 回ずつ。yz_protocol）に、
  段階 1: 迷う二択（概念 pp_x と pp_y。文脈 studio。結果は空）を 3 回。選ぶ側は人格の乱数（自然な選択）。J の最大の概念を「好まれる側 fav」、他方を oth とする。
  再び選ぶ（段階 2）: 選択肢 0 = {fav, 友人 ally, connection 0.8 と support 0.6（warm_persona で ally から得ていたものと同じ。予期どおりの良い結果）}、選択肢 1 = {oth, 敵 foe, harm 0.6}
      （易しい二択: |ΔQ| ≈ 0.5、葛藤 c ≈ 1e-11）。
      「再び選ぶ」ことを台本どおりに起こすため、選ぶ側を選択肢 0 に強制する（B1 の強制再生と同じ仕組み npn.force_decisions = [{"choice": 0}]。乱数は同じだけ引く）。
  反証: 選択肢 0 = {fav, ally, harm 0.8（行為者 ally。選んだ相手に裏切られる。人モデルの負の教師が立つ）}、選択肢 1 は段階 2 と同じ。同様に選択肢 0 を強制する（反証を受ける選択を、自分で選んだ経験として与える）。

【P1 源の追跡】 次の混合の履歴（段階 1 → 再び選ぶ 6 回 → 反証 3 回 → 迷う二択 6 回 → 通常の経験 40 → 反対側（oth）を選ぶ 3 回）の後に、(i) 台帳の kind = "justification" の出来事を
      手がかりごと・源ごとに足した値が、人格の源ごとの和 just_src に完全に一致（≤ 1e-12）、(ii) 手がかりごとに台帳の全源の和が J に一致（≤ 1e-12）、
      (iii) 出来事の源は {spread, reaffirm, counter, overwhelm, cap} のどれか、(iv) 4 つの源（spread, reaffirm, counter, overwhelm）がどれも 1 回以上起きている（検査が空でない）。
      reaffirm の変化量は、そのときの（二択の前の）J から η_r (1 − |J|/J_num) と一致（≤ 1e-12）。
【P2 無条件の正帰還がない】 段階 1 と再び選ぶ 6 回の後、300 個の経験（W6 の経験史 R の 100 個、cue を含む通常の経験 100 個、optional の申し出 100 個）。二択はなし。
      J・源ごとの和・chosen_before が一つも変わらない（==。厳密に 0）。台帳に kind = "justification" が 1 つも増えない。前提: 学習した経験と回避した経験が両方ある、J の大きさ > 0.01（試走の後に 0.1 から緩めた。CHANGELOG の S 節 #14）。
      対照（報告のみ）: 同じ区間を orig と persist_with_decay に与えたときの J の変化（時間の減衰）。
【P3 数値的な健全性】 二択 2000 回（b_protocol の 5 つの鋳型を巡る + 7 回に 1 回は反証つきの二択。選ぶ側は自然）で、毎回、有限・|J| ≤ J_num・B2 の宣言の範囲、最後に B3 の規則検査。
      さらに、B1〜B3 の 4 条件のうち Z-persist（W6 の経験史 + 二択 + 訂正、2000 経験の B 系の実行。test_b_health.py）の B2・B3 も合格していること。
【P4 持続の由来】 段階 1 のあと、H0 = そのまま、H10 = 再び選ぶ 10 回（段階 2）。どちらにも同じ反証を 5 回（選択肢 0 を強制）。
      J_X = 手がかり concept:fav の J。残存の差 Δ = J_X(H10) − J_X(H0)。台帳から reaffirm・overwhelm の和の差 E = (Σreaffirm_H10 − Σreaffirm_H0) + (Σoverwhelm_H10 − Σoverwhelm_H0)。
      合格: |Δ − E| ≤ 0.05 |Δ|（95% 以上が説明される）。前提: 台帳の reaffirm の回数の差 = 10（fav の手がかり）、|Δ| > 1e-3。残存の大きさ・比は報告のみ。
【P5 抵抗の由来】 H10 の状態（段階 1 + 再び選ぶ 10 回）の写しを 2 つ。正当化つき（そのまま）と、対照（同じ J・同じ状態で、抵抗だけを切った写し: ρ_dis = 0）。最初の反証（選択肢 0 を強制）を同じに与える（状態が同じなので、
      人モデルの教師の違いは割り引きだけ）。W の学習の減少 R_meas = ΔW_人(対照) − ΔW_人(正当化つき)（特徴 × 欲求）。台帳だけからの予測 R_pred = K·moved: K = 対照の台帳の ΔW / 予測誤差
      （カルマン利得。欲求ごと・特徴ごと）、moved = 正当化つきの台帳の割り引きの記録（discount.moved）。合格: ‖R_meas − R_pred‖ ≤ 0.01 ‖R_meas‖。
      前提: 割り引きの割合 q > 0、‖R_meas‖ > 1e-6。状況モデルの学習の増加も同じ形で報告。続けて反証を合計 10 回与え、何回で概念の選好（fav と oth の状況プローブ。Q と根拠のみ）が
      反転するかを、正当化つき・対照で報告（合否なし）。
【P6 増幅の由来】 段階 1 のあと、再び選ぶことを 30 回（強制）。「J の増加」は、手がかりごとの正の出来事の和（総増加。負の出来事 = overwhelm・counter は別に数える）。
      手がかりごとに: 総増加 = 台帳の reaffirm の和（≤ 1e-9）。reaffirm 以外の源の正の出来事の和 ≤ 1e-9（他の源から増えない）。純変化 = 全出来事の和（≤ 1e-12）。
      reaffirm の各変化量が η_r (1 − |J|/J_num)（J は二択の前の値）と一致（≤ 1e-12）。前提: reaffirm の和（fav）> 0.2、葛藤 c < 1e-7（spread が混ざる量は上の 1e-9 の検査で見る。1e-9 から緩めた。S 節 #14）。減った分（overwhelm）は報告する。
      対照（報告のみ）: orig は同じ履歴で J が J_max に抑えられ減衰する。
【P7 経験史による個人差】 合否なし。W6 の経験史に二択・訂正を挟んだ実行（B 系の Z-persist 条件）の最終の J を、人格の型（R/K/X）ごとに、源ごとの和に分解して報告する。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np

import b_protocol as B
import yz_protocol as Y
from helpers import make
from npnv3 import NPN

SOURCES = ("spread", "reaffirm", "counter", "overwhelm", "cap")
CTX_STUDIO = "studio"


def run_forced(n: NPN, ch: Dict[str, Any], side: int = 0) -> Dict[str, Any]:
    """選ぶ側を台本どおりに強制して二択を与える（B1 の強制再生の仕組み。乱数は同じだけ引く）。"""
    n.force_decisions = [{"choice": side}]
    n._force_i = 0
    try:
        return n.experience(copy.deepcopy(ch))
    finally:
        n.force_decisions = None
        n._force_i = 0


def rechoice(fav: str, oth: str, cid: str, harm: bool = False) -> Dict[str, Any]:
    return {"id": cid, "kind": "choice",
            "options": [{"context": Y.Z_CTX, "concepts": [fav], "participants": [dict(Y.ALLY)]},
                        {"context": Y.Z_CTX, "concepts": [oth], "participants": [dict(Y.FOE)]}],
            "outcomes_by_option": {"0": [{"type": "harm", "actor": "ally", "magnitude": 0.8}] if harm
                                   else [{"type": "connection", "actor": "ally", "magnitude": 0.8}, {"type": "support", "actor": "ally", "magnitude": 0.6}],
                                   "1": [{"type": "harm", "actor": "foe", "magnitude": 0.6}]}}


def phase1(n: NPN, k: int = 3, tag: str = "p1") -> Tuple[str, str]:
    for i in range(k):
        n.experience(Y.hesitant_choice(f"{tag}{i}", "pp_x", "pp_y"))
    jx, jy = n.just.get("concept:pp_x", 0.0), n.just.get("concept:pp_y", 0.0)
    return ("pp_x", "pp_y") if jx >= jy else ("pp_y", "pp_x")


def base_persona(seed: int, mode: str = "persist", **kw: Any) -> Tuple[NPN, str, str]:
    n = Y.warm_persona(seed, justification_mode=mode, **kw)
    fav, oth = phase1(n)
    return n, fav, oth


def event_sums(n: NPN, cue: str = None, start: int = 0) -> Dict[str, float]:
    """台帳の kind = "justification" の出来事の、源ごとの和（cue を指定すればその手がかりだけ）。"""
    out: Dict[str, float] = {s: 0.0 for s in SOURCES + ("decay", "bound")}
    for en in n.ledger.entries[start:]:
        if en["kind"] != "justification":
            continue
        for ev in en["events"]:
            if cue is None or ev["cue"] == cue:
                out[ev["source"]] = out.get(ev["source"], 0.0) + ev["delta"]
    return out


def event_counts(n: NPN, cue: str, source: str, start: int = 0) -> int:
    return sum(1 for en in n.ledger.entries[start:] if en["kind"] == "justification" for ev in en["events"] if ev["cue"] == cue and ev["source"] == source)


def ledger_j_sums(n: NPN) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """台帳から: 手がかりごと・源ごとの和、手がかりごとの全源の和。"""
    by: Dict[str, Dict[str, float]] = {}
    for en in n.ledger.entries:
        if en["kind"] != "justification":
            continue
        for ev in en["events"]:
            row = by.setdefault(ev["cue"], {})
            row[ev["source"]] = row.get(ev["source"], 0.0) + ev["delta"]
    return by, {c: float(sum(r.values())) for c, r in by.items()}


def reaffirm_formula_error(n: NPN) -> Tuple[float, int]:
    """reaffirm の各変化量が η_r (1 − |J|/J_num)（J は二択の前の値）と一致するか。台帳の出来事を順に足し直して調べる。"""
    run: Dict[str, float] = {}
    err, cnt = 0.0, 0
    for en in n.ledger.entries:
        if en["kind"] != "justification":
            continue
        start = dict(run)
        for ev in en["events"]:
            if ev["source"] == "reaffirm":
                expect = n.cfg.eta_r * (1.0 - abs(start.get(ev["cue"], 0.0)) / n.cfg.j_num)
                err = max(err, abs(ev["delta"] - expect))
                cnt += 1
            run[ev["cue"]] = run.get(ev["cue"], 0.0) + ev["delta"]
    return err, cnt


def p1_worker(seed: int) -> Dict[str, Any]:
    n, fav, oth = base_persona(seed)
    for i in range(6):
        run_forced(n, rechoice(fav, oth, f"p1r{i}"), 0)
    for i in range(3):
        run_forced(n, rechoice(fav, oth, f"p1f{i}", harm=True), 0)
    for i in range(6):
        n.experience(Y.hesitant_choice(f"p1h{i}", "pp_x", "pp_y"))
    for i, e in enumerate(Y.unrelated_to_A(seed, 40, tag="p1u")):
        n.experience(e)
    for i in range(3):
        run_forced(n, rechoice(fav, oth, f"p1o{i}"), 1)                      # 反対側（oth）を選ぶ: counter
    by, tot = ledger_j_sums(n)
    err_src = max([abs(sum(by.get(c, {}).get(s, 0.0) for s in SOURCES) - sum(row.values())) for c, row in n.just_src.items()] or [0.0])
    err_src_each = 0.0
    for c, row in n.just_src.items():
        for s, v in row.items():
            err_src_each = max(err_src_each, abs(by.get(c, {}).get(s, 0.0) - v))
    for c, row in by.items():
        for s, v in row.items():
            err_src_each = max(err_src_each, abs(n.just_src.get(c, {}).get(s, 0.0) - v))
    err_j = max([abs(tot.get(c, 0.0) - n.just.get(c, 0.0)) for c in set(tot) | set(n.just)] or [0.0])
    sources_seen = sorted({s for row in by.values() for s in row})
    n_events = {s: sum(event_counts(n, c, s) for c in by) for s in SOURCES}
    ferr, fcnt = reaffirm_formula_error(n)
    return {"seed": seed, "fav": fav, "err_source_sums": err_src_each, "err_cue_total_vs_J": err_j, "err_src_total": err_src, "sources_seen": sources_seen,
            "event_counts": n_events, "reaffirm_formula_error": ferr, "reaffirm_events": fcnt,
            "unknown_sources": sorted(set(sources_seen) - set(SOURCES)), "J_fav": n.just.get(f"concept:{fav}", 0.0)}


def _p2_segment(seed: int, fav: str, oth: str) -> List[Dict[str, Any]]:
    from helpers import mh
    out = [dict(e, id="p2h-" + e["id"]) for e in mh.generate_extended_history("R", seed, 100, praise_rate=0.25)]
    for i in range(100):                        # cue を含む通常の経験（二択ではない）
        if i % 2 == 0:
            out.append({"id": f"p2c{i}", "context": Y.Z_CTX, "concepts": [fav], "participants": [dict(Y.ALLY)],
                        "outcomes": [{"type": "connection", "actor": "ally", "magnitude": 0.5}], "session": f"p2cs{i}"})
        else:
            out.append({"id": f"p2c{i}", "context": Y.Z_CTX, "concepts": [oth], "participants": [dict(Y.FOE)],
                        "outcomes": [{"type": "harm", "actor": "foe", "magnitude": 0.5}], "session": f"p2cs{i}"})
    rng = np.random.default_rng([seed, 7002])
    for i in range(100):                        # optional の申し出（接近または回避）
        out.append({"id": f"p2o{i}", "context": Y.Z_CTX, "concepts": [fav if i % 2 == 0 else oth], "participants": [],
                    "outcomes": [{"type": "praise" if i % 2 == 0 else "criticism", "actor": "world", "magnitude": round(float(rng.uniform(0.3, 0.7)), 2)}],
                    "optional": True, "announced": ["praise" if i % 2 == 0 else "criticism"], "session": f"p2os{i}"})
    return out


def p2_worker(seed: int) -> Dict[str, Any]:
    res: Dict[str, Any] = {"seed": seed}
    for mode in ("persist", "persist_with_decay", "orig"):
        n, fav, oth = base_persona(seed, mode)
        for i in range(6):
            run_forced(n, rechoice(fav, oth, f"p2r{i}"), 0)
        j0, src0, cb0 = dict(n.just), copy.deepcopy(n.just_src), list(n.chosen_before)
        nled = sum(1 for en in n.ledger.entries if en["kind"] == "justification")
        learned0, avoided0 = n.n_experiences, n.n_avoided
        for e in _p2_segment(seed, fav, oth):
            n.experience(copy.deepcopy(e))
        res[mode] = {"J_before_max": max(abs(v) for v in j0.values()), "J_unchanged": bool(n.just == j0), "sources_unchanged": bool(n.just_src == src0),
                     "chosen_before_unchanged": bool(n.chosen_before == cb0),
                     "new_justification_entries": sum(1 for en in n.ledger.entries if en["kind"] == "justification") - nled,
                     "learned": n.n_experiences - learned0, "avoided": n.n_avoided - avoided0,
                     "J_after_max": max(abs(v) for v in n.just.values()), "J_ratio_after_before": max(abs(v) for v in n.just.values()) / max(abs(v) for v in j0.values())}
    return res


def p3_worker(seed: int) -> Dict[str, Any]:
    n = make(seed, lc_design="spec_literal", justification_mode="persist")
    mon = B.Monitor(n.cfg.j_num, n.cond.K)
    for i in range(2000):
        if i % 7 == 6:
            ch = B._choice_template(i)
            ch = copy.deepcopy(ch)
            ch["id"] = f"p3h{i}"
            ch["outcomes_by_option"] = {"0": [{"type": "harm", "actor": "world", "magnitude": 0.9}], "1": [{"type": "harm", "actor": "world", "magnitude": 0.9}]}
        else:
            ch = copy.deepcopy(B._choice_template(i))
            ch["id"] = f"p3c{i}"
        r = n.experience(ch)
        mon.cheap(n)
        if i % 20 == 0:
            mon.psd(n)
    mon.psd(n)
    rules = B.check_rules(n)
    viol = B.violations_of(rules)
    by, _tot = ledger_j_sums(n)
    return {"seed": seed, "B2": mon.summary(), "B3": rules, "B3_violations": viol, "max_abs_J": mon.max_abs_J, "J_num": n.cfg.j_num,
            "event_counts": {s: sum(event_counts(n, c, s) for c in by) for s in SOURCES}, "n_choices": 2000, "finite": all(np.isfinite(v) for v in n.just.values()),
            "n_cues_with_J": len(n.just)}


def p4_worker(seed: int) -> Dict[str, Any]:
    base, fav, oth = base_persona(seed)
    x = f"concept:{fav}"
    h0, h10 = copy.deepcopy(base), copy.deepcopy(base)
    for i in range(10):
        run_forced(h10, rechoice(fav, oth, f"p4r{i}"), 0)
    out: Dict[str, Any] = {"seed": seed, "fav": fav}
    chosen_ok = True
    for tag, h in (("H0", h0), ("H10", h10)):
        peak = h.just.get(x, 0.0)
        for i in range(5):
            r = run_forced(h, rechoice(fav, oth, f"p4f{i}", harm=True), 0)
            chosen_ok = chosen_ok and r["choice"]["chosen"] == 0
        sums = event_sums(h, x)
        out[tag] = {"J_peak_before_refutation": peak, "J_residual": h.just.get(x, 0.0), "sums": sums, "reaffirm_events": event_counts(h, x, "reaffirm"),
                    "overwhelm_events": event_counts(h, x, "overwhelm"), "remaining_fraction": h.just.get(x, 0.0) / peak if peak else None}
    delta = out["H10"]["J_residual"] - out["H0"]["J_residual"]
    expl = (out["H10"]["sums"]["reaffirm"] - out["H0"]["sums"]["reaffirm"]) + (out["H10"]["sums"]["overwhelm"] - out["H0"]["sums"]["overwhelm"])
    other = {s: out["H10"]["sums"][s] - out["H0"]["sums"][s] for s in ("spread", "counter", "cap")}
    out.update({"delta": delta, "explained": expl, "unexplained": delta - expl, "unexplained_fraction": abs(delta - expl) / abs(delta) if delta else None,
                "other_sources_difference": other, "reaffirm_count_difference": out["H10"]["reaffirm_events"] - out["H0"]["reaffirm_events"], "refutation_chosen_ok": chosen_ok})
    return out


def _w_learning(ent: Dict[str, Any], model: str) -> Dict[str, np.ndarray]:
    return {f: np.array(v) for f, v in ent["dW"][model].items()}


def p5_worker(seed: int) -> Dict[str, Any]:
    base, fav, oth = base_persona(seed)
    for i in range(10):
        run_forced(base, rechoice(fav, oth, f"p5r{i}"), 0)
    jst = copy.deepcopy(base)
    ctl = copy.deepcopy(base)
    ctl.cfg = ctl.cfg.replace(rho_dis=0.0)           # 対照: 同じ J・同じ状態で、抵抗（割り引き）だけを切る
    ref0 = rechoice(fav, oth, "p5f0", harm=True)
    ap_c = run_forced(ctl, ref0, 0)
    ap_j = run_forced(jst, ref0, 0)
    ent_c = [e for e in ctl.ledger.entries if e["kind"] == "experience"][-1]
    ent_j = [e for e in jst.ledger.entries if e["kind"] == "experience"][-1]
    disc = ap_j.get("discount")
    out: Dict[str, Any] = {"seed": seed, "fav": fav, "discount_q": None if disc is None else disc["q"], "J_chosen": None if disc is None else disc["J_chosen"]}
    if disc is None:
        return out
    needs = list(ap_c["value_change"]["prediction_error"]["person"]["ally"].keys())
    err = np.array([ap_c["value_change"]["prediction_error"]["person"]["ally"][k] for k in needs])
    moved = np.array(disc["moved"]["ally"])
    # 人モデル
    wc, wj = _w_learning(ent_c, "person"), _w_learning(ent_j, "person")
    meas, pred = [], []
    for f in wc:
        red = wc[f] - wj.get(f, np.zeros(5))
        K = np.divide(wc[f], err, out=np.zeros(5), where=np.abs(err) > 1e-6)
        meas.append(red)
        pred.append(K * moved)
    meas, pred = np.concatenate(meas), np.concatenate(pred)
    out["person"] = {"R_meas_norm": float(np.linalg.norm(meas)), "R_pred_norm": float(np.linalg.norm(pred)), "rel_error": float(np.linalg.norm(meas - pred) / np.linalg.norm(meas)) if np.linalg.norm(meas) > 0 else None,
                     "moved_abs_sum": float(np.abs(moved).sum()), "n_needs_with_moved": int((moved != 0).sum()), "dW_control_norm": float(np.linalg.norm(np.concatenate(list(wc.values()))))}
    # 状況モデル（増加）: 割り引いた量が回ってくる
    sc, sj = _w_learning(ent_c, "situation"), _w_learning(ent_j, "situation")
    err_s = np.array([ap_c["value_change"]["prediction_error"]["situation"][k] for k in needs])
    gain, g_pred = [], []
    for f in sc:
        K = np.divide(sc[f], err_s, out=np.zeros(5), where=np.abs(err_s) > 1e-6)
        gain.append(sj.get(f, np.zeros(5)) - sc[f])
        g_pred.append(K * (-moved))                       # 場面の教師は moved（負）だけ増える: y_j = y_c + moved、利得 K → ΔW_j − ΔW_c = K · moved
    gain, g_pred = np.concatenate(gain), np.concatenate(g_pred)
    out["situation"] = {"gain_norm": float(np.linalg.norm(gain)), "rel_error_if_sign_flipped_pred": float(np.linalg.norm(gain + g_pred) / np.linalg.norm(gain)) if np.linalg.norm(gain) > 0 else None}
    out["teachers_total_equal"] = bool(np.allclose(np.array(ent_c["teachers"]["situation"]) + sum(np.array(t) for t in ent_c["teachers"]["person"].values()),
                                                   np.array(ent_j["teachers"]["situation"]) + sum(np.array(t) for t in ent_j["teachers"]["person"].values()), atol=1e-12))
    # 何回で反転するか（報告のみ）: 反証を合計 10 回。概念の選好 = 状況プローブ fav と oth（文脈 studio、同じ種）
    rev: Dict[str, Any] = {}
    for tag, h, first in (("with_justification", jst, ap_j), ("control", ctl, ap_c)):
        gaps_q, gaps_g, r_q, r_g = [], [], None, None
        for k in range(1, 11):
            if k > 1:
                run_forced(h, rechoice(fav, oth, f"p5f{k - 1}", harm=True), 0)
            ex = h.explain_preference([{"context": CTX_STUDIO, "concepts": [fav]}, {"context": CTX_STUDIO, "concepts": [oth]}], key="p5-pref")
            gaps_q.append(ex["delta"]["Q"]); gaps_g.append(ex["delta"]["grounded"])
            if r_q is None and ex["delta"]["Q"] < 0:
                r_q = k
            if r_g is None and ex["delta"]["grounded"] < 0:
                r_g = k
        rev[tag] = {"reversed_at_Q": r_q, "reversed_at_grounded": r_g, "gap_Q": gaps_q, "gap_grounded": gaps_g, "J_fav_end": h.just.get(f"concept:{fav}", 0.0)}
    out["reversal"] = rev
    return out


def p6_worker(seed: int) -> Dict[str, Any]:
    n, fav, oth = base_persona(seed)
    x = f"concept:{fav}"
    led0 = len(n.ledger.entries)
    j0 = dict(n.just)
    traj = []
    cmax = 0.0
    for i in range(30):
        r = run_forced(n, rechoice(fav, oth, f"p6r{i}"), 0)
        cmax = max(cmax, r["choice"]["conflict"])
        traj.append(n.just.get(x, 0.0))
    # 手がかりごとに: 増加（正の出来事の和）を源ごとに、減少（負の出来事の和）を源ごとに
    gross: Dict[str, Dict[str, float]] = {}
    drop: Dict[str, Dict[str, float]] = {}
    for en in n.ledger.entries[led0:]:
        if en["kind"] == "justification":
            for ev in en["events"]:
                tgt = gross if ev["delta"] > 0 else drop
                tgt.setdefault(ev["cue"], {})
                tgt[ev["cue"]][ev["source"]] = tgt[ev["cue"]].get(ev["source"], 0.0) + ev["delta"]
    err_gross, other_pos_max, err_net = 0.0, 0.0, 0.0
    cues = set(gross) | set(drop) | set(n.just) | set(j0)
    for c in cues:
        g, d = gross.get(c, {}), drop.get(c, {})
        err_gross = max(err_gross, abs(sum(g.values()) - g.get("reaffirm", 0.0)))
        other_pos_max = max(other_pos_max, sum(v for s_, v in g.items() if s_ != "reaffirm"))
        err_net = max(err_net, abs((n.just.get(c, 0.0) - j0.get(c, 0.0)) - (sum(g.values()) + sum(d.values()))))
    # 式: 各 reaffirm の変化量（個別に）
    run = dict(j0)
    ferr = 0.0
    nre = 0
    for en in n.ledger.entries[led0:]:
        if en["kind"] != "justification":
            continue
        start = dict(run)
        for ev in en["events"]:
            if ev["source"] == "reaffirm":
                ferr = max(ferr, abs(ev["delta"] - n.cfg.eta_r * (1.0 - abs(start.get(ev["cue"], 0.0)) / n.cfg.j_num)))
                nre += 1
            run[ev["cue"]] = run.get(ev["cue"], 0.0) + ev["delta"]
    # 対照（報告のみ）: orig で同じ段階 1 + 再び選ぶ 30 回
    of = Y.warm_persona(seed, justification_mode="orig")
    fo, oo = phase1(of)
    for i in range(30):
        run_forced(of, rechoice(fo, oo, f"p6o{i}"), 0)
    return {"seed": seed, "fav": fav, "J_fav_before": j0.get(x, 0.0), "J_fav_after": n.just.get(x, 0.0),
            "gross_increase_fav": sum(gross.get(x, {}).values()), "reaffirm_sum_fav": gross.get(x, {}).get("reaffirm", 0.0),
            "erased_by_other_sources_fav": {s_: v for s_, v in drop.get(x, {}).items()},
            "err_gross_increase_vs_reaffirm": err_gross, "other_sources_max_positive": other_pos_max, "err_net_vs_all_events": err_net,
            "reaffirm_formula_error": ferr, "reaffirm_events": nre, "max_conflict": cmax, "J_fav_trajectory_every5": traj[4::5],
            "orig_J_fav_after": of.just.get(f"concept:{fo}", 0.0), "orig_J_max": of.cfg.j_max}


def p7_worker(seed: int) -> Dict[str, Any]:
    """P7（報告のみ）: W6 の経験史 + 二択 + 訂正（B 系の Z-persist 条件と同じ 2270 イベント。摂動なし）の最終の J を、手がかりごと・源ごとに分解する。
    持続 = 最終の |J| の和、増幅 = reaffirm の和、抵抗 = 割り引き（人モデルの負の教師を場面へ回した量）と overwhelm で消えた量。"""
    n = make(seed, lc_design="spec_literal", justification_mode="persist")
    for e in B.b_events(seed, True):
        n.experience(copy.deepcopy(e))
    by, tot = ledger_j_sums(n)
    src_signed = {s: float(sum(row.get(s, 0.0) for row in by.values())) for s in SOURCES}
    src_abs = {s: float(sum(abs(row.get(s, 0.0)) for row in by.values())) for s in SOURCES}
    pos_only = float(sum(max(v, 0.0) for v in n.just.values()))
    disc = [en["discount"] for en in n.ledger.entries if en["kind"] == "justification" and en.get("discount")]
    moved_abs = float(sum(abs(x) for d in disc for m in d["moved"].values() for x in m))
    return {"seed": seed, "J": {k: float(v) for k, v in n.just.items()}, "by_source": {c: {s: float(v) for s, v in row.items()} for c, row in by.items()},
            "sum_abs_J": float(sum(abs(v) for v in n.just.values())), "sum_positive_J": pos_only, "n_cues": len(n.just),
            "source_sums_signed": src_signed, "source_sums_abs_over_cues": src_abs, "n_choices": sum(1 for e in n.ledger.entries if e["kind"] == "choice"),
            "n_discounted_choices": len(disc), "q_mean_when_discounted": float(np.mean([d["q"] for d in disc])) if disc else 0.0, "moved_abs_total": moved_abs,
            "max_abs_J_ever_final": max((abs(v) for v in n.just.values()), default=0.0)}
