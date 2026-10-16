"""機構の追跡（SPEC-v3.3 6.3 の M1〜M5）の測定。tests/test_m_trace.py と tests/diagnostics_e_revised.py が使う。

目的: 数値として E3〜E6 が合格しても、「機構として嫌悪が戻った」のか「係数で嫌悪を強めただけ」なのかを区別する。割当の記録（台帳の kind = "assignment" / "reinstate"）と、
応答の内訳（Conditioning.explain）から、消去の直後の状態からテスト時点までの応答の変化を分ける。

【定義（M の結果を計算する前に固定した細部。SPEC 6.3 の表は変えていない）】
  獲得の原因   獲得 12 試行の直後に、手がかり x の原因のうち、割り当てられた試行が最も多いもの（同数なら添字が小さい方）。原因の通し番号 uid で追う
              （満員のときの再利用で別の原因が同じ添字に入っても、uid で見分けられる。消えていたら価値 0、重み 0 とみなす）。
  応答の変化の分解  応答 = Σ_k p_k C_k。消去直後の状態 S0（文脈 c0、時刻 t0）からテスト状態 S1（c1, t1）まで:
              Δresp = Σ_k (p1_k − p0_k) C0_k   [重み付けの移り変わり]  +  Σ_k p1_k (C1_k − C0_k)   [価値の変化]
              （恒等式。S0 に無い原因の C0 = 0、S1 で消えた原因の C1 = C0）。
              E3: S1 = 同じ状態を文脈 A で読む。E4: 未見の文脈 C。E5: 無関係な経験 50 回の後に B で読む。E6: 手がかりなしの害 2 回の後に B で読む。S0 は消去直後を B で読む。
  M1 の数値    重み付けの項の合計 / Δresp の合計（20 本の合計どうしの比）。M2: 全種・全プロトコルでの最大の増加。
  獲得の原因の事前の対数比  ρ = log w_獲得 − log(Σ_{k≠獲得} w_k + α)（w_k は事前の重み。獲得の原因が他の全ての原因と新しい原因に対して持つ対数オッズ）。
              S0 → S1 の変化 Δρ を、順に 3 段に分ける:  (1) 文脈  S0 の状態を c1 で読む  (2) 時間  読む時刻を t0 → t1 に  (3) 試行・計数  S1 の状態（加わった試行）で読む。
              「時間」= 同じ試行の集合で経過した時間だけ（質量 Σ(t−t_i+1)^(−d) の変化）。「文脈」= 同じ試行の集合で文脈の項だけ。「試行・計数」= 加わった試行が質量と計数に入る分。
              M3: E3・E4 は (1) の割合、E5 は (2) の割合、E6 は (3) の割合が 50% 以上（Δρ > 0 のとき）。
  M4          消去 20 試行のうち、割当の記録で獲得の原因が MAP に選ばれた試行の数、と、消去中の獲得の原因の |ΔC| の和（記録の学習前後の C_k から）。
  M5          同じ種で 2 回走らせた割当の記録の完全一致、と、消去直後の保存 → 復元で assignment_trace と explain_response が完全一致。
"""
from __future__ import annotations

import copy
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from e_protocol import CTX_A, CTX_B, CTX_C, CUE_NAME, run_variant, states_equal
from helpers import deep_equal
from npnv3 import NPN


def table(cd, ctx: Optional[str], t: float) -> Dict[int, Dict[str, Any]]:
    ex = cd.explain(CUE_NAME, ctx, t)
    return {int(c["uid"]): c for c in ex["causes"]}


def acq_uid_of(npn: NPN) -> int:
    cd = npn.cond
    i = cd.idx[CUE_NAME]
    n = cd.hist_n[i]
    best = None
    for k in range(cd.K):
        if cd.AL[i, k] > 0.5:
            cnt = int((cd.hist_k[i][:n] == k).sum())
            if best is None or cnt > best[0]:
                best = (cnt, k)
    return int(cd.UID[i, best[1]])


def c_of_uid(npn: NPN, uid: int) -> Optional[float]:
    cd = npn.cond
    i = cd.idx[CUE_NAME]
    hit = np.nonzero((cd.UID[i] == uid) & (cd.AL[i] > 0.5))[0]
    return float(cd.CK[i, hit[0]]) if hit.size else None


def decompose(cd0, ctx0: str, t0: float, cd1, ctx1: str, t1: float) -> Dict[str, float]:
    a, b = table(cd0, ctx0, t0), table(cd1, ctx1, t1)
    r0 = sum(c["contribution"] for c in a.values())
    r1 = sum(c["contribution"] for c in b.values())
    weight = value = 0.0
    for u in set(a) | set(b):
        p0, c0 = (a[u]["p"], a[u]["value"]) if u in a else (0.0, 0.0)
        p1, c1 = (b[u]["p"], b[u]["value"]) if u in b else (0.0, c0)
        weight += (p1 - p0) * c0
        value += p1 * (c1 - c0)
    return {"resp0": r0, "resp1": r1, "delta": r1 - r0, "weighting": weight, "value": value, "identity_error": (r1 - r0) - (weight + value)}


def rho(cd, ctx: str, t: float, uid: int) -> Tuple[Optional[float], Dict[str, float]]:
    """獲得の原因の事前の対数比 ρ（None = 獲得の原因が無い）と、その要因（log 時間の項、log 文脈の項、log 他の重みの和）。"""
    ex = cd.explain(CUE_NAME, ctx, t)
    mine = [c for c in ex["causes"] if c["uid"] == uid]
    if not mine:
        return None, {}
    c = mine[0]
    others = sum(x["weight"] for x in ex["causes"] if x["uid"] != uid) + ex["alpha"]
    parts = {"log_time": float(np.log(c["time_term"])), "log_context": float(np.log(c["context_term"])), "log_others": float(np.log(others))}
    return float(np.log(c["weight"]) - np.log(others)), parts


def rho_steps(cd0, ctx0: str, t0: float, cd1, ctx1: str, t1: float, uid: int) -> Dict[str, Any]:
    r0, p0 = rho(cd0, ctx0, t0, uid)
    ra, _ = rho(cd0, ctx1, t0, uid)         # (1) 文脈
    rb, _ = rho(cd0, ctx1, t1, uid)         # (2) 時間
    rc, p1 = rho(cd1, ctx1, t1, uid)        # (3) 試行・計数
    if None in (r0, ra, rb, rc):
        return {"defined": False}
    total = rc - r0
    out = {"defined": True, "rho0": r0, "rho1": rc, "delta": total, "step_context": ra - r0, "step_time": rb - ra, "step_trials": rc - rb,
           "parts0": p0, "parts1": p1}
    for nm in ("context", "time", "trials"):
        out[f"share_{nm}"] = (out[f"step_{nm}"] / total) if total > 1e-9 else None
    return out


def assignments(npn: NPN, start: int = 0, end: Optional[int] = None) -> List[Dict[str, Any]]:
    return [e for e in npn.ledger.entries[start:end] if e["kind"] == "assignment" and e["cue"] == CUE_NAME]


def m_worker(seed: int) -> Dict[str, Any]:
    """1 つの人格（潜在原因、既定の設計）について、E 系のプロトコルを走らせ、M1〜M5 の測定をする。"""
    box: Dict[str, Any] = {}
    e7_series: List[float] = []

    def cb_e7(i: int, npn: NPN) -> None:
        v = c_of_uid(npn, box["uid"])
        e7_series.append(abs(v) if v is not None else 0.0)

    # 獲得の原因の uid は獲得直後に決まる。cb_E7 はそのあとで呼ばれるので、先に箱の uid を入れる仕組み: acq の写しができたら uid を入れる
    class Box(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            if k == "acq":
                super().__setitem__("uid", acq_uid_of(v))
    box = Box()
    box["cb_E7"] = cb_e7
    out = run_variant(seed, collect=box)
    uid = box["uid"]
    acq, ext = box["acq"], box["ext"]
    c_post = c_of_uid(acq, uid)
    T = float(ext._clock())
    res: Dict[str, Any] = {"seed": seed, "acq_uid": uid, "C_acq_post": c_post, "E": {k: out[k] for k in ("E1", "E2_B", "ABA", "ABC", "E5_after", "E6_reinstated", "E6_control", "E7_first", "E7_re")}}
    # --- M2: 全プロトコルのテスト時点の獲得の原因の |C|
    states = {"after_extinction (E2, E3, E4)": ext, "E1": box["E1"], "E5": box["E5"], "E6": box["E6"]}
    m2 = {}
    for nm, st in states.items():
        v = c_of_uid(st, uid)
        m2[nm] = abs(v) if v is not None else 0.0
    idx = min(int(out["E7_re"]), len(e7_series)) - 1
    m2["E7 (at the trial reaching 70%)"] = e7_series[idx]
    res["M2"] = {"C_acq_post_abs": abs(c_post), "abs_C_at_test": m2, "max_increase": max(v - abs(c_post) for v in m2.values()),
                 "acq_cause_alive_after_extinction": c_of_uid(ext, uid) is not None}
    # --- M4: 消去中の割当
    n0, n1 = len(acq.ledger.entries), len(ext.ledger.entries)
    recs = assignments(ext, n0, n1)
    picked = [r for r in recs if r["map_uid"] == uid]
    dsum = 0.0
    for r in recs:
        if uid in r["uid_before"]:
            k = r["uid_before"].index(uid)
            dsum += abs(r["C_after"][k] - r["C_before"][k])
    res["M4"] = {"n_extinction_updates": len(recs), "acq_cause_selected": len(picked), "acq_cause_abs_dC_sum": dsum,
                 "assigned_to_new_cause": int(sum(1 for r in recs if r["created"] is not None)),
                 "reuse_events": int(sum(1 for r in recs if r["reused"])), "replaced_acq_cause": bool(any(r["replaced_uid"] == uid for r in recs))}
    # --- M1・M3: 応答の変化の分解（消去直後を B で読んだ状態 S0 から）
    cases = {"E3": (CTX_A, ext, T), "E4": (CTX_C, ext, T), "E5": (CTX_B, box["E5"], float(box["E5"]._clock())), "E6": (CTX_B, box["E6"], float(box["E6"]._clock()))}
    res["M1"], res["M3"] = {}, {}
    for nm, (ctx1, st1, t1) in cases.items():
        res["M1"][nm] = decompose(ext.cond, CTX_B, T, st1.cond, ctx1, t1)
        res["M3"][nm] = rho_steps(ext.cond, CTX_B, T, st1.cond, ctx1, t1, uid)
    # --- M5: 決定性と保存・復元
    box2: Dict[str, Any] = {}
    run_variant(seed, collect=box2)
    same = deep_equal([e for e in box["ext"].ledger.entries if e["kind"] in ("assignment", "reinstate")],
                      [e for e in box2["ext"].ledger.entries if e["kind"] in ("assignment", "reinstate")])
    same &= deep_equal([e for e in box["E6"].ledger.entries if e["kind"] in ("assignment", "reinstate")],
                       [e for e in box2["E6"].ledger.entries if e["kind"] in ("assignment", "reinstate")])
    with tempfile.TemporaryDirectory() as d:
        box["E6"].save(d)
        loaded = NPN.load(d)
    restored = deep_equal(box["E6"].assignment_trace(CUE_NAME), loaded.assignment_trace(CUE_NAME)) and states_equal(box["E6"], loaded)
    for ctx in (CTX_A, CTX_B, CTX_C):
        restored &= deep_equal(box["E6"].explain_response(CUE_NAME, ctx), loaded.explain_response(CUE_NAME, ctx))
    res["M5"] = {"deterministic": bool(same), "restored_equal": bool(restored),
                 "n_records": int(sum(1 for e in box["E6"].ledger.entries if e["kind"] in ("assignment", "reinstate")))}
    res["narrative"] = narrative(box, uid, out) if seed == 0 else None
    return res


def narrative(box: Dict[str, Any], uid: int, out: Dict[str, Any]) -> Dict[str, List[str]]:
    """1 つの種の割当の記録の要約（人が読む用）。"""
    acq, ext = box["acq"], box["ext"]
    lines: Dict[str, List[str]] = {}
    ra = assignments(acq)[-12:]                                  # 獲得の 12 試行（手がかり x の更新 12 件）
    lines["acquisition"] = [f"獲得 {len(ra)} 試行の割当: " + ", ".join(f"#{r['map_uid']}{'(新)' if r['created'] is not None else ''}" for r in ra),
                            f"獲得の原因 = #{uid}（C = {c_of_uid(acq, uid):+.3f}）。他の原因: " + ", ".join(
                                f"#{int(acq.cond.UID[acq.cond.idx[CUE_NAME], k])}（C = {acq.cond.CK[acq.cond.idx[CUE_NAME], k]:+.3f}、β_k = {acq.cond.BK[acq.cond.idx[CUE_NAME], k]:g}）"
                                for k in range(acq.cond.K) if acq.cond.AL[acq.cond.idx[CUE_NAME], k] > 0.5 and int(acq.cond.UID[acq.cond.idx[CUE_NAME], k]) != uid)]
    n0, n1 = len(acq.ledger.entries), len(ext.ledger.entries)
    re_ = assignments(ext, n0, n1)
    lines["extinction"] = [f"消去 {len(re_)} 試行の割当: " + ", ".join(f"#{r['map_uid']}{'(新)' if r['created'] is not None else ''}{'(再利用)' if r['reused'] else ''}" for r in re_),
                           f"獲得の原因 #{uid} が選ばれた試行 {sum(1 for r in re_ if r['map_uid'] == uid)} 件、|ΔC| の和 "
                           f"{sum(abs(r['C_after'][r['uid_before'].index(uid)] - r['C_before'][r['uid_before'].index(uid)]) for r in re_ if uid in r['uid_before']):.4f}"]
    first = re_[0]
    lines["extinction_first_trial"] = [
        f"消去の最初の試行（文脈 {first['context']}、目標 y = {first['y_x']:+.3f}）: 事前 p(k) = " + ", ".join(
            f"#{u}: {p:.3f}" for u, p in zip(first['uid_before'], first['p_prior']) if u >= 0) + f", 新 α: {first['p_prior_new']:.3f}"
        + "（時間の項 " + ", ".join(f"{x:.3f}" for x, u in zip(first['time_term'], first['uid_before']) if u >= 0)
        + "、文脈の項 " + ", ".join(f"{x:.3f}" for x, u in zip(first['context_term'], first['uid_before']) if u >= 0) + "）",
        "  尤度 " + ", ".join(f"#{u}: {l:.3f}" for u, l in zip(first['uid_before'], first['likelihood']) if u >= 0) + f", 新: {first['likelihood_new']:.3f}"
        + f" → 事後 " + ", ".join(f"#{u}: {p:.3f}" for u, p in zip(first['uid_before'], first['post']) if u >= 0) + f", 新: {first['post_new']:.3f}"
        + f" → MAP = #{first['map_uid']}" + ("（新しい原因）" if first['created'] is not None else "")]
    T = float(ext._clock())
    for nm, st, ctx, t1 in (("E3", ext, CTX_A, T), ("E4", ext, CTX_C, T), ("E5", box["E5"], CTX_B, float(box["E5"]._clock())), ("E6", box["E6"], CTX_B, float(box["E6"]._clock()))):
        b0 = table(ext.cond, CTX_B, T)
        b1 = table(st.cond, ctx, t1)
        ctx0 = f"B, t = {T:.0f}"
        lines[nm] = [f"{nm}: 読む前 ({ctx0}) 応答 {sum(c['contribution'] for c in b0.values()):+.4f}  →  テスト ({ctx}, t = {t1:.0f}) 応答 {sum(c['contribution'] for c in b1.values()):+.4f}",
                     "  重み p(k): " + ", ".join(f"#{u}{'*' if u == uid else ''}: {b0.get(u, {'p': 0.0})['p']:.3f} → {b1.get(u, {'p': 0.0})['p']:.3f}" for u in sorted(set(b0) | set(b1))) + f", 新: "
                     f"{ext.cond.explain(CUE_NAME, CTX_B, T)['p_new']:.3f} → {st.cond.explain(CUE_NAME, ctx, t1)['p_new']:.3f}（* = 獲得の原因 C = {c_of_uid(st, uid) if c_of_uid(st, uid) is not None else float('nan'):+.3f}）"]
        rs = rho_steps(ext.cond, CTX_B, T, st.cond, ctx, t1, uid)
        if rs["defined"]:
            lines[nm].append(f"  獲得の原因の事前の対数比 ρ {rs['rho0']:+.3f} → {rs['rho1']:+.3f}（文脈 {rs['step_context']:+.3f}、時間 {rs['step_time']:+.3f}、試行・計数 {rs['step_trials']:+.3f}）")
    rein = [e for e in box["E6"].ledger.entries[len(ext.ledger.entries):] if e["kind"] == "reinstate"]
    for r in rein:
        ch = r.get("chosen")
        lines["E6"].append(f"  復位の推論 (t = {r['t']:.0f}, 文脈 {r['context']}, 嫌悪 r = {r['r']:+.3f}): " + (
            f"原因 #{ch['uid']}（C = {ch['C']:+.3f}）に (t, 文脈 {r['context']}) の試行を加えた。事後 {ch['posterior']:.2f}（どの原因でもない: {r['null_posterior']:.2f}）" if ch else "どの原因でもない"))
    return lines
