"""応答の内訳と割当の記録の、人が読む形（v3.3 第 1 部の設計の改訂。SPEC-v3.3 6.2）。NPN.explain_response / assignment_trace の戻り値を整形する。CLI の explain とデモが使う。"""
from __future__ import annotations

from typing import Any, Dict, List


def format_explanation(ex: Dict[str, Any]) -> List[str]:
    """応答 C(cue | 文脈, t) = Σ_k p(k) C_k の内訳。p(k) = w_k/(Σ w + α)、w_k = 時間の項 × 文脈の項 × 計数の項。"""
    ctx = ex["context"] if ex["context"] is not None else "（文脈を問わない）"
    lines = [f"手がかり {ex['cue']}  文脈 {ctx}  時刻 {ex['t']:.0f}  設計 {ex['design']}: 応答 C = Σ p(k)·C_k = {ex['response']:+.4f}"]
    if not ex["known"]:
        lines.append("  （この手がかりはまだ経験していない）")
        return lines
    rev = ex["design"] == "revised"
    lines.append(f"  {'原因':<6}{'C_k':>9}{'p(k)':>8}{'p·C':>9}   {'時間の項':>9}{'文脈の項':>9}{'計数の項':>9}   試行数  文脈での計数  β_k   最後の試行")
    for c in ex["causes"]:
        nid = f"#{c['uid']}" if c["uid"] >= 0 else f"k{c['k']}"
        nin = "-" if c["n_in_context"] is None else f"{c['n_in_context']:.0f}"
        ntr = "-" if c["n_trials"] is None else f"{c['n_trials']:.0f}"
        lines.append(f"  {nid:<6}{c['value']:>+9.4f}{c['p']:>8.3f}{c['contribution']:>+9.4f}   {c['time_term']:>9.3f}{c['context_term']:>9.3f}{c['count_term']:>9.3f}"
                     f"   {ntr:>4}  {nin:>10}  {c.get('beta', float('nan')):>5.1f}  {c.get('last_trial_t', float('nan')):>6.0f}")
    lines.append(f"  {'新':<6}{0.0:>+9.4f}{ex['p_new']:>8.3f}{0.0:>+9.4f}   （新しい原因の重み α = {ex['alpha']:g}。価値 0）")
    if rev:
        lines.append("  時間の項 = Σ_{i∈k} (t − t_i + 1)^(−d)、文脈の項 = (n_k(c) + β_k)/(N_k + β_k S)。")
    return lines


def format_assignment(rec: Dict[str, Any]) -> str:
    """割当の記録 1 件（kind = "assignment"）。事前（内訳）→ 事後 → MAP の原因 → 学習前後の C。"""
    if rec["kind"] == "reinstate":
        ch = rec.get("chosen")
        if ch is None:
            return f"t={rec['t']:.0f} 復位の推論（文脈 {rec['context']}、嫌悪 r={rec['r']:+.3f}）: どの原因でもない（事後 {rec['null_posterior']:.2f}）"
        return (f"t={rec['t']:.0f} 復位の推論（文脈 {rec['context']}、嫌悪 r={rec['r']:+.3f}）: {ch['cue']} の原因 #{ch['uid']}（C={ch['C']:+.3f}）に"
                f"試行 (t, {rec['context']}) を加えた。事後 {ch['posterior']:.2f}（どの原因でもない {rec['null_posterior']:.2f}）。価値は更新しない")
    alive = [(u, i) for i, u in enumerate(rec["uid_before"]) if u >= 0]
    pri = " ".join(f"#{u}:{rec['p_prior'][i]:.3f}" for u, i in alive)
    tm = " ".join(f"{rec['time_term'][i]:.3f}" for u, i in alive)
    cx = " ".join(f"{rec['context_term'][i]:.3f}" for u, i in alive)
    post = " ".join(f"#{u}:{rec['post'][i]:.3f}" for u, i in alive)
    k = rec["map"]
    mapn = f"#{rec['map_uid']}" + ("（新しい原因" + ("。満員なので #%d を再利用" % rec["replaced_uid"] if rec["reused"] else "") + "）" if rec["created"] is not None else "")
    c0 = 0.0 if rec["created"] is not None else rec["C_before"][k]
    return (f"t={rec['t']:.0f} 文脈 {rec['context']} 目標 y={rec['y_x']:+.3f}: 事前 {pri} 新(α):{rec['p_prior_new']:.3f} [時間の項 {tm} | 文脈の項 {cx}]"
            f" → 事後 {post} 新:{rec['post_new']:.3f} → MAP {mapn}: C {c0:+.4f} → {rec['C_after'][k]:+.4f}")
