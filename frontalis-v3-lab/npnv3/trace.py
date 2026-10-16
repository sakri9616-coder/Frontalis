"""κ（情報源の信用）と τ（自分の解釈への信頼）の変化の連鎖を、台帳だけから再構成する（v3.4 Stage 3。SPEC-v3.4 第 4.2〜4.5 節）。

使う台帳の項目:
  kind = "correction_experience"（訂正の出来事）: eid, source, g, llr, kappa_bar [前, 後]（訂正は κ̄ を動かさない）, log_tau {鍵: [前, 後]}, tau_effect, refs
  kind = "verification"（検証の出来事）        : eid, source, llr_ver, kappa [前, 後], d_alpha, d_beta, refs
  refs は [先行する出来事の eid, 経路] の対。経路は既にある結合だけ（凍結した仕様に新しい結合を足さない）:
    "kappa->g"（κ̄_s が吸収率 g_i を決める）、"tau->interpretation->llr"（τ が解釈の重みを変え、信念を通って LLR_i を変える）、
    "tau->interpretation->llr_ver"（検証の LLR_ver に効く当時の読み）、"verifies"（検証される訂正）、"kappa(prev)"（同じ情報源の前の検証）。

「増幅」の定義（観察のための道具で、判定ではない）: 受け入れる向きの符号を κ̄ の変化は +、log τ の変化は − とする（τ が下がる = 自分の読みに頼らなくなる = 受け入れる向き）。
参照の連鎖でつながった出来事が、同じ符号の変化を続けて起こしたとき、その連なりを「自己強化に見える連鎖」と呼び、最初の出来事を起点として報告する。
自己強化は故障ではない。上限も罰則もここにはない。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _events(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in entries if e.get("kind") in ("correction_experience", "verification")]


def tau_trajectory(entries: List[Dict[str, Any]], final_log_tau: Optional[Dict[str, float]] = None) -> List[Dict[str, float]]:
    """訂正の通し番号ごとの、その訂正の評価の後の log τ の全体（τ-yoke に渡す軌跡）。台帳の log_tau の [前, 後] から再構成する。"""
    cur: Dict[str, float] = {}
    out: List[Dict[str, float]] = []
    for e in entries:
        if e.get("kind") != "correction_experience":
            continue
        for k, (b, a) in e["log_tau"].items():
            cur[k] = float(a)
        out.append(dict(cur))
    return out


def trace_loop(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """台帳の出来事から κ̄ と log τ の変化を再生し、因果の連鎖を返す。
    返すもの: nodes（出来事ごと）、edges（参照）、kappa / log_tau（再生した最終値。状態と一致するはず）、consistent / problems（前の値の連続性）、
    runs（符号の同じ連鎖。長さの降順）、origin（最長の連鎖の起点の eid）。"""
    evs = _events(entries)
    ab: Dict[str, List[float]] = {}
    lt: Dict[str, float] = {}
    problems: List[str] = []
    nodes: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for e in evs:
        eid = e["eid"]
        if eid in nodes:
            problems.append(f"eid の重複: {eid}")
        node: Dict[str, Any] = {"eid": eid, "kind": e["kind"], "source": e.get("source"), "refs": [list(r) for r in e.get("refs", [])], "changes": []}
        if e["kind"] == "verification":
            s = e["source"]
            a0, b0 = ab.get(s, [1.0, 1.0])
            k0 = a0 / (a0 + b0)
            if abs(k0 - e["kappa"][0]) > 1e-9:
                problems.append(f"{eid}: κ̄ の前の値が再生と合わない ({k0} vs {e['kappa'][0]})")
            a1, b1 = a0 + e["d_alpha"], b0 + e["d_beta"]
            ab[s] = [a1, b1]
            k1 = a1 / (a1 + b1)
            node.update({"llr": e["llr_ver"], "kappa": [k0, k1], "tau_snapshot": e.get("tau_snapshot", {})})
            d = k1 - k0
            if d != 0.0:
                node["changes"].append({"var": "kappa:" + s, "delta": d, "accept_sign": 1 if d > 0 else -1})
        else:
            node.update({"llr": e["llr"], "g": e["g"], "kappa": e["kappa_bar"], "log_tau": e["log_tau"], "tau_effect": e.get("tau_effect", {}),
                         "own": e["own"], "guilt": e["guilt"], "regret": e["regret"], "defend": e["defend"], "tau_source": e["tau_source"]})
            for k, (b, a) in e["log_tau"].items():
                if abs(lt.get(k, 0.0) - b) > 1e-12:
                    problems.append(f"{eid}: log τ[{k}] の前の値が再生と合わない ({lt.get(k, 0.0)} vs {b})")
                lt[k] = float(a)
                d = a - b
                if d != 0.0:
                    node["changes"].append({"var": "tau:" + k, "delta": d, "accept_sign": -1 if d > 0 else 1})
        nodes[eid] = node
        order.append(eid)
    edges: List[Dict[str, str]] = []
    for eid in order:
        for r, via in nodes[eid]["refs"]:
            if r not in nodes:
                problems.append(f"{eid}: 参照先 {r} が台帳にない")
            edges.append({"from": r, "to": eid, "via": via})
    # 祖先の閉包
    # 同じ情報源の前の検証への参照（kappa(prev)）は、同じ量の積み重ねなので連鎖の判定には使わない（edges には残す）
    parents: Dict[str, List[str]] = {eid: [r for r, via in nodes[eid]["refs"] if r in nodes and via != "kappa(prev)"] for eid in order}
    anc: Dict[str, set] = {}
    for eid in order:
        s_: set = set()
        for p in parents[eid]:
            s_ |= {p} | anc.get(p, set())
        anc[eid] = s_
    # 同じ符号の変化が参照でつながって続く連鎖（最長の連なり）
    best: Dict[Tuple[str, int], Tuple[int, str, List[str]]] = {}
    runs: List[Dict[str, Any]] = []
    for eid in order:
        for ch in nodes[eid]["changes"]:
            sg = ch["accept_sign"]
            cand = (1, eid, [eid])
            for a_ in anc[eid]:
                for ch2 in nodes[a_]["changes"]:
                    if ch2["accept_sign"] == sg:
                        n_, o_, path = best[(a_, ch2["var"])]
                        if n_ + 1 > cand[0]:
                            cand = (n_ + 1, o_, path + [eid])
            best[(eid, ch["var"])] = cand
            runs.append({"end": eid, "var": ch["var"], "accept_sign": sg, "length": cand[0], "origin": cand[1], "path": cand[2]})
    runs.sort(key=lambda r: (-r["length"], order.index(r["end"])))
    kappa = {s: a / (a + b) for s, (a, b) in ab.items()}
    return {"nodes": [nodes[e] for e in order], "edges": edges, "kappa": kappa, "alpha_beta": {s: list(v) for s, v in ab.items()}, "log_tau": lt,
            "consistent": not problems, "problems": problems, "runs": runs, "origin": (runs[0]["origin"] if runs and runs[0]["length"] > 1 else None),
            "cross_edges": [e for e in edges if (e["via"] == "kappa->g") or e["via"].startswith("tau->")]}
