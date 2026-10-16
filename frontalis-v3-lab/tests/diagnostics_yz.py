"""診断（テストではない）: Y1・Y2 の数値の切り分けと、Z4 の感度。判定には使わず、**どれも既定・プロトコルに入れていない**。結果は tests/results/diag_yz.json。

背景: v3.3 第 2 部の実装で、Y1（事実の訂正: 対照への近づき 0.80 以上）が 0.772 で不合格、Y2（残る割合 0.3〜0.9）が 0.302 で境界上の合格、
Z4（5 回以内の反転）が 20 本中 10 本で不合格だった。この診断は、数値を作っている機構を分けて見せる。
  A. Y1: p_person（スコープの混合: 個体 + 文脈）と p_ind（個体の部分）での近づき、想起の掃引（recall_after_correction。SPEC にない実験設定）の効果、
     取り消した後の責任 φ（訂正後の観測の人のせい / 場面のせい）と、対照のオンラインの φ の比較。
  B. Y2: 再評価の強さ w_reapp を変えたときの残る割合（SPEC の w_reapp = 0.5 は変えない。動きの大きさを見るだけ）。
  C. Z4: 害の大きさ・害を与える方法（概念だけの状況 / 参加者が行為者）・J を作る回数を変えたときの、反転までの害の回数。
実行: python3 tests/diagnostics_yz.py
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yz_protocol as Y  # noqa: E402
from helpers import SEEDS, dump, make, pmap  # noqa: E402
from npnv3.innate import TYPE_INDEX  # noqa: E402

CT = TYPE_INDEX["credit_taken"]


def phi_share(npn):
    """経験 j0..j7 の credit_taken の観測の、人のせい φ の平均（取り消し・再計算後の記録 rec_phi。行為者の行だけ）と、π_person。"""
    b = npn.belief
    vals = []
    for r in range(b.rec_n):
        if b.rec_y[r, CT] > 0.5 and b.rec_m[r, CT] > 0.5:
            vals.append(b.rec_phi[r, CT])
    return float(np.mean(vals)) if vals else None, float(b.pi[CT, 0] / b.pi[CT].sum())


def a_job(seed):
    mis, ctrl = make(seed), make(seed)
    Y.feed(mis, Y.y_history(seed, True))
    Y.feed(ctrl, Y.y_history(seed, False))
    cor = copy.deepcopy(mis)
    Y.feed(cor, Y.y_corrections(seed))
    rec = make(seed, recall_after_correction=True)
    Y.feed(rec, Y.y_history(seed, True))
    Y.feed(rec, Y.y_corrections(seed))
    ctrl_sweep = copy.deepcopy(ctrl)
    ctrl_sweep.belief.reattribute()
    b = lambda n: Y.belief_A(n)
    bi = lambda n: Y.belief_A_individual(n)
    return {"P": {"mis": b(mis), "ctrl": b(ctrl), "corrected": b(cor), "corrected_with_recall_option": b(rec), "ctrl_after_one_sweep": b(ctrl_sweep)},
            "Pind": {"mis": bi(mis), "ctrl": bi(ctrl), "corrected": bi(cor), "corrected_with_recall_option": bi(rec)},
            "phi_credit_taken_and_pi": {"mis": phi_share(mis), "ctrl": phi_share(ctrl), "corrected": phi_share(cor)}}


def b_job(args):
    seed, w = args
    mis, ctrl = make(seed, w_reapp=w), make(seed, w_reapp=w)
    Y.feed(mis, Y.y_history(seed, True))
    Y.feed(ctrl, Y.y_history(seed, False))
    d0 = Y.like_A(mis) - Y.like_A(ctrl)
    Y.feed(mis, Y.y_corrections(seed))
    return {"remaining": (Y.like_A(mis) - Y.like_A(ctrl)) / d0}


def z4_job(args):
    seed, harm, build, mode, justification = args
    n = make(seed, no_justification=not justification)
    pa = {"id": "z4pa", "class": "friend"}
    pb = {"id": "z4pb", "class": "rival"}
    opt = lambda c, p: {"context": "studio", "concepts": [c], "participants": [p] if (mode == "participant" and p) else []}
    for i in range(build):
        n.experience({"id": f"z4c{i}", "kind": "choice", "options": [opt("z4_a", pa), opt("z4_b", pb)], "outcomes_by_option": {"0": [], "1": []}})
    probe = lambda c, p: n.probe({"context": "studio", "concepts": [c], "participants": [p] if (mode == "participant") else []})["Q"]
    qa, qb = probe("z4_a", pa), probe("z4_b", pb)
    pref = "a" if qa >= qb else "b"
    pc, pp = ("z4_a", pa) if pref == "a" else ("z4_b", pb)
    oc, op = ("z4_b", pb) if pref == "a" else ("z4_a", pa)
    reversed_at = None
    for k in range(1, 11):
        out = [{"type": "harm", "actor": pp["id"] if mode == "participant" else "world", "magnitude": harm}]
        n.experience({"id": f"z4h{k}", "context": "studio", "concepts": [pc], "participants": [pp] if mode == "participant" else [], "outcomes": out, "session": f"z4hs{k}"})
        if probe(pc, pp) - probe(oc, op) < 0.0:
            reversed_at = k
            break
    return {"gap0": abs(qa - qb), "reversed_at": reversed_at}


def main():
    res = {}
    rows = pmap(a_job, SEEDS)
    m = lambda f: float(np.mean([f(r) for r in rows]))
    P = {k: m(lambda r: r["P"][k]) for k in rows[0]["P"]}
    Pi = {k: m(lambda r: r["Pind"][k]) for k in rows[0]["Pind"]}
    res["A_Y1"] = {"p_person_means": P, "closeness_p_person": (P["mis"] - P["corrected"]) / (P["mis"] - P["ctrl"]),
                   "p_ind_means": Pi, "closeness_p_ind": (Pi["mis"] - Pi["corrected"]) / (Pi["mis"] - Pi["ctrl"]),
                   "closeness_with_recall_option (vs ctrl without sweep; overshoots)": (P["mis"] - P["corrected_with_recall_option"]) / (P["mis"] - P["ctrl"]),
                   "closeness_with_recall_option (vs ctrl after one sweep)": (P["mis"] - P["corrected_with_recall_option"]) / (P["mis"] - P["ctrl_after_one_sweep"]),
                   "phi_of_credit_taken_observations_and_pi_person (seed 0)": rows[0]["phi_credit_taken_and_pi"]}
    res["B_Y2_w_reapp"] = {}
    for w in (0.25, 0.5, 1.0, 2.0):
        r = pmap(b_job, [(s, w) for s in SEEDS])
        res["B_Y2_w_reapp"][str(w)] = float(np.mean([x["remaining"] for x in r]))
    res["C_Z4"] = {}
    for harm in (0.8, 1.0):
        for build in (3, 6, 12):
            for mode in ("concept_only", "participant"):
                for just in (True, False):
                    r = pmap(z4_job, [(s, harm, build, mode, just) for s in SEEDS])
                    ra = [x["reversed_at"] for x in r]
                    res["C_Z4"][f"harm={harm} build={build} {mode} justification={just}"] = {
                        "gap_before_mean": float(np.mean([x["gap0"] for x in r])), "reversed_at": ra,
                        "within_5": int(sum(a is not None and a <= 5 for a in ra)), "never_within_10": int(sum(a is None for a in ra))}
    dump("diag_yz", res)
    print("Y1 closeness p_person %.3f  p_ind %.3f  recall-option %.3f / %.3f" % (res["A_Y1"]["closeness_p_person"], res["A_Y1"]["closeness_p_ind"],
          res["A_Y1"]["closeness_with_recall_option (vs ctrl without sweep; overshoots)"], res["A_Y1"]["closeness_with_recall_option (vs ctrl after one sweep)"]))
    print("Y2 remaining by w_reapp:", res["B_Y2_w_reapp"])
    for k, v in res["C_Z4"].items():
        print(k, "within 5:", v["within_5"], "never:", v["never_within_10"], "gap0 %.3f" % v["gap_before_mean"])


if __name__ == "__main__":
    main()
