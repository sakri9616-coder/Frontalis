"""診断（テストではない）: Z-persist（SPEC-v3.3 8.3）の係数と条件の感度。判定には使わず、**どれも既定・プロトコルに入れていない**。結果は tests/results/diag_persist.json。

  A. Z4 のプロトコル（迷う二択 6 回で J を作り、好む側に害を 1 回ずつ（通常の経験として）与え、Q の差が反転するまで）を、Z-orig・Z-persist・persist_with_decay で比べる。
     通常の経験は二択ではないので、persist の抵抗（割り引き）と overwhelm は働かない（J は時間で減衰しないだけ）。
  B. overwhelm の強さ η_ov（SPEC に比例の係数の指定がなく、試走の後に 1.0 の乗算型に決めた。CHANGELOG の S 節）の感度: 段階 1（3 回）→ 再び選ぶ 10 回 → 反証（harm 0.8 / 0.3）5 回。
     反証の後に残る J（fav）の割合と、reaffirm・overwhelm の和。
実行: python3 tests/diagnostics_persist.py
"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p_protocol as P  # noqa: E402
import yz_protocol as Y  # noqa: E402
from helpers import SEEDS, dump, pmap  # noqa: E402


def z4_job(args):
    seed, mode = args
    return Y.z4_worker(seed, justification_mode=mode)


def b_job(args):
    seed, eta_ov, harm_mag = args
    base = Y.warm_persona(seed, justification_mode="persist", eta_ov=eta_ov)
    fav, oth = P.phase1(base)
    x = f"concept:{fav}"
    for i in range(10):
        P.run_forced(base, P.rechoice(fav, oth, f"b{i}"), 0)
    peak = base.just[x]
    ref = P.rechoice(fav, oth, "rf", harm=True)
    ref["outcomes_by_option"]["0"] = [{"type": "harm", "actor": "ally", "magnitude": harm_mag}]
    traj = []
    for i in range(5):
        r = copy.deepcopy(ref)
        r["id"] = f"rf{i}"
        P.run_forced(base, r, 0)
        traj.append(base.just.get(x, 0.0))
    s = P.event_sums(base, x)
    return {"peak": peak, "residual": traj[-1], "fraction": traj[-1] / peak, "after_first": traj[0] / peak, "reaffirm": s["reaffirm"], "overwhelm": s["overwhelm"]}


def main():
    res = {"A_Z4_by_mode": {}, "B_eta_ov": {}}
    for mode in ("orig", "persist", "persist_with_decay"):
        r = pmap(z4_job, [(s, mode) for s in SEEDS])
        ra = [x["reversed_at"] for x in r]
        res["A_Z4_by_mode"][mode] = {"reversed_at": ra, "within_5": int(sum(a is not None and a <= 5 for a in ra)), "never_within_10": int(sum(a is None for a in ra)),
                                     "J_pref_mean": float(np.mean([x["J_pref"] for x in r])), "gap_before_mean": float(np.mean([x["gap_before"] for x in r]))}
    for eta in (0.1, 0.3, 1.0):
        for harm in (0.8, 0.3):
            r = pmap(b_job, [(s, eta, harm) for s in SEEDS])
            res["B_eta_ov"][f"eta_ov={eta} harm={harm}"] = {k: float(np.mean([x[k] for x in r])) for k in ("peak", "residual", "fraction", "after_first", "reaffirm", "overwhelm")}
    dump("diag_persist", res)
    for k, v in res["A_Z4_by_mode"].items():
        print("Z4", k, "within5", v["within_5"], "never", v["never_within_10"], "J_pref %.3f" % v["J_pref_mean"])
    for k, v in res["B_eta_ov"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
