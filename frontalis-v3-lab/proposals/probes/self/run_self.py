"""Run v13 (DESIGN-self v3.1).

usage:
  python run_self.py --smoke
  python run_self.py --dev A B                 # learning sanity only; no B quantity
  python run_self.py --seeds A B --out res.json
"""
import argparse
import json
import math
import sys
import time

import numpy as np

import sim_self as S

sys.path.insert(0, "../credit")
from run_g import binom_two_sided, two_prop_p  # noqa: E402


def follow(f, a, b):
    return float(f[a:b].mean())


def per_seed(s):
    out = {}
    for form in (0, 1):
        for g in (0.0, 2.0, 4.0, 8.0):
            for tau in (1.0, 3.0):
                c, f, T, _ = S.life(form, "Q", s, gamma=g, tau=tau)
                out[f"Q_f{form}_g{g:g}_t{tau:g}"] = {"late": follow(f, -1000, None), "first": follow(f, 0, 500),
                                                     "T": float(T[-1]), "top": int(np.argmax(c)), "c": c.round(3).tolist()}
        for mech in ("mirror", "notrust"):
            c, f, T, _ = S.life(form, "Q", s, mech=mech)
            out[f"Q_f{form}_{mech}"] = {"late": follow(f, -1000, None), "top": int(np.argmax(c))}
        for mech in ("self", "notrust"):
            c, f, T, _ = S.life(form, "Q", s, mech=mech, switch_to="N")
            out[f"QthenN_f{form}_{mech}"] = {"first500_of_N": follow(f, 4000, 4500), "last": follow(f, -1000, None)}
            c, f, T, _ = S.life(form, "N", s, mech=mech)                  # no prior relation: N from the start
            out[f"Nonly_f{form}_{mech}"] = {"first500_of_N": follow(f, 0, 500), "last": follow(f, -1000, None)}
    for harm in (False, True):
        c, f, T, _ = S.life(0, "W", s, harm=harm)
        out[f"W_h{int(harm)}"] = {"late": follow(f, -1000, None), "c_overwork": float(c[2])}
    c, f, T, _ = S.life(0, "C", s, betray_at=3000)
    out["betray"] = {"late": follow(f, -1000, None), "T": float(T[-1])}
    for g in (0.0, 4.0):
        c, f, T, _ = S.life(0, "Q", s, gamma=g, coop_bad_at=1000)
        tops = S.life.tops
        first = next((i for i, x in enumerate(tops) if x != 0), len(tops))
        out[f"reason_g{g:g}"] = {"late": follow(f, -1000, None), "top": int(np.argmax(c)), "change_time": first * 10}
    return out


def judge(rows):
    n = len(rows)
    g = lambda k, f: np.array([r[k][f] for r in rows])
    inter = ((g("Q_f1_g4_t3", "late") - g("Q_f1_g0_t3", "late"))
             - (g("Q_f0_g4_t3", "late") - g("Q_f0_g0_t3", "late")))
    b1_pos = int((inter > 0).sum()); b1_neg = int((inter < 0).sum())
    b2 = int(((g("Q_f0_g4_t3", "T") < 0) & (g("Q_f1_g4_t3", "T") > 0)).sum())
    b3d = g("QthenN_f1_self", "first500_of_N") - g("Nonly_f1_self", "first500_of_N")
    b3c = g("QthenN_f1_notrust", "first500_of_N") - g("Nonly_f1_notrust", "first500_of_N")
    ch_reason = int((g("reason_g4", "top") != 0).sum())
    ch_none = int((g("Q_f0_g4_t3", "top") != 0).sum())
    res = {
        "B1_interaction_pos": b1_pos, "B1_interaction_neg": b1_neg, "B1_interaction_mean": float(inter.mean()),
        "B1_pass": bool(b1_pos >= 30 and binom_two_sided(b1_pos, b1_pos + b1_neg) < 0.01),
        "B2_count": b2, "B2_pass": bool(b2 >= 36),
        "B3_count": int((b3d >= 0.2).sum()), "B3_ctrl_count": int((b3c >= 0.2).sum()),
        "B3_pass": bool((b3d >= 0.2).sum() >= 30 and binom_two_sided(int((b3c > 0).sum()), int((b3c != 0).sum()) or 1) >= 0.01),
        "B6_top_changed_with_reason": ch_reason, "B6_top_changed_without_reason": ch_none,
        "B6_p": two_prop_p(ch_reason, n, ch_none, n),
    }
    res["B6_pass"] = bool(ch_reason / n - ch_none / n >= 0.5 and res["B6_p"] < 0.01)
    # B6b (partner's contribution to the o-driven change): the top changes earlier with the partner (g4) than without (g0)
    tg4, tg0 = g("reason_g4", "change_time"), g("reason_g0", "change_time")
    e_pos, e_neg = int((tg4 < tg0).sum()), int((tg4 > tg0).sum())
    res["B6b_earlier"] = e_pos; res["B6b_later"] = e_neg
    res["B6b_pass"] = bool(e_pos >= 30 and binom_two_sided(e_pos, e_pos + e_neg) < 0.01)
    desc = {}
    for k in rows[0]:
        for f in rows[0][k]:
            v = [r[k][f] for r in rows]
            if isinstance(v[0], (int, float)):
                desc[f"{k}.{f}"] = round(float(np.mean(v)), 3)
    res["describe_means"] = desc
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev", nargs=2, type=int)
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.smoke:
        c, f, T, ch = S.life(0, "Q", 1)
        assert np.isfinite(c).all() and np.abs(T).max() <= S.T_CLIP and len(f) == S.N_DYAD
        print("smoke ok")
    elif a.dev:
        t0, cm = time.time(), []
        for s in range(*a.dev):
            for form in (0, 1):
                c, f, T, _ = S.life(form, "Q", s)
                cm.append(np.abs(c).max())
                assert np.isfinite(c).all()
        print(f"agents={len(cm)} max|c|={max(cm):.2f} finite=True time={time.time() - t0:.0f}s")
    else:
        rows = [per_seed(s) for s in range(*a.seeds)]
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "judgment": judge(rows), "rows": rows}, fh, indent=1)
        print("written", a.out)
