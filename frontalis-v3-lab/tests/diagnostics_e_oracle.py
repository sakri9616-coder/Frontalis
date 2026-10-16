"""診断（テストではない）: E3〜E6・E9 の不合格が、学習（事後の分離）の問題か、SPEC-v3.3 4.2 の事前・応答の式そのものの構造か、を切り分ける。
判定には使わない。結果は tests/results/diag_e_oracle.json。

背景: E 系を SPEC どおりの式と係数で走らせると、E3・E4・E5・E6・E9 が不合格になった（tests/results/E01_E09.json）。tests/diagnostics_e.py の変種（σ_r、冪乗則、
復位の変種）で一部は動くが、全部は動かない。ここでは「消去の学習が古い原因を壊さない」理想の状態（oracle）を手で置き、式だけの寄与を調べる。

  A. oracle の状態: 獲得（文脈 studio で 12 試行 = 24 経験）の後、古い原因（価値 C_a = −0.204、計数 11.3、最後に使われた時刻 22）が消去の学習で壊れずに残り、
     消去の原因（価値 0、計数 20、文脈 cafe_chat、最後に使われた時刻 62）が別にできた状態。実測の E 系（seed 0）の獲得直後の値を使った。
     この状態に SPEC の事前・応答の式（α = 0.5, τ = 30, β = 1）を当てて、E3・E4・E5・E6 に当たる比を出す。
  B. 時間の単調性: 学習がない間、応答の絶対値は時間とともに増えない（すべての原因の重みが同じ exp(−t/τ) で縮み、新しい原因 α の質量だけが一定だから）。
     ランダムな状態で確かめ、E5（自発的回復）が SPEC の式では原理的に起きないことを示す。
  C. 未見の文脈での原因の重み N_k · (0 + β)/(N_k + β S): 計数 N_k が大きいと β に飽和し、消去の文脈の数（S）に依らない。E9（3 つの文脈での消去の般化）が式から出ない理由。
実行: python3 tests/diagnostics_e_oracle.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import dump  # noqa: E402
from npnv3 import Config  # noqa: E402
from npnv3.value import Conditioning  # noqa: E402

A, B, C = "studio", "cafe_chat", "garden"


def place(cd, cue, causes, ctxs):
    """causes: (価値, {文脈: 計数}, 最後に使われた時刻) のリスト。"""
    i = cd._register(cue)
    for c in ctxs:
        cd._ctx_col(c, True)
    for k, (v, cnt, t) in enumerate(causes):
        cd.CK[i, k] = v
        cd.AL[i, k] = 1.0
        cd.TK[i, k] = t
        for c, n in cnt.items():
            cd.CNT[i, k, cd.ctx_idx[c]] = n
    return i


def oracle_state(ext_counts):
    cd = Conditioning(Config())
    ctxs = [A] + list(ext_counts)
    place(cd, "x", [(-0.204, {A: 11.3}, 22.0), (0.0, dict(ext_counts), 62.0)], ctxs)
    return cd


def parta():
    """A. oracle の状態での E3〜E6・E9 に当たる比（獲得直後の studio の応答 −0.195 を 1 とする）。"""
    base = Conditioning(Config())
    place(base, "x", [(-0.204, {A: 11.3}, 22.0)], [A])
    r_acq = base.value_of("x", A, 24.0)
    out = {"response_after_acquisition_A": r_acq}
    cd = oracle_state({B: 20.0})
    t = 64.0
    out["E2_B"] = cd.value_of("x", B, t) / r_acq
    out["E3_ABA"] = cd.value_of("x", A, t) / r_acq
    out["E4_ABC"] = cd.value_of("x", C, t) / r_acq
    out["E5_B_after_50_unrelated"] = cd.value_of("x", B, t + 50) / r_acq
    out["E5_increase"] = out["E5_B_after_50_unrelated"] - out["E2_B"]
    # E6: 復位（負の価値の原因の時刻を現在 t + 2 に更新）の後の B と、同じ時間だけ経った対照
    ctl = cd.value_of("x", B, t + 2) / r_acq
    cd.reinstate([], t + 2)
    out["E6_reinstated"] = cd.value_of("x", B, t + 2) / r_acq
    out["E6_control"] = ctl
    out["E6_difference"] = out["E6_reinstated"] - ctl
    # E9: 消去を 3 つの文脈（7 + 7 + 6 試行）で行った状態の ABC
    c3 = oracle_state({"cafe_chat": 7.0, "workshop": 7.0, "travel": 6.0})
    out["E9_ABC_one_context"] = out["E4_ABC"]
    out["E9_ABC_three_contexts"] = c3.value_of("x", C, t) / r_acq
    out["E9_difference"] = out["E9_ABC_one_context"] - out["E9_ABC_three_contexts"]
    # 時間の減衰（τ = 30）を外した（τ → 大）変種: 式のどの部分が効いているか
    for tau in (30.0, 300.0, 3000.0):
        cdt = Conditioning(Config(lc_tau=tau))
        place(cdt, "x", [(-0.204, {A: 11.3}, 22.0), (0.0, {B: 20.0}, 62.0)], [A, B])
        bt = Conditioning(Config(lc_tau=tau))
        place(bt, "x", [(-0.204, {A: 11.3}, 22.0)], [A])
        ra = bt.value_of("x", A, 24.0)
        out[f"tau_{int(tau)}"] = {"E3_ABA": cdt.value_of("x", A, 64.0) / ra, "E4_ABC": cdt.value_of("x", C, 64.0) / ra,
                                  "E5_increase": (cdt.value_of("x", B, 114.0) - cdt.value_of("x", B, 64.0)) / ra}
    out["thresholds"] = {"E3": ">= 0.5", "E4": ">= 0.4", "E5_increase": ">= 0.15", "E6_difference": ">= 0.15", "E9_difference": ">= 0.1"}
    return out


def partb(trials=4000):
    """B. 学習のない間の応答の絶対値は時間とともに増えない（価値がすべて ≤ 0 のとき。符号を反転すれば ≥ 0 でも同じ）。最大の増加量を探す。"""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(trials):
        K = int(rng.integers(1, 6))
        cd = Conditioning(Config(lc_max_causes=5))
        vals = -rng.uniform(0.0, 0.6, size=K) * (rng.random(K) < 0.8)
        causes = [(float(v), {A: float(rng.uniform(0, 20)), B: float(rng.uniform(0, 20))}, float(rng.uniform(0, 60))) for v in vals]
        place(cd, "x", causes, [A, B])
        t0 = 60.0
        ctx = [A, B, C][int(rng.integers(3))]
        prev = abs(cd.value_of("x", ctx, t0))
        for dt in (1, 5, 20, 50, 200):
            cur = abs(cd.value_of("x", ctx, t0 + dt))
            worst = max(worst, cur - prev)
            prev = cur
    return {"trials": trials, "largest_increase_of_abs_response_over_time": worst}


def partc():
    """C. 未見の文脈での原因の重み N · β/(N + β S)（β = 1）。"""
    out = {}
    for N in (5, 10, 20, 50, 200):
        out[f"N={N}"] = {f"S={S}": N * 1.0 / (N + 1.0 * S) for S in (2, 3, 5)}
    return out


def main():
    res = {"A_oracle_state": parta(), "B_time_monotonicity": partb(), "C_unseen_context_weight": partc()}
    a = res["A_oracle_state"]
    print("A. oracle の状態（古い原因が壊れていない）での比")
    for k in ("E2_B", "E3_ABA", "E4_ABC", "E5_increase", "E6_difference", "E9_ABC_one_context", "E9_ABC_three_contexts", "E9_difference"):
        print(f"   {k:24s} {a[k]:+.3f}")
    print("   しきい値:", a["thresholds"])
    for tau in (30, 300, 3000):
        print(f"   τ = {tau:5d}: ", {k: round(v, 3) for k, v in a[f'tau_{tau}'].items()})
    print("B. 学習のない間の |応答| の最大の増加:", res["B_time_monotonicity"]["largest_increase_of_abs_response_over_time"])
    print("C. 未見の文脈での重み N·β/(N + βS):", {k: {s: round(v, 3) for s, v in d.items()} for k, d in res["C_unseen_context_weight"].items()})
    dump("diag_e_oracle", res)


if __name__ == "__main__":
    main()
