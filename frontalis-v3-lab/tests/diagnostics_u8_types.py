"""診断（テストではない）: U8 の Q の持続が低い理由を、結果の型ごとの信念と、価値への寄与に分ける。

未見の共同作業者 N' の型 k の確率 P_k（music_session）の P2 − P1 の差と、その価値への寄与 P_k · g_k（g_k = 型 k が 1 回起きたときの、
その人格の h と関心の重み a での効用変化）の差を、N との経験 m = 0, 8, 32 で見る（種 0〜9 の平均）。
信念の差が残っていても、寄与の差が縮むなら、状態（欲求の状態 h・関心の重み a）の結合が原因。
実行: python3 tests/diagnostics_u8_types.py（結果は tests/results/diag_u8_types.json）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from helpers import RESULTS, feed, make, mh  # noqa: E402
import npnv3.value as V  # noqa: E402
from npnv3.innate import TYPE_INDEX  # noqa: E402

TYPES = ["connection", "credit_taken", "betrayal", "support", "achievement"]


def main():
    acc = {}
    for seed in range(10):
        p1_h, p2_h, n_h = mh.generate_t4(seed)
        p1, p2 = make(seed), make(seed)
        feed(p1, p1_h); feed(p2, p2_h)
        done = 0
        for m in (0, 8, 32):
            feed(p1, n_h[done:m]); feed(p2, n_h[done:m]); done = m
            a = p1.belief.predict("music_session", [{"id": "zz", "class": "collaborator"}])
            b = p2.belief.predict("music_session", [{"id": "zz", "class": "collaborator"}])
            g1 = V.type_effect(p1.h, p1.belief.mag, p1.M, p1.a)
            g2 = V.type_effect(p2.h, p2.belief.mag, p2.M, p2.a)
            acc.setdefault(m, []).append({t: (b.p_person[0, TYPE_INDEX[t]] - a.p_person[0, TYPE_INDEX[t]],
                                              b.p_person[0, TYPE_INDEX[t]] * g2[TYPE_INDEX[t]] - a.p_person[0, TYPE_INDEX[t]] * g1[TYPE_INDEX[t]])
                                          for t in TYPES})
    res = {}
    for t in TYPES:
        dp = [float(np.mean([r[t][0] for r in acc[m]])) for m in (0, 8, 32)]
        dv = [float(np.mean([r[t][1] for r in acc[m]])) for m in (0, 8, 32)]
        res[t] = {"dP_m0_m8_m32": dp, "contribution_m0_m8_m32": dv}
        print(f"{t:13s} P2−P1: {dp[0]:+.3f} {dp[1]:+.3f} {dp[2]:+.3f} | 価値への寄与 P·g の差: {dv[0]:+.3f} {dv[1]:+.3f} {dv[2]:+.3f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "diag_u8_types.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
