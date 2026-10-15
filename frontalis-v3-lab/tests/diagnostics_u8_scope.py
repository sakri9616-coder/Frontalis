"""診断（テストではない）: U8 の持続が低い原因は、信念の「人の部分」か「文脈の部分」か。

未見の共同作業者 N' の P(credit_taken) = π̄_person p_ind + π̄_ctx p_ctx^person を、人の部分と文脈の部分に分け、
P2 − P1 の差が N との経験（m = 8, 32）の後にどれだけ残るか（後の差 / 前の差）を、種 0〜9 の平均で見る。
music_session は N と搾取が共有する文脈、cafe_chat はどちらの履歴にも出ない文脈。
実行: python3 tests/diagnostics_u8_scope.py（結果は tests/results/diag_u8_scope.json）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from helpers import RESULTS, feed, make, mh  # noqa: E402
from npnv3.innate import TYPE_INDEX  # noqa: E402

K = TYPE_INDEX["credit_taken"]


def run(seed, **kw):
    p1_h, p2_h, n_h = mh.generate_t4(seed)
    p1, p2 = make(seed, **kw), make(seed, **kw)
    feed(p1, p1_h); feed(p2, p2_h)
    out, done = {}, 0
    for m in (0, 8, 32):
        feed(p1, n_h[done:m]); feed(p2, n_h[done:m]); done = m
        r = {}
        for ctx in ("music_session", "cafe_chat"):
            a = p1.belief.predict(ctx, [{"id": "zz", "class": "collaborator"}])
            b = p2.belief.predict(ctx, [{"id": "zz", "class": "collaborator"}])
            r[ctx] = {"P": b.p_person[0, K] - a.p_person[0, K],
                      "person_part": b.pi_p[K] * b.p_ind[0, K] - a.pi_p[K] * a.p_ind[0, K],
                      "ctx_part": b.pi_c[K] * b.p_ctxp[0, K] - a.pi_c[K] * a.p_ctxp[0, K]}
        out[m] = r
    return out


def main():
    res = {}
    for name, kw in (("default", {}), ("no_scope", {"no_scope": True})):
        rows = [run(s, **kw) for s in range(10)]
        res[name] = {}
        for ctx in ("music_session", "cafe_chat"):
            res[name][ctx] = {}
            for comp in ("P", "person_part", "ctx_part"):
                v = [float(np.mean([r[m][ctx][comp] for r in rows])) for m in (0, 8, 32)]
                res[name][ctx][comp] = {"diff_m0": v[0], "diff_m8": v[1], "diff_m32": v[2],
                                        "persistence_m8": v[1] / v[0] if v[0] else None,
                                        "persistence_m32": v[2] / v[0] if v[0] else None}
                print(f"{name:8s} {ctx:14s} {comp:12s} 差 {v[0]:+.3f} → {v[1]:+.3f} → {v[2]:+.3f}"
                      + (f"  持続 {v[1] / v[0]:.2f} / {v[2] / v[0]:.2f}" if v[0] else ""))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "diag_u8_scope.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
