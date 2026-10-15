"""診断（テストではない）: U8（全体の評価 Q の持続）が不合格になった原因の切り分け。

実行: python3 tests/diagnostics_u8.py   結果は tests/results/diag_u8.json。
未見の共同作業者 N' への Q の P2 − P1 の差を、MF の部分と MB の部分に分け、調停の重み ω と一緒に見る。
設定を変えた変種（固定 κ_e、スコープ帰属なし、関心の重みなし）は、原因の切り分けのためだけで、既定は変えていない。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from helpers import RESULTS, SEEDS, feed, make, mh, pmap  # noqa: E402

VARIANTS = {"default": {}, "estimate_kappa=False": {"estimate_kappa": False}, "no_scope": {"no_scope": True},
            "no_concern": {"no_concern": True}}


def worker(seed):
    p1_h, p2_h, n_h = mh.generate_t4(seed)
    sit = {"context": "music_session", "concepts": ["collaboration", "music"],
           "participants": [{"id": f"diag-np-{seed}", "class": "collaborator"}]}
    out = {}
    for name, kw in VARIANTS.items():
        p1, p2 = make(seed, **kw), make(seed, **kw)
        feed(p1, p1_h); feed(p2, p2_h)
        rows = {}
        done = 0
        for m in (0, 8, 32):
            feed(p1, n_h[done:m]); feed(p2, n_h[done:m]); done = m
            a, b = p1.probe(sit), p2.probe(sit)
            rows[m] = {"dQ": b["Q"] - a["Q"], "dMF": b["V_MF"] - a["V_MF"], "dMB": b["V_MB"] - a["V_MB"],
                       "omega1": a["omega"], "omega2": b["omega"]}
        rows["kappa_P1"] = p1.belief.kappa_of("collaborator")
        out[name] = rows
    return out


def main():
    rows = pmap(worker, SEEDS)
    res = {}
    for name in VARIANTS:
        res[name] = {}
        for m in (0, 8, 32):
            res[name][str(m)] = {k: float(np.mean([r[name][m][k] for r in rows])) for k in rows[0][name][m]}
        d0 = res[name]["0"]
        for m in (8, 32):
            res[name][str(m)]["persistence_Q"] = res[name][str(m)]["dQ"] / d0["dQ"]
            res[name][str(m)]["persistence_MB_part"] = res[name][str(m)]["dMB"] / d0["dMB"]
            res[name][str(m)]["persistence_MF_part"] = res[name][str(m)]["dMF"] / d0["dMF"]
        res[name]["kappa_P1_final_mean"] = float(np.mean([r[name]["kappa_P1"] for r in rows]))
        print(f"{name:22s} Q の持続 m=8: {res[name]['8']['persistence_Q']:.3f}  m=32: {res[name]['32']['persistence_Q']:.3f} | "
              f"MB 部分 {res[name]['8']['persistence_MB_part']:.3f} / {res[name]['32']['persistence_MB_part']:.3f} | "
              f"MF 部分 {res[name]['8']['persistence_MF_part']:.3f} / {res[name]['32']['persistence_MF_part']:.3f} | "
              f"ω(P1) {res[name]['0']['omega1']:.2f}→{res[name]['32']['omega1']:.2f} | κ_e(P1) {res[name]['kappa_P1_final_mean']:.2f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "diag_u8.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
