"""診断（テストではない）: R/K/X の経験史で、事前ヘッド（mlp）は評価と帰属を実際に動かしているか。

v3.0 では R/K/X の履歴でヘッドが 50 件の解禁に届かず、no_head との差は厳密に 0 だった（批評 S1）。
v3.1 は ρ_h = n/(n+30) で混ぜる。同じ種・同じ経験史で head_kind = "mlp" と "none" を比べ、新しい経験への |ΔQ|、|ΔV_MB|、
責任 φ_person の最大の違いを 20 本で平均する。実行: python3 tests/diagnostics_head.py（結果は tests/results/diag_head.json）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from helpers import KINDS, RESULTS, SEEDS, load_json, persona_gen, pmap  # noqa: E402

NEW = load_json("new_event.json")


def worker(seed):
    out = {}
    for k in KINDS:
        a, b = persona_gen(k, seed), persona_gen(k, seed, head_kind="none")
        pa, pb = a.appraise(NEW), b.appraise(NEW)
        out[k] = {"dQ": abs(pa["Q"] - pb["Q"]), "dMB": abs(pa["V_MB"] - pb["V_MB"]),
                  "dphi": max(abs(x["phi_person"] - y["phi_person"]) for x, y in zip(pa["phi"], pb["phi"])),
                  "rho": a.belief.rho()}
    return out


def main():
    rows = pmap(worker, SEEDS)
    res = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in ("dQ", "dMB", "dphi", "rho")} for k in KINDS}
    for k, v in res.items():
        print(f"{k}: ρ_h {v['rho']:.2f}  |ΔQ| {v['dQ']:.4f}  |ΔV_MB| {v['dMB']:.4f}  max|Δφ_person| {v['dphi']:.3f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "diag_head.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
