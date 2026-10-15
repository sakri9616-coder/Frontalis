"""診断（判定ではない）: v3.2 の 3 つの新しい切り替えを外すと v3.1（改訂 2 の後）の数値に戻るか、1 つずつ外すとどう動くか、habituate では何が起きるか。

実行: python3 tests/diagnostics_v32_ablations.py   （約 30 秒。結果は tests/results/diag_v32_ablations.json）
指標: U1(a)（h・m を揃えた f の人格間の最大差、20 本の平均）、U5（申し出の接近率、K 型・R 型、20 本の平均）。

  v3.1 相当 = no_precision_weighting（R_k = R）+ no_cq（C を Q に入れない）+ head_kind = "mlp"（単一ヘッド）
  これが改訂 2 の後の v3.1 の値（U1(a) 0.2360、U5 K 0.3100 / R 0.8025）と一致する = v3.2 の追加は、この 3 つの切り替えで完全に外せる。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from helpers import KINDS, SEEDS, align_shell, dump, load_json, mh, persona_gen, pmap  # noqa: E402

NEW = load_json("new_event.json")
CFGS = {
    "v3.2 既定": {},
    "v3.1 相当（no_precision_weighting, no_cq, head_kind=mlp）": {"no_precision_weighting": True, "no_cq": True, "head_kind": "mlp"},
    "no_cq だけ": {"no_cq": True},
    "no_precision_weighting だけ": {"no_precision_weighting": True},
    "head_kind=mlp だけ": {"head_kind": "mlp"},
    "habituate（関心の重みの方向）": {"concern_direction": "habituate"},
}


def work(args):
    name, seed = args
    kw = CFGS[name]
    f = {}
    for k in KINDS:
        f[k] = align_shell(persona_gen(k, seed, **kw)).appraise(NEW)["f"]
    rates = {}
    for kind in ("K", "R"):
        n = persona_gen(kind, seed, **kw)
        rates[kind] = float(np.mean([bool(n.experience(e)["approached"]) for e in mh.generate_offers(kind, seed, 40)]))
    return {"name": name, "seed": seed, "gap": max(f.values()) - min(f.values()), "rK": rates["K"], "rR": rates["R"]}


if __name__ == "__main__":
    rows = pmap(work, [(nm, s) for nm in CFGS for s in SEEDS])
    out = {}
    for nm in CFGS:
        rr = [r for r in rows if r["name"] == nm]
        out[nm] = {"U1a_gap": float(np.mean([r["gap"] for r in rr])), "U5_K": float(np.mean([r["rK"] for r in rr])),
                   "U5_R": float(np.mean([r["rR"] for r in rr])), "U5_diff": float(np.mean([r["rR"] - r["rK"] for r in rr]))}
        print(f'{nm:<58} U1(a) {out[nm]["U1a_gap"]:.4f}   U5 K {out[nm]["U5_K"]:.4f}  R {out[nm]["U5_R"]:.4f}  差 {out[nm]["U5_diff"]:.4f}')
    dump("diag_v32_ablations", out)
