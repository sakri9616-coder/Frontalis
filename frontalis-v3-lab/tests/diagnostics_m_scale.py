"""診断（テストではない）: 効果表 M の大きさ gM に対する T2 の指標の感度。

実行: python3 tests/diagnostics_m_scale.py
T2(c)（fixed_teacher の cos ≥ 0.95）が不合格になった原因の切り分けに使った。結果は tests/results/diag_m_scale.json。
M の大きさを変えるのは診断だけで、実装の既定（gM = 0.3）は変えていない。
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from helpers import RESULTS, cos, flat_update, load_json, persona, update_keys  # noqa: E402
from npnv3 import innate  # noqa: E402

BASE = np.array(innate.M)


def run(scale, **kw):
    innate.M = BASE * (scale / 0.3)
    new = load_json("new_event.json")
    aps, ents = {}, []
    for name in "RKX":
        n = persona(name, 0, **kw)
        aps[name] = n.experience(new)
        ents.append(n.ledger.entries[-1])
    keys = update_keys(ents)
    vec = {nm: flat_update(e, keys) for nm, e in zip("RKX", ents)}
    cs = {f"{a}-{b}": cos(vec[a], vec[b]) for a, b in itertools.combinations("RKX", 2)}
    f = [aps[n]["f"] for n in "RKX"]
    return {"cos": cs, "cos_min": min(cs.values()), "f_max_diff": max(f) - min(f),
            "h_after_history": None}


def main():
    out = {}
    for sc in (0.3, 0.2, 0.15, 0.1):
        out[str(sc)] = {"kalman": run(sc), "fixed_teacher": run(sc, fixed_teacher=True)}
        print(f"gM={sc}: (a) f 最大差 {out[str(sc)]['kalman']['f_max_diff']:.3f} | (b) cos 最小 "
              f"{out[str(sc)]['kalman']['cos_min']:.3f} | (c) fixed cos 最小 {out[str(sc)]['fixed_teacher']['cos_min']:.3f}")
    innate.M = BASE
    h = {}
    for name in "RKX":
        h[name] = [float(x) for x in persona(name, 0).h]
    out["h_after_history_gM0.3"] = h
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "diag_m_scale.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("h（履歴の後, gM=0.3）:", {k: [round(x, 2) for x in v] for k, v in h.items()})


if __name__ == "__main__":
    main()
