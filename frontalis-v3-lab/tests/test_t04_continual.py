"""T4 継続学習（持続と可塑性の両立）。
P1（8 人の共同作業者に 1 回ずつ搾取された）と P2（普通に協働）に、新しい共同作業者 N との良い経験を m = 8, 32 回。
  評価 = エンティティプローブの「好悪 like」（価値ネット MF の人モデルの予測。docs/04 第 3.3 節の Like(x)）。
  持続   = 未見の共同作業者 N' への P1 と P2 の評価差が、N との経験の後にどれだけ残るか（後の差 / 前の差）。
           前の差が 0（flat_value では未見の個体に何も引き継がれない）なら、持続は 0 と定義する。
  可塑性 = P1 の N への評価が真値にどれだけ近づいたか（(後 - 前) / (真値 - 前)）
           真値 = その時点の h で、N の結果（connection, support）に帰属する効用変化の総和
合格条件（SPEC 第 8 節）: 持続 ≥ 0.7 かつ 可塑性 ≥ 0.7（m = 8, 32）。flat_value では 持続 < 0.5。
数値は種 0〜19 の平均。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, feed, load_jsonl, make
from npnv3 import innate

NPRIME = {"entity": "T4_Nprime", "class": "collaborator"}
N_PROBE = {"entity": "T4_N", "class": "collaborator"}
N_OUTCOMES = [("connection", 1.0), ("support", 0.7), ("achievement", 0.6)]   # T4_N.jsonl の中身（achievement は自分）


def true_value_of_N(h):
    """N の結果に帰属する効用変化の総和（φ = 1 の会計）。T4_N.jsonl の 1 経験を h で評価する。"""
    types = [t for t, _ in N_OUTCOMES]
    rows = innate.outcome_rows(types, [m for _, m in N_OUTCOMES])
    _, _, contrib = innate.contributions(h, rows)
    return float(contrib[:2].sum())          # connection, support（actor = N）


def run(seed, m, **kw):
    p1, p2 = make(seed, **kw), make(seed, **kw)
    feed(p1, load_jsonl("T4_P1.jsonl")); feed(p2, load_jsonl("T4_P2.jsonl"))
    d0 = p2.probe(NPRIME)["like"] - p1.probe(NPRIME)["like"]
    n0 = p1.probe(N_PROBE)["like"]
    q0 = p2.probe({**NPRIME, "context": "music_session"})["Q"] - p1.probe({**NPRIME, "context": "music_session"})["Q"]
    exps = load_jsonl("T4_N.jsonl")[:m]
    feed(p1, exps); feed(p2, exps)
    dm = p2.probe(NPRIME)["like"] - p1.probe(NPRIME)["like"]
    qm = p2.probe({**NPRIME, "context": "music_session"})["Q"] - p1.probe({**NPRIME, "context": "music_session"})["Q"]
    n_after = p1.probe(N_PROBE)["like"]
    truth = true_value_of_N(p1.h)
    return {"seed": seed, "m": m, "d0": d0, "dm": dm,
            "persistence": dm / d0 if abs(d0) > 1e-12 else 0.0,     # 引き継ぐ差が最初からなければ、持続は 0
            "plasticity": (n_after - n0) / (truth - n0) if truth != n0 else float("nan"),
            "persistence_Q": qm / q0 if q0 != 0 else float("nan"),
            "N_before": n0, "N_after": n_after, "truth": truth}


class T4Continual(unittest.TestCase):
    def test_persistence_and_plasticity(self):
        out = {}
        for m in (8, 32):
            rows = [run(s, m) for s in SEEDS]
            out[f"m{m}"] = {k: float(np.mean([r[k] for r in rows])) for k in
                            ("d0", "dm", "persistence", "plasticity", "persistence_Q", "N_before", "N_after", "truth")}
        flat = {}
        for m in (8, 32):
            rows = [run(s, m, flat_value=True) for s in SEEDS]
            flat[f"m{m}"] = {k: float(np.mean([r[k] for r in rows])) for k in ("d0", "dm", "persistence", "plasticity")}
        dump("T04", {"hierarchical": out, "flat_value": flat})
        for m in (8, 32):
            self.assertGreaterEqual(out[f"m{m}"]["persistence"], 0.7, out)
            self.assertGreaterEqual(out[f"m{m}"]["plasticity"], 0.7, out)
            self.assertLess(flat[f"m{m}"]["persistence"], 0.5, flat)


if __name__ == "__main__":
    unittest.main()
