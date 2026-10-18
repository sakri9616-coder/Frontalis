"""U8 全体の評価の持続: T4 の課題を、価値の like ではなく、全体の予期 Q で測る。

SPEC-v3.1 第 3 節: m = 8 と 32 で、Q の持続が 0.7 以上、可塑性が 0.7 以上。

測り方（実装前に固定した細部）:
  T4 の課題の確率的な版（generate_t4(種)）: P1（8 人の共同作業者に 1 回ずつ搾取された）と P2（普通に協働）に、新しい共同作業者 N との
  良い経験を 8 回、続けて計 32 回（大きさ・文脈は種ごとに抽出）。評価 = 状況プローブの Q（music_session、概念 collaboration・music、
  参加者 1 人。MB と MF を調停した全体の予期）。
  持続   = 未見の共同作業者 N' への Q の P2 − P1 の差が、N との経験の後にどれだけ残るか（後の差 / 前の差）。
  可塑性 = P1 の N への Q が真値に近づいた割合（(後 − 前) / (真値 − 前)）。真値 = 経験した N の 1 回分の、P1 の現在の h・関心の重み a での
           a で重み付けた効用変化の総和（appraise の du_weighted。N との最後の経験と同じ結果、新しい id）。
  20 本（種ごとに別の P1・P2・N）の平均。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, feed, make, mh, pmap


def worker(seed):
    p1_h, p2_h, n_h = mh.generate_t4(seed)
    nprime = f"u8-nprime-{seed}"
    nid = f"t4n-{seed}"

    def sit(pid):
        return {"context": "music_session", "concepts": ["collaboration", "music"],
                "participants": [{"id": pid, "class": "collaborator"}]}
    p1, p2 = make(seed), make(seed)
    feed(p1, p1_h); feed(p2, p2_h)
    d0 = p2.probe(sit(nprime))["Q"] - p1.probe(sit(nprime))["Q"]
    q_n0 = p1.probe(sit(nid))["Q"]
    out = {"seed": seed, "d0": d0, "q_n0": q_n0}
    done = 0
    for m in (8, 32):
        feed(p1, n_h[done:m]); feed(p2, n_h[done:m]); done = m
        dm = p2.probe(sit(nprime))["Q"] - p1.probe(sit(nprime))["Q"]
        last = dict(n_h[m - 1]); last["id"] = f"truth-{seed}-{m}"
        truth = p1.appraise(last)["du_weighted"]
        q_n = p1.probe(sit(nid))["Q"]
        out[m] = {"persistence": dm / d0 if abs(d0) > 1e-12 else 0.0,
                  "plasticity": (q_n - q_n0) / (truth - q_n0) if abs(truth - q_n0) > 1e-12 else 0.0,
                  "dm": dm, "q_n": q_n, "truth": truth}
    return out


class U8TotalEvaluation(unittest.TestCase):
    def test_total_evaluation_persists_and_stays_plastic(self):
        rows = pmap(worker, SEEDS)
        res = {}
        for m in (8, 32):
            res[f"m{m}"] = {k: float(np.mean([r[m][k] for r in rows])) for k in ("persistence", "plasticity", "dm", "q_n", "truth")}
            res[f"m{m}"]["persistence_median"] = float(np.median([r[m]["persistence"] for r in rows]))
            res[f"m{m}"]["plasticity_median"] = float(np.median([r[m]["plasticity"] for r in rows]))
        res["d0_mean"] = float(np.mean([r["d0"] for r in rows]))
        res["q_N_before_mean"] = float(np.mean([r["q_n0"] for r in rows]))
        dump("U08", res)
        for m in (8, 32):
            self.assertGreaterEqual(res[f"m{m}"]["persistence"], 0.7, res)
            self.assertGreaterEqual(res[f"m{m}"]["plasticity"], 0.7, res)


if __name__ == "__main__":
    unittest.main()
