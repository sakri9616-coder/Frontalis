"""W3 型の間と型の内: 人格の差は、型（経験史の世界）の差が、同じ型の中の個体差より大きいか。

SPEC-v3.2 第 2 節: U1 の手順で、同じ型の 3 人格（別の種）の f の最大差（型の内）と、異なる型の 3 人格の最大差（型の間）を比べる。
合格: 型の間の平均 ≥ 2 × 型の内の平均。

測り方（実装前に固定した細部）:
  U1 の手順 = 型 R/K/X の経験史（generate_history、36 件）で育て、h と m を基準値に揃え、new_event.json を appraise した f。
  3 型 × 乱数種 0〜59 = 180 人格を作る（NPN の種も履歴の種も同じ s）。
  型の内: 各型で、種 (3t, 3t+1, 3t+2)（t = 0〜19）の 3 人格の f の最大 − 最小。3 型 × 20 組 = 60 組の平均。
  型の間: t = 0〜19 と 3 通りの回転（R, K, X に種 (3t, 3t+1, 3t+2) を割り当て、さらに 1 つずつずらす）で、R・K・X 各 1 人の f の最大 − 最小。60 組の平均。
  各人格はどの組でも同じ f を使う（作り直さない）。
"""
import unittest

import numpy as np

from helpers import KINDS, align_shell, dump, load_json, persona_gen, pmap

NEW = load_json("new_event.json")
N_SEEDS = 60


def f_of(args):
    kind, s = args
    return align_shell(persona_gen(kind, s)).appraise(NEW)["f"]


class W3WithinBetween(unittest.TestCase):
    def test_between_type_gap_is_at_least_twice_within_type_gap(self):
        jobs = [(k, s) for k in KINDS for s in range(N_SEEDS)]
        vals = pmap(f_of, jobs)
        f = {(k, s): v for (k, s), v in zip(jobs, vals)}
        within = []
        for k in KINDS:
            for t in range(N_SEEDS // 3):
                g = [f[(k, 3 * t + j)] for j in range(3)]
                within.append(max(g) - min(g))
        between = []
        for t in range(N_SEEDS // 3):
            for rot in range(3):
                g = [f[(k, 3 * t + (j + rot) % 3)] for j, k in enumerate(KINDS)]
                between.append(max(g) - min(g))
        ratio = float(np.mean(between) / np.mean(within))
        dump("W03", {"mean_within_type_gap": float(np.mean(within)), "mean_between_type_gap": float(np.mean(between)),
                     "ratio": ratio, "n_personas": len(jobs), "n_within_groups": len(within), "n_between_groups": len(between),
                     "within_by_type": {k: float(np.mean(within[i * 20:(i + 1) * 20])) for i, k in enumerate(KINDS)},
                     "f_mean_by_type": {k: float(np.mean([f[(k, s)] for s in range(N_SEEDS)])) for k in KINDS},
                     "f_std_by_type": {k: float(np.std([f[(k, s)] for s in range(N_SEEDS)])) for k in KINDS},
                     "between_ge_2x_within_in_share_of_groups": float(np.mean([b >= 2 * np.mean(within) for b in between]))})
        self.assertGreaterEqual(ratio, 2.0)


if __name__ == "__main__":
    unittest.main()
