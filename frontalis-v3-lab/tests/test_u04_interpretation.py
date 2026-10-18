"""U4 解釈: 既存の人格が、曖昧な出来事を、自分の信念で受け止める。

SPEC-v3.1 第 3 節: 曖昧な「案が使われなかった（手柄を取られた／共有された）」を、K 型と R 型が受け止める。
合格: K 型の「手柄を取られた」の解釈の重みが、R 型より 0.2 以上大きい（20 本平均）。
明確な反証（手柄を共有された）を 8 回与えると、K 型の信念 P(credit_taken) が 30% 以上下がる。

測り方（実装前に固定した細部）:
  K 型・R 型は乱数種ごとの生成器の経験史（36 件）で育てる。
  曖昧な出来事 = project_meeting で、新しい共同作業者 amb_partner から
    {"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": amb_partner, "magnitude": 0.8}（LLM の素の読み）。
  解釈の重み = appraise の interpretations[0].weights.credit_taken（w_I ∝ q_LLM(I) · P_belief(I | actor, context)）。
  反証の前に、その曖昧な出来事を 1 回経験させる（信念は曖昧な型を計数しない）。次に、同じ相手 amb_partner との
  明確な出来事（project_meeting。credit_shared 0.7 ＋ connection 0.5）を 8 回。
  P(credit_taken) = エンティティプローブ（amb_partner, project_meeting）の p_person.credit_taken。下がった割合 = (前 − 後) / 前。
  参考（判定しない）: 8 人の別々の相手からの反証後の、未見の共同作業者への P の下がり方。
"""
import copy
import unittest

import numpy as np

from helpers import SEEDS, dump, persona_gen, pmap

PARTNER = {"id": "amb_partner", "class": "collaborator"}
AMBIGUOUS = {"id": "amb-1", "context": "project_meeting", "concepts": ["collaboration"], "participants": [PARTNER],
             "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "amb_partner",
                           "magnitude": 0.8}], "optional": False}
PROBE = {"entity": "amb_partner", "class": "collaborator", "context": "project_meeting"}


def refutation(i, pid="amb_partner"):
    return {"id": f"refute-{pid}-{i}", "context": "project_meeting", "concepts": ["collaboration"],
            "participants": [{"id": pid, "class": "collaborator"}],
            "outcomes": [{"type": "credit_shared", "actor": pid, "magnitude": 0.7},
                         {"type": "connection", "actor": pid, "magnitude": 0.5}], "optional": False}


def worker(seed):
    out = {"seed": seed}
    for kind in ("K", "R"):
        n = persona_gen(kind, seed)
        ap = n.appraise(AMBIGUOUS)
        out[f"w_{kind}"] = ap["interpretations"][0]["weights"]["credit_taken"]
        out[f"du_{kind}"] = ap["du_weighted"]
        if kind == "K":
            m = copy.deepcopy(n)
            m.experience(AMBIGUOUS)
            before = m.probe(PROBE)["p_person"]["credit_taken"]
            for i in range(8):
                m.experience(refutation(i))
            after = m.probe(PROBE)["p_person"]["credit_taken"]
            out["P_before"], out["P_after"] = before, after
            # 参考: 8 人の別々の相手からの反証後の、未見の共同作業者への P
            q = copy.deepcopy(n)
            u0 = q.probe({"entity": "unseen", "class": "collaborator", "context": "project_meeting"})["p_person"]["credit_taken"]
            for i in range(8):
                q.experience(refutation(i, pid=f"other{i}"))
            out["P_unseen_before"] = u0
            out["P_unseen_after"] = q.probe({"entity": "unseen", "class": "collaborator", "context": "project_meeting"})["p_person"]["credit_taken"]
    return out


class U4Interpretation(unittest.TestCase):
    def test_weights_differ_by_belief_and_clear_refutation_changes_belief(self):
        rows = pmap(worker, SEEDS)
        wk = float(np.mean([r["w_K"] for r in rows])); wr = float(np.mean([r["w_R"] for r in rows]))
        drop = float(np.mean([(r["P_before"] - r["P_after"]) / r["P_before"] for r in rows]))
        drop_unseen = float(np.mean([(r["P_unseen_before"] - r["P_unseen_after"]) / r["P_unseen_before"] for r in rows]))
        dump("U04", {"weight_credit_taken_K": wk, "weight_credit_taken_R": wr, "difference": wk - wr,
                     "P_before_mean": float(np.mean([r["P_before"] for r in rows])),
                     "P_after_mean": float(np.mean([r["P_after"] for r in rows])), "relative_drop": drop,
                     "relative_drop_unseen_collaborator_8_partners": drop_unseen,
                     "seeds_drop_ge_30pct": int(sum((r["P_before"] - r["P_after"]) / r["P_before"] >= 0.3 for r in rows))})
        self.assertGreaterEqual(wk - wr, 0.2)
        self.assertGreaterEqual(drop, 0.3)


if __name__ == "__main__":
    unittest.main()
