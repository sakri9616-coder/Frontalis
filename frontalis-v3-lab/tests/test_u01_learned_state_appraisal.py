"""U1 入れ替え（主判定）: 受け止め方 f は、学習した状態に依存するか。

SPEC-v3.1 第 3 節:
  人格の型 R・K・X の経験史で育てた後、欲求の状態 h と気分 m を全員同じ値（基準値）に揃え、同じ新しい経験を評価する。
  次に、学習した状態（W、Σ、C、a、信念）だけを人格間で入れ替える。
  (a) h と m を揃えても、f の人格間の最大差が 0.10 以上（20 本の平均）。
  (b) 学習した状態を入れ替えると、f の差の 80% 以上が状態と一緒に移る。

測り方（実装前に固定した細部）:
  経験史は乱数種ごとに生成器が作る別の履歴（data/make_histories.py の generate_history。36 件、型の世界の真の確率から抽出）。
  新しい経験は data/histories/new_event.json（v3.0 の T2 と同じ）。
  (a) 3 人格の f の最大 − 最小を、種ごとに求めて平均する。
  (b) 受け手 r の殻（h, m は揃えた値）に、提供者 d の学習した状態（価値ネット W・Σ、条件づけ C、関心の重み a、信念、所有する埋め込み）を
      丸ごと移した NPN の f を f_swap とする。移った割合 = (f_swap − f_r) / (f_d − f_r)（|f_d − f_r| ≥ 0.02 の順序対）。
      全順序対（6 対 × 20 本）の平均が 0.8 以上。
  参考（判定しない）: 部品ごと（a だけ、C だけ、価値ネットだけ、信念だけ）の入れ替えで移る割合、空の NPN との f の差、アブレーション。
"""
import copy
import itertools
import unittest

import numpy as np

from helpers import KINDS, SEEDS, align_shell, dump, load_json, persona_gen, pmap, make

NEW = load_json("new_event.json")
ABLATIONS = {"full": {}, "no_concern": {"no_concern": True}, "no_conditioning": {"no_conditioning": True},
             "no_reference": {"no_reference": True}}


def transplant(recipient, donor, parts):
    """donor の学習した状態を recipient の殻（h, m, 乱数、設定）に移す。parts: 移す部品の集合。"""
    s = copy.deepcopy(recipient)
    d = copy.deepcopy(donor)
    if "all" in parts:
        s.emb, s.belief, s.value, s.concern, s.cond = d.emb, d.belief, d.value, d.concern, d.cond
        return s
    if "a" in parts:
        s.concern = d.concern
    if "C" in parts:
        s.cond = d.cond
    if "value" in parts:
        d.value.emb = s.emb
        s.value = d.value
    if "belief" in parts:
        d.belief.emb = s.emb
        s.belief = d.belief
    return s


def worker(seed):
    out = {"seed": seed}
    for name, kw in ABLATIONS.items():
        nets = {k: align_shell(persona_gen(k, seed, **kw)) for k in KINDS}
        aps = {k: n.appraise(NEW) for k, n in nets.items()}
        f = {k: aps[k]["f"] for k in KINDS}
        out[name] = {"f": f, "gap": max(f.values()) - min(f.values()),
                     "parts": {k: aps[k]["f_parts"] for k in KINDS}, "Q": {k: aps[k]["Q"] for k in KINDS}}
        if name != "full":
            continue
        empty = align_shell(make(seed)).appraise(NEW)["f"]
        out["empty_f"] = empty
        ratios, part_ratios = [], {p: [] for p in ("a", "C", "value", "belief")}
        for r, d in itertools.permutations(KINDS, 2):
            gap = f[d] - f[r]
            if abs(gap) < 0.02:
                continue
            ratios.append((transplant(nets[r], nets[d], {"all"}).appraise(NEW)["f"] - f[r]) / gap)
            for p in part_ratios:
                part_ratios[p].append((transplant(nets[r], nets[d], {p}).appraise(NEW)["f"] - f[r]) / gap)
        out["ratios"] = ratios
        out["part_ratios"] = part_ratios
    return out


class U1LearnedState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pmap(worker, SEEDS)

    def test_a_f_differs_with_shell_aligned(self):
        gaps = [r["full"]["gap"] for r in self.rows]
        abl = {k: float(np.mean([r[k]["gap"] for r in self.rows])) for k in ABLATIONS}
        vs_empty = float(np.mean([max(abs(r["full"]["f"][k] - r["empty_f"]) for k in KINDS) for r in self.rows]))
        parts = {k: float(np.mean([r["full"]["parts"][k][c] for r in self.rows]))
                 for k in KINDS for c in ("innate", "conditioned", "reference")} if False else {
            c: {k: float(np.mean([r["full"]["parts"][k][c] for r in self.rows])) for k in KINDS}
            for c in ("innate", "conditioned", "reference")}
        dump("U01a", {"mean_gap": float(np.mean(gaps)), "min_gap": float(np.min(gaps)), "gaps": gaps,
                      "ablation_mean_gap": abl, "mean_abs_f_minus_empty_NPN": vs_empty, "f_parts_mean": parts,
                      "Q_mean": {k: float(np.mean([r["full"]["Q"][k] for r in self.rows])) for k in KINDS},
                      "f_mean": {k: float(np.mean([r["full"]["f"][k] for r in self.rows])) for k in KINDS}})
        self.assertGreaterEqual(float(np.mean(gaps)), 0.10)

    def test_b_f_moves_with_learned_state(self):
        ratios = [x for r in self.rows for x in r["ratios"]]
        parts = {p: float(np.mean([x for r in self.rows for x in r["part_ratios"][p]])) for p in ("a", "C", "value", "belief")}
        dump("U01b", {"mean_ratio": float(np.mean(ratios)), "min_ratio": float(np.min(ratios)), "n_pairs": len(ratios),
                      "partial_swap_ratio_mean": parts})
        self.assertGreaterEqual(float(np.mean(ratios)), 0.8)


if __name__ == "__main__":
    unittest.main()
