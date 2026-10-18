"""T2 履歴差 と T3 学習の向き（v3.0 の判定基準。v3.1 でも維持）。
同じ初期状態から 3 つの経験史（協働に救われた R、手柄を取られた K、探索者 X）を通し、同じ新しい経験を与える。
T2 (a) 新しい経験の感じた価値 f が 3 人格で異なる（最大差 > 0.1）            ← v3.1: 記録用（主判定は U1）
T2 (b) 価値の更新ベクトル（全特徴×欲求）の人格間 cos の最小値 < 0.5
T2 (c) fixed_teacher では cos ≥ 0.95                                         ← v3.1: expectedFailure（記録用）
T3     相手 A への評価の更新の符号が、少なくとも 1 人格で他と逆になる
数値は種 0〜19 の 20 本。(a)(b)(c) は平均と全本の両方で満たすことを要求する。

v3.1 での位置づけ（SPEC-v3.1 第 1 節の冒頭、批評 V1・S3）:
  T2(a) と T2(c) は「人格ではなく状態（欲求の状態 h と気分 m）を測っていた」。v3.0 では f が h と気分だけで決まり、
  h と気分を揃えると人格間の f の差は 0.000 だった（批評 exp1）。T2(a) の合格は人格形成の証拠ではなかった。
  主判定は U1（h と m を揃えた f、学習した状態の入れ替え）に移した。T2(a) は記録として残す（v3.0: 0.336 → v3.1: 下の結果）。
  T2(c) の不合格の原因は、v3.0 REPORT に書いたとおり Δu の h 依存（人格間の h の違い）で、v3.1 でも変わらない。
  h と気分を揃えれば cos は 1.000 になる（批評 exp2）。判定基準の想定の誤りとして expectedFailure のまま残す。
"""
import itertools
import unittest

import numpy as np

from helpers import SEEDS, cos, dump, feed, flat_update, load_json, persona, update_keys

NAMES = ["R", "K", "X"]


def run_new_event(seed, **kw):
    new = load_json("new_event.json")
    aps, entries = {}, []
    for name in NAMES:
        n = persona(name, seed, **kw)
        aps[name] = n.experience(new)
        entries.append(n.ledger.entries[-1])
    keys = update_keys(entries)
    vecs = {name: flat_update(e, keys) for name, e in zip(NAMES, entries)}
    return aps, vecs


class T2HistoryDifference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default = [run_new_event(s) for s in SEEDS]
        cls.fixed = [run_new_event(s, fixed_teacher=True) for s in SEEDS]

    def test_t2a_felt_value_differs(self):
        """記録用（v3.1 では主判定ではない。h と気分の違いを含んだ f の差を測っている）。"""
        spans = [max(ap[n]["f"] for n in NAMES) - min(ap[n]["f"] for n in NAMES) for ap, _ in self.default]
        dump("T02a", {"f_max_diff_by_seed": spans, "mean": float(np.mean(spans)), "min": float(np.min(spans)),
                      "f_seed0": {n: self.default[0][0][n]["f"] for n in NAMES}})
        self.assertGreater(float(np.mean(spans)), 0.1)
        self.assertGreater(float(np.min(spans)), 0.1)

    def test_t2b_update_vector_directions_differ(self):
        mins, pair0 = [], {}
        for i, (_, vecs) in enumerate(self.default):
            cs = {(a, b): cos(vecs[a], vecs[b]) for a, b in itertools.combinations(NAMES, 2)}
            mins.append(min(cs.values()))
            if i == 0:
                pair0 = {f"{a}-{b}": v for (a, b), v in cs.items()}
        dump("T02b", {"min_cos_by_seed": mins, "mean": float(np.mean(mins)), "max": float(np.max(mins)),
                      "pairs_seed0": pair0})
        self.assertLess(float(np.mean(mins)), 0.5)
        self.assertLess(float(np.max(mins)), 0.5)

    @unittest.expectedFailure
    def test_t2c_fixed_teacher_directions_agree(self):
        """記録用・既知の不合格（v3.0 から変わらず cos の最小 0.845）。理由: fixed_teacher の人格間の差は Δu の差だけで、
        Δu は欲求の状態 h（効用の傾き exp(−h/3)）に依存する。docs/04 の R/K/X の履歴（同じ方向の事象が 12 回続く）を
        M の大きさ 0.30 で通すと、履歴の後の h が人格で大きく違う。SPEC の想定（h の影響はわずか）が成り立たない。
        閾値は緩めない。h と気分を揃えた比較は U1 が担う。"""
        mins, pair0 = [], {}
        for i, (_, vecs) in enumerate(self.fixed):
            cs = {(a, b): cos(vecs[a], vecs[b]) for a, b in itertools.combinations(NAMES, 2)}
            mins.append(min(cs.values()))
            if i == 0:
                pair0 = {f"{a}-{b}": v for (a, b), v in cs.items()}
        dump("T02c", {"min_cos_by_seed": mins, "mean": float(np.mean(mins)), "min": float(np.min(mins)),
                      "pairs_seed0": pair0})
        self.assertGreaterEqual(float(np.mean(mins)), 0.95)
        self.assertGreaterEqual(float(np.min(mins)), 0.95)

    def test_t3_direction_of_learning_flips(self):
        deltas, flips = [], 0
        for ap, _ in self.default:
            d = {n: ap[n]["value_change"]["entities"]["friendA"]["delta"] for n in NAMES}
            deltas.append(d)
            signs = {np.sign(v) for v in d.values()}
            flips += int(len(signs) > 1)
        dump("T03", {"delta_like_friendA_seed0": deltas[0], "seeds_with_sign_disagreement": flips,
                     "n_seeds": len(SEEDS)})
        self.assertEqual(flips, len(SEEDS), deltas[0])

    # ---- 補助（SPEC 第 8 節の外）----
    def test_extra_expectation_and_surprise_differ_by_history(self):
        """補助: 学習した価値が効くのは、感じた価値 f ではなく、予期 Q と驚き δ である（f は h と気分で決まる）。
        同じ新しい経験への予期 Q が人格で 0.1 を超えて違い（最小の本でも）、驚き δ も人格で違う。"""
        qs = [{n: ap[n]["Q"] for n in NAMES} for ap, _ in self.default]
        ds = [{n: ap[n]["delta"] for n in NAMES} for ap, _ in self.default]
        span_q = [max(q.values()) - min(q.values()) for q in qs]
        dump("T02_extra", {"Q_seed0": qs[0], "delta_seed0": ds[0], "Q_span_min_over_seeds": float(np.min(span_q))})
        self.assertGreater(float(np.min(span_q)), 0.1)
        # δ の符号の分岐は v3.1 では起きない（参照点項で全員正: R +0.267, K +1.245, X +0.071。v3.0: −0.070, +1.054, +0.294）。
        # 検査としては外し、数値は T02_extra.json に記録する。
        self.assertGreater(max(max(d.values()) - min(d.values()) for d in ds), 0.1)

    def test_extra_subtyping_prediction_of_docs04_section6(self):
        """補助（docs/04 第 6 節の予測。デモを見たあとに検査として追加）: 傷ついた履歴の K は、予期より良い個人 A を
        『例外』として高く評価し、類（共同作業者）への見方はほとんど変えない。"""
        new = load_json("new_event.json")
        n = persona("K", 0)
        kal = n.value.person
        cls_before = float(kal.W[kal.idx["cls:collaborator"]].sum())
        a = n.experience(new)
        cls_after = float(kal.W[kal.idx["cls:collaborator"]].sum())
        v = a["value_change"]["entities"]["friendA"]
        dump("T03_subtyping", {"A_before": v["before"], "A_after": v["after"], "cls_before": cls_before,
                               "cls_after": cls_after})
        self.assertLess(v["before"], 0.0)
        self.assertGreater(v["after"], 0.0)                              # A は例外として高く評価される
        self.assertLess(abs(cls_after - cls_before), 0.2 * abs(v["after"] - v["before"]))   # 類はほとんど動かない


if __name__ == "__main__":
    unittest.main()
