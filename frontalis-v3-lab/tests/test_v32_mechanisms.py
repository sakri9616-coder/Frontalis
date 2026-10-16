"""v3.2 の機構の単体検査（SPEC-v3.2 1.1〜1.4）。判定基準 W1〜W7 そのものは test_w*.py。ここは「機構が仕様どおりに動くか」。

  1.1 欲求ごとの観測雑音 R_k = R / a_k（Σ は欲求ごと）
  1.2 Q = U_val + λ_CQ ΣC(手がかり)、announced
  1.3 concern_direction ∈ {sensitize, habituate}
  1.4 ヘッドの混合（前向きの対数尤度の softmax）
"""
import copy
import tempfile
import unittest

import numpy as np

from helpers import align_shell, feed, load_json, load_jsonl, make, mh, persona
from npnv3 import NPN, innate
from npnv3.belief import HEAD_NAMES
from npnv3.schema import SchemaError
from npnv3.value import KalmanModel


# ----------------------------------------------------------------------
# 1.1 カルマンの欲求ごとの雑音
# ----------------------------------------------------------------------
def _feats(rng, n_ind=3):
    ind = int(rng.integers(n_ind))
    return [("people", "people", 1.0), ("cls:c", "cls", 1.0), (f"ind:{ind}", "ind", 1.0), ("emb:0", "emb", 0.5), ("emb:1", "emb", -0.3)]


def _ref_update(W, S, ii, z, teacher, R, qv):
    """v3.1 の共有 Σ の更新（参照実装）。"""
    Sz = S[:, ii] @ z
    den = float(z @ Sz[ii]) + R
    K = Sz / den
    err = teacher - z @ W[ii]
    W2 = W + np.outer(K, err)
    S2 = S - np.outer(K, Sz)
    S2 = 0.5 * (S2 + S2.T)
    for i, q in zip(ii, qv):
        S2[i, i] += q
    return W2, S2


class PerNeedKalman(unittest.TestCase):
    VAR = {"people": 0.01, "cls": 0.04, "ind": 0.25, "emb": 0.02}
    Q = {"people": 5e-6, "cls": 2e-5, "ind": 1e-4, "emb": 5e-6}

    def test_a_none_reproduces_shared_sigma_update(self):
        """a = None（または a = 1）なら、欲求ごとの Σ は全部同じで、更新は v3.1 の共有 Σ と一致する。"""
        rng = np.random.default_rng(0)
        k = KalmanModel(self.VAR, self.Q, 0.01)
        k_ones = KalmanModel(self.VAR, self.Q, 0.01)
        names, layers = [], []
        Wr, Sr = np.zeros((0, 5)), np.zeros((0, 0))
        for t in range(40):
            feats = _feats(rng)
            teacher = rng.normal(size=5) * 0.3
            k.update(feats, teacher)
            k_ones.update(feats, teacher, np.ones(5))
            for name, layer, v in feats:                     # 参照実装（v3.1: 全欲求で共有の Σ）
                if name not in names:
                    names.append(name); layers.append(layer)
                    Wr = np.vstack([Wr, np.zeros((1, 5))])
                    S2 = np.zeros((len(names), len(names))); S2[:-1, :-1] = Sr; S2[-1, -1] = self.VAR[layer]; Sr = S2
            ii = np.array([names.index(n) for n, _, _ in feats])
            z = np.array([v for _, _, v in feats])
            Wr, Sr = _ref_update(Wr, Sr, ii, z, teacher, 0.01, [self.Q[layers[i]] for i in ii])
        self.assertEqual(k.names, names)
        self.assertTrue(np.allclose(k.W, Wr, atol=1e-12))
        for j in range(5):
            self.assertTrue(np.allclose(k.S[j], Sr, atol=1e-12))
        self.assertTrue(np.array_equal(k.W, k_ones.W))        # a = None と a = 1 は完全に同じ

    def test_larger_a_learns_faster_and_more_precisely(self):
        """a_k が大きい欲求ほど、同じ教師から大きく動き、事後分散が小さくなる。"""
        a = np.array([2.0, 1.0, 1.0, 1.0, 0.25])
        k = KalmanModel(self.VAR, self.Q, 0.01)
        feats = [("people", "people", 1.0), ("cls:c", "cls", 1.0), ("ind:x", "ind", 1.0)]
        for _ in range(6):
            k.update(feats, np.ones(5), a)
        v = k.predict(feats)[0]
        self.assertGreater(v[0], v[1])
        self.assertGreater(v[1], v[4])
        ii = [k.idx["ind:x"]]
        var_ind = k.S[:, ii[0], ii[0]]
        self.assertLess(var_ind[0], var_ind[1])
        self.assertLess(var_ind[1], var_ind[4])
        self.assertTrue(np.allclose(k.S[1], k.S[2]) and np.allclose(k.S[1], k.S[3]))     # a が同じ欲求は Σ も同じ

    def test_covariances_stay_psd_and_bounded(self):
        rng = np.random.default_rng(1)
        k = KalmanModel(self.VAR, self.Q, 0.01)
        for t in range(300):
            a = rng.dirichlet(np.ones(5)) * 5
            k.update(_feats(rng), rng.normal(size=5), a)
        for j in range(5):
            ev = np.linalg.eigvalsh(k.S[j])
            self.assertGreater(ev.min(), -1e-9)
            self.assertLessEqual(float(np.diag(k.S[j]).max()), 0.25 + 1e-9)

    def test_no_concern_and_no_precision_weighting_give_identical_w(self):
        """申し出のない履歴では、no_concern（a = 1 固定）と no_precision_weighting（a は動くが雑音は R）の価値の重みは完全に同じ
        （教師 Δu は a に依存しない。a → W は R_k の経路だけ）。"""
        hist = load_jsonl("K.jsonl") + load_jsonl("R.jsonl")
        n1, n2 = make(0, no_concern=True), make(0, no_precision_weighting=True)
        feed(n1, hist); feed(n2, hist)
        self.assertFalse(np.allclose(n2.concern.a, 1.0))
        self.assertTrue(np.array_equal(n1.value.person.W, n2.value.person.W))
        self.assertTrue(np.array_equal(n1.value.sit.W, n2.value.sit.W))
        n3 = make(0)
        feed(n3, hist)
        self.assertFalse(np.array_equal(n1.value.person.W, n3.value.person.W))      # 既定（R/a_k）は違う

    def test_a_does_not_depend_on_w(self):
        """ループの確認（SPEC-v3.2 1.1）: a の目標は |Δu|（外生）で W に依存しない。W を壊しても a の軌跡は変わらない。"""
        hist = load_jsonl("R.jsonl")
        n1, n2 = make(0), make(0)
        feed(n1, hist[:10]); feed(n2, hist[:10])
        n2.value.person.W = n2.value.person.W + 5.0
        n2.value.sit.W = n2.value.sit.W - 5.0
        for e in hist[10:]:
            n1.experience(e); n2.experience(e)
        self.assertTrue(np.array_equal(n1.concern.a, n2.concern.a))


# ----------------------------------------------------------------------
# 1.2 C → Q、announced
# ----------------------------------------------------------------------
SIT = {"context": "music_session", "concepts": ["collaboration"],
       "participants": [{"id": "x1", "class": "collaborator"}]}


class ConditionedValueEntersQ(unittest.TestCase):
    def test_q_is_uval_plus_lambda_times_c_of_cues(self):
        n = persona("K", 0)
        base = n.probe(SIT)
        self.assertAlmostEqual(base["Q"], base["Q_val"] + base["C_Q"], places=12)
        cues = ["concept:collaboration", "ind:x1", "cls:collaborator"]
        # v3.3: 潜在原因の事前は文脈に依存するので、手がかりのプローブにも文脈を渡す
        expected = sum(n.probe({"cue": c, "context": SIT["context"]})["C"] for c in cues)
        self.assertAlmostEqual(base["C_Q"], expected, places=12)
        # single_cause（K = 1、v3.2 と同じ）: C を手で動かすと、Q がそのぶん動く（U_val は変わらない）
        s1 = persona("K", 0, single_cause=True)
        base1 = s1.probe(SIT)
        m = copy.deepcopy(s1)
        m.cond.nudge("concept:collaboration", 0.5)
        shifted = m.probe(SIT)
        self.assertAlmostEqual(shifted["Q"] - base1["Q"], 0.5, places=12)
        self.assertAlmostEqual(shifted["Q_val"], base1["Q_val"], places=12)
        # 既定（潜在原因）: 全ての原因の価値を動かすと、応答は事前の重みの和（< 1）倍だけ動く
        m2 = copy.deepcopy(n)
        m2.cond.nudge("concept:collaboration", 0.5)
        d = m2.probe(SIT)["C_Q"] - base["C_Q"]
        self.assertTrue(0.0 < d <= 0.5 + 1e-12)

    def test_lambda_cq_and_ablations(self):
        n = persona("K", 0)
        m = copy.deepcopy(n)
        m.cond.nudge("concept:collaboration", 0.5)
        q1 = m.probe(SIT)["C_Q"]
        m.cfg.lambda_cq = 2.0
        self.assertAlmostEqual(m.probe(SIT)["C_Q"], 2.0 * q1, places=12)
        for kw in ({"no_cq": True}, {"no_conditioning": True}):
            z = make(0, **kw)
            feed(z, load_jsonl("K.jsonl"))
            self.assertEqual(z.probe(SIT)["C_Q"], 0.0)
            self.assertEqual(z.probe(SIT)["Q"], z.probe(SIT)["Q_val"])

    def test_announced_type_is_a_cue_of_q_only(self):
        n = persona("K", 0, single_cause=True)
        n.cfg.mc_samples = 2048                  # MB の乱数はプローブの内容で決まる。announced で Q_val が動かないことを見るため、サンプルを増やして雑音を小さくする
        n.cond.set_single("type:praise", -0.4)
        a = n.probe({**SIT, "announced": ["praise"]})
        b = n.probe(SIT)
        self.assertAlmostEqual(a["C_Q"] - b["C_Q"], -0.4, places=12)
        self.assertAlmostEqual(a["Q"] - a["Q_val"], a["C_Q"], places=12)
        self.assertLess(abs(a["Q_val"] - b["Q_val"]), 0.05)        # U_val は announced に依らない（MB の乱数はプローブの内容で決まるので、わずかな差）
        e = {"id": "o1", "context": "music_session", "concepts": ["collaboration"], "participants": [],
             "outcomes": [{"type": "praise", "actor": "world", "magnitude": 0.5}], "optional": True, "announced": ["praise"]}
        ap = n.appraise(e)
        self.assertIn("type:praise", ap["cues_Q"])
        self.assertAlmostEqual(ap["C_Q"], n.cond.value(ap["cues_Q"], "music_session", n._clock()), places=12)

    def test_schema_validates_announced(self):
        n = make(0)
        base = {"id": "a", "context": "music_session", "concepts": [], "participants": [], "outcomes": []}
        with self.assertRaises(SchemaError):
            n.appraise({**base, "announced": ["no_such_type"]})
        with self.assertRaises(SchemaError):
            n.appraise({**base, "announced": ["praise", "praise"]})
        with self.assertRaises(SchemaError):
            n.appraise({**base, "announced": "praise"})
        n.appraise({**base, "announced": ["praise"]})

    def test_conditioned_value_changes_approach_probability(self):
        n = make(0, single_cause=True)
        e = {"id": "o1", "context": "studio", "concepts": ["music"], "participants": [],
             "outcomes": [{"type": "praise", "actor": "world", "magnitude": 0.5}], "optional": True, "announced": ["praise"]}
        p0 = n.appraise(e)["p_approach"]
        n.cond.set_single("type:praise", -0.6)
        p1 = n.appraise(e)["p_approach"]
        n.cond.set_single("type:praise", +0.6)
        p2 = n.appraise(e)["p_approach"]
        self.assertLess(p1, p0)
        self.assertLess(p0, p2)

    def test_person_part_of_q(self):
        n = persona("K", 0)
        pr = n.probe(SIT)
        self.assertAlmostEqual(pr["Q_person"], pr["omega"] * pr["V_MF_person"] + (1 - pr["omega"]) * pr["V_MB_person"], places=12)
        self.assertAlmostEqual(pr["V_MF_person"], sum(x["value"] for x in pr["persons"]), places=12)
        e = n.probe({"context": "solo_practice", "concepts": ["music"], "participants": []})
        self.assertEqual(e["Q_person"], 0.0)
        self.assertEqual(e["V_MB_person"], 0.0)
        self.assertLess(pr["Q_person"], 0.0)               # K は共同作業者に手柄を取られた


# ----------------------------------------------------------------------
# 1.3 関心の重みの方向
# ----------------------------------------------------------------------
class ConcernDirection(unittest.TestCase):
    def _run(self, direction):
        n = make(0, concern_direction=direction)
        feed(n, load_jsonl("K.jsonl"))
        return n

    def test_sum_bounded_and_opposite_ordering(self):
        s, h = self._run("sensitize"), self._run("habituate")
        for n in (s, h):
            self.assertAlmostEqual(float(n.concern.a.sum()), 5.0, places=12)
            self.assertTrue(np.all(n.concern.a >= 0) and np.all(n.concern.a < 5.0))
        self.assertFalse(np.allclose(s.concern.a, h.concern.a))
        # 揺さぶられやすい欲求（K の履歴で効用変化の累積が大きい欲求）の順位が逆向きになる
        shake = np.zeros(5)
        n = make(0)
        for e in load_jsonl("K.jsonl"):
            shake += np.abs(np.array([n.experience(e)["du"][k] for k in innate.NEEDS]))
        top = int(np.argmax(shake))
        self.assertGreater(s.concern.a[top], 1.0)
        self.assertLess(h.concern.a[top], 1.0)
        self.assertGreater(s.concern.a[top], h.concern.a[top])

    def test_habituate_target_formula(self):
        """habituate の目標 = clip(run / max(|Δu_k|, run/5), 0, 5)。1 回の更新で a が目標へ η_a だけ寄る。"""
        n = make(0, concern_direction="habituate")
        du = np.array([0.4, 0.0, 0.1, 0.0, 0.1])
        n.concern.update(du)
        run = float(np.abs(du).mean())         # 最初の更新は rate = 1: run = mean|Δu|
        target = np.clip(run / np.maximum(np.abs(du), run / 5.0), 0.0, 5.0)
        a = np.ones(5) + 0.02 * (target - 1.0)
        self.assertTrue(np.allclose(n.concern.a, 5.0 * a / a.sum(), atol=1e-12))

    def test_direction_changes_how_much_is_learned_from_the_same_experience(self):
        """同じ経験から学ぶ量（ΔW のノルム）が、関心の重みで変わる。"""
        s, h = self._run("sensitize"), self._run("habituate")
        e = load_json("new_event.json")
        s.experience(copy.deepcopy(e)); h.experience(copy.deepcopy(e))
        ds = np.linalg.norm(s.ledger.entries[-1]["dW"]["person"]["ind:friendA"])
        dh = np.linalg.norm(h.ledger.entries[-1]["dW"]["person"]["ind:friendA"])
        self.assertNotAlmostEqual(float(ds), float(dh), places=4)

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            make(0, concern_direction="amplify")


# ----------------------------------------------------------------------
# 1.4 ヘッドの混合
# ----------------------------------------------------------------------
class HeadMixture(unittest.TestCase):
    def test_default_is_mixture_and_weights_form_a_distribution(self):
        n = make(0)
        self.assertEqual(n.cfg.head_kind, "mixture")
        self.assertEqual(set(n.belief.heads), {"linear"})                    # v3.3: X3 の結論で、既定の mixture から mlp を外した（学習するヘッドは linear だけ）
        self.assertEqual(n.cfg.mixture_heads, ("linear", "knn1", "none"))
        w0 = n.belief.head_weights()
        self.assertEqual(float(w0[HEAD_NAMES.index("none")]), 1.0)          # 観測 0 件: どのヘッドにも実績がない → 一般の率
        self.assertEqual(n.belief.rho(), 0.0)
        feed(n, load_jsonl("T8_train.jsonl")[:40])
        w = n.belief.head_weights()
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        for nm in HEAD_NAMES:
            self.assertEqual(w[HEAD_NAMES.index(nm)] > 0, nm in n.cfg.mixture_heads)
        self.assertAlmostEqual(n.belief.rho(), 1.0 - float(w[HEAD_NAMES.index("none")]), places=12)
        # mlp は設定として残してある
        self.assertEqual(set(make(0, mixture_heads=("mlp", "linear", "knn1", "none")).belief.heads), {"mlp", "linear"})

    def test_weights_are_softmax_of_transfer_loglik_ema(self):
        n = make(0)
        feed(n, load_jsonl("T8_train.jsonl")[:60])
        act = [HEAD_NAMES.index(h) for h in n.cfg.mixture_heads]
        z = n.belief.ll_ema[act]
        e = np.exp(z - z.max())
        expected = np.zeros(len(HEAD_NAMES)); expected[act] = e / e.sum()
        self.assertTrue(np.allclose(n.belief.head_weights(), expected, atol=1e-12))
        self.assertEqual(n.belief.ll_cnt, 60)
        self.assertTrue(np.all(n.belief.ll_ema[[i for i in range(len(HEAD_NAMES)) if i not in act]] == 0.0))

    def test_forward_loglik_uses_the_state_before_the_observation(self):
        """最初の経験: 計数も学習もまだない。none ヘッドの予測は全型 0.2（事前の平均）で、移動平均は rate = 1 なので一回分の対数尤度そのもの。"""
        n = make(0)
        e = {"id": "f1", "context": "music_session", "concepts": ["collaboration"], "participants": [{"id": "p", "class": "collaborator"}],
             "outcomes": [{"type": "connection", "actor": "p", "magnitude": 0.8}, {"type": "credit_taken", "actor": "p", "magnitude": 0.5}],
             "optional": False}
        n.experience(e)
        p = 0.2
        ll = 2 * np.log(p) + (innate.N_TYPES - 2) * np.log(1 - p)
        self.assertAlmostEqual(float(n.belief.ll_ema[HEAD_NAMES.index("none")]), ll, places=10)

    def test_masked_cells_do_not_enter_the_loglik(self):
        """曖昧な結果の型（観測なし）は、前向きの対数尤度にも入らない。"""
        e = {"id": "f2", "context": "music_session", "concepts": ["collaboration"], "participants": [{"id": "p", "class": "collaborator"}],
             "outcomes": [{"type": "connection", "actor": "p", "magnitude": 0.8},
                          {"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "p", "magnitude": 0.5}],
             "optional": False}
        n = make(0)
        n.experience(e)
        ll = 1 * np.log(0.2) + (innate.N_TYPES - 3) * np.log(0.8)           # connection=1、マスク 2 型を除く 16 型は 0
        self.assertAlmostEqual(float(n.belief.ll_ema[HEAD_NAMES.index("none")]), ll, places=10)

    def test_t8_three_class_world_cannot_support_transfer_so_none_dominates(self):
        """v3.3: 重みは転移性能（類を除いたヘッドで、その類の観測を予測した対数尤度）で決まる。T8 の世界は類が 3 つしかなく、1 つの類を除くと
        残りの 2 つからは似た類の率を学べない（共同作業者を除くと師と見知らぬ人だけ）ので、どのヘッドも転移できず、none（他の類をまとめた率）が最大になる。
        （v3.2 では自分の類を含むヘッドが有利で none の重みは 0.08 だった。転移できる世界で類を見るヘッドが重みを得ることは test_x_nn_transfer.py）"""
        n = make(0)
        feed(n, load_jsonl("T8_train.jsonl"))
        w = n.belief.head_weights_dict()
        self.assertEqual(max(w, key=w.get), "none")
        self.assertGreater(w["none"], 0.5)

    def test_save_and_restore_keep_head_state(self):
        n = make(0)
        feed(n, load_jsonl("T8_train.jsonl")[:50])
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        self.assertTrue(np.array_equal(n.belief.ll_ema, m.belief.ll_ema))
        self.assertEqual(n.belief.ll_cnt, m.belief.ll_cnt)
        self.assertEqual(n.belief.head_weights_dict(), m.belief.head_weights_dict())
        e = load_jsonl("T8_train.jsonl")[50]
        self.assertEqual(n.experience(copy.deepcopy(e))["Q"], m.experience(copy.deepcopy(e))["Q"])

    def test_single_head_kinds_are_the_v31_baselines(self):
        for kind in ("mlp", "linear", "knn1", "none"):
            n = make(0, head_kind=kind)
            feed(n, load_jsonl("T8_train.jsonl")[:30])
            self.assertEqual(n.belief.ll_cnt, 0)                  # 前向き評価は mixture のときだけ
            if kind in ("mlp", "linear"):
                self.assertAlmostEqual(n.belief.rho(), 30 / 60)
            elif kind == "none":
                self.assertEqual(n.belief.rho(), 0.0)


class V32Equivalence(unittest.TestCase):
    def test_v33_additions_can_be_switched_off_to_reproduce_v32(self):
        """v3.3 の追加（類を除いたヘッド、潜在原因の条件づけ）は、head_kind = "none" と single_cause = True で完全に外せる。
        外すと、v3.2 の head_kind = "none" の数値（v3.2 のコードで測った値）に戻る（種 0 の f と H1 の C(praise)、1e-12 以内）。
        （v3.2 の既定の mixture / mlp へのヘッドの SGD は、学び直しに置き換えたので、v3.1 の数値への復帰の検査は廃止した。）"""
        ref = {"R": 0.47996245797296794, "K": 0.772816079486168, "X": 0.34298498586776577}
        new = load_json("new_event.json")
        for k, f0 in ref.items():
            n = persona(k, 0, head_kind="none", single_cause=True)
            self.assertAlmostEqual(n.appraise(new)["f"], f0, places=12)
        n = make(0, head_kind="none", single_cause=True)
        for e in mh.generate_praise_history("H1", 0):
            n.experience(e)
        self.assertAlmostEqual(n.probe({"cue": "type:praise"})["C"], -0.14606508077008556, places=12)
        # 既定（v3.3）では違う
        self.assertGreater(abs(persona("K", 0).appraise(new)["f"] - ref["K"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
