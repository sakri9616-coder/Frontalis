"""潜在原因による条件づけ価値（SPEC-v3.3 4.2）の機構の単体検査。判定基準 E1〜E9 は test_e_extinction.py。ここは「式どおりに動くか」。

設計は 2 つある（lc_design）。PriorAndResponse・StructuralLimits・UpdateRules・Reinstatement・SingleCauseIsV32 は **spec_literal**（SPEC-v3.3 4.2 のとおり。改訂前の実装）を検査する。
Revised* は **revised**（実験設定。v3.3 第 1 部の設計の改訂。オーケストレーター承認、結果を見た後。第 2 部で既定から外した）を検査する。
IntegrationInNPN と ExplainAndTraceSpecLiteral は既定（spec_literal）の NPN を通す。

  spec_literal  事前 p(k) ∝ N_k · (n_k(c) + β)/(N_k + β S) · exp(−(t − t_k)/τ)，新しい原因は ∝ α。応答 Σ_k p(k) C_k
                事後 事前 × N(y_x; C_k, σ_r²)，新しい原因が最大なら作る。学習 ΔC_k = (α_C / 手がかりの数) 事後_k (y_x − C_k)（柔らかい）
                復位 手がかりなしの嫌悪の結果で、負の価値を持つ全ての原因の t_k を現在に
  revised       事前 w_k = Σ_{i∈k} (t − t_i + 1)^(−d) · (n_k(c) + β_k)/(N_k + β_k S)，β_k = β_first（最初の原因）| β_later（2 番目以降）。新しい原因 ∝ α
                学習は局所 MAP（事後が最大の原因だけを TD の誤差で更新し、その履歴に (t, 文脈) を加える）。満員なら質量最小の原因を再利用
                復位は、嫌悪の結果を尤度 N(r; C_{x,k}, σ_r²) で原因に帰属し、最も確からしい 1 つの原因に (t, 文脈) の試行を加える（価値は更新しない）
  single_cause（K = 1）は v3.2 の算術と完全に同じ
"""
import copy
import tempfile
import unittest

import numpy as np

from helpers import deep_equal, feed, load_jsonl, make, mh, persona
from npnv3 import NPN, Config
from npnv3.value import Conditioning


def cond(**kw):
    kw.setdefault("lc_design", "spec_literal")      # このファイルの古い検査は SPEC のとおりの式（改訂前）を見る
    return Conditioning(Config(**kw))


def rcond(**kw):
    kw.setdefault("lc_design", "revised")            # 既定は spec_literal（v3.3 第 2 部）。revised は実験設定
    return Conditioning(Config(**kw))


def place(cd, cue, causes, ctxs=("A", "B")):
    """revised: 手がかり cue に、原因を手で置く。causes = [(価値, β, [(時刻, 文脈), ...]), ...]（試行の履歴から文脈の計数も作る）。"""
    i = cd._register(cue)
    for c in ctxs:
        cd._ctx_col(c, True)
    for k, (v, beta, trials) in enumerate(causes):
        cd.CK[i, k] = v
        cd.AL[i, k] = 1.0
        cd.BK[i, k] = beta
        cd.CRE[i] += 1
        for (t, c) in trials:
            cd._hist_add(i, k, float(t), cd.ctx_idx[c])
            cd.CNT[i, k, cd.ctx_idx[c]] += 1.0
            cd.TK[i, k] = max(cd.TK[i, k], t)
    return i


def put_causes(cd, cue, values, counts, times, ctxs=("A", "B")):
    """手がかり cue に、原因を手で置く（white-box）。counts[k][j] = 原因 k の文脈 ctxs[j] の計数。"""
    i = cd._register(cue)
    for c in ctxs:
        cd._ctx_col(c, True)
    for k, (v, cnt, t) in enumerate(zip(values, counts, times)):
        cd.CK[i, k] = v
        cd.AL[i, k] = 1.0
        cd.TK[i, k] = t
        for j, c in enumerate(ctxs):
            cd.CNT[i, k, cd.ctx_idx[c]] = cnt[j]
    return i


class PriorAndResponse(unittest.TestCase):
    def test_prior_weight_formula_and_response(self):
        cd = cond()
        put_causes(cd, "x", [-0.2, 0.0], [[10.0, 0.0], [0.0, 6.0]], [10.0, 40.0])
        cfg = cd.cfg
        for ctx, t in (("A", 50.0), ("B", 50.0), ("A", 100.0)):
            S = 2.0                                                    # 見た文脈は A, B の 2 つ（ctx は既知）
            w = []
            for N, nc, tk in ((10.0, 10.0 if ctx == "A" else 0.0, 10.0), (6.0, 0.0 if ctx == "A" else 6.0, 40.0)):
                w.append(N * (nc + cfg.lc_beta) / (N + cfg.lc_beta * S) * np.exp(-(t - tk) / cfg.lc_tau))
            p = np.array(w) / (sum(w) + cfg.lc_alpha)
            self.assertAlmostEqual(cd.value_of("x", ctx, t), float(p[0] * -0.2 + p[1] * 0.0), places=14)

    def test_unseen_context_counts_as_one_more_category(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[9.0, 1.0]], [10.0])         # A と B を見た（計数 9 と 1）
        cfg = cd.cfg
        t = 12.0
        N = 10.0
        w_seen = N * (1.0 + cfg.lc_beta) / (N + cfg.lc_beta * 2.0) * np.exp(-(t - 10.0) / cfg.lc_tau)      # B は見た文脈
        w_new = N * (0.0 + cfg.lc_beta) / (N + cfg.lc_beta * 3.0) * np.exp(-(t - 10.0) / cfg.lc_tau)       # C は未見: 範疇が 1 つ増える
        self.assertAlmostEqual(cd.value_of("x", "B", t), -0.2 * w_seen / (w_seen + cfg.lc_alpha), places=14)
        self.assertAlmostEqual(cd.value_of("x", "C", t), -0.2 * w_new / (w_new + cfg.lc_alpha), places=14)

    def test_context_free_read_ignores_the_context_term(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[10.0, 0.0]], [10.0])
        t = 12.0
        w = 10.0 * np.exp(-(t - 10.0) / cd.cfg.lc_tau)
        self.assertAlmostEqual(cd.value_of("x", None, t), -0.2 * w / (w + cd.cfg.lc_alpha), places=14)

    def test_time_decay_shrinks_the_response_toward_zero_exponentially_in_the_weight(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[10.0, 0.0]], [0.0])
        v = [cd.value_of("x", "A", t) for t in (0.0, 30.0, 60.0, 120.0)]
        self.assertTrue(all(a < 0.0 for a in v))
        self.assertTrue(abs(v[0]) > abs(v[1]) > abs(v[2]) > abs(v[3]))

    def test_unknown_cue_and_no_cause_give_zero(self):
        cd = cond()
        self.assertEqual(cd.value_of("nothing", "A", 5.0), 0.0)
        self.assertEqual(cd.value(["a", "b"], "A", 5.0), 0.0)

    def test_sum_over_cues(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[10.0, 0.0]], [10.0])
        put_causes(cd, "y", [0.1], [[3.0, 5.0]], [11.0])
        self.assertAlmostEqual(cd.value(["x", "y", "z"], "A", 12.0), cd.value_of("x", "A", 12.0) + cd.value_of("y", "A", 12.0), places=14)


class StructuralLimits(unittest.TestCase):
    """SPEC-v3.3 4.2 の式そのものの性質（判定ではない）。E5・E9 が不合格になる原因の一つを、式の性質として固定する（tests/diagnostics_e_oracle.py）。"""

    def test_without_learning_the_response_magnitude_never_grows_with_time(self):
        """すべての原因の重みは同じ exp(−t/τ) で縮み、新しい原因 α の質量だけが一定なので、学習のない間の |応答| は増えない（価値がすべて ≤ 0 のとき）。
        したがって E5（自発的回復: 消去の後、無関係な経験だけを与えると応答が戻る）は、この式では原理的に起きない。"""
        rng = np.random.default_rng(7)
        for _ in range(300):
            cd = cond()
            K = int(rng.integers(1, 6))
            vals = -rng.uniform(0.0, 0.6, size=K) * (rng.random(K) < 0.8)
            put_causes(cd, "x", list(vals), [[float(rng.uniform(0, 20)), float(rng.uniform(0, 20))] for _ in range(K)],
                       [float(rng.uniform(0, 60)) for _ in range(K)])
            ctx = ["A", "B", "C"][int(rng.integers(3))]
            prev = abs(cd.value_of("x", ctx, 60.0))
            for dt in (1, 5, 20, 50, 200):
                cur = abs(cd.value_of("x", ctx, 60.0 + dt))
                self.assertLessEqual(cur, prev + 1e-15)
                prev = cur

    def test_weight_in_an_unseen_context_saturates_so_it_barely_depends_on_how_counts_are_spread(self):
        """未見の文脈での原因の重み N_k (0 + β)/(N_k + β S) は、N_k が大きいと β に飽和する。消去の計数を 1 つの文脈に集めても 3 つの文脈に散らしても
        （S が 3 → 5）、N_k = 20 で 0.87 → 0.80 の違いしかない。E9（3 つの文脈で消去すると ABC の復元が小さい）がこの式から出にくい理由。"""
        beta = Config().lc_beta
        w1 = 20.0 * beta / (20.0 + beta * 3.0)
        w3 = 20.0 * beta / (20.0 + beta * 5.0)
        self.assertGreater(w3, 0.75)
        self.assertLess(w1 - w3, 0.1)


class UpdateRules(unittest.TestCase):
    def test_first_update_creates_a_cause_and_learns_with_alpha_c(self):
        cd = cond()
        dC, det = cd.update(["x"], -0.3, "A", 5.0)
        i = cd.idx["x"]
        self.assertEqual(int(cd.AL[i].sum()), 1)
        self.assertEqual(det["x"]["created"], 0)
        self.assertAlmostEqual(cd.CK[i, 0], cd.cfg.alpha_c * 1.0 * (-0.3 - 0.0), places=14)    # 事後 1、新しい原因の価値 0 から
        self.assertAlmostEqual(cd.CNT[i, 0, cd.ctx_idx["A"]], 1.0, places=14)                  # 文脈の計数 += 事後
        self.assertEqual(cd.TK[i, 0], 5.0)
        self.assertAlmostEqual(dC["x"], cd.CK[i, 0], places=14)

    def test_posterior_is_prior_times_gaussian_likelihood(self):
        cd = cond()
        put_causes(cd, "x", [-0.2, 0.0], [[10.0, 0.0], [0.0, 6.0]], [10.0, 40.0])
        i = cd.idx["x"]
        cfg = cd.cfg
        t, y = 50.0, 0.0
        w = np.array([10.0 * (0.0 + 1.0) / (10.0 + 2.0) * np.exp(-(t - 10.0) / cfg.lc_tau),
                      6.0 * (6.0 + 1.0) / (6.0 + 2.0) * np.exp(-(t - 40.0) / cfg.lc_tau)])
        lik = np.exp(-0.5 * ((y - np.array([-0.2, 0.0])) / cfg.lc_sigma_r) ** 2)
        new = cfg.lc_alpha * np.exp(-0.5 * (y / cfg.lc_sigma_r) ** 2)
        z = float((w * lik).sum() + new)
        post = w * lik / z
        before = cd.CK[i].copy(); cnt = cd.CNT[i].copy()
        _, det = cd.update(["x"], y, "B", t)
        self.assertTrue(np.allclose(det["x"]["post"][:2], post, atol=1e-14))
        self.assertAlmostEqual(det["x"]["post_new"], new / z, places=14)
        self.assertIsNone(det["x"]["created"])                                                  # 新しい原因は最大でない
        self.assertTrue(np.allclose(cd.CK[i, :2], before[:2] + cfg.alpha_c * post * (y - before[:2]), atol=1e-14))
        self.assertTrue(np.allclose(cd.CNT[i, :2, cd.ctx_idx["B"]], cnt[:2, cd.ctx_idx["B"]] + post, atol=1e-14))
        self.assertEqual(cd.TK[i, int(np.argmax(post))], t)                                      # 事後が最大の原因だけ時刻を更新
        self.assertEqual(cd.TK[i, 1 - int(np.argmax(post))], [10.0, 40.0][1 - int(np.argmax(post))])

    def test_new_cause_is_created_when_old_causes_cannot_explain_the_target(self):
        cd = cond()
        put_causes(cd, "x", [-0.5], [[3.0, 0.0]], [10.0])         # 強く条件づけられた原因（価値 −0.5）。N が小さい
        _, det = cd.update(["x"], 0.0, "B", 11.0)                  # 何も起きない: z = 3.3 → 尤度が小さい
        self.assertEqual(det["x"]["created"], 1)
        self.assertEqual(int(cd.AL[cd.idx["x"]].sum()), 2)
        self.assertAlmostEqual(cd.CK[cd.idx["x"], 0], -0.5, delta=0.01)    # 古い原因はほとんど動かない（事後が小さい）

    def test_full_set_reuses_the_cause_with_smallest_decayed_mass(self):
        cd = cond(lc_max_causes=2)
        put_causes(cd, "x", [-0.5, 0.3], [[20.0, 0.0], [2.0, 0.0]], [10.0, 10.0])
        i = cd.idx["x"]
        _, det = cd.update(["x"], 0.0, "B", 11.0)                  # どちらにも説明できない目標 → 新しい原因が最大 → 満員なので N が最小の原因（添字 1）を再利用
        self.assertTrue(det["x"]["reused"])
        self.assertEqual(det["x"]["created"], 1)
        self.assertEqual(int(cd.AL[i].sum()), 2)
        self.assertAlmostEqual(cd.CK[i, 0], -0.5, delta=0.01)
        self.assertLess(abs(cd.CK[i, 1]), 0.05)                    # 価値 0 から始まった新しい原因

    def test_number_of_causes_never_exceeds_K(self):
        rng = np.random.default_rng(0)
        for K in (2, 3, 5):
            cd = cond(lc_max_causes=K)
            for t in range(200):
                cd.update(["x"], float(rng.choice([-0.6, -0.3, 0.0, 0.3, 0.6])), ["A", "B", "C"][t % 3], float(t))
                self.assertLessEqual(int(cd.AL[cd.idx["x"]].sum()), K)

    def test_multiple_cues_share_one_target_by_explaining_away(self):
        cd = cond()
        put_causes(cd, "x", [-0.2], [[10.0, 0.0]], [10.0])
        put_causes(cd, "z", [0.0], [[1.0, 0.0]], [10.0])
        rx, rz = cd.value_of("x", "A", 12.0), cd.value_of("z", "A", 12.0)
        _, det = cd.update(["x", "z"], -0.3, "A", 12.0)
        self.assertAlmostEqual(det["x"]["y_x"], -0.3 - rz, places=14)       # 他の手がかりの応答を引いた目標
        self.assertAlmostEqual(det["z"]["y_x"], -0.3 - rx, places=14)
        self.assertAlmostEqual(det["x"]["resp_before"], rx, places=14)

    def test_dC_is_the_sum_of_cause_changes(self):
        cd = cond()
        rng = np.random.default_rng(1)
        for t in range(40):
            before = cd.CK.copy()
            dC, det = cd.update(["x", "y"], float(rng.normal(0, 0.3)), ["A", "B"][t % 2], float(t))
            for c in ("x", "y"):
                i = cd.idx[c]
                b = before[i] if i < before.shape[0] else np.zeros(cd.K)
                self.assertAlmostEqual(dC[c], float((cd.CK[i] - b).sum()), places=13)
                self.assertTrue(np.allclose(det[c]["dCk"], cd.CK[i] - b, atol=1e-13))


class SingleCauseIsV32(unittest.TestCase):
    def test_single_cause_update_is_the_v32_nlms_rule(self):
        cd = cond(single_cause=True)
        rng = np.random.default_rng(2)
        ref = {}                                                  # v3.2 の更新の参照実装
        names = ["a", "b", "c", "d"]
        for t in range(60):
            cues = list(rng.choice(names, size=int(rng.integers(1, 4)), replace=False))
            y = float(rng.normal(0, 0.4))
            err = y - sum(ref.get(c, 0.0) for c in cues)
            d = cd.cfg.alpha_c * err / len(cues)
            for c in cues:
                ref[c] = ref.get(c, 0.0) + d
            cd.update(cues, y, ["A", "B"][t % 2], float(t))
        for c, v in ref.items():
            self.assertEqual(cd.value_of(c, "A", 99.0), v)       # 完全に同じ（== で）
            self.assertEqual(cd.value_of(c, "Z", 0.0), v)         # 文脈・時刻に依存しない
        self.assertEqual(cd.CNT.shape[2], 0)                       # 文脈の計数は使わない

    def test_single_cause_has_exactly_one_alive_cause_per_cue(self):
        n = persona("K", 0, single_cause=True)
        self.assertEqual(n.cond.K, 1)
        self.assertTrue(np.all(n.cond.AL == 1.0))


class Reinstatement(unittest.TestCase):
    def test_only_negative_causes_of_other_cues_are_refreshed(self):
        cd = cond()
        put_causes(cd, "neg", [-0.2, 0.1], [[5.0, 0.0], [5.0, 0.0]], [3.0, 3.0])
        put_causes(cd, "pos", [0.2], [[5.0, 0.0]], [3.0])
        put_causes(cd, "tiny", [-0.01], [[5.0, 0.0]], [3.0])
        put_causes(cd, "present", [-0.3], [[5.0, 0.0]], [3.0])
        refreshed = cd.reinstate(["present"], 40.0)
        self.assertEqual(refreshed, [("neg", 0)])
        self.assertEqual(cd.TK[cd.idx["neg"], 0], 40.0)
        self.assertEqual(cd.TK[cd.idx["neg"], 1], 3.0)
        self.assertEqual(cd.TK[cd.idx["pos"], 0], 3.0)
        self.assertEqual(cd.TK[cd.idx["tiny"], 0], 3.0)             # 閾値（−0.02）より浅い
        self.assertEqual(cd.TK[cd.idx["present"], 0], 3.0)          # 提示された手がかりは対象外

    def test_reinstatement_raises_the_response_through_recency(self):
        cd = cond()
        put_causes(cd, "x", [-0.2, 0.0], [[10.0, 0.0], [0.0, 20.0]], [10.0, 80.0])
        before = cd.value_of("x", "B", 90.0)
        cd.reinstate([], 90.0)
        after = cd.value_of("x", "B", 90.0)
        self.assertLess(after, before)                              # 負の価値の原因の重みが増えたので、応答はより負になる
        self.assertLess(abs(before), 0.2)

    def test_single_cause_never_reinstates(self):
        cd = cond(single_cause=True)
        cd.update(["x"], -0.5, "A", 1.0)
        self.assertEqual(cd.reinstate([], 10.0), [])

    def test_aversive_event_without_the_cue_triggers_it_in_the_npn(self):
        n = make(0)
        # x を条件づける
        for k in range(6):
            s = f"s{k}"
            n.experience({"id": f"c{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": s})
            n.experience({"id": f"u{k}", "context": "studio", "concepts": [], "participants": [],
                          "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": s})
        i = n.cond.idx["concept:praise_from_X"]
        t0 = float(n.cond.TK[i].max())
        for k in range(8):                                           # 時間を進める
            n.experience({"id": f"f{k}", "context": "solo_practice", "concepts": ["music"], "participants": [],
                          "outcomes": [], "session": f"f{k}"})
        self.assertEqual(float(n.cond.TK[i].max()), t0)
        n.experience({"id": "h1", "context": "cafe_chat", "concepts": [], "participants": [],
                      "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": "h"})
        self.assertGreater(float(n.cond.TK[i].max()), t0)
        self.assertEqual(float(n.cond.TK[i].max()), float(n.n_experiences + n.n_avoided - 1))
        entry = n.ledger.entries[-1]
        self.assertIn("concept:praise_from_X", [c for c, k in entry["td"]["reinstated"]])


class RevisedPriorAndResponse(unittest.TestCase):
    def test_prior_weight_is_power_law_mass_times_context_prediction(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(-0.2, c.lc_beta_first, [(0, "A"), (2, "A"), (4, "A")]), (0.0, c.lc_beta_later, [(10, "B"), (12, "B")])])
        for ctx, t in (("A", 14.0), ("B", 14.0), ("A", 50.0)):
            S = 2.0
            w = []
            for times, n_a, n_b, beta in (([0, 2, 4], 3.0, 0.0, c.lc_beta_first), ([10, 12], 0.0, 2.0, c.lc_beta_later)):
                mass = sum((t - ti + 1.0) ** (-c.lc_decay_d) for ti in times)
                n_c, N = (n_a if ctx == "A" else n_b), n_a + n_b
                w.append(mass * (n_c + beta) / (N + beta * S))
            p = np.array(w) / (sum(w) + c.lc_alpha)
            self.assertAlmostEqual(cd.value_of("x", ctx, t), float(p[0] * -0.2), places=14)

    def test_old_cause_with_many_trials_regains_share_as_time_passes(self):
        """冪乗則: 最近の少数の試行は急に減り、古い多数の試行はゆっくり減る。原因どうしの比は時間で動き、古い原因が戻る
        （指数関数の spec_literal では比が動かない: StructuralLimits）。"""
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(-0.2, c.lc_beta_first, [(2 * i, "A") for i in range(12)]), (0.0, c.lc_beta_later, [(24 + 2 * j, "B") for j in range(3)])])
        def share(t):
            w = cd._prior_weights(np.array([0]), "B", t)[0]
            return w[0] / w[1]
        r = [share(t) for t in (31.0, 60.0, 120.0, 400.0)]
        self.assertTrue(r[0] < r[1] < r[2] < r[3])

    def test_context_prediction_concentration_first_broad_later_narrow(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(-0.2, c.lc_beta_first, [(0, "A")] * 10), (0.0, c.lc_beta_later, [(5, "B")] * 10)], ctxs=("A", "B", "C"))
        pred, N = cd._ctx_pred(np.array([0]), "C", cd.BK[[0]])
        self.assertGreater(pred[0, 0], 10 * pred[0, 1])                  # 未見の文脈 C: 広い原因は確率を残し、狭い原因はほとんど残さない
        pred_b, _ = cd._ctx_pred(np.array([0]), "B", cd.BK[[0]])
        self.assertGreater(pred_b[0, 1], 0.9)                            # 狭い原因は作られた文脈 B に強く結びつく

    def test_unknown_cue_and_empty_history_give_zero(self):
        cd = rcond()
        self.assertEqual(cd.value_of("nothing", "A", 5.0), 0.0)
        cd._register("x")
        self.assertEqual(cd.value_of("x", "A", 5.0), 0.0)


class RevisedUpdateRules(unittest.TestCase):
    def test_first_update_creates_the_first_cause_with_wide_context_concentration(self):
        cd = rcond()
        dC, det = cd.update(["x"], -0.3, "A", 5.0)
        i = cd.idx["x"]
        self.assertEqual(det["x"]["created"], 0)
        self.assertEqual(cd.BK[i, 0], cd.cfg.lc_beta_first)
        self.assertAlmostEqual(cd.CK[i, 0], cd.cfg.alpha_c * -0.3, places=14)               # 局所 MAP: 重み 1 で α_C (y − 0)
        self.assertEqual(cd.hist_n[i], 1)
        self.assertEqual((cd.hist_t[i][0], cd.hist_k[i][0], cd.hist_c[i][0]), (5.0, 0, cd.ctx_idx["A"]))
        self.assertEqual(cd.CNT[i, 0, cd.ctx_idx["A"]], 1.0)
        self.assertAlmostEqual(dC["x"], cd.CK[i, 0], places=14)

    def test_only_the_map_cause_is_updated_and_gets_the_trial(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(-0.2, c.lc_beta_first, [(0, "A"), (2, "A"), (4, "A")]), (0.0, c.lc_beta_later, [(10, "B"), (11, "B"), (12, "B"), (12, "B"), (13, "B"), (13, "B")])])
        i = cd.idx["x"]
        before = cd.CK[i].copy()
        t, y = 14.0, 0.0
        w = cd._prior_weights(np.array([i]), "B", t)[0]
        lik = np.exp(-0.5 * ((y - before) / c.lc_sigma_r) ** 2)
        post = w[:2] * lik[:2]
        new = c.lc_alpha * np.exp(-0.5 * (y / c.lc_sigma_r) ** 2)
        k = int(np.argmax(post))
        self.assertGreater(post[k], new)                                                     # この状態では既存の原因が最大
        _, det = cd.update(["x"], y, "B", t)
        self.assertEqual(det["x"]["map"], k)
        self.assertIsNone(det["x"]["created"])
        other = 1 - k
        self.assertEqual(cd.CK[i, other], before[other])                                     # 他の原因は動かない
        self.assertAlmostEqual(cd.CK[i, k], before[k] + c.alpha_c * (y - before[k]), places=14)
        self.assertEqual(cd.hist_n[i], 10)
        self.assertEqual(int((cd.hist_k[i][:10] == k).sum()), (3 if k == 0 else 6) + 1)
        self.assertEqual(cd.TK[i, k], t)

    def test_surprise_creates_a_later_cause_with_narrow_concentration(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(-0.5, c.lc_beta_first, [(0, "A"), (2, "A")])])
        _, det = cd.update(["x"], 0.0, "B", 5.0)                  # 何も起きない: 古い原因（−0.5）では説明できない
        i = cd.idx["x"]
        self.assertEqual(det["x"]["created"], 1)
        self.assertEqual(cd.BK[i, 1], c.lc_beta_later)
        self.assertEqual(cd.CK[i, 0], -0.5)                                                  # 古い原因は動かない（柔らかい更新ではない）
        self.assertEqual(int(cd.CRE[i]), 2)

    def test_full_set_reuses_the_smallest_mass_cause_and_drops_its_history(self):
        cd = rcond(lc_max_causes=2)
        c = cd.cfg
        place(cd, "x", [(-0.5, c.lc_beta_first, [(2 * i, "A") for i in range(10)]), (0.3, c.lc_beta_later, [(20, "A")])])
        i = cd.idx["x"]
        masses = cd._mass_row(i, 21.0)
        self.assertLess(masses[1], masses[0])
        _, det = cd.update(["x"], 0.0, "B", 21.0)                  # どちらにも説明できない目標 → 新しい原因 → 満員 → 質量最小（添字 1）を再利用
        self.assertTrue(det["x"]["reused"])
        self.assertEqual(det["x"]["created"], 1)
        self.assertEqual(int(cd.AL[i].sum()), 2)
        self.assertEqual(cd.CK[i, 0], -0.5)
        self.assertLess(abs(cd.CK[i, 1]), 0.05)
        self.assertEqual(int((cd.hist_k[i][:cd.hist_n[i]] == 1).sum()), 1)                   # 古い履歴は捨てられ、新しい試行だけ
        self.assertEqual(cd.BK[i, 1], c.lc_beta_later)

    def test_number_of_causes_never_exceeds_K_and_counts_match_history(self):
        rng = np.random.default_rng(0)
        for K in (2, 3, 5):
            cd = rcond(lc_max_causes=K)
            for t in range(200):
                cd.update(["x", "y"], float(rng.choice([-0.6, -0.3, 0.0, 0.3, 0.6])), ["A", "B", "C"][t % 3], float(t))
                for cue in ("x", "y"):
                    i = cd.idx[cue]
                    self.assertLessEqual(int(cd.AL[i].sum()), K)
            for cue in ("x", "y"):
                i = cd.idx[cue]
                n = cd.hist_n[i]
                for k in range(K):
                    inh = cd.hist_c[i][:n][cd.hist_k[i][:n] == k]
                    for j in range(len(cd.ctx_names)):
                        self.assertEqual(float(cd.CNT[i, k, j]), float((inh == j).sum()))       # 文脈の計数 = 履歴の文脈の数

    def test_multiple_cues_share_one_target_by_explaining_away(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(-0.2, c.lc_beta_first, [(0, "A"), (2, "A"), (4, "A")])])
        place(cd, "z", [(0.0, c.lc_beta_first, [(0, "A")])])
        rx, rz = cd.value_of("x", "A", 6.0), cd.value_of("z", "A", 6.0)
        _, det = cd.update(["x", "z"], -0.3, "A", 6.0)
        self.assertAlmostEqual(det["x"]["y_x"], -0.3 - rz, places=14)
        self.assertAlmostEqual(det["z"]["y_x"], -0.3 - rx, places=14)

    def test_dC_is_the_sum_of_cause_changes(self):
        cd = rcond()
        rng = np.random.default_rng(1)
        for t in range(40):
            before = cd.CK.copy()
            dC, det = cd.update(["x", "y"], float(rng.normal(0, 0.3)), ["A", "B"][t % 2], float(t))
            for c in ("x", "y"):
                i = cd.idx[c]
                b = before[i] if i < before.shape[0] else np.zeros(cd.K)
                self.assertAlmostEqual(dC[c], float((cd.CK[i] - b).sum()), places=13)
                self.assertTrue(np.allclose(det[c]["dCk"], cd.CK[i] - b, atol=1e-13))

    def test_state_roundtrip_keeps_history_and_concentrations(self):
        cd = rcond()
        rng = np.random.default_rng(2)
        for t in range(60):
            cd.update(["x", "y"], float(rng.normal(0, 0.3)), ["A", "B", "C"][t % 3], float(t))
        meta, arrays = cd.get_state()
        cd2 = rcond()
        cd2.set_state(meta, arrays)
        for name in cd.names:
            for t in (61.0, 100.0):
                for ctx in ("A", "B", "C", "D"):
                    self.assertEqual(cd.value_of(name, ctx, t), cd2.value_of(name, ctx, t))
        m2, a2 = cd2.get_state()
        self.assertTrue(deep_equal(meta, m2) and all(np.array_equal(arrays[k], a2[k]) for k in arrays))


class RevisedReinstatement(unittest.TestCase):
    def test_unsignaled_aversive_outcome_is_attributed_to_the_cause_whose_value_predicts_it(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(-0.2, c.lc_beta_first, [(0, "A"), (2, "A")]), (0.0, c.lc_beta_later, [(10, "B")] * 8)])
        place(cd, "other", [(0.05, c.lc_beta_first, [(1, "A")] * 20)])
        i = cd.idx["x"]
        before = cd.CK.copy()
        out = cd.reinstate([], 30.0, "B", -0.3)
        self.assertEqual(out, [("x", 0)])                                    # 価値 −0.2 の原因が −0.3 に最も近い。質量の大きい価値 0 の原因（消去）や他の手がかりは選ばない
        self.assertEqual(cd.hist_n[i], 11)
        self.assertEqual((cd.hist_t[i][10], cd.hist_k[i][10], cd.hist_c[i][10]), (30.0, 0, cd.ctx_idx["B"]))
        self.assertEqual(cd.CNT[i, 0, cd.ctx_idx["B"]], 1.0)
        self.assertTrue(np.array_equal(cd.CK, before))                       # 価値は更新しない
        self.assertEqual(cd.hist_n[cd.idx["other"]], 20)

    def test_no_cause_predicts_the_outcome_so_nothing_happens(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(0.0, c.lc_beta_first, [(0, "A")] * 5)])
        place(cd, "y", [(0.05, c.lc_beta_first, [(0, "A")] * 5)])            # 価値が 0 以上の原因は、「どの原因でもない」（価値 0）より嫌悪を説明しない
        self.assertEqual(cd.reinstate([], 10.0, "B", -0.3), [])
        place(cd, "z", [(-0.01, c.lc_beta_first, [(0, "A")] * 5)])           # わずかでも負なら、価値 0 より近い（|r − C| < |r|）ので選ばれる
        self.assertEqual(cd.reinstate([], 10.0, "B", -0.3), [("z", 0)])

    def test_present_cues_are_skipped(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "present", [(-0.3, c.lc_beta_first, [(0, "A")] * 5)])
        place(cd, "absent", [(-0.2, c.lc_beta_first, [(0, "A")] * 5)])
        self.assertEqual(cd.reinstate(["present"], 10.0, "B", -0.3), [("absent", 0)])

    def test_only_one_cause_is_reinforced_not_every_cue(self):
        cd = rcond()
        c = cd.cfg
        for name, v in (("a", -0.22), ("b", -0.2), ("c", -0.19)):
            place(cd, name, [(v, c.lc_beta_first, [(0, "A")] * 3)])
        out = cd.reinstate([], 10.0, "B", -0.3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "a")                                       # 最も近い
        self.assertEqual(sum(cd.hist_n[cd.idx[n]] for n in "abc"), 10)

    def test_ties_go_to_the_larger_prior_weight(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "small", [(-0.2, c.lc_beta_first, [(0, "A")] * 2)])
        place(cd, "large", [(-0.2, c.lc_beta_first, [(0, "A")] * 9)])
        self.assertEqual(cd.reinstate([], 5.0, "A", -0.3), [("large", 0)])

    def test_reinstatement_raises_the_response_in_the_test_context(self):
        cd = rcond()
        c = cd.cfg
        place(cd, "x", [(-0.2, c.lc_beta_first, [(2 * i, "A") for i in range(12)]), (0.0, c.lc_beta_later, [(24 + 2 * j, "B") for j in range(20)])])
        before = cd.value_of("x", "B", 66.0)
        cd.reinstate([], 66.0, "B", -0.3)
        cd.reinstate([], 67.0, "B", -0.3)
        self.assertLess(cd.value_of("x", "B", 67.0), before - 0.0)

    def test_single_cause_never_reinstates(self):
        cd = rcond(single_cause=True)
        cd.update(["x"], -0.5, "A", 1.0)
        self.assertEqual(cd.reinstate([], 10.0, "A", -0.5), [])


class ExplainAndTrace(unittest.TestCase):
    """revised の割当の記録（台帳の kind = "assignment" / "reinstate"）と、応答の内訳（explain / explain_response / cue プローブの breakdown）。SPEC-v3.3 6.2。
    spec_literal（既定）の割当の記録は ExplainAndTraceSpecLiteral。"""

    design = "revised"

    def _history(self, n_acq=6, n_ext=6):
        n = make(0, lc_design=self.design)
        for k in range(n_acq):
            s = f"a{k}"
            n.experience({"id": f"ac{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": s})
            n.experience({"id": f"au{k}", "context": "studio", "concepts": [], "participants": [],
                          "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": s})
        for k in range(n_ext):
            s = f"x{k}"
            n.experience({"id": f"xc{k}", "context": "cafe_chat", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": s})
            n.experience({"id": f"xn{k}", "context": "cafe_chat", "concepts": [], "participants": [], "outcomes": [], "session": s})
        return n

    def test_explain_decomposes_the_response_exactly(self):
        n = self._history()
        for ctx in ("studio", "cafe_chat", "garden", None):
            for t in (None, 30.0, 200.0):
                ex = n.explain_response("concept:praise_from_X", ctx, t)
                self.assertAlmostEqual(ex["response"], sum(c["contribution"] for c in ex["causes"]), places=14)
                self.assertAlmostEqual(sum(c["p"] for c in ex["causes"]) + ex["p_new"], 1.0, places=14)
                for c in ex["causes"]:
                    self.assertAlmostEqual(c["contribution"], c["p"] * c["value"], places=14)
                    self.assertAlmostEqual(c["weight"], c["time_term"] * c["context_term"] * c["count_term"], places=14)
                    self.assertAlmostEqual(c["p"], c["weight"] / (ex["total_weight"] + ex["alpha"]), places=14)
                self.assertAlmostEqual(ex["response"], n.cond.value_of("concept:praise_from_X", ctx, n._clock() if t is None else t), places=14)

    def test_explain_factors_are_the_stated_formulas(self):
        n = self._history()
        cd = n.cond
        i = cd.idx["concept:praise_from_X"]
        t = n._clock()
        ex = n.explain_response("concept:praise_from_X", "studio")
        for c in ex["causes"]:
            k = c["k"]
            times = cd.hist_t[i][:cd.hist_n[i]][cd.hist_k[i][:cd.hist_n[i]] == k]
            self.assertAlmostEqual(c["time_term"], float(((t - times + 1.0) ** (-cd.cfg.lc_decay_d)).sum()), places=12)
            self.assertEqual(c["n_trials"], times.size)
            j = cd.ctx_idx["studio"]
            S = ex["causes"][0]["n_contexts"]
            self.assertAlmostEqual(c["context_term"], (cd.CNT[i, k, j] + cd.BK[i, k]) / (cd.CNT[i, k].sum() + cd.BK[i, k] * S), places=12)
            self.assertEqual(c["n_in_context"], float(cd.CNT[i, k, j]))

    def test_explain_does_not_change_state(self):
        n = self._history()
        before = copy.deepcopy(n)
        n.explain_response("concept:praise_from_X", "garden", 99.0)
        n.probe({"cue": "concept:praise_from_X", "context": "studio"})
        ma, aa = n.get_state(); mb, ab = before.get_state()
        self.assertTrue(deep_equal(ma, mb) and all(np.array_equal(aa[k], ab[k]) for k in aa))

    def test_cue_probe_returns_the_breakdown(self):
        n = self._history()
        r = n.probe({"cue": "concept:praise_from_X", "context": "studio"})
        self.assertAlmostEqual(r["breakdown"]["response"], r["C"], places=14)
        self.assertGreaterEqual(len(r["breakdown"]["causes"]), 2)
        self.assertEqual(n.probe({"cue": "concept:never_seen"})["breakdown"]["causes"], [])

    def test_assignment_records_follow_the_update_rule(self):
        n = self._history()
        recs = n.assignment_trace("concept:praise_from_X")
        self.assertEqual(len(recs), 12)                                           # 手がかりの経験ごとに 1 件（獲得 6 + 消去 6）
        cfg = n.cfg
        seen_uids = set()
        for r in recs:
            K = len(r["weight"])
            alive = [u >= 0 for u in r["uid_before"]]
            tot = sum(r["weight"]) + r["alpha"]
            for q in range(K):
                self.assertAlmostEqual(r["p_prior"][q], r["weight"][q] / tot, places=12)
                if alive[q]:
                    self.assertAlmostEqual(r["weight"][q], r["time_term"][q] * r["context_term"][q] * r["count_term"][q], places=12)
            self.assertAlmostEqual(r["p_prior_new"], r["alpha"] / tot, places=12)
            # 事後 ∝ 事前 × 尤度（新しい原因は α × 尤度_新）
            z = sum(r["weight"][q] * r["likelihood"][q] for q in range(K)) + r["alpha"] * r["likelihood_new"]
            for q in range(K):
                self.assertAlmostEqual(r["post"][q], r["weight"][q] * r["likelihood"][q] / z, places=10)
            self.assertAlmostEqual(r["post_new"], r["alpha"] * r["likelihood_new"] / z, places=10)
            # MAP: 事後が最大（新しい原因を含む）
            best_old = max(r["post"])
            if r["created"] is not None:
                self.assertGreater(r["post_new"], best_old)
                self.assertNotIn(r["map_uid"], seen_uids)                         # 新しい原因の通し番号は初めて
                seen_uids.add(r["map_uid"])
            else:
                self.assertEqual(r["map"], int(np.argmax(r["post"])))
                self.assertGreaterEqual(best_old, r["post_new"])
            # 学習: MAP の原因だけが動く。ΔC = α_C (y_x − C_before)（新しい原因は C_before が 0 から）
            k = r["map"]
            c0 = 0.0 if r["created"] is not None else r["C_before"][k]
            self.assertAlmostEqual(r["C_after"][k], c0 + cfg.alpha_c * (r["y_x"] - c0), places=12)
            for q in range(K):
                if q != k:
                    self.assertEqual(r["C_after"][q], r["C_before"][q])

    def test_assignment_records_precede_the_experience_entry_and_reconstruct_the_values(self):
        n = self._history()
        self.assertEqual(n.ledger.entries[-1]["kind"], "experience")
        ent = n.ledger.entries
        for i, e in enumerate(ent):
            if e["kind"] == "assignment":
                nxt = next(x for x in ent[i + 1:] if x["kind"] == "experience")
                self.assertEqual(nxt["id"], e["after"])                           # 記録は、その更新を起こした経験の台帳の直前
        # 記録の学習後の C_k から、現在の原因の価値が復元できる（uid で追う）
        cd = n.cond
        i = cd.idx["concept:praise_from_X"]
        last = {}
        for r in n.assignment_trace("concept:praise_from_X"):
            last[r["map_uid"]] = r["C_after"][r["map"]]
        for k in range(cd.K):
            if cd.AL[i, k] > 0.5:
                self.assertAlmostEqual(cd.CK[i, k], last[int(cd.UID[i, k])], places=12)

    def test_reinstate_record_names_the_inferred_cause_and_its_posterior(self):
        n = self._history()
        n.experience({"id": "h1", "context": "cafe_chat", "concepts": [], "participants": [],
                      "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": "h"})
        rr = [e for e in n.ledger.entries if e["kind"] == "reinstate"][-1]
        self.assertEqual(rr["after"], "h1")
        self.assertLess(rr["r"], -0.05)
        self.assertEqual(rr["context"], "cafe_chat")
        ch = rr["chosen"]
        self.assertIsNotNone(ch)
        self.assertEqual(ch["cue"], "concept:praise_from_X")
        self.assertGreater(ch["C"], -1.0)
        self.assertTrue(0.0 < ch["posterior"] <= 1.0)
        self.assertEqual(ch["added_trial"], {"t": rr["t"], "context": "cafe_chat"})
        post_sum = rr["null_posterior"] + sum(x["posterior"] for x in rr["top"])
        self.assertLessEqual(post_sum, 1.0 + 1e-12)
        # 価値は更新されない（記録された原因の C は復位の前後で同じ）
        cd = n.cond
        i = cd.idx["concept:praise_from_X"]
        k = int(np.nonzero(cd.UID[i] == ch["uid"])[0][0])
        self.assertEqual(float(cd.CK[i, k]), ch["C"])

    def test_no_records_without_the_ledger(self):
        n = make(0, record_ledger=False, lc_design="revised")
        for k in range(3):
            n.experience({"id": f"c{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": f"s{k}"})
            n.experience({"id": f"u{k}", "context": "studio", "concepts": [], "participants": [],
                          "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": f"s{k}"})
        self.assertEqual(len(n.ledger.entries), 0)
        self.assertLess(n.cond.value_of("concept:praise_from_X", "studio"), 0.0)       # 学習は同じ（記録だけ省く）
        m = make(0, lc_design="revised")
        for k in range(3):
            m.experience({"id": f"c{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": f"s{k}"})
            m.experience({"id": f"u{k}", "context": "studio", "concepts": [], "participants": [],
                          "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": f"s{k}"})
        self.assertEqual(n.cond.value_of("concept:praise_from_X", "studio"), m.cond.value_of("concept:praise_from_X", "studio"))

    def test_explain_for_spec_literal_and_single_cause(self):
        n = make(0, lc_design="spec_literal")
        for k in range(4):
            n.experience({"id": f"c{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": f"s{k}"})
            n.experience({"id": f"u{k}", "context": "studio", "concepts": [], "participants": [],
                          "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": f"s{k}"})
        ex = n.explain_response("concept:praise_from_X", "studio")
        self.assertEqual(ex["design"], "spec_literal")
        self.assertAlmostEqual(ex["response"], n.cond.value_of("concept:praise_from_X", "studio", n._clock()), places=14)
        self.assertAlmostEqual(sum(c["p"] for c in ex["causes"]) + ex["p_new"], 1.0, places=14)
        s = make(0, single_cause=True)
        s.experience({"id": "c", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": "s"})
        s.experience({"id": "u", "context": "studio", "concepts": [], "participants": [], "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": "s"})
        ex = s.explain_response("concept:praise_from_X", "studio")
        self.assertEqual(ex["design"], "single_cause")
        self.assertEqual(len(ex["causes"]), 1)
        self.assertEqual(ex["response"], s.cond.value_of("concept:praise_from_X", "studio"))

    def test_records_survive_save_and_restore(self):
        n = self._history()
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        self.assertTrue(deep_equal(n.assignment_trace("concept:praise_from_X"), m.assignment_trace("concept:praise_from_X")))
        for ctx in ("studio", "cafe_chat", "garden"):
            self.assertTrue(deep_equal(n.explain_response("concept:praise_from_X", ctx), m.explain_response("concept:praise_from_X", ctx)))
        self.assertEqual(n.cond.next_uid, m.cond.next_uid)
        self.assertTrue(np.array_equal(n.cond.UID, m.cond.UID))


class ExplainAndTraceSpecLiteral(unittest.TestCase):
    """spec_literal（既定）の割当の記録（台帳の kind = "assignment" / "reinstate"）。v3.3 第 2 部で、既定を spec_literal に戻したときに、revised と同じ追跡ができるようにした
    （ユーザーの要求: 柔らかい事後の重みと、各原因の ΔC を記録する）。"""

    def _history(self, n_acq=6, n_ext=6):
        n = make(0)
        self.assertEqual(n.cfg.lc_design, "spec_literal")
        for k in range(n_acq):
            s = f"a{k}"
            n.experience({"id": f"ac{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": s})
            n.experience({"id": f"au{k}", "context": "studio", "concepts": [], "participants": [],
                          "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": s})
        for k in range(n_ext):
            s = f"x{k}"
            n.experience({"id": f"xc{k}", "context": "cafe_chat", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": s})
            n.experience({"id": f"xn{k}", "context": "cafe_chat", "concepts": [], "participants": [], "outcomes": [], "session": s})
        return n

    def test_one_record_per_cue_event_and_ordered_before_the_experience_entry(self):
        n = self._history()
        recs = n.assignment_trace("concept:praise_from_X")
        self.assertEqual(len([r for r in recs if r["kind"] == "assignment"]), 12)
        ent = n.ledger.entries
        for i, e in enumerate(ent):
            if e["kind"] == "assignment":
                nxt = next(x for x in ent[i + 1:] if x["kind"] == "experience")
                self.assertEqual(nxt["id"], e["after"])
                self.assertEqual(e["design"], "spec_literal")

    def test_prior_breakdown_and_soft_posterior_follow_the_stated_formulas(self):
        n = self._history()
        for r in [x for x in n.assignment_trace("concept:praise_from_X") if x["kind"] == "assignment"]:
            K = len(r["weight"])
            alive = [u >= 0 for u in r["uid_before"]]
            tot = sum(r["weight"]) + r["alpha"]
            for q in range(K):
                self.assertAlmostEqual(r["p_prior"][q], r["weight"][q] / tot, places=12)
                if alive[q]:
                    self.assertAlmostEqual(r["weight"][q], r["time_term"][q] * r["context_term"][q] * r["count_term"][q], places=12)
            self.assertAlmostEqual(r["p_prior_new"], r["alpha"] / tot, places=12)
            z = sum(r["weight"][q] * r["likelihood"][q] for q in range(K)) + r["alpha"] * r["likelihood_new"]
            for q in range(K):
                self.assertAlmostEqual(r["post_raw"][q], r["weight"][q] * r["likelihood"][q] / z, places=10)
            self.assertAlmostEqual(r["post_new"], r["alpha"] * r["likelihood_new"] / z, places=10)
            self.assertAlmostEqual(sum(r["post_raw"]) + r["post_new"], 1.0, places=12)

    def test_update_weights_are_the_soft_posterior_and_every_cause_moves_by_its_own_share(self):
        n = self._history()
        cfg = n.cfg
        created_seen = 0
        for r in [x for x in n.assignment_trace("concept:praise_from_X") if x["kind"] == "assignment"]:
            K = len(r["weight"])
            used = list(r["post_raw"])
            c0 = list(r["C_before"])
            if r["created"] is not None:                                  # 新しい原因の事後が最大 → その場所の事後を新しい原因のものに置き換えて正規化
                used[r["created"]] = r["post_new"]
                c0[r["created"]] = 0.0
                created_seen += 1
                tot = sum(used)
                used = [u / tot for u in used]
            else:                                                         # 作らないときは、新しい原因の事後の分（post_new）を除いた柔らかい事後のまま更新に使う
                self.assertAlmostEqual(sum(used) + r["post_new"], 1.0, places=12)
            for q in range(K):
                self.assertAlmostEqual(r["post_used"][q], used[q], places=12)
            for q in range(K):
                alive_after = r["uid_before"][q] >= 0 or q == r["created"]
                expect = c0[q] + (cfg.alpha_c * used[q] * (r["y_x"] - c0[q]) if alive_after else 0.0)
                self.assertAlmostEqual(r["C_after"][q], expect, places=12)
                self.assertAlmostEqual(r["dC_k"][q], r["C_after"][q] - c0[q], places=12)        # 各原因の ΔC（新しい原因は 0 から）
            self.assertEqual(r["map"], int(np.argmax(r["post_used"])))
        self.assertGreaterEqual(created_seen, 1)

    def test_the_last_record_gives_the_current_cause_values_and_uids_are_distinct(self):
        n = self._history()
        cd = n.cond
        i = cd.idx["concept:praise_from_X"]
        last = [x for x in n.assignment_trace("concept:praise_from_X") if x["kind"] == "assignment"][-1]
        self.assertTrue(np.allclose(last["C_after"], cd.CK[i], atol=1e-14))
        uids = [int(cd.UID[i, k]) for k in range(cd.K) if cd.AL[i, k] > 0.5]
        self.assertEqual(len(uids), len(set(uids)))
        self.assertTrue(all(u >= 0 for u in uids))
        self.assertEqual(cd.next_uid, max(int(cd.UID.max()) + 1, cd.next_uid))

    def test_extinction_updates_move_the_acquisition_cause_which_is_the_known_failure_of_the_literal_formula(self):
        """B.1 の切り分け（消去の学習が古い原因を壊す）を、記録で確かめる: 消去中に獲得の原因が受けた更新の重みの総和が大きく、C が 0 へ動く。"""
        n = self._history(n_acq=12, n_ext=20)
        recs = [x for x in n.assignment_trace("concept:praise_from_X") if x["kind"] == "assignment"]
        acq, ext = recs[:12], recs[12:]
        cnt = {}
        for r in acq:
            cnt[r["map_uid"]] = cnt.get(r["map_uid"], 0) + 1
        uid = max(cnt, key=lambda u: cnt[u])
        mass = sum(r["post_used"][r["uid_before"].index(uid)] for r in ext if uid in r["uid_before"])
        dc = sum(r["dC_k"][r["uid_before"].index(uid)] for r in ext if uid in r["uid_before"])
        self.assertGreater(mass, 5.0)                           # 20 試行のうち大半の重みを獲得の原因が受けている
        self.assertGreater(dc, 0.05)                            # 負の価値が 0 へ戻る方向に動く（獲得の記憶が削られる）

    def test_reinstate_record_lists_the_refreshed_causes(self):
        n = self._history()
        n.experience({"id": "h1", "context": "cafe_chat", "concepts": [], "participants": [],
                      "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": "h"})
        rr = [e for e in n.ledger.entries if e["kind"] == "reinstate"][-1]
        self.assertEqual(rr["design"], "spec_literal")
        self.assertEqual(rr["after"], "h1")
        self.assertEqual(rr["context"], "cafe_chat")
        self.assertIsNone(rr["chosen"])
        hit = [x for x in rr["refreshed"] if x["cue"] == "concept:praise_from_X"]
        self.assertTrue(hit)
        cd = n.cond
        i = cd.idx["concept:praise_from_X"]
        for x in hit:
            self.assertLess(x["C"], -cd.cfg.lc_neg_threshold)
            self.assertEqual(x["t_after"], rr["t"])
            self.assertEqual(float(cd.TK[i, x["k"]]), rr["t"])           # 時刻が現在に更新された
            self.assertLess(x["t_before"], rr["t"])
        self.assertTrue(n.assignment_trace("concept:praise_from_X")[-1] is rr)       # assignment_trace が復位の記録も返す

    def test_explain_response_and_cue_probe_breakdown_work_and_decompose_exactly(self):
        n = self._history()
        for ctx in ("studio", "cafe_chat", "garden", None):
            ex = n.explain_response("concept:praise_from_X", ctx)
            self.assertEqual(ex["design"], "spec_literal")
            self.assertAlmostEqual(ex["response"], sum(c["contribution"] for c in ex["causes"]), places=14)
            self.assertAlmostEqual(sum(c["p"] for c in ex["causes"]) + ex["p_new"], 1.0, places=14)
            for c in ex["causes"]:
                self.assertAlmostEqual(c["weight"], c["time_term"] * c["context_term"] * c["count_term"], places=14)
                self.assertGreaterEqual(c["uid"], 0)
            r = n.probe({"cue": "concept:praise_from_X", "context": ctx} if ctx else {"cue": "concept:praise_from_X"})
            self.assertAlmostEqual(r["breakdown"]["response"], r["C"], places=14)

    def test_formatting_handles_every_record(self):
        from npnv3.explain import format_assignment, format_explanation
        n = self._history()
        n.experience({"id": "h1", "context": "cafe_chat", "concepts": [], "participants": [],
                      "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": "h"})
        for r in n.assignment_trace("concept:praise_from_X"):
            self.assertIsInstance(format_assignment(r), str)
        self.assertTrue(format_explanation(n.explain_response("concept:praise_from_X", "studio")))

    def test_records_are_deterministic_and_survive_save_and_restore(self):
        a, b = self._history(), self._history()
        self.assertTrue(deep_equal(a.assignment_trace("concept:praise_from_X"), b.assignment_trace("concept:praise_from_X")))
        with tempfile.TemporaryDirectory() as d:
            a.save(d)
            m = NPN.load(d)
        self.assertTrue(deep_equal(a.assignment_trace("concept:praise_from_X"), m.assignment_trace("concept:praise_from_X")))
        for ctx in ("studio", "cafe_chat", "garden"):
            self.assertTrue(deep_equal(a.explain_response("concept:praise_from_X", ctx), m.explain_response("concept:praise_from_X", ctx)))

    def test_recording_does_not_change_learning(self):
        a = make(0, record_ledger=False)
        b = self._history()
        for k in range(6):
            a.experience({"id": f"ac{k}", "context": "studio", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": f"a{k}"})
            a.experience({"id": f"au{k}", "context": "studio", "concepts": [], "participants": [], "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": f"a{k}"})
        for k in range(6):
            a.experience({"id": f"xc{k}", "context": "cafe_chat", "concepts": ["praise_from_X"], "participants": [], "outcomes": [], "session": f"x{k}"})
            a.experience({"id": f"xn{k}", "context": "cafe_chat", "concepts": [], "participants": [], "outcomes": [], "session": f"x{k}"})
        self.assertEqual(len(a.ledger.entries), 0)
        for ctx in ("studio", "cafe_chat"):
            self.assertEqual(a.cond.value_of("concept:praise_from_X", ctx), b.cond.value_of("concept:praise_from_X", ctx))


class SpecLiteralIsSelectable(unittest.TestCase):
    def test_spec_literal_npn_runs_and_roundtrips(self):
        n = make(0, lc_design="spec_literal")
        for e in mh.generate_praise_history("H1", 3, 12):
            n.experience(e)
        self.assertEqual(sum(h for h in n.cond.hist_n), 0)                     # 履歴は使わない
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        self.assertEqual(m.cfg.lc_design, "spec_literal")
        for cue in n.cond.names:
            self.assertEqual(n.cond.value_of(cue, "studio"), m.cond.value_of(cue, "studio"))

    def test_spec_literal_is_the_default_and_revised_is_selectable(self):
        self.assertEqual(Config().lc_design, "spec_literal")
        self.assertEqual(make(0).cond.cfg.lc_design, "spec_literal")
        n = make(0, lc_design="revised")
        for e in mh.generate_praise_history("H1", 3, 12):
            n.experience(e)
        self.assertGreater(sum(h for h in n.cond.hist_n), 0)                   # revised は試行の履歴を使う
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        self.assertEqual(m.cfg.lc_design, "revised")

    def test_invalid_design_is_refused(self):
        with self.assertRaises(ValueError):
            Config(lc_design="nonsense")


class IntegrationInNPN(unittest.TestCase):
    def test_ledger_cause_totals_equal_cause_values(self):
        n = make(0)
        for e in mh.generate_praise_history("H1", 3, 12):
            n.experience(e)
        tot = {}
        for en in n.ledger.entries:
            if en["kind"] != "experience":
                continue
            for cue, d in en["td"]["lc"].items():
                v = np.array(d["dCk"])
                tot[cue] = tot[cue] + v if cue in tot else v
        self.assertTrue(len(tot) > 3)
        for cue, v in tot.items():
            self.assertTrue(np.allclose(n.cond.CK[n.cond.idx[cue]], v, atol=1e-12), cue)

    def test_appraise_and_probe_do_not_change_conditioning_state(self):
        n = persona("K", 1)
        before = copy.deepcopy(n)
        n.appraise(load_jsonl("R.jsonl")[3])
        n.probe({"cue": "type:harm", "context": "music_session"})
        n.probe({"context": "music_session", "concepts": ["collaboration"], "participants": [{"id": "p", "class": "collaborator"}]})
        for key in ("CK", "TK", "AL", "CNT"):
            self.assertTrue(np.array_equal(getattr(n.cond, key), getattr(before.cond, key)), key)
        self.assertEqual(n.cond.ctx_names, before.cond.ctx_names)

    def test_clock_counts_learned_and_avoided_experiences(self):
        n = make(0)
        self.assertEqual(n._clock(), 0.0)
        e = {"id": "o1", "context": "studio", "concepts": ["music"], "participants": [],
             "outcomes": [{"type": "praise", "actor": "world", "magnitude": 0.5}], "optional": True, "announced": ["praise"]}
        n.cond.set_single("type:praise", -2.0)
        learned = avoided = 0
        for i in range(30):
            ee = copy.deepcopy(e); ee["id"] = f"o{i}"
            r = n.experience(ee)
            learned += int(r["approached"]); avoided += int(not r["approached"])
        self.assertGreater(avoided, 0)
        self.assertEqual(n.n_experiences, learned)
        self.assertEqual(n.n_avoided, avoided)
        self.assertEqual(n._clock(), float(learned + avoided))

    def test_pending_carries_context_and_time(self):
        n = make(0)
        n.experience({"id": "a", "context": "cafe_chat", "concepts": ["conversation"], "participants": [], "outcomes": [], "session": "s"})
        self.assertEqual(n.pending["context"], "cafe_chat")
        self.assertEqual(n.pending["t"], 0.0)
        n.experience({"id": "b", "context": "studio", "concepts": [], "participants": [], "outcomes": [], "session": "s"})
        self.assertEqual(n.pending["t"], 1.0)

    def test_save_restore_keeps_causes_counts_times_and_continues_identically(self):
        n = persona("K", 2)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        for key in ("CK", "TK", "AL", "CNT"):
            self.assertTrue(np.array_equal(getattr(n.cond, key), getattr(m.cond, key)), key)
        self.assertEqual(n.cond.ctx_names, m.cond.ctx_names)
        rest = load_jsonl("R.jsonl")
        self.assertTrue(deep_equal(feed(n, copy.deepcopy(rest)), feed(m, copy.deepcopy(rest))))
        self.assertEqual(n.cond.value_of("type:harm", "music_session"), m.cond.value_of("type:harm", "music_session"))

    def test_loading_with_a_different_K_is_refused(self):
        n = persona("K", 0)
        meta, arrays = n.get_state()
        m = make(0, single_cause=True)
        with self.assertRaises(ValueError):
            m.cond.set_state(meta["conditioning"], arrays)

    def test_no_conditioning_leaves_no_state(self):
        n = make(0, no_conditioning=True)
        feed(n, load_jsonl("K.jsonl"))
        self.assertEqual(len(n.cond.names), 0)
        self.assertEqual(n.cond.CNT.shape[0], 0)

    def test_cue_probe_reports_causes_and_takes_a_context(self):
        n = persona("K", 0)
        r = n.probe({"cue": "concept:collaboration", "context": "music_session"})
        self.assertEqual(r["context"], "music_session")
        self.assertGreaterEqual(r["n_causes"], 1)
        self.assertTrue(r["known"])
        self.assertNotEqual(n.probe({"cue": "concept:collaboration"})["C"], 0.0)
        self.assertFalse(n.probe({"cue": "concept:never_seen"})["known"])


if __name__ == "__main__":
    unittest.main()
