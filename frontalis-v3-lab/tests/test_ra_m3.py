"""v3.4 Stage 3 の単体テスト: M3（訂正の主観的評価 → 双方向で有界の τ → 解釈の重み → 次の学習）、訂正を時間の出来事にする変更、試験用の入口（SPEC-v3.4 第 4.1・4.4・4.5・6 節、第 8 節 RA3・RA8(c)・RA10 H3・H7〜H9）。

乱数種: 開発用の 3490〜3499 だけ（使うのは 3490〜3492）。評価 3400〜3439、変異 3450〜3459、診断 3460〜3479、予約 3500〜3539 は使わない（先頭で確認する）。
ここで新しく作る閾値はない（1e-9、1e-12、SPEC の値のまま）。**効果の向きや大きさは検査しない**（それは観察であり、判定ではない）。検査するのは、規則が自分の定義どおりに働くこと（F2）、
有界・有限（F1）、台帳から再構成できること（F3）、当時の記録が書き換わらないこと（F4）、経路の分離である。

  M3 の式     own_i・罪悪感・後悔・自己正当化を台帳の値から再構成して一致、1 回の変化 ≤ η_τ、符号の和を [−1, 1] に切る（S3-3）
  負の対照     報告の誤りは own_i = 0 で τ が動かない（RA3(5)）。曖昧でも、解釈が訂正後の型と同じなら誤読ではない
  τ の更新     訂正の出来事でだけ変わる（H3/H9）、境界 [1/3, 3]、境界でも反対向きに押せば動く（RA8(c)）、上下どちらにも動きうる
  解釈の重み   w_I ∝ q_LLM P^τ（τ = 1 で v3.3 と同じ式）、鍵（類 × 文脈）の特異性、v33 では τ を使わない、次の学習の教師が変わる
  f_c・安堵    気分にだけ入る: clamp-mood で後続の学習の状態がビット単位で同じ（RA3(4)）、H8（気分の摂動の比 ≤ 1）
  条件の入口   clamp-τ / clamp-mood / clamp-both / τ-yoke / A-τ-off / A-clock-off が、SPEC の言う量だけを変える。RA3(3)。val-swap・hm-eq・swap-G（G5 を含む）
  時間の出来事 v34 は訂正ごとに時計 +1・気分更新・correction_experience を追記、v33 は変わらない
  追跡         trace_loop が台帳だけから κ・τ の最終値を再構成し、状態と一致する
"""
import copy
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

from helpers import ROOT, cos  # noqa: F401

sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "tests"))
import make_histories_v34 as H  # noqa: E402
import ra_protocol as RP  # noqa: E402
from npnv3 import NPN  # noqa: E402
from npnv3.config import Config  # noqa: E402
from npnv3.innate import N_NEEDS, NEEDS  # noqa: E402
from npnv3.npn import _digest  # noqa: E402
from npnv3.trace import tau_trajectory, trace_loop  # noqa: E402

DEV_SEEDS = tuple(range(3490, 3500))
FORBIDDEN = set(range(3400, 3440)) | set(range(3450, 3460)) | set(range(3500, 3540))
USED_SEEDS = (3490, 3491, 3492)
assert set(USED_SEEDS) <= set(DEV_SEEDS) and not (set(USED_SEEDS) & FORBIDDEN)

TOL = 1e-9
KEY = "collaborator|project_meeting"          # A（類 collaborator）の曖昧な出来事の τ の鍵
KEY_OTHER = "collaborator|music_session"      # 別の文脈（r9 の対照）
ETA = Config().eta_tau
_CACHE = {}


def need(d):
    return np.array([d[k] for k in NEEDS], dtype=np.float64)


def feed(npn, seq):
    RP.feed_items(npn, seq)


def build(seed, bg="betray", src="none", err="interp", n=4, mode="v34", variant="none", upto="all", extra_cfg=None, kappa_push=None):
    """経験史を与えた人格（共有・書き込み禁止）。variant は COR の直前から効く（ra_protocol の cor_overrides と同じ）。"""
    key = (seed, bg, src, err, n, mode, variant, upto, tuple(sorted((extra_cfg or {}).items())), kappa_push)
    if key not in _CACHE:
        _CACHE[key] = build_fresh(seed, bg, src, err, n, mode, variant, upto, extra_cfg, kappa_push)
    return _CACHE[key]


def build_fresh(seed, bg="betray", src="none", err="interp", n=4, mode="v34", variant="none", upto="all", extra_cfg=None, kappa_push=None):
    h = H.make_history(seed, bg=bg, src=src, err=err, n=n)
    npn = NPN(seed=seed, reappraisal_mode=mode, **(extra_cfg or {}))
    feed(npn, h["pre"] + h["mis"] + h["extra"])
    if kappa_push is not None:                                    # 試験用の入口 verify_source で κ̄ を動かす（M2 の規則そのもの）
        for _ in range(30):
            npn.verify_source(H.SOURCE, kappa_push)
    if upto == "mis":
        return npn
    npn.cfg.m3_variant = variant
    feed(npn, h["cor"])
    if upto == "all":
        feed(npn, h["fol"])
    return npn


def led(npn, kind):
    return [e for e in npn.ledger.entries if e["kind"] == kind]


def snapshot(npn):
    meta, arrays = npn.get_state()
    return json.dumps(meta, sort_keys=True, default=str), {k: np.array(v) for k, v in arrays.items()}


def same_state(a, b):
    ma, aa = snapshot(a)
    mb, ab = snapshot(b)
    return ma == mb and set(aa) == set(ab) and all(np.array_equal(aa[k], ab[k]) for k in aa)


LEARN_META = ("belief", "value", "concern", "conditioning", "justification", "justification_sources")


def learning_equal(a, b, with_kappa_tau=True):
    """学習の状態（W・Σ・信念・C・a・h・J）がビット単位で同じか。with_kappa_tau なら κ の Beta と τ も含める。気分・台帳・時計は含めない。"""
    ma, aa = a.get_state()
    mb, ab = b.get_state()
    if set(aa) != set(ab) or not all(np.array_equal(aa[k], ab[k]) for k in aa):
        return False
    for k in LEARN_META:
        if json.dumps(ma[k], sort_keys=True, default=str) != json.dumps(mb[k], sort_keys=True, default=str):
            return False
    if with_kappa_tau:
        return a.src_beta == b.src_beta and a.log_tau == b.log_tau
    return a.src_beta == b.src_beta


def amb_weights(npn, actor, ctx, concept, q=0.5):
    ev = {"id": "amb_t", "context": ctx, "concepts": [concept], "participants": [{"id": actor, "class": "collaborator"}],
          "outcomes": [{"interpretations": {"credit_taken": q, "credit_shared": 1.0 - q}, "actor": actor, "magnitude": 0.8}], "optional": False}
    ap = npn.appraise(ev)
    return ap["interpretations"][0]["weights"], ev


# ======================================================================
class M3Evaluation(unittest.TestCase):
    """訂正の主観的評価（f_c・安堵・own_i・罪悪感・後悔・自己正当化）が SPEC の式どおりか（F2）。"""

    def test_guilt_regret_defend_are_reconstructed_from_the_ledger(self):
        s_f = Config().s_f
        for seed, bg in ((3490, "betray"), (3491, "neutral"), (3492, "trust")):
            n = build(seed, bg=bg, err="interp", upto="cor")
            cors = {e["id"]: e for e in led(n, "correction")}
            for ce in led(n, "correction_experience"):
                c = cors[ce["id"]]
                like = ce["like"]
                guilt = ce["own"] * math.tanh(sum(max(l1 - l0, 0.0) for l0, l1 in like.values() if l0 < 0.0) / s_f)
                regret = ce["own"] * math.tanh(sum(max(l0 - l1, 0.0) for l0, l1 in like.values() if l0 > 0.0) / s_f)
                self.assertAlmostEqual(ce["guilt"], guilt, places=12)
                self.assertAlmostEqual(ce["regret"], regret, places=12)
                # 自己正当化: own (1 − g) tanh(Σ_e Σ_k a_k |D_e,k| / s_f)。a は台帳の訂正の項に残らないので、現在の a を使う（訂正は a を変えない）
                m1 = c["m1"]
                a = n.concern.a
                dmag = sum(float(a @ np.abs(need(m1["D"][pid]))) for pid in m1["D"])
                self.assertAlmostEqual(ce["defend"], ce["own"] * (1.0 - m1["g"]) * math.tanh(dmag / s_f), places=12)
                self.assertAlmostEqual(ce["g"], m1["g"], places=15)
                # 台帳の emotions にも同じ値（v33 の読み出し guilt を置き換える）
                self.assertEqual(c["emotions"]["guilt"], ce["guilt"])
                self.assertEqual((c["emotions"]["regret"], c["emotions"]["defend"]), (ce["regret"], ce["defend"]))
                self.assertTrue(0.0 <= ce["own"] <= 1.0 and ce["guilt"] >= 0.0 and ce["regret"] >= 0.0 and 0.0 <= ce["defend"] < 1.0)

    def test_f_c_is_tanh_of_Q_gain_over_sf_plus_beta_mood_and_relief_follows(self):
        cfg = Config()
        n = build_fresh(3490, "betray", "none", "interp", 4, "v34", upto="mis")
        h = H.make_history(3490, "betray", "none", "interp", 4)
        for cor in h["cor"]:
            m0 = n.mood
            n.experience(copy.deepcopy(cor))
            c, ce = led(n, "correction")[-1], led(n, "correction_experience")[-1]
            q0, q1 = c["Q"]
            self.assertAlmostEqual(ce["f_c"], math.tanh((q1 - q0) / cfg.s_f + cfg.beta_mood * m0), places=12)
            self.assertEqual(ce["relief"], min(1.0, ce["f_c"]) if (ce["f_c"] > 0.0 and q0 < 0.0) else 0.0)
            self.assertAlmostEqual(ce["delta_c"], ce["f_c"] - math.tanh(q0 / cfg.s_f), places=12)
            self.assertAlmostEqual(n.mood, 0.8 * m0 + 0.2 * ce["delta_c"], places=14)         # m ← 0.8 m + 0.2 δ_c

    def test_own_is_the_fraction_of_changed_outcomes_the_person_misread(self):
        for bg in H.BGS:
            n = build(3490, bg=bg, err="interp", upto="cor")
            for ce in led(n, "correction_experience"):
                units = ce["units"]
                self.assertEqual(len(units), 1)                          # 変わった結果は曖昧な credit_taken/credit_shared の 1 つ（achievement は同じ）
                u = units[0]
                self.assertTrue(u["ambiguous"])
                self.assertEqual(u["misread"], u["prev_type"] != u["new_type"])
                self.assertEqual(ce["own"], 1.0 if u["misread"] else 0.0)
                self.assertEqual(u["new_type"], "credit_shared")
        # 同じ訂正でも、人格（背景）によって誤読だったかどうかが違う: 解釈を決めたのは各人格自身の信念
        owns = {bg: [e["own"] for e in led(build(3490, bg=bg, err="interp", upto="cor"), "correction_experience")] for bg in H.BGS}
        self.assertTrue(len({tuple(v) for v in owns.values()}) > 1, owns)

    def test_negative_control_report_errors_that_the_actor_did_not_cause_give_own_zero(self):
        """RA3(5)（単体版）: 曖昧でない誤った報告の訂正は、好悪が動いても own = 0 → 罪悪感・後悔・自己正当化 = 0 → τ は厳密に動かない。"""
        for err in ("report", "report-x"):
            for bg in ("betray", "trust"):
                n = build(3490, bg=bg, err=err, upto="all")
                self.assertEqual(n.log_tau, {})
                cs = led(n, "correction_experience")
                self.assertEqual(len(cs), 4)
                for ce in cs:
                    self.assertEqual((ce["own"], ce["guilt"], ce["regret"], ce["defend"], ce["tau_step"]), (0.0, 0.0, 0.0, 0.0, 0.0))
                    self.assertEqual(ce["log_tau"], {})
                self.assertTrue(any(abs(l1 - l0) > 1e-6 for ce in cs for l0, l1 in ce["like"].values()))     # 好悪は動いている（対照が空でない）
        # 自分の誤読ではなく、曖昧でない誤りを、人格自身の読みが決めた曖昧な結果の訂正と同じ好悪の動きの中で比べる
        n = NPN(seed=3490, reappraisal_mode="v34")
        feed(n, H.make_bg(3490, "betray"))
        part = [{"id": "A", "class": "collaborator"}, {"id": "O", "class": "collaborator"}]
        rep = {"id": "r1", "context": H.CTX_MIS, "concepts": [H.CON_MIS], "participants": part,
               "outcomes": [H._o("credit_taken", "A", 0.8), H._o("achievement", "self", 0.4)]}
        cor = {"id": "r1x", "kind": "correction", "refers_to": "r1", "source": "witness",
               "corrected_outcomes": [H._o("credit_taken", "O", 0.8), H._o("achievement", "self", 0.4)]}
        feed(n, [rep, cor])
        ce = led(n, "correction_experience")[-1]
        self.assertEqual((ce["own"], ce["guilt"], ce["regret"], ce["defend"]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(n.log_tau, {})

    def test_ambiguous_outcome_that_the_person_read_correctly_is_not_a_misreading(self):
        """人格自身の解釈の最大の型が訂正後の型と同じなら、曖昧でも own = 0（例: 解釈が credit_shared に傾いた人格）。"""
        n = build_fresh(3490, "trust", "none", "interp", 4, "v34", upto="mis")
        rec = n.exp_log[[r["id"] for r in n.exp_log].index("j0")]
        w = np.array(rec["w_out"][0])
        # 解釈の重みを credit_shared 側に傾けた事実として台帳の外から確かめる: 訂正後の型は credit_shared
        n2 = copy.deepcopy(n)
        rec2 = n2.exp_log[[r["id"] for r in n2.exp_log].index("j0")]
        from npnv3.innate import TYPE_INDEX
        k_shared, k_taken = TYPE_INDEX["credit_shared"], TYPE_INDEX["credit_taken"]
        rec2["w_out"][0] = [0.0] * len(rec2["w_out"][0])
        rec2["w_out"][0][k_shared] = 1.0
        n2.experience(copy.deepcopy(H.make_history(3490, "trust", "none", "interp", 4)["cor"][0]))
        ce = led(n2, "correction_experience")[-1]
        self.assertFalse(ce["units"][0]["misread"])
        self.assertEqual((ce["own"], ce["guilt"], ce["regret"], ce["defend"]), (0.0, 0.0, 0.0, 0.0))
        rec3 = copy.deepcopy(n)
        r3 = rec3.exp_log[[r["id"] for r in rec3.exp_log].index("j0")]
        r3["w_out"][0] = [0.0] * len(r3["w_out"][0])
        r3["w_out"][0][k_taken] = 1.0
        rec3.experience(copy.deepcopy(H.make_history(3490, "trust", "none", "interp", 4)["cor"][0]))
        self.assertTrue(led(rec3, "correction_experience")[-1]["units"][0]["misread"])
        self.assertEqual(led(rec3, "correction_experience")[-1]["own"], 1.0)
        self.assertTrue(np.isfinite(w).all())

    def test_f_c_and_relief_do_not_enter_tau_or_guilt(self):
        """τ の更新は defend − guilt − regret だけから決まる（気分も f_c も使わない）: 気分を切っても（clamp-mood）τ の軌跡は C-v34 と同じ。"""
        a = build(3490, bg="betray", err="interp", upto="cor")
        b = build(3490, bg="betray", err="interp", variant="clamp_mood", upto="cor")
        self.assertEqual(a.log_tau, b.log_tau)
        for ca, cb in zip(led(a, "correction_experience"), led(b, "correction_experience")):
            for k in ("own", "guilt", "regret", "defend", "tau_step", "log_tau"):
                self.assertEqual(ca[k], cb[k], k)
        self.assertNotEqual([c["f_c"] for c in led(a, "correction_experience")], [c["f_c"] for c in led(b, "correction_experience")])


# ======================================================================
class TauUpdate(unittest.TestCase):
    def test_H3_H9_tau_changes_only_at_correction_events_and_by_at_most_eta(self):
        h = H.make_history(3490, "betray", "reliable", "interp", 4)
        n = NPN(seed=3490, reappraisal_mode="v34")
        seq = h["pre"] + h["mis"] + h["extra"] + h["cor"] + h["fol"]
        changed_at = []
        for it in seq:
            before = dict(n.log_tau)
            n_ce = len(led(n, "correction_experience"))
            feed(n, [it])
            if it.get("kind") == "correction":
                self.assertEqual(len(led(n, "correction_experience")), n_ce + 1)
                ce = led(n, "correction_experience")[-1]
                for k in set(before) | set(n.log_tau):
                    self.assertLessEqual(abs(n.log_tau.get(k, 0.0) - before.get(k, 0.0)), ETA + 1e-15, (it["id"], k))      # H9
                self.assertEqual({k: [before.get(k, 0.0), n.log_tau[k]] for k in ce["log_tau"]}, {k: v for k, v in ce["log_tau"].items()})
                if n.log_tau != before:
                    changed_at.append(it["id"])
            else:
                self.assertEqual(n.log_tau, before, it["id"])                        # H3: 訂正以外では動かない（検証・経験・刻み）
                self.assertEqual(len(led(n, "correction_experience")), n_ce)
        self.assertTrue(changed_at)
        # 台帳の和 = 状態（H2）
        lt = {}
        for ce in led(n, "correction_experience"):
            for k, (b, a) in ce["log_tau"].items():
                lt[k] = lt.get(k, 0.0) + (a - b)
        self.assertEqual(set(lt), set(n.log_tau))
        for k in lt:
            self.assertAlmostEqual(lt[k], n.log_tau[k], places=12)

    def test_tau_stays_in_bounds_and_moves_off_the_boundary_when_pushed_back_RA8c(self):
        lim = math.log(3.0)
        # 境界: 強く押し続けても [−log 3, log 3] を出ない（η_τ を大きくして短い履歴で境界に当てる。符号つきの和は [−1, 1] に切られる）
        for sign in (+1, -1):
            n = NPN(seed=3490, reappraisal_mode="v34", eta_tau=50.0)
            h = H.make_history(3490, "betray", "none", "interp", 4)
            feed(n, h["pre"] + h["mis"])
            n._tau_step = lambda d, g, r, e, s=sign: s * 50.0                          # 押す向きを固定する（規則の clip の検査）
            feed(n, h["cor"])
            self.assertAlmostEqual(n.log_tau[KEY], sign * lim, places=14)
            self.assertTrue(abs(n.log_tau[KEY]) <= lim + 1e-15)
        # 境界にある τ を反対向きに押す訂正が来ると動く（RA8(c)）。同じ向きなら境界に留まる
        base = NPN(seed=3490, reappraisal_mode="v34", eta_tau=0.5)
        h = H.make_history(3490, "betray", "none", "interp", 4)
        feed(base, h["pre"] + h["mis"])
        base.log_tau[KEY] = lim
        for push, expect_moves in ((-0.5, True), (+0.5, False)):
            n = copy.deepcopy(base)
            n._tau_step = lambda d, g, r, e, p=push: p
            feed(n, h["cor"][:1])
            moved = n.log_tau[KEY] != lim
            self.assertEqual(moved, expect_moves)
            if expect_moves:
                self.assertAlmostEqual(n.log_tau[KEY], lim - 0.5, places=14)
        # 別のキーは動かない
        self.assertEqual(set(n.log_tau), {KEY})

    def test_tau_step_is_clipped_so_that_the_per_correction_change_never_exceeds_eta(self):
        f = NPN._tau_step
        self.assertEqual(f(0.0, 0.0, 0.0, ETA), 0.0)
        self.assertEqual(f(0.9, 0.0, 0.0, ETA), ETA * 0.9)
        self.assertEqual(f(0.0, 0.9, 0.9, ETA), -ETA)             # SPEC の式 (defend − guilt − regret) は −1.8 になりうる → −1 に切る（S3-3）
        self.assertEqual(f(1.0, 0.0, 0.0, ETA), ETA)
        for d, g, r in ((0.5, 0.2, 0.1), (0.0, 0.99, 0.99), (0.99, 0.0, 0.0)):
            self.assertLessEqual(abs(f(d, g, r, ETA)), ETA)

    def test_tau_goes_down_with_more_credence_in_the_source_and_up_with_less(self):
        """向きを開発者が決めない: 同じ履歴・同じ訂正で、情報源の信用 κ̄ が高いほど τ は下がる向きに（自己正当化が減り罪悪感が増える）、低いほど上がる向きに動く。
        どちらの符号になるかは人格（経験史と評価状態）が決める。ここでは更新量が κ̄ について単調に減ることだけを検査する。"""
        steps = {}
        for name, push in (("high", 6.0), ("low", -6.0)):
            n = build(3490, bg="betray", err="interp", upto="cor", kappa_push=push)
            steps[name] = [ce["tau_step"] for ce in led(n, "correction_experience")]
            if name == "high":
                self.assertTrue(all(ce["g"] > 0.5 for ce in led(n, "correction_experience")))
        self.assertLess(steps["high"][0], steps["low"][0])
        self.assertLess(sum(steps["high"]), sum(steps["low"]))

    def test_tau_can_move_in_both_directions(self):
        """η_τ の符号つきの押しが上にも下にも作用する（Δ log τ の符号が defend − guilt − regret の符号）。"""
        lim = math.log(3.0)
        for push in (+0.3, -0.3):
            n = NPN(seed=3490, reappraisal_mode="v34")
            h = H.make_history(3490, "betray", "none", "interp", 4)
            feed(n, h["pre"] + h["mis"])
            n._tau_step = lambda d, g, r, e, p=push: p
            feed(n, h["cor"][:2])
            self.assertAlmostEqual(n.log_tau[KEY], 2 * push, places=14)
            self.assertLess(abs(n.log_tau[KEY]), lim)

    def test_H1_tau_and_interpretation_weights_are_finite_and_in_range(self):
        for seed, bg in ((3490, "betray"), (3491, "trust"), (3492, "neutral")):
            n = build(seed, bg=bg, src="unreliable", err="interp", n=8, upto="all")
            for v in n.log_tau.values():
                self.assertTrue(math.isfinite(v) and 1.0 / 3.0 - 1e-12 <= math.exp(v) <= 3.0 + 1e-12)
            self.assertLess(abs(n.mood), 2.0)
            for ce in led(n, "correction_experience"):
                self.assertTrue(all(math.isfinite(ce[k]) for k in ("f_c", "delta_c", "guilt", "regret", "defend", "own", "tau_step")))
            for e in led(n, "experience"):
                for i in e.get("emotions", {}).values():
                    self.assertTrue(math.isfinite(i))
            for net in (n.value.person, n.value.sit):
                self.assertTrue(np.isfinite(net.W).all() and np.abs(net.W).max() <= 10.0)


# ======================================================================
class InterpretationWeight(unittest.TestCase):
    def setUp(self):
        self.n = build_fresh(3490, "trust", "none", "interp", 4, "v34", upto="mis")

    def test_weight_is_proportional_to_q_times_belief_to_the_tau(self):
        n = self.n
        for tau in (0.5, 1.0, 2.0, 3.0):
            m = copy.deepcopy(n)
            m.log_tau = {KEY: math.log(tau)}
            w, ev = amb_weights(m, H.A, H.CTX_MIS, H.CON_MIS, q=0.5)
            p = m.belief.predict(H.CTX_MIS, ev["participants"]).p_person[0]
            from npnv3.innate import TYPE_INDEX
            kt, ks = TYPE_INDEX["credit_taken"], TYPE_INDEX["credit_shared"]
            num = np.array([0.5 * p[kt] ** tau, 0.5 * p[ks] ** tau])
            num = num / num.sum()
            self.assertAlmostEqual(w["credit_taken"], num[0], places=12)
            self.assertAlmostEqual(w["credit_shared"], num[1], places=12)

    def test_tau_one_is_the_v33_formula_bitwise_and_v33_mode_ignores_tau(self):
        a, b = copy.deepcopy(self.n), copy.deepcopy(self.n)
        b.log_tau = {KEY: 0.0}                                         # τ = 1 を明示
        wa, _ = amb_weights(a, H.A, H.CTX_MIS, H.CON_MIS)
        wb, _ = amb_weights(b, H.A, H.CTX_MIS, H.CON_MIS)
        self.assertEqual(wa, wb)
        c = copy.deepcopy(self.n)
        c.cfg.reappraisal_mode = "v33"
        c.log_tau = {KEY: math.log(3.0)}
        d = copy.deepcopy(self.n)
        d.cfg.reappraisal_mode = "v33"
        wc, _ = amb_weights(c, H.A, H.CTX_MIS, H.CON_MIS)
        wd, _ = amb_weights(d, H.A, H.CTX_MIS, H.CON_MIS)
        self.assertEqual(wc, wd)

    def test_tau_above_one_leans_on_the_prior_reading_and_below_one_relaxes_it(self):
        n = self.n
        w1, ev = amb_weights(n, H.A, H.CTX_MIS, H.CON_MIS)
        p = n.belief.predict(H.CTX_MIS, ev["participants"]).p_person[0]
        from npnv3.innate import TYPE_INDEX
        big = "credit_taken" if p[TYPE_INDEX["credit_taken"]] > p[TYPE_INDEX["credit_shared"]] else "credit_shared"
        lo, hi = copy.deepcopy(n), copy.deepcopy(n)
        lo.log_tau, hi.log_tau = {KEY: math.log(1 / 3)}, {KEY: math.log(3)}
        wl, _ = amb_weights(lo, H.A, H.CTX_MIS, H.CON_MIS)
        wh, _ = amb_weights(hi, H.A, H.CTX_MIS, H.CON_MIS)
        self.assertLess(wl[big], w1[big])
        self.assertLess(w1[big], wh[big])
        self.assertAlmostEqual(sum(wh.values()), 1.0, places=12)

    def test_key_is_class_times_context(self):
        """τ は（類 × 文脈）の鍵。同じ類・同じ文脈の A と A2 には同じ τ が効き、別の文脈（music_session）や別の類には効かない。"""
        n = copy.deepcopy(self.n)
        n.log_tau = {KEY: math.log(3)}
        base = self.n
        for actor, ctx, con, same in ((H.A, H.CTX_MIS, H.CON_MIS, False), (H.A2, H.CTX_MIS, H.CON_MIS, False),
                                      (H.A2, H.CTX_INS, H.CON_INS, True)):
            wa, _ = amb_weights(n, actor, ctx, con)
            wb, _ = amb_weights(base, actor, ctx, con)
            self.assertEqual(wa == wb, same, (actor, ctx))
        # 別の類
        ev = {"id": "x", "context": H.CTX_MIS, "concepts": [H.CON_MIS], "participants": [{"id": "M1", "class": "mentor"}],
              "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "M1", "magnitude": 0.8}], "optional": False}
        self.assertEqual(n.appraise(ev)["interpretations"][0]["weights"], base.appraise(ev)["interpretations"][0]["weights"])

    def test_interpretation_reaches_the_next_learning(self):
        """経路の存在（効果の向きは検査しない）: τ だけが違う 2 つの人格に同じ曖昧な出来事を与えると、人モデルの教師（Δu の φ 分割）と学ぶ W が変わる。τ が同じなら変わらない。"""
        a, b, c = copy.deepcopy(self.n), copy.deepcopy(self.n), copy.deepcopy(self.n)
        b.log_tau = {KEY: math.log(3)}
        c.log_tau = {KEY: 0.0}
        ev = H._exp("tx", H.CTX_MIS, H.CON_MIS, [H._p(H.A2, "collaborator")], [H._amb(H.A2, 0.5, 0.7)])
        ra, rb, rc = (x.experience(copy.deepcopy(ev)) for x in (a, b, c))
        self.assertNotEqual(ra["teachers"], rb["teachers"])
        self.assertFalse(np.array_equal(a.value.person.W, b.value.person.W))
        self.assertEqual(ra["teachers"], rc["teachers"])
        self.assertTrue(np.array_equal(a.value.person.W, c.value.person.W))

    def test_H5_with_tau_equal_one_ordinary_experiences_follow_the_v33_path(self):
        """τ ≡ 1（A-τ-off）の v34 の人格と、同じ状態の設定だけを v33 にした人格で、後続の通常の経験の出力と状態がビット単位で同じ。"""
        x = build(3490, bg="betray", err="interp", variant="tau_off", upto="cor")
        self.assertEqual(x.log_tau, {})
        y = copy.deepcopy(x)
        z = copy.deepcopy(x)
        z.cfg.reappraisal_mode = "v33"
        fol = H.make_history(3490, "betray", "none", "interp", 4)["fol"]
        ry = [y.experience(copy.deepcopy(e)) for e in fol]
        rz = [z.experience(copy.deepcopy(e)) for e in fol]
        self.assertEqual(json.dumps(ry, sort_keys=True, default=str), json.dumps(rz, sort_keys=True, default=str))
        self.assertTrue(np.array_equal(y.value.person.W, z.value.person.W) and np.array_equal(y.value.person.S, z.value.person.S))


# ======================================================================
class PathSeparation(unittest.TestCase):
    """RA3(3)(4)、H7、H8、f_c・安堵の学習への効果が構造上ゼロ。"""

    def test_RA3_3_clamp_tau_changes_only_tau_right_after_COR(self):
        for bg in ("betray", "neutral"):
            a = build(3490, bg=bg, err="interp", upto="cor")
            b = build(3490, bg=bg, err="interp", variant="clamp_tau", upto="cor")
            self.assertTrue(learning_equal(a, b, with_kappa_tau=False))           # W・Σ・信念・C・a・h・J・κ がビット単位で同じ
            self.assertEqual(a.mood, b.mood)                                       # 気分の経路は残る
            self.assertEqual(a.n_ticks, b.n_ticks)
            self.assertEqual(b.log_tau, {})
            self.assertNotEqual(a.log_tau, b.log_tau)
            for ca, cb in zip(led(a, "correction_experience"), led(b, "correction_experience")):
                for k in ("own", "guilt", "regret", "defend", "f_c", "relief", "g", "mood"):
                    self.assertEqual(ca[k], cb[k], k)                              # 台帳に元の値が残る
                self.assertTrue(cb["tau_clamped"])
                self.assertEqual(cb["log_tau"], {})
            self.assertTrue(any(ce["tau_step"] != 0.0 for ce in led(b, "correction_experience")))       # 渡さなかった元の押しは記録されている

    def test_RA3_4_clamp_mood_leaves_the_following_learning_bit_identical_to_C_v34(self):
        """f_c・安堵が学習に与える効果は構造上ゼロ: 気分の更新を切っても、FOL の後の学習の状態も読み出し R もビット単位で同じ（変わるのは気分と、気分を含む読み出しの f・δ だけ）。"""
        for bg, src in (("betray", "none"), ("trust", "unreliable")):
            a = build(3490, bg=bg, src=src, err="interp", upto="all")
            b = build(3490, bg=bg, src=src, err="interp", variant="clamp_mood", upto="all")
            self.assertTrue(learning_equal(a, b, with_kappa_tau=True))
            self.assertTrue(np.array_equal(RP.read_R(a, "collaborator"), RP.read_R(b, "collaborator")))
            self.assertNotEqual(a.mood, b.mood)
            self.assertEqual(a.n_ticks, b.n_ticks)                                  # 時計の経路は残る
            # 教師 teachers も同じ（台帳の経験の項）
            ta = [e["teachers"] for e in led(a, "experience")]
            tb = [e["teachers"] for e in led(b, "experience")]
            self.assertEqual(json.dumps(ta, sort_keys=True), json.dumps(tb, sort_keys=True))
            # 変わったのは気分と、f・δ（読み出し）だけ
            fa = [e["f"] for e in led(a, "experience")]
            fb = [e["f"] for e in led(b, "experience")]
            self.assertNotEqual(fa, fb)

    def test_clamp_both_cuts_both_and_clock_is_kept(self):
        a = build(3490, bg="betray", err="interp", upto="all")
        both = build(3490, bg="betray", err="interp", variant="clamp_both", upto="all")
        pre = build(3490, bg="betray", err="interp", upto="mis")
        self.assertEqual(both.log_tau, {})
        mood_pre = led(both, "correction_experience")[0]["mood"][0]
        for ce in led(both, "correction_experience"):
            self.assertEqual(ce["mood"], [mood_pre, mood_pre])
        self.assertEqual(both.n_ticks, a.n_ticks)

    def test_H7_beliefs_read_by_r4_r5_are_the_same_with_and_without_value_reappraisal(self):
        """予測と価値の分離: COR の直後の信念の読み出し r4・r5（p_person）が、C-v34 と C-none（価値を変えない対照。同じ v34 の上）でビット単位で同じ。"""
        for err in H.ERRS:
            for bg in ("betray", "trust"):
                a = build(3490, bg=bg, err=err, upto="cor")
                b = build(3490, bg=bg, err=err, upto="cor", extra_cfg={"no_reappraisal": True})
                ra, rb = RP.read_R(a, H.o_class(err)), RP.read_R(b, H.o_class(err))
                self.assertEqual((ra[3], ra[4]), (rb[3], rb[4]), (err, bg))
                self.assertNotEqual(ra[0], rb[0])                                       # 価値の読み出しは違う（対照が空でない）

    def test_H8_mood_loop_is_a_contraction(self):
        """m → f → δ → m の 1 回の更新の導関数の上限は 0.8 + 0.2 β = 0.86 < 1。COR の直後に 1e-3 の摂動を気分に与え、FOL の間の摂動の比が 1 以下。"""
        n = build_fresh(3490, "betray", "none", "interp", 4, "v34", upto="cor")
        m = copy.deepcopy(n)
        m.mood += 1e-3
        fol = H.make_history(3490, "betray", "none", "interp", 4)["fol"]
        d_prev = abs(m.mood - n.mood)
        self.assertAlmostEqual(d_prev, 1e-3, places=12)
        for e in fol:
            n.experience(copy.deepcopy(e))
            m.experience(copy.deepcopy(e))
            d = abs(m.mood - n.mood)
            self.assertLessEqual(d / d_prev, 0.8 + 0.2 * Config().beta_mood + 1e-12)
            self.assertLessEqual(d / d_prev, 1.0)
            d_prev = d
        self.assertTrue(learning_equal(n, m, with_kappa_tau=True))                       # 気分の摂動は学習に入らない
        # 訂正の更新の中でも
        a = build_fresh(3490, "betray", "none", "interp", 4, "v34", upto="mis")
        b = copy.deepcopy(a)
        b.mood += 1e-3
        for cor in H.make_history(3490, "betray", "none", "interp", 4)["cor"]:
            a.experience(copy.deepcopy(cor))
            b.experience(copy.deepcopy(cor))
        self.assertLessEqual(abs(a.mood - b.mood), 1e-3)

    def test_each_variant_changes_only_what_the_spec_says(self):
        ref = build(3490, bg="betray", err="interp", upto="cor")
        pre = build(3490, bg="betray", err="interp", upto="mis")
        # A-τ-off: τ ≡ 1。時計・気分は C-v34 と同じ。評価は計算・記録される
        v = build(3490, bg="betray", err="interp", variant="tau_off", upto="cor")
        self.assertEqual(v.log_tau, {})
        self.assertEqual((v.mood, v.n_ticks), (ref.mood, ref.n_ticks))
        self.assertTrue(learning_equal(v, ref, with_kappa_tau=False))
        self.assertTrue(all(ce["tau_source"] == "off" for ce in led(v, "correction_experience")))
        # A-clock-off: 訂正が時計と気分を進めない。τ は C-v34 と同じ
        v = build(3490, bg="betray", err="interp", variant="clock_off", upto="cor")
        self.assertEqual(v.log_tau, ref.log_tau)
        self.assertEqual((v.n_ticks, v.mood, v._clock()), (pre.n_ticks, pre.mood, pre._clock()))
        self.assertTrue(learning_equal(v, ref, with_kappa_tau=False))                         # W・Σ・信念・a・h は同じ（時計は C の時刻に効くだけ）
        self.assertEqual(ref.n_ticks, pre.n_ticks + 4)
        # clamp-mood: 気分は COR の前のまま、τ・時計は C-v34 と同じ
        v = build(3490, bg="betray", err="interp", variant="clamp_mood", upto="cor")
        self.assertEqual((v.mood, v.log_tau, v.n_ticks), (pre.mood, ref.log_tau, ref.n_ticks))
        # 台帳: 訂正の経験はちょうど N 件。他の項目（kind = experience）は変わらない
        for vv in ("tau_off", "clock_off", "clamp_mood", "clamp_tau", "clamp_both"):
            x = build(3490, bg="betray", err="interp", variant=vv, upto="cor")
            self.assertEqual(len(led(x, "correction_experience")), 4)
            self.assertEqual(json.dumps(led(x, "experience"), sort_keys=True), json.dumps(led(pre, "experience"), sort_keys=True))

    def test_tau_yoke_gives_the_donor_trajectory_not_the_persons_own(self):
        donor = build(3490, bg="betray", err="interp", upto="cor")
        traj = tau_trajectory(donor.ledger.entries)
        self.assertEqual(len(traj), 4)
        self.assertEqual(traj[-1], donor.log_tau)
        n = build_fresh(3490, "neutral", "none", "interp", 4, "v34", upto="mis")
        n.cfg.m3_variant = "tau_yoke"
        n.set_tau_yoke(traj)
        h = H.make_history(3490, "neutral", "none", "interp", 4)
        for i, cor in enumerate(h["cor"]):
            n.experience(copy.deepcopy(cor))
            self.assertEqual(n.log_tau, traj[i])
            self.assertEqual(led(n, "correction_experience")[-1]["tau_source"], "yoke")
        own = build(3490, bg="neutral", err="interp", upto="cor")
        self.assertNotEqual(own.log_tau, n.log_tau)                       # 自分の評価で動いた τ とは違う（背景が違うので）
        # 軌跡がなければ動かさない（yoke_missing）
        m = build_fresh(3490, "neutral", "none", "interp", 4, "v34", upto="mis")
        m.cfg.m3_variant = "tau_yoke"
        feed(m, h["cor"][:1])
        self.assertEqual(m.log_tau, {})
        self.assertEqual(led(m, "correction_experience")[-1]["tau_source"], "yoke_missing")


# ======================================================================
class CorrectionAsTimeEvent(unittest.TestCase):
    def test_v34_correction_advances_the_clock_and_adds_one_record(self):
        pre = build(3490, bg="betray", err="interp", upto="mis")
        n = build(3490, bg="betray", err="interp", upto="cor")
        self.assertEqual(n.n_ticks, pre.n_ticks + 4)
        self.assertEqual(n._clock(), pre._clock() + 4.0)
        self.assertEqual(n.cond.now, n._clock())
        self.assertEqual(n.n_experiences, pre.n_experiences)                  # 想起の掃引の予定（n_experiences）は進めない
        self.assertEqual(len(led(n, "correction_experience")), 4)
        self.assertEqual(sum(1 for r in n.exp_log if r.get("how") == "correction"), 4)
        self.assertEqual(sum(1 for e in n.ledger.entries if e["kind"] == "tick"), 4)           # SRC-none の空の刻み 4 つだけ。訂正の時計は kind = tick を足さない
        for ce in led(n, "correction_experience"):
            self.assertEqual(ce["t"][1] - ce["t"][0], 1.0)
            self.assertTrue(ce["clock_on"] and ce["mood_on"])
        # 当時の記録（経験の項と original_outcomes）は変わらない
        self.assertEqual(json.dumps(led(n, "experience"), sort_keys=True), json.dumps(led(pre, "experience"), sort_keys=True))
        self.assertEqual([r["original_outcomes"] for r in n.exp_log if r.get("how") != "correction"], [r["original_outcomes"] for r in pre.exp_log])
        # 価値の再生は訂正の経験の記録を飛ばす（教師を持たない）
        nv = n._replay_values()
        self.assertTrue(np.isfinite(nv.person.W).all())

    def test_v33_is_unchanged_by_the_m3_state(self):
        n = build(3490, mode="v33", bg="betray", err="interp", upto="all")
        self.assertEqual((n.n_ticks, n.log_tau, n.last_ev, n.G, n.tau_yoke), (0, {}, {}, {}, None))
        self.assertEqual(led(n, "correction_experience"), [])
        self.assertTrue(all(r.get("how") != "correction" for r in n.exp_log))
        for c in led(n, "correction"):
            self.assertNotIn("m3", c)
            self.assertEqual(set(c["emotions"]), {"relief", "guilt"})

    def test_mood_variants_control_nothing_in_v33_mode(self):
        a = build(3490, mode="v33", bg="betray", err="interp", upto="all")
        b = build(3490, mode="v33", bg="betray", err="interp", variant="clamp_both", upto="all")
        self.assertTrue(same_state(a, b) or json.dumps(a.get_state()[0]["config"], sort_keys=True) != json.dumps(b.get_state()[0]["config"], sort_keys=True))
        self.assertTrue(np.array_equal(a.value.person.W, b.value.person.W))
        self.assertEqual(a.mood, b.mood)

    def test_controls_do_not_get_the_m3_path(self):
        h = H.make_history(3490, "betray", "none", "interp", 2)
        for flag in ("no_reappraisal", "retroactive_value"):
            n = NPN(seed=3490, reappraisal_mode="v34", **{flag: True})
            feed(n, h["pre"] + h["mis"] + h["extra"] + h["cor"])
            self.assertEqual(led(n, "correction_experience"), [])
            self.assertEqual((n.n_ticks, n.log_tau), (4, {}))                  # 4 は SRC-none の空の刻み（訂正は時計を進めていない）

    def test_C_OR_accumulates_G_from_guilt_and_regret(self):
        n = build(3491, bg="neutral", err="interp", mode="v34_or", upto="cor")
        ces = led(n, "correction_experience")
        tot = sum(ce["guilt"] + ce["regret"] for ce in ces)
        self.assertGreater(tot, 0.0)
        for pid in (H.A, H.O):
            self.assertAlmostEqual(n.G[pid], tot, places=12)
        cors = led(n, "correction")
        G = 0.0
        for c, ce in zip(cors, ces):
            self.assertAlmostEqual(c["m1"]["G_e"][H.A], G, places=12)                                # 訂正 i の g を決める時点の G は i より前の累積
            self.assertAlmostEqual(c["m1"]["g_e"][H.A], c["m1"]["g"] * (1.0 + 0.25 * G), places=12)
            G += ce["guilt"] + ce["regret"]
        # C-v34 でも同じ評価の量が記録される（G は C-OR の吸収率にだけ使われる）
        v = build(3491, bg="neutral", err="interp", upto="cor")
        self.assertEqual(led(v, "correction_experience")[0]["guilt"], ces[0]["guilt"])
        for c in led(v, "correction"):
            self.assertEqual(c["m1"]["g_e"][H.A], c["m1"]["g"])


# ======================================================================
class SaveRestore(unittest.TestCase):
    def test_H4_bit_exact_save_restore_with_the_new_state_and_identical_continuation(self):
        h = H.make_history(3490, "betray", "unreliable", "interp", 3)
        for upto in ("cor", "all"):
            n = build_fresh(3490, "betray", "unreliable", "interp", 3, "v34", upto=upto)
            self.assertTrue(n.log_tau and n.last_ev)
            n.set_tau_yoke([{KEY: 0.1}])
            with tempfile.TemporaryDirectory() as d:
                digest = n.save(d)
                m = NPN.load(d)
                self.assertTrue(same_state(n, m))
                self.assertEqual(digest, _digest(*m.get_state()))
                self.assertEqual((m.log_tau, m.tau_yoke, m.last_ev, m.G), (n.log_tau, n.tau_yoke, n.last_ev, n.G))
                tail = (h["fol"] if upto == "cor" else [H._insert("zz9")]) + [H._insert("zz10")]
                ra = [n.experience(copy.deepcopy(x)) for x in tail]
                rb = [m.experience(copy.deepcopy(x)) for x in tail]
                self.assertEqual(json.dumps(ra, sort_keys=True, default=str), json.dumps(rb, sort_keys=True, default=str))
                self.assertTrue(same_state(n, m))
        # 訂正の続きも同じ（τ の更新が復元後も同じ）
        n = build_fresh(3490, "betray", "none", "interp", 4, "v34", upto="mis")
        cors = H.make_history(3490, "betray", "none", "interp", 4)["cor"]
        n.experience(copy.deepcopy(cors[0]))
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        for c in cors[1:]:
            n.experience(copy.deepcopy(c))
            m.experience(copy.deepcopy(c))
        self.assertTrue(same_state(n, m))
        self.assertEqual(m.exp_log, n.exp_log)
        self.assertEqual(set(m._exp_pos), {r["id"] for r in m.exp_log if r.get("how") != "correction"})

    def test_H4_runs_are_deterministic_for_every_variant(self):
        for vv in ("none", "clamp_tau", "clamp_mood", "clamp_both", "tau_off", "clock_off"):
            a = build_fresh(3491, "trust", "reliable", "interp", 3, "v34", variant=vv)
            b = build_fresh(3491, "trust", "reliable", "interp", 3, "v34", variant=vv)
            self.assertTrue(same_state(a, b), vv)

    def test_a_state_saved_before_stage_3_loads_with_empty_m3_state(self):
        n = build(3490, bg="betray", err="report", n=2, upto="all")
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            p = os.path.join(d, "state.json")
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            meta = doc["meta"]
            for k in ("log_tau", "tau_yoke", "last_ev"):
                meta["reappraisal"].pop(k)
            meta["config"].pop("m3_variant"), meta["config"].pop("eta_tau"), meta["config"].pop("tau_max")
            with np.load(os.path.join(d, "arrays.npz"), allow_pickle=False) as z:
                arrays = {k: np.array(z[k]) for k in z.files}
            doc["content_hash"] = _digest(meta, arrays)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
            m = NPN.load(d)
            self.assertEqual((m.log_tau, m.tau_yoke, m.last_ev, m.cfg.m3_variant), ({}, None, {}, "none"))

    def test_invalid_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            NPN(seed=3490, reappraisal_mode="v34", m3_variant="clamp_all")


# ======================================================================
class TraceLoop(unittest.TestCase):
    def test_trace_reconstructs_kappa_and_tau_from_the_ledger_alone(self):
        for seed, bg, src in ((3490, "betray", "reliable"), (3491, "neutral", "unreliable"), (3492, "trust", "none")):
            n = build(seed, bg=bg, src=src, err="interp", upto="all")
            tr = n.trace_loop()
            self.assertTrue(tr["consistent"], tr["problems"])
            for s_, k in tr["kappa"].items():
                self.assertAlmostEqual(k, n.kappa_bar(s_), places=12)
                self.assertAlmostEqual(tr["alpha_beta"][s_][0], n.src_beta[s_][0], places=12)
            self.assertEqual(set(tr["kappa"]), set(n.src_beta))
            self.assertEqual(set(tr["log_tau"]), set(n.log_tau))
            for k, v in tr["log_tau"].items():
                self.assertAlmostEqual(v, n.log_tau[k], places=12)
            ids = {x["eid"] for x in tr["nodes"]}
            self.assertEqual(ids, {e["eid"] for e in n.ledger.entries if e["kind"] in ("correction_experience", "verification")})
            for e in tr["edges"]:
                self.assertIn(e["from"], ids)
                self.assertIn(e["to"], ids)
            for x in tr["nodes"]:                       # 検証は検証される訂正を参照する
                if x["kind"] == "verification":
                    self.assertIn("verifies", [v for r, v in x["refs"]])

    def test_trace_uses_only_existing_paths_and_finds_the_chain_between_kappa_and_tau(self):
        n = build(3490, bg="betray", src="reliable", err="interp", n=4, upto="all")
        tr = n.trace_loop()
        vias = {e["via"] for e in tr["edges"]}
        self.assertTrue(vias <= {"kappa->g", "tau->interpretation->llr", "tau->interpretation->llr_ver", "verifies", "kappa(prev)"}, vias)
        self.assertIn("kappa->g", vias)               # 情報源の履歴で作られた κ̄ が、主実験の訂正の吸収率 g を決めた
        # 各訂正の台帳に、訂正前後の κ̄・τ、g、LLR、τ が効いた解釈の重みの変化、参照した先行の出来事がある
        for ce in led(n, "correction_experience"):
            for k in ("eid", "source", "kappa_bar", "log_tau", "g", "llr", "tau_effect", "refs"):
                self.assertIn(k, ce)
            for key, eff in ce["tau_effect"].items():
                self.assertEqual(len(eff["w_before"]), len(eff["w_after"]))
                self.assertNotEqual(eff["w_before"], eff["w_after"])
        for v in led(n, "verification"):
            for k in ("eid", "source", "kappa", "llr_ver", "refs"):
                self.assertIn(k, v)
        # 連鎖: 最長の連鎖は参照でつながっている
        if tr["runs"]:
            r = tr["runs"][0]
            ids = {x["eid"] for x in tr["nodes"]}
            self.assertTrue(set(r["path"]) <= ids and r["path"][0] == r["origin"] and r["path"][-1] == r["end"])

    def test_trace_detects_a_tampered_ledger(self):
        n = build(3490, bg="betray", src="none", err="interp", upto="cor")
        entries = copy.deepcopy(n.ledger.entries)
        for e in entries:
            if e["kind"] == "correction_experience" and e["log_tau"]:
                k = next(iter(e["log_tau"]))
                e["log_tau"][k][0] += 0.1
                break
        self.assertFalse(trace_loop(entries)["consistent"])
        entries = copy.deepcopy(n.ledger.entries)
        self.assertTrue(trace_loop(entries)["consistent"])

    def test_synthetic_verifications_chain_through_kappa_prev(self):
        n = NPN(seed=3490, reappraisal_mode="v34")
        n.verify_source("s", -3.0)
        n.verify_source("s", -3.0)
        tr = n.trace_loop()
        self.assertTrue(tr["consistent"])
        self.assertEqual([e["via"] for e in tr["edges"]], ["kappa(prev)"])
        self.assertAlmostEqual(tr["kappa"]["s"], n.kappa_bar("s"), places=12)

    def test_tau_trajectory_matches_state_and_ordinals(self):
        n = build(3490, bg="betray", src="reliable", err="interp", upto="cor")
        traj = tau_trajectory(n.ledger.entries)
        self.assertEqual(len(traj), n.n_corrections)
        self.assertEqual(traj[-1], n.log_tau)


# ======================================================================
class ProtocolEntrances(unittest.TestCase):
    """tests/ra_protocol.py の条件の入口（NotImplementedError が残っていない）。測定の値は検査しない。"""

    def test_stage3_conditions_run_on_development_seeds_and_start_from_the_same_pre_COR_state(self):
        conds = ["C-v34", "C-OR", "C-nocorr-v34", "A-τ-off", "A-clock-off", "clamp-τ", "clamp-mood", "clamp-both"]
        out = RP.run_individual(3490, "betray", "none", "interp", 3, 0, conds, "dev")
        pre = out["cond"]["C-v34"]["R_pre"]
        for c in conds:
            r = out["cond"][c]
            self.assertTrue(np.isfinite(r["R_cor"]).all() and np.isfinite(r["R_fol"]).all(), c)
            self.assertTrue(np.array_equal(r["R_pre"], pre), c)                    # COR の直前の状態は全条件で同じ
        nc = out["cond"]["C-nocorr-v34"]
        self.assertTrue(np.array_equal(nc["R_cor"][:7], pre[:7]))                  # 訂正なしの枝: 価値・信念・解釈は動かない（時計だけ進む）
        self.assertFalse(np.array_equal(out["cond"]["C-v34"]["R_cor"][:4], pre[:4]))
        self.assertTrue(np.array_equal(out["cond"]["clamp-mood"]["R_fol"], out["cond"]["C-v34"]["R_fol"]))   # RA3(4)
        idx = [0, 1, 2, 3, 4, 7, 8]                                                  # τ に依らない成分（r6・r7 は τ が効く曖昧な出来事の解釈の重み）
        self.assertTrue(np.array_equal(out["cond"]["clamp-τ"]["R_cor"][idx], out["cond"]["C-v34"]["R_cor"][idx]))      # RA3(3): COR の直後、τ 以外は同じ
        for alias in ("clamp-tau", "A-tau-off", "tau-yoke"):
            RP.get_condition(alias)
        with self.assertRaises(NotImplementedError):
            RP.get_condition("A-g-const")

    def test_val_swap_hm_eq_swap_G_and_the_literal_sensitivity(self):
        r = RP.run_val_swap(3490, "none", "interp", "betray", "trust", "dev", n=3)
        self.assertIsNotNone(r)
        plain = RP.run_individual(3490, "betray", "none", "interp", 3, 0, ["C-v34"], "dev")["cond"]["C-v34"]
        self.assertTrue(np.isfinite(r["R_cor"]).all())
        self.assertFalse(np.array_equal(r["R_cor"], plain["R_cor"]))               # 価値の評価状態を入れ替えると COR の直後の読み出しが変わる
        self.assertIsNotNone(RP.run_hm_eq(3490, "none", "interp", "betray", "trust", "dev", n=3))
        lit = RP.run_val_swap(3490, "none", "interp", "betray", "trust", "dev", n=3, literal=True)
        self.assertTrue(lit is None or np.isfinite(lit["R_cor"]).all())              # 文面どおりの読みは除かれうる（A1）。除かれた数は報告する
        for g in (["G5"], ["G1", "G5"]):
            self.assertIsNotNone(RP.run_swap(3490, "none", "interp", "betray", "trust", g, "C-v34", "dev", n=3))
        # G5 の入れ替え: κ・τ が相手のものになり、元の 2 つは変わらない
        t = build(3490, bg="betray", src="reliable", err="interp", upto="cor")
        s = build(3490, bg="trust", src="reliable", err="interp", upto="cor")
        t0 = copy.deepcopy(t.log_tau), copy.deepcopy(t.src_beta)
        sw = RP.swap_groups(t, s, ["G5"])
        self.assertEqual((sw.log_tau, sw.src_beta), (s.log_tau, s.src_beta))
        self.assertEqual((t.log_tau, t.src_beta), t0)
        self.assertIsNot(sw.log_tau, s.log_tau)

    def test_tau_yoke_protocol_and_guards(self):
        r = RP.run_tau_yoke(3490, "none", "interp", "neutral", "betray", "dev", n=3)
        self.assertEqual(r["log_tau"], r["donor_log_tau"])
        with self.assertRaises(ValueError):
            RP.run_individual(3490, "betray", "none", "interp", 3, 0, ["τ-yoke"], "dev")
        with self.assertRaises(PermissionError):
            RP.run_individual(3400, "betray", "none", "interp", 3, 0, ["C-v34"], "dev")      # 評価の種は拒否される
        with self.assertRaises(PermissionError):
            RP.run_tau_yoke(3450, "none", "interp", "neutral", "betray", "dev", n=3)

    def test_only_development_seeds_are_used(self):
        self.assertTrue(set(USED_SEEDS) <= set(DEV_SEEDS))
        self.assertFalse(set(USED_SEEDS) & FORBIDDEN)


if __name__ == "__main__":
    unittest.main()
