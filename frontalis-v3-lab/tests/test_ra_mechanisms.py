"""v3.4 Stage 2 の単体テスト: M1（含意された変化の吸収）、M2（情報源の信用）、C-OR、スキーマ 8（SPEC-v3.4 第 4.2・4.3・4.6 節、第 8 節 RA5・RA8(d)・RA10 H1〜H5）。

範囲: M3（罪悪感・後悔・τ・訂正を時計の出来事にする・気分）は Stage 3 なので、ここでは検査しない（H7〜H9、RA3、RA8(c) も Stage 3）。
乱数種: 開発用の 3490〜3499 だけ（SPEC 第 5.1 節）。評価 3400〜3439、変異 3450〜3459、診断 3460〜3479、予約 3500〜3539 は使わない（モジュールの先頭で確認する）。
判定の閾値はここで新しく作らない: 1e-9（RA5 の欲求ごとの誤差）、(0, 1)（g の範囲）、|W| ≤ 10（H1）は SPEC の値のまま。

  M1 / RA5   C-v34 で g_i ∈ (0, 1)、吸収後の予測の変化 = g_i D_i（欲求ごと 1e-9 以内）、吸収で Σ・C・a がビット単位で不変、台帳から再構成できる、残りの記録
  C-OR       g^OR = σ(·)(1 + λ_g G_e)（上限なし）。g > 1 でも RA5（吸収 = g D、Σ 不変、台帳で再構成）・H1（有限、|W| ≤ 10）・H2・H4・RA6 が破れない
  M2 / H3    κ̄ が変わるのは検証の出来事だけ（回数が台帳と一致）。訂正 1 回につき検証は最大 1 回。信用の固定化（RA8(d)）: 反対向きの検証で κ̄ が動き、計数が増えるほど動きが小さくなる
  H1〜H5     有限・範囲、台帳の和 = 状態、κ の出来事、保存・復元がビット単位・決定的、通常の経験の経路が既定のコードと一致（τ は Stage 3。ここでは経路の同一性）
  既定       reappraisal_mode = "v33" で v3.3 の挙動、スキーマ 7 → 8（7 は既定で読める、6 は拒否）
"""
import copy
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

from helpers import ROOT, cos  # noqa: F401  (helpers が sys.path に ROOT と data を足す)

sys.path.insert(0, str(ROOT / "data"))
import make_histories_v34 as H  # noqa: E402
from npnv3 import NPN  # noqa: E402
from npnv3.config import Config  # noqa: E402
from npnv3.innate import N_NEEDS, NEEDS  # noqa: E402
from npnv3.npn import READABLE_SCHEMAS, SCHEMA_VERSION, StateVersionError, _digest  # noqa: E402
from npnv3.value import KalmanModel  # noqa: E402

DEV_SEEDS = tuple(range(3490, 3500))
FORBIDDEN = set(range(3400, 3440)) | set(range(3450, 3460)) | set(range(3500, 3540))
USED_SEEDS = (3490, 3491, 3492)
assert set(USED_SEEDS) <= set(DEV_SEEDS) and not (set(USED_SEEDS) & FORBIDDEN)

TOL = 1e-9            # RA5: 欲求ごとの誤差
W_LIMIT = 10.0        # H1


def need(d):
    return np.array([d[k] for k in NEEDS], dtype=np.float64)


_CACHE = {}


def history(seed, bg="trust", src="reliable", err="report", n=4):
    return H.make_history(seed, bg=bg, src=src, err=err, n=n)


def build(seed, mode="v34", bg="trust", src="reliable", err="report", n=4, upto="all", G=None):
    """経験史を与えた人格。upto = "pre"（背景と情報源）| "mis"（誤帰属まで）| "cor"（訂正の直後。検証待ちが残る）| "all"（後続まで）。
    結果は書き込み禁止として共有される（変更するテストは build_fresh を使う）。"""
    key = (seed, mode, bg, src, err, n, upto, tuple(sorted((G or {}).items())))
    if key not in _CACHE:
        _CACHE[key] = build_fresh(seed, mode, bg, src, err, n, upto, G)
    return _CACHE[key]


def build_fresh(seed, mode="v34", bg="trust", src="reliable", err="report", n=4, upto="all", G=None):
    h = history(seed, bg, src, err, n)
    npn = NPN(seed=seed, reappraisal_mode=mode)
    seq = list(h["pre"])
    if upto in ("mis", "cor", "all"):
        seq += list(h["mis"]) + list(h["extra"])
    feed(npn, seq)
    for pid, v in (G or {}).items():
        npn._seed_G(pid, v)
    if upto in ("cor", "all"):
        feed(npn, h["cor"])
    if upto == "all":
        feed(npn, h["fol"])
    return npn


def feed(npn, seq):
    """経験・訂正を与える。空の刻み（kind = "tick"）は、v34 では advance_clock、v33 では無視（v3.3 の T.2: 訂正も刻みも時計を進めない）。"""
    for it in seq:
        if it.get("kind") == "tick":
            if npn.cfg.reappraisal_mode != "v33":
                npn.advance_clock(1)
            continue
        npn.experience(copy.deepcopy(it))


def corrections(npn):
    return [e for e in npn.ledger.entries if e["kind"] == "correction"]


def verifications(npn):
    return [e for e in npn.ledger.entries if e["kind"] == "verification"]


def snapshot(npn):
    """保存形式と同じ内容（メタと配列）。ビット単位の比較用。"""
    meta, arrays = npn.get_state()
    return json.dumps(meta, sort_keys=True, default=str), {k: np.array(v) for k, v in arrays.items()}


def same_state(a, b):
    ma, aa = snapshot(a)
    mb, ab = snapshot(b)
    return ma == mb and set(aa) == set(ab) and all(np.array_equal(aa[k], ab[k]) for k in aa)


def ledger_person_total(npn, model):
    return npn.ledger.totals(model)


class M1Absorption(unittest.TestCase):
    """RA5 の不変量（単体版）。"""

    def test_g_is_strictly_inside_unit_interval_for_all_err_and_bg(self):
        for seed in USED_SEEDS[:2]:
            for err in H.ERRS:
                for bg in ("trust", "betray"):
                    n = build(seed, "v34", bg=bg, src="unreliable", err=err)
                    cs = corrections(n)
                    self.assertEqual(len(cs), 8)                      # 情報源の履歴の訂正 4 + 主実験の訂正 4
                    for c in cs:
                        m1 = c["m1"]
                        self.assertTrue(math.isfinite(m1["llr"]))
                        self.assertTrue(0.0 < m1["g"] < 1.0, (seed, err, bg, m1["g"]))
                        self.assertTrue(all(0.0 < x < 1.0 for x in m1["g_e"].values()))
                        self.assertFalse(m1["or"])

    def test_g_follows_the_formula_sigma_of_logit_kappa_plus_llr(self):
        n = build(3490, "v34", src="reliable", err="interp")
        for c in corrections(n):
            m1 = c["m1"]
            a, b = m1["kappa_alpha_beta"]
            z = math.log(a / b) + m1["llr"]
            self.assertAlmostEqual(m1["g"], 1.0 / (1.0 + math.exp(-z)), places=12)
            self.assertAlmostEqual(m1["kappa_bar"], a / (a + b), places=12)

    def test_absorbed_prediction_change_equals_g_times_D_to_1e9_per_need(self):
        """欲求ごと・対象ごと（参加者 A と O、状況）。台帳に残る値と、外から測った訂正の前後の予測の差の両方で確かめる。"""
        for err in H.ERRS:
            h = history(3490, "trust", "reliable", err, 4)
            n = build_fresh(3490, "v34", src="reliable", err=err, upto="mis")
            ctx_concepts = (H.CTX_MIS, [H.CON_MIS])
            parts = [{"id": H.A, "class": "collaborator"}, {"id": H.O, "class": H.o_class(err)}]
            for cor in h["cor"]:
                before = [_pred(n, p) for p in parts]
                before_s = _pred_sit(n, *ctx_concepts)
                n.experience(copy.deepcopy(cor))
                m1 = corrections(n)[-1]["m1"]
                after = [_pred(n, p) for p in parts]
                after_s = _pred_sit(n, *ctx_concepts)
                for j, p in enumerate(parts):
                    d = need(m1["D"][p["id"]])
                    g = m1["g_e"][p["id"]]
                    self.assertLessEqual(np.abs((after[j] - before[j]) - g * d).max(), TOL, (err, cor["id"], p["id"]))
                    self.assertLessEqual(np.abs(need(m1["delta"][p["id"]]) - g * d).max(), 1e-15)
                    self.assertLessEqual(np.abs(need(m1["pred_change"][p["id"]]) - (after[j] - before[j])).max(), 1e-12)
                self.assertLessEqual(np.abs((after_s - before_s) - m1["g"] * need(m1["D_sit"])).max(), TOL)

    def test_unabsorbed_remainder_is_logged_and_complements_the_absorbed_part(self):
        n = build(3491, "v34", bg="betray", src="unreliable", err="report-x")
        for c in corrections(n):
            m1 = c["m1"]
            for pid in m1["D"]:
                d, dl, rem = need(m1["D"][pid]), need(m1["delta"][pid]), need(m1["remainder"][pid])
                self.assertLessEqual(np.abs(d - dl - rem).max(), 1e-15)
                self.assertLessEqual(np.abs(rem - (1.0 - m1["g_e"][pid]) * d).max(), 1e-15)
            self.assertLessEqual(np.abs(need(m1["D_sit"]) - need(m1["delta_sit"]) - need(m1["remainder_sit"])).max(), 1e-15)

    def test_sigma_conditioning_and_concern_are_unchanged_bitwise_by_a_v34_correction(self):
        """吸収は Σ を変えない（過程雑音も足さない）。C と a も変えない。v3.3 の再評価では Σ が変わる（検査が検出できることの確認）。"""
        for mode, expect_same_S in (("v34", True), ("v34_or", True), ("v33", False)):
            n = build_fresh(3490, mode, upto="mis")
            S_p, S_s = n.value.person.S.copy(), n.value.sit.S.copy()
            names_p, names_s = list(n.value.person.names), list(n.value.sit.names)
            C, a = n.cond.C.copy(), n.concern.a.copy()
            h = history(3490, "trust", "reliable", "report", 4)
            n.experience(copy.deepcopy(h["cor"][0]))
            self.assertEqual(names_p, list(n.value.person.names))
            self.assertEqual(names_s, list(n.value.sit.names))
            same = np.array_equal(S_p, n.value.person.S) and np.array_equal(S_s, n.value.sit.S)
            self.assertEqual(same, expect_same_S, mode)
            self.assertTrue(np.array_equal(C, n.cond.C))
            self.assertTrue(np.array_equal(a, n.concern.a))

    def test_effect_is_reconstructable_from_the_ledger(self):
        """台帳の ΔW（訂正の項）と特徴ベクトルから、like(A) の変化が 1e-9 以内で再構成できる。台帳の ΔW の総和 = 現在の W（H2）。"""
        h = history(3492, "neutral", "unreliable", "interp", 4)
        n = build_fresh(3492, "v34", bg="neutral", src="unreliable", err="interp", upto="mis")
        a = n.concern.a.copy()
        for cor in h["cor"]:
            l0 = n.probe({"entity": H.A, "class": "collaborator", "context": H.CTX_MIS})["like"]
            n.experience(copy.deepcopy(cor))
            l1 = n.probe({"entity": H.A, "class": "collaborator", "context": H.CTX_MIS})["like"]
            ent = corrections(n)[-1]
            z = {f: v for f, _l, v in n.value.person_feats(H.A, "collaborator")}
            dW = ent["dW"]["person"]
            # 台帳の ΔW に、参加者 A と O の両方が共有する特徴（people・cls・emb）が混ざる: A の特徴ベクトルで読み出すのが like(A) の変化
            dl = sum(z[f] * float(np.dot(a, np.array(dW[f]))) for f in z if f in dW)
            self.assertLessEqual(abs(dl - (l1 - l0)), TOL)
        for model in ("person", "situation"):
            net = n.value.person if model == "person" else n.value.sit
            tot = ledger_person_total(n, model)
            for nm, row in tot.items():
                self.assertLessEqual(np.abs(row - net.W[net.idx[nm]]).max(), TOL, (model, nm))
            for nm in net.names:                               # 台帳にない特徴の W は 0
                if nm not in tot:
                    self.assertLessEqual(np.abs(net.W[net.idx[nm]]).max(), TOL)

    def test_kalman_absorb_matches_the_spec_formula_for_one_target_and_is_exact_for_many(self):
        rng = np.random.default_rng(3490)
        m = KalmanModel({"people": 0.01, "cls": 0.04, "ind": 0.25, "emb": 0.02}, {"people": 5e-6, "cls": 2e-5, "ind": 1e-4, "emb": 5e-6}, 0.01)
        feats = [("people", "people", 1.0), ("cls:c", "cls", 1.0), ("ind:x", "ind", 1.0), ("ind:y", "ind", 1.0), ("emb:0", "emb", 0.3), ("emb:1", "emb", -0.5)]
        for _ in range(6):                                                   # 非対角の共分散を作る
            sel = [feats[i] for i in (0, 1, 2, 4, 5)] if rng.random() < 0.5 else [feats[i] for i in (0, 1, 3, 4, 5)]
            m.update(sel, rng.normal(size=N_NEEDS), np.ones(N_NEEDS))
        S0 = m.S.copy()
        # 1 つの対象: W_k += Σ_k z (zᵀ Σ_k z)⁻¹ Δ_k
        t1 = [feats[i] for i in (0, 1, 2, 4, 5)]
        W0 = m.W.copy()
        ii = np.array([m.idx[f[0]] for f in t1])
        z = np.array([f[2] for f in t1])
        delta = rng.normal(size=(1, N_NEEDS))
        m.absorb([t1], delta)
        for k in range(N_NEEDS):
            Sz = m.S[k][:, ii] @ z
            expect = Sz / (z @ m.S[k][np.ix_(ii, ii)] @ z) * delta[0, k]
            self.assertLessEqual(np.abs((m.W - W0)[:, k] - expect).max(), 1e-12)
        self.assertTrue(np.array_equal(S0, m.S))
        # 2 つの対象（共有の特徴を持つ）: 両方の予測がちょうど Δ だけ動く
        t2 = [feats[i] for i in (0, 1, 3, 4, 5)]
        W1 = m.W.copy()
        d2 = rng.normal(size=(2, N_NEEDS))
        m.absorb([t1, t2], d2)
        for r, tg in enumerate((t1, t2)):
            zi = np.array([m.idx[f[0]] for f in tg]); zv = np.array([f[2] for f in tg])
            self.assertLessEqual(np.abs(zv @ (m.W - W1)[zi] - d2[r]).max(), TOL)
        self.assertTrue(np.array_equal(S0, m.S))

    def test_no_value_net_output_enters_g_or_D(self):
        """§4.5: M1 の g_i は信念だけ、D_i は記録された教師だけから決まる。価値の W を撹乱しても、LLR・g・D はビット単位で変わらない（吸収後の W は変わる）。"""
        h = history(3490, "trust", "reliable", "report", 4)
        a = build_fresh(3490, "v34", upto="mis")
        b = build_fresh(3490, "v34", upto="mis")
        rng = np.random.default_rng(3491)
        b.value.person.W = b.value.person.W + rng.normal(size=b.value.person.W.shape)
        b.value.sit.W = b.value.sit.W + rng.normal(size=b.value.sit.W.shape)
        a.experience(copy.deepcopy(h["cor"][0]))
        b.experience(copy.deepcopy(h["cor"][0]))
        ma, mb = corrections(a)[-1]["m1"], corrections(b)[-1]["m1"]
        for k in ("llr", "g", "z", "kappa_bar", "D", "D_sit", "g_e"):
            self.assertEqual(ma[k], mb[k], k)
        self.assertFalse(np.array_equal(a.value.person.W, b.value.person.W))

    def test_original_records_are_not_rewritten_RA6(self):
        h = history(3490, "betray", "reliable", "interp", 4)
        n = build_fresh(3490, "v34", bg="betray", err="interp", upto="mis")
        led0 = copy.deepcopy([e for e in n.ledger.entries if e["kind"] == "experience"])
        orig0 = copy.deepcopy([r["original_outcomes"] for r in n.exp_log])
        for cor in h["cor"]:
            n.experience(copy.deepcopy(cor))
        led1 = [e for e in n.ledger.entries if e["kind"] == "experience"][: len(led0)]
        self.assertEqual(json.dumps(led0, sort_keys=True), json.dumps(led1, sort_keys=True))
        self.assertEqual(orig0, [r["original_outcomes"] for r in n.exp_log][: len(orig0)])
        self.assertEqual(len(corrections(n)), 8)

    def test_controls_keep_their_own_rules_in_v34_mode(self):
        """no_reappraisal と retroactive_value は v34 の上でも対照として働く（吸収しない・m1 を持たない）。"""
        h = history(3490, "trust", "reliable", "report", 2)
        for flag in ("no_reappraisal", "retroactive_value"):
            n = NPN(seed=3490, reappraisal_mode="v34", **{flag: True})
            feed(n, h["pre"] + h["mis"] + h["extra"] + h["cor"])
            for c in corrections(n):
                self.assertNotIn("m1", c)
            self.assertEqual(n.ver_pending, [])


def _pred(npn, p):
    return npn.value.predict_person(p)


def _pred_sit(npn, ctx, concepts):
    return npn.value.predict_situation(ctx, concepts)


class COR(unittest.TestCase):
    """C-OR（上限のない吸収）。過剰な反省は故障ではない（F1・F2・F3 だけを検査する）。"""

    def test_cor_equals_cv34_bitwise_when_the_accumulated_G_is_zero(self):
        """Stage 2 は M3 を持たないので G_e は自然には増えない（0）。そのとき C-OR は C-v34 と同じ（g^OR = σ(·)）。"""
        a = build(3490, "v34", src="unreliable", err="interp")
        b = build(3490, "v34_or", src="unreliable", err="interp")
        self.assertTrue(np.array_equal(a.value.person.W, b.value.person.W))
        self.assertTrue(np.array_equal(a.value.sit.W, b.value.sit.W))
        for ca, cb in zip(corrections(a), corrections(b)):
            self.assertEqual(ca["m1"]["g_e"], cb["m1"]["g_e"])
            self.assertTrue(cb["m1"]["or"])

    def test_cor_g_formula_has_no_upper_bound_and_exceeds_one(self):
        G = {H.A: 12.0, H.O: 3.0}
        n = build(3490, "v34_or", src="reliable", err="report", upto="all", G=G)
        saw_above_one = False
        for c in corrections(n):
            m1 = c["m1"]
            g = m1["g"]
            self.assertTrue(math.isfinite(g) and 0.0 < g < 1.0)               # 基底の σ は (0, 1)
            if H.A not in m1["g_e"]:                                           # 情報源の履歴の訂正（対象は Qk）: G を持たないので g のまま
                self.assertTrue(all(abs(v - g) < 1e-15 for v in m1["g_e"].values()))
                continue
            for pid, Ge in ((H.A, 12.0), (H.O, 3.0)):
                self.assertAlmostEqual(m1["g_e"][pid], g * (1.0 + 0.25 * Ge), places=12)       # λ_g = 0.25
                self.assertAlmostEqual(m1["G_e"][pid], Ge, places=12)
                saw_above_one |= m1["g_e"][pid] > 1.0
            self.assertGreater(m1["g_e"][H.A], m1["g_e"][H.O])
        self.assertTrue(saw_above_one, "g > 1 の経路が動いていない（G が足りない）")

    def test_cor_with_g_above_one_satisfies_RA5_H1_H2_H4_and_RA6(self):
        G = {H.A: 12.0, H.O: 3.0}
        h = history(3491, "betray", "unreliable", "report-x", 4)
        n = build_fresh(3491, "v34_or", bg="betray", src="unreliable", err="report-x", upto="mis", G=G)
        led0 = copy.deepcopy([e for e in n.ledger.entries if e["kind"] == "experience"])
        parts = [{"id": H.A, "class": "collaborator"}, {"id": H.O, "class": H.o_class("report-x")}]
        S0 = (n.value.person.S.copy(), n.value.sit.S.copy())
        for cor in h["cor"]:
            before = [_pred(n, p) for p in parts]
            n.experience(copy.deepcopy(cor))
            m1 = corrections(n)[-1]["m1"]
            after = [_pred(n, p) for p in parts]
            for j, p in enumerate(parts):                                        # RA5: 吸収された変化 = g^OR D（上限なしでも一致）
                self.assertLessEqual(np.abs((after[j] - before[j]) - m1["g_e"][p["id"]] * need(m1["D"][p["id"]])).max(), TOL)
            self.assertTrue(all(math.isfinite(x) for x in m1["g_e"].values()))
            self.assertLessEqual(np.abs(need(m1["D"][H.A]) - need(m1["delta"][H.A]) - need(m1["remainder"][H.A])).max(), 1e-15)
        self.assertTrue(np.array_equal(S0[0], n.value.person.S) and np.array_equal(S0[1], n.value.sit.S))      # Σ 不変
        # H1: 有限・|W| ≤ 10
        for net in (n.value.person, n.value.sit):
            self.assertTrue(np.isfinite(net.W).all() and np.abs(net.W).max() <= W_LIMIT, np.abs(net.W).max())
        # H2: 台帳の和 = W
        for model, net in (("person", n.value.person), ("situation", n.value.sit)):
            for nm, row in n.ledger.totals(model).items():
                self.assertLessEqual(np.abs(row - net.W[net.idx[nm]]).max(), TOL)
        # RA6 / F4
        led1 = [e for e in n.ledger.entries if e["kind"] == "experience"][: len(led0)]
        self.assertEqual(json.dumps(led0, sort_keys=True), json.dumps(led1, sort_keys=True))
        # H4: 保存・復元（G を含む）
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        self.assertEqual(m.G, n.G)
        self.assertTrue(same_state(m, n))

    def test_cor_situation_target_uses_the_base_g(self):
        """状況の対象は人ではないので G を掛けない（SPEC の曖昧さ。CHANGELOG-v3.4 S2-3）。"""
        n = build(3490, "v34_or", src="reliable", err="report", upto="all", G={H.A: 12.0, H.O: 3.0})
        for c in corrections(n):
            m1 = c["m1"]
            self.assertLessEqual(np.abs(need(m1["delta_sit"]) - m1["g"] * need(m1["D_sit"])).max(), 1e-15)

    def test_seed_G_rejects_invalid_values_and_the_m3_hook_is_a_stub(self):
        n = NPN(seed=3490, reappraisal_mode="v34_or")
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                n._seed_G("A", bad)
        self.assertEqual(n._m3_guilt_regret({}, {}, []), (0.0, 0.0))          # Stage 3 の入口
        self.assertEqual(n.G, {})


class M2SourceCredibility(unittest.TestCase):
    def test_kappa_changes_only_at_verification_events_H3(self):
        h = history(3490, "trust", "reliable", "interp", 4)
        n = NPN(seed=3490, reappraisal_mode="v34")
        seq = h["pre"] + h["mis"] + h["extra"] + h["cor"] + h["fol"]
        prev = json.dumps(n.src_beta, sort_keys=True)
        changes = 0
        for it in seq:
            nv0 = len(verifications(n))
            feed(n, [it])
            cur = json.dumps(n.src_beta, sort_keys=True)
            nv1 = len(verifications(n))
            if cur != prev:
                changes += 1
                self.assertGreater(nv1, nv0, it["id"])             # 変わったのは検証の出来事があったときだけ
            else:
                self.assertEqual(nv1, nv0)
            prev = cur
        self.assertEqual(n.n_verifications, len(verifications(n)))
        self.assertGreater(changes, 0)
        # H2: 台帳の Σ d_alpha, Σ d_beta = α − α0, β − β0
        for s, (al, be) in n.src_beta.items():
            self.assertAlmostEqual(al - 1.0, sum(e["d_alpha"] for e in verifications(n) if e["source"] == s), places=12)
            self.assertAlmostEqual(be - 1.0, sum(e["d_beta"] for e in verifications(n) if e["source"] == s), places=12)
        # 訂正の g の計算に使う κ̄ は、その時点の Beta（訂正が κ̄ を動かさない）: 訂正の台帳エントリの前後で同じ
        for c in corrections(n):
            self.assertEqual(c["m1"]["source"], H.SOURCE)

    def test_at_most_one_verification_per_correction_at_first_appearance(self):
        n = build(3490, "v34", src="reliable", err="report", n=4, upto="all")
        vs = verifications(n)
        ids = [v["correction"] for v in vs]
        self.assertEqual(len(ids), len(set(ids)))                  # 訂正 1 回につき最大 1 回
        # SRC エピソードの訂正 src{k}x は Qk の直接の経験 src{k}c で検証される
        by = {v["correction"]: v for v in vs}
        for k in range(4):
            self.assertEqual(by[f"src{k}x"]["experience"], f"src{k}c")
            self.assertEqual(by[f"src{k}x"]["target"], f"Q{k + 1}")
        # 主実験の訂正 c{i}（対象 A・O）は、後続の最初の A との経験（fol0）で検証される
        for i in range(4):
            self.assertEqual(by[f"c{i}"]["experience"], "fol0")
            self.assertEqual(by[f"c{i}"]["target"], H.A)
        self.assertEqual(n.ver_pending, [])
        self.assertTrue(all(v["synthetic"] is False for v in vs))

    def test_verification_is_not_triggered_by_ambiguous_only_appearance_or_by_avoided_experiences(self):
        """対象が曖昧な結果だけで現れた経験・参加者として現れない経験では検証しない。"""
        h = history(3490, "trust", "reliable", "report", 2)
        n = build_fresh(3490, "v34", err="report", n=2, upto="cor")
        npend = len(n.ver_pending)
        self.assertEqual(npend, 2)
        # B との経験（対象でない）・A2 の曖昧な出来事（対象でない）では検証されない
        n.experience(copy.deepcopy(h["fol"][1]))                       # (b) A2 の曖昧な出来事
        n.experience(copy.deepcopy(H._insert("zz1")))                  # B との経験
        self.assertEqual(len(n.ver_pending), npend)
        # A が曖昧な結果だけで現れる経験: 検証されない
        amb = {"id": "ambA", "context": H.CTX_MIS, "concepts": [H.CON_MIS], "participants": [{"id": H.A, "class": "collaborator"}],
               "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": H.A, "magnitude": 0.7}], "optional": False}
        n.experience(copy.deepcopy(amb))
        self.assertEqual(len(n.ver_pending), npend)
        # A の明確な結果を伴う経験: 検証される（2 件とも）
        n.experience(copy.deepcopy(h["fol"][0]))
        self.assertEqual(len(n.ver_pending), 0)

    def test_sign_of_the_verification_follows_agreement_with_the_corrected_fact(self):
        """訂正（credit_shared）と整合する直接の経験は LLR_ver > 0（信用が上がる）、元の報告（credit_taken）と整合する経験は < 0。"""
        def run(kind):
            n = NPN(seed=3490, reappraisal_mode="v34")
            feed(n, history(3490, "trust", "none", "report", 2)["pre"])
            q = {"id": "Q9", "class": "peer"}
            rep = {"id": "rep", "context": H.CTX_SRC, "concepts": [H.CON_SRC], "participants": [q], "outcomes": [H._o("credit_taken", "Q9", 0.7)], "optional": False}
            cor = {"id": "repx", "kind": "correction", "refers_to": "rep", "source": "witness", "corrected_outcomes": [H._o("credit_shared", "Q9", 0.7)]}
            d = {"id": "dir", "context": H.CTX_SRC, "concepts": [H.CON_SRC], "participants": [q], "outcomes": [H._o(kind, "Q9", 0.6)], "optional": False}
            feed(n, (rep, cor, d))
            return [v for v in verifications(n) if v["source"] == "witness"][0]
        agree, disagree = run("credit_shared"), run("credit_taken")
        self.assertGreater(agree["llr_ver"], 0.0)
        self.assertLess(disagree["llr_ver"], 0.0)
        self.assertGreater(agree["kappa"][1], agree["kappa"][0])
        self.assertLess(disagree["kappa"][1], disagree["kappa"][0])
        self.assertAlmostEqual(agree["d_alpha"] + agree["d_beta"], 1.0, places=12)

    def test_reverting_the_belief_works_on_a_copy_and_leaves_the_main_belief_untouched(self):
        n = build_fresh(3490, "v34", err="report", n=2, upto="cor")
        pend = n.ver_pending[0]
        meta0, arr0 = n.belief.get_state()
        s0 = json.dumps(meta0, sort_keys=True, default=str)
        arr0 = {k: np.array(v) for k, v in arr0.items()}
        rev = n._reverted_belief(pend)
        self.assertIsNot(rev, n.belief)
        meta1, arr1 = n.belief.get_state()
        self.assertEqual(s0, json.dumps(meta1, sort_keys=True, default=str))
        self.assertTrue(all(np.array_equal(arr0[k], arr1[k]) for k in arr0))
        # 戻した信念は、訂正前の事実（A が取った）に近い: p_person(A, credit_taken) が現在の信念より高い
        P = [{"id": H.A, "class": "collaborator"}, {"id": H.O, "class": "collaborator"}]
        from npnv3.innate import OUTCOME_TYPES
        k = OUTCOME_TYPES.index("credit_taken")
        self.assertGreater(rev.predict(H.CTX_MIS, P).p_person[0, k], n.belief.predict(H.CTX_MIS, P).p_person[0, k])

    def test_kappa_stays_inside_the_open_unit_interval_for_extreme_llr(self):
        n = NPN(seed=3490, reappraisal_mode="v34")
        for llr in (1e9, -1e9, 800.0, -800.0, 1e300):
            n.verify_source("s", llr)
        al, be = n.src_beta["s"]
        self.assertTrue(math.isfinite(al) and math.isfinite(be) and al > 0 and be > 0)
        self.assertTrue(0.0 < n.kappa_bar("s") < 1.0)
        self.assertEqual(n.kappa_bar("unseen"), 0.5)                    # Beta(1, 1)

    def test_RA8d_kappa_moves_for_an_opposite_verification_and_hardens_with_the_count(self):
        """規則が働いているか（反対向きの検証で κ̄ が 0.01 以上動く）と、固定化（計数が増えるほど 1 回で動く量が小さい）の報告。"""
        n = build(3490, "v34", src="reliable", err="report", n=4, upto="all")
        s = H.SOURCE
        k0 = n.kappa_bar(s)
        n2 = copy.deepcopy(n)
        e = n2.verify_source(s, -5.0)                      # 反対向き（元の報告が正しかったことを示す検証）
        self.assertLessEqual(e["kappa"][1] - e["kappa"][0], -0.01)
        n3 = copy.deepcopy(n)
        e3 = n3.verify_source(s, +5.0)
        self.assertGreaterEqual(e3["kappa"][1] - e3["kappa"][0], 0.0)
        self.assertEqual(e["synthetic"], True)
        self.assertEqual(n2.n_verifications, n.n_verifications + 1)
        # 固定化: 計数が小さいときの 1 回の動き > 大きいときの動き（同じ LLR）
        small = NPN(seed=3490, reappraisal_mode="v34")
        big = NPN(seed=3490, reappraisal_mode="v34")
        for _ in range(30):
            big.verify_source("t", 2.0)
        ds = small.verify_source("t", -2.0)
        db = big.verify_source("t", -2.0)
        self.assertGreater(abs(ds["kappa"][1] - ds["kappa"][0]), abs(db["kappa"][1] - db["kappa"][0]))
        self.assertGreater(sum(n.src_beta[s]), 2.0)           # 終わりの α + β（報告用）
        self.assertTrue(0.0 < k0 < 1.0)


class NumericalHealth(unittest.TestCase):
    """RA10 H1〜H5 の単体版（M3 に依る H7〜H9 は Stage 3）。"""

    def test_H1_finite_and_in_range_for_v34_and_cor(self):
        for mode in ("v34", "v34_or"):
            for seed in USED_SEEDS[:2]:
                n = build(seed, mode, bg="betray", src="unreliable", err="report-x", n=8)
                for net in (n.value.person, n.value.sit):
                    self.assertTrue(np.isfinite(net.W).all() and np.isfinite(net.S).all())
                    self.assertLessEqual(np.abs(net.W).max(), W_LIMIT)
                for s, (al, be) in n.src_beta.items():
                    self.assertTrue(0.0 < al / (al + be) < 1.0)
                for c in corrections(n):
                    self.assertTrue(math.isfinite(c["m1"]["llr"]) and math.isfinite(c["m1"]["z"]))
                    if mode == "v34":
                        self.assertTrue(0.0 < c["m1"]["g"] < 1.0)
                self.assertLess(abs(n.mood), 2.0)
                self.assertTrue(np.isfinite(n.h).all())

    def test_H2_ledger_sums_match_the_state(self):
        n = build(3491, "v34", bg="betray", src="reliable", err="interp", n=4)
        for model, net in (("person", n.value.person), ("situation", n.value.sit)):
            tot = n.ledger.totals(model)
            for nm in net.names:
                row = tot.get(nm, np.zeros(N_NEEDS))
                self.assertLessEqual(np.abs(row - net.W[net.idx[nm]]).max(), TOL, (model, nm))
        for s, (al, be) in n.src_beta.items():
            self.assertAlmostEqual(al, 1.0 + sum(e["d_alpha"] for e in verifications(n) if e["source"] == s), places=12)
            self.assertAlmostEqual(be, 1.0 + sum(e["d_beta"] for e in verifications(n) if e["source"] == s), places=12)

    def test_H4_save_restore_is_bit_exact_including_pending_and_continues_identically(self):
        h = history(3490, "trust", "unreliable", "interp", 3)
        for upto in ("cor", "all"):          # cor = 検証待ちが残っている状態
            n = build_fresh(3490, "v34", src="unreliable", err="interp", n=3, upto=upto)
            if upto == "cor":
                self.assertGreater(len(n.ver_pending), 0)
            with tempfile.TemporaryDirectory() as d:
                digest = n.save(d)
                m = NPN.load(d)
                self.assertTrue(same_state(n, m))
                self.assertEqual(digest, _digest(*m.get_state()))
                self.assertEqual(m.src_beta, n.src_beta)
                self.assertEqual(m.ver_pending, n.ver_pending)
                self.assertEqual((m.n_ticks, m.n_verifications), (n.n_ticks, n.n_verifications))
                tail = (h["fol"] if upto == "cor" else [H._insert("zz9")]) + [H._insert("zz10")]
                ra = [n.experience(copy.deepcopy(x)) for x in tail]
                rb = [m.experience(copy.deepcopy(x)) for x in tail]
                self.assertEqual(json.dumps(ra, sort_keys=True, default=str), json.dumps(rb, sort_keys=True, default=str))
                self.assertTrue(same_state(n, m))

    def test_H4_runs_are_deterministic(self):
        a = build_fresh(3491, "v34_or", bg="neutral", src="reliable", err="report", n=3, G={H.A: 2.0})
        b = build_fresh(3491, "v34_or", bg="neutral", src="reliable", err="report", n=3, G={H.A: 2.0})
        self.assertTrue(same_state(a, b))

    def test_H5_ordinary_experiences_use_the_same_path_as_the_default_mode(self):
        """Stage 2 の単体版: (a) 訂正の前は v33 と v34 の状態がビット単位で同じ。(b) 同じ状態から、モードだけを替えて通常の経験を与えると、
        出力（教師・f・感情を含む）と状態がビット単位で同じ（τ は Stage 3。通常の経験の経路に v3.4 の分岐がないことの確認）。"""
        bgmis = history(3490, "trust", "none", "report", 4)
        seq = bgmis["pre"][:H.N_BG] + bgmis["mis"] + bgmis["extra"]          # 背景 + 誤帰属（訂正も空の刻みもなし）
        a, b = NPN(seed=3490, reappraisal_mode="v33"), NPN(seed=3490, reappraisal_mode="v34")
        feed(a, seq)
        feed(b, seq)
        ma, aa = snapshot(a)
        mb, ab = snapshot(b)
        cfg_a, cfg_b = json.loads(ma)["config"], json.loads(mb)["config"]
        self.assertEqual({k: v for k, v in cfg_a.items() if k != "reappraisal_mode"}, {k: v for k, v in cfg_b.items() if k != "reappraisal_mode"})
        self.assertTrue(all(np.array_equal(aa[k], ab[k]) for k in aa))
        self.assertEqual(json.dumps(json.loads(ma)["exp_log"], sort_keys=True), json.dumps(json.loads(mb)["exp_log"], sort_keys=True))
        # (b)
        h = history(3490, "trust", "reliable", "report", 4)
        x = build_fresh(3490, "v33", upto="cor")                       # v33 で訂正まで進めた状態
        with tempfile.TemporaryDirectory() as d:
            x.save(d)
            y = NPN.load(d)
        y.cfg.reappraisal_mode = "v34"                                  # 設定だけを替える（ver_pending は空: v33 では登録されない）
        self.assertEqual(y.ver_pending, [])
        ra = [x.experience(copy.deepcopy(e)) for e in h["fol"]]
        rb = [y.experience(copy.deepcopy(e)) for e in h["fol"]]
        self.assertEqual(json.dumps(ra, sort_keys=True, default=str), json.dumps(rb, sort_keys=True, default=str))
        self.assertTrue(np.array_equal(x.value.person.W, y.value.person.W) and np.array_equal(x.value.person.S, y.value.person.S))


class DefaultAndSchema(unittest.TestCase):
    def test_default_mode_is_v33_and_creates_no_v34_state(self):
        self.assertEqual(Config().reappraisal_mode, "v33")
        n = build(3490, "v33", src="reliable", err="report", upto="all")
        self.assertEqual(n.cfg.reappraisal_mode, "v33")
        self.assertEqual((n.src_beta, n.ver_pending, n.G, n.n_ticks, n.n_verifications), ({}, [], {}, 0, 0))
        self.assertEqual(verifications(n), [])
        for c in corrections(n):
            self.assertEqual(c["reappraisal"], "kalman")
            self.assertNotIn("m1", c)
        d = NPN(seed=3490)
        self.assertEqual(d.cfg.reappraisal_mode, "v33")

    def test_invalid_reappraisal_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            NPN(seed=3490, reappraisal_mode="v35")

    def test_schema_is_8_and_a_schema_7_state_loads_by_default(self):
        self.assertEqual(SCHEMA_VERSION, 8)
        self.assertEqual(READABLE_SCHEMAS, (7, 8))
        n = build(3490, "v33", src="none", err="report", n=2, upto="all")
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            p = os.path.join(d, "state.json")
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            meta = doc["meta"]
            # スキーマ 7 の状態を作る: v3.4 で足した項目を取り除く（設定のキーと meta["reappraisal"]）
            for k in ("reappraisal_mode", "lambda_g", "kappa_alpha0", "kappa_beta0", "g_logit_clip"):
                meta["config"].pop(k)
            meta.pop("reappraisal")
            with np.load(os.path.join(d, "arrays.npz"), allow_pickle=False) as z:
                arrays = {k: np.array(z[k]) for k in z.files}
            doc["schema_version"] = 7
            doc["content_hash"] = _digest(meta, arrays)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
            m = NPN.load(d)
            self.assertEqual(m.cfg.reappraisal_mode, "v33")
            self.assertEqual((m.src_beta, m.ver_pending, m.G, m.n_ticks), ({}, [], {}, 0))
            self.assertTrue(np.array_equal(m.value.person.W, n.value.person.W))
            self.assertEqual(json.dumps(m.exp_log, sort_keys=True), json.dumps(n.exp_log, sort_keys=True))
            # 7 の状態の続きは、8 の状態の続きと同じ
            e = H._insert("zz3")
            self.assertEqual(json.dumps(m.experience(copy.deepcopy(e)), sort_keys=True, default=str),
                             json.dumps(n.experience(copy.deepcopy(e)), sort_keys=True, default=str))
            # 6 は拒否する
            doc["schema_version"] = 6
            with open(p, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False)
            with self.assertRaises(StateVersionError):
                NPN.load(d)

    def test_advance_clock_moves_only_the_clock(self):
        n = build_fresh(3490, "v34", src="none", err="report", n=2, upto="mis")
        before = snapshot(n)
        t0, k0 = n._clock(), n.n_ticks
        n.advance_clock(3)
        self.assertEqual((n.n_ticks, n._clock()), (k0 + 3, t0 + 3.0))
        self.assertEqual(n.cond.now, n._clock())
        m1, a1 = snapshot(n)
        meta0, meta1 = json.loads(before[0]), json.loads(m1)
        for k in ("belief", "value", "concern", "conditioning", "exp_log", "mood"):
            self.assertEqual(json.dumps(meta0[k], sort_keys=True), json.dumps(meta1[k], sort_keys=True), k)
        self.assertTrue(all(np.array_equal(before[1][k], a1[k]) for k in a1))
        self.assertEqual(sum(1 for e in n.ledger.entries if e["kind"] == "tick"), k0 + 3)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            self.assertEqual(NPN.load(d).n_ticks, k0 + 3)

    def test_only_development_seeds_are_used(self):
        self.assertTrue(set(USED_SEEDS) <= set(DEV_SEEDS))
        self.assertFalse(set(USED_SEEDS) & FORBIDDEN)


if __name__ == "__main__":
    unittest.main()
