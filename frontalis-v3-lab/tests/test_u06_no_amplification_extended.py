"""U6 自己増幅なし（拡張）: 全機構を有効にした閉ループで、状態の摂動が増幅しない。

SPEC-v3.1 第 3 節: 全機構を有効にして、同じ経験史を 2000 経験まで延長し、状態に 1e-3 の摂動を入れて共通乱数で再生する。
合格: 全ての値が有界（|C| < 10、a の各成分 < 5）。最終状態の摂動の比 ‖Δ最終‖/‖Δ初期‖ が 20 本の中央値で 1 以下。

測り方（実装前に固定した細部）:
  全機構 = 既定の設定（関心の重み a、条件づけ C、参照点、曖昧な結果の解釈、事前ヘッド、κ_e の推定、スコープ帰属、optional の接近・回避）。
  経験史 = generate_extended_history(型, 種, 2000)。型は種に応じて R, K, X を巡回（種 0 = R, 1 = K, 2 = X, 3 = R, ...）。
    相手は 12 人の個体が繰り返し現れ、8% の協働に曖昧な結果、20% の協働は optional な申し出。
  摂動の入れ方: 最初の 200 経験の後の状態に、価値ネットの重み W（人・状況）、条件づけ C、関心の重み a、気分 m、欲求の状態 h へ
    独立なガウス雑音を入れ、全体のノルムを 1e-3 に正規化する（a は摂動の後に和 5 へ再正規化）。残りの 1800 経験を、
    基準と摂動の 2 つの NPN に同じ順序で与える。optional の接近・回避の乱数は同じ状態から始める（共通乱数）。
  ‖Δ‖ = 上の 5 種の状態（特徴名で対応づけ、片方にしかない特徴は 0）の差の L2 ノルム。Σ・信念の計数は摂動しない。
  有界の検査: 基準の再生の 100 経験ごとに max|C| と max a_k を記録し、最大値が上の上限未満。

v3.2 の変更（W6 を同じ実行に統合）:
  W6（SPEC-v3.2: U6 の手順を、v3.2 の全機構と申し出を含む経験史で行う。有界、摂動の比の中央値 ≤ 1、選択が反転した本の数を報告）は、
  この U6 の実行と同じ計算で測る（テスト時間のため。別の実行にすると 2 倍かかる）。そのため経験史に praise_rate = 0.25 を加えた
  （協働の 25% で、協働の前に同じ相手に褒められる事象が先行する。申し出のときは announced = ["praise"]。C → Q → 選択 → 経験 → C のループを使う）。
  U6 の摂動 A（W、C、a、m、h）と判定（有界、比の中央値 ≤ 1）は v3.1 のまま。変更前（praise なし、v3.2 の機構）: A の比の中央値 0.700、B 0.531。
  摂動 B は、v3.2 の状態（ヘッドの前向き対数尤度 ll_ema、ヘッドの重みの状態）にも雑音を入れる形に広げた = W6 の摂動。
  W6 = 摂動 B の結果（クラス W6NoAmplificationV32）。

設計の改訂 2（想起・再帰属）の後の追加: 想起は既定で有効（K_r = 10）。U6 の判定（上の摂動 A）は想起ありのまま測る。
  摂動 B（追加の検査。判定基準は U6 と同じ）: A に加えて、想起の記録の φ（参加者が行為者の観測、全体で 1 ノルムの雑音）を摂動し、
  計数を記録から作り直す。‖Δ‖ には記録の φ の差も加える。想起のループ（φ → 計数 → 信念 → φ）の増幅を測る。
  v3.2: B はさらにヘッドの前向き対数尤度 ll_ema（ヘッドの重みの状態）にも雑音を入れ、‖Δ‖ にその差も加える（= W6）。
"""
import copy
import unittest

import numpy as np

from helpers import KINDS, SEEDS, dump, make, mh, pmap

WARMUP = 200
TOTAL = 2000
EPS = 1e-3
PRAISE_RATE = 0.25      # v3.2: 協働の 25% で褒められる事象が先行する（W6 / C → Q → 選択のループを使う）


def state_vec(n):
    d = {}
    for tag, kal in (("pW", n.value.person), ("sW", n.value.sit)):
        for i, name in enumerate(kal.names):
            for k in range(5):
                d[(tag, name, k)] = float(kal.W[i, k])
    for name, c in zip(n.cond.names, n.cond.C):
        d[("C", name)] = float(c)
    for k in range(5):
        d[("a", k)] = float(n.concern.a[k]); d[("h", k)] = float(n.h[k])
    d[("m",)] = float(n.mood)
    return d


def distance(n1, n2):
    a, b = state_vec(n1), state_vec(n2)
    return float(np.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in set(a) | set(b))))


def perturb(n, rng, with_head=False):
    """価値ネット W・C・a・m・h（with_head=True ではヘッドの ll_ema も）に、全体のノルムが EPS のガウス雑音を入れる。"""
    shapes = [n.value.person.W.shape, n.value.sit.W.shape, n.cond.C.shape, (5,), (), (5,)]
    if with_head:
        shapes.append(n.belief.ll_ema.shape)
    noise = [rng.normal(size=s) for s in shapes]
    norm = np.sqrt(sum(float(np.sum(x ** 2)) for x in noise))
    noise = [x * EPS / norm for x in noise]
    n.value.person.W = n.value.person.W + noise[0]
    n.value.sit.W = n.value.sit.W + noise[1]
    n.cond.C = n.cond.C + noise[2]
    a = n.concern.a + noise[3]
    n.concern.a = 5.0 * a / a.sum()
    n.mood = n.mood + float(noise[4])
    n.h = n.h + noise[5]
    if with_head:
        n.belief.ll_ema = n.belief.ll_ema + noise[6]


def perturb_records(n, rng):
    """想起の記録の φ（観測されたセル）に、ノルム EPS のガウス雑音を入れ、計数を記録から作り直す。"""
    b = n.belief
    R = b.rec_n
    mask = b.rec_m[:R] > 0.5
    noise = rng.normal(size=int(mask.sum()))
    noise *= EPS / np.linalg.norm(noise)
    phi = b.rec_phi[:R].copy()
    phi[mask] = np.clip(phi[mask] + noise, 1e-6, 1.0 - 1e-6)
    b.rec_phi[:R] = phi
    b._rebuild_counts()


def phi_distance(n1, n2):
    a, b = n1.belief, n2.belief
    R = max(a.rec_n, b.rec_n)
    pa, pb = np.zeros((R, a.rec_phi.shape[1])), np.zeros((R, a.rec_phi.shape[1]))
    pa[: a.rec_n] = a.rec_phi[: a.rec_n] * (a.rec_m[: a.rec_n] > 0.5)
    pb[: b.rec_n] = b.rec_phi[: b.rec_n] * (b.rec_m[: b.rec_n] > 0.5)
    return float(np.sqrt(np.sum((pa - pb) ** 2)))


def full_distance(n1, n2):
    """摂動 B / W6 の ‖Δ‖: 状態の差 ＋ 想起の記録の φ の差 ＋ ヘッドの ll_ema の差。"""
    d2 = distance(n1, n2) ** 2 + phi_distance(n1, n2) ** 2 + float(np.sum((n1.belief.ll_ema - n2.belief.ll_ema) ** 2))
    return float(np.sqrt(d2))


def worker(seed):
    kind = KINDS[seed % 3]
    hist = mh.generate_extended_history(kind, seed, TOTAL, praise_rate=PRAISE_RATE)
    base = make(seed, record_ledger=False)
    for e in hist[:WARMUP]:
        base.experience(e)
    pertA = copy.deepcopy(base)
    perturb(pertA, np.random.default_rng([seed, 77]))
    pertB = copy.deepcopy(base)
    perturb(pertB, np.random.default_rng([seed, 77]), with_head=True)
    perturb_records(pertB, np.random.default_rng([seed, 78]))
    d0A = distance(base, pertA)
    d0B = full_distance(base, pertB)
    for p in (pertA, pertB):
        p.rng.bit_generator.state = base.rng.bit_generator.state          # 共通乱数
    max_c, max_a, flips, flipsB, offers, praise_offers, praise_avoided = 0.0, 0.0, 0, 0, 0, 0, 0
    for i, e in enumerate(hist[WARMUP:]):
        r1 = base.experience(e)
        r2 = pertA.experience(e)
        r3 = pertB.experience(e)
        if e["optional"]:
            offers += 1
            flips += int(r1["approached"] != r2["approached"] or r1["approached"] != r3["approached"])
            flipsB += int(r1["approached"] != r3["approached"])
            if e.get("announced"):
                praise_offers += 1
                praise_avoided += int(not r1["approached"])
        if i % 100 == 0 or i == TOTAL - WARMUP - 1:
            for q in (base, pertA, pertB):
                max_c = max(max_c, float(np.abs(q.cond.C).max()))
                max_a = max(max_a, float(q.concern.a.max()))
    piv = base.belief.pi[:, 0] / base.belief.pi.sum(axis=1)
    return {"seed": seed, "kind": kind, "ratio": distance(base, pertA) / d0A, "d0": d0A, "max_C": max_c, "max_a": max_a,
            "ratio_with_records": full_distance(base, pertB) / d0B,
            "d0_with_records": d0B, "pi_person_range": [float(piv.min()), float(piv.max())],
            "records": base.belief.rec_n,
            "choice_flips": flips, "choice_flips_B": flipsB, "offers": offers, "praise_offers": praise_offers,
            "praise_offers_avoided": praise_avoided,
            "C_praise_final": base.cond.value_of("type:praise") if "type:praise" in base.cond.idx else 0.0,
            "head_weights": base.belief.head_weights_dict(),
            "n_avoided": base.n_avoided, "n_exp": base.n_experiences,
            "n_features": [len(base.value.person.names), len(base.value.sit.names), len(base.cond.names)]}


_ROWS = None


def get_rows():
    """U6 と W6 で同じ実行を共有する（モジュール内で 1 回だけ）。"""
    global _ROWS
    if _ROWS is None:
        _ROWS = pmap(worker, SEEDS)
    return _ROWS


class U6NoAmplificationExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = get_rows()

    def test_perturbation_of_recall_records_does_not_grow(self):
        """摂動 B（想起の記録の φ を含む）。判定基準は U6 と同じ: 比の中央値 ≤ 1。"""
        r2 = [r["ratio_with_records"] for r in self.rows]
        dump("U06b", {"median_ratio": float(np.median(r2)), "max_ratio": float(np.max(r2)), "ratios": r2,
                      "pi_person_range_min": min(r["pi_person_range"][0] for r in self.rows),
                      "pi_person_range_max": max(r["pi_person_range"][1] for r in self.rows),
                      "records_mean": float(np.mean([r["records"] for r in self.rows]))})
        self.assertLessEqual(float(np.median(r2)), 1.0)

    def test_bounded_and_perturbation_does_not_grow(self):
        rows = self.rows
        ratios = [r["ratio"] for r in rows]
        dump("U06", {"median_ratio": float(np.median(ratios)), "max_ratio": float(np.max(ratios)),
                     "ratios": ratios, "max_abs_C": max(r["max_C"] for r in rows), "max_a": max(r["max_a"] for r in rows),
                     "choice_flips_total": int(sum(r["choice_flips"] for r in rows)),
                     "praise_rate_in_history": PRAISE_RATE,
                     "kinds": [r["kind"] for r in rows], "n_experiences": rows[0]["n_exp"],
                     "avoided_mean": float(np.mean([r["n_avoided"] for r in rows])),
                     "features_seed0": rows[0]["n_features"]})
        self.assertLess(max(r["max_C"] for r in rows), 10.0)
        self.assertLess(max(r["max_a"] for r in rows), 5.0)
        self.assertLessEqual(float(np.median(ratios)), 1.0)


class W6NoAmplificationV32(unittest.TestCase):
    """W6（SPEC-v3.2）: v3.2 の全機構（a → R_k → W、C → Q → 選択 → 経験 → C、ヘッドの重み、想起）と申し出を含む経験史で、状態の摂動が増幅しないか。
    合格: 有界（|C| < 10、a < 5）、摂動の比の中央値 ≤ 1。選択が反転した本の数を報告する（合否には使わない）。
    実行は U6 と共有（上の docstring）。摂動 = 摂動 B（W、C、a、m、h、ll_ema、想起の記録の φ）。"""

    @classmethod
    def setUpClass(cls):
        cls.rows = get_rows()

    def test_bounded_and_perturbation_does_not_grow_with_all_v32_mechanisms(self):
        rows = self.rows
        ratios = [r["ratio_with_records"] for r in rows]
        by_kind = {k: float(np.median([r["ratio_with_records"] for r in rows if r["kind"] == k])) for k in KINDS}
        dump("W06", {"median_ratio": float(np.median(ratios)), "max_ratio": float(np.max(ratios)), "ratios": ratios,
                     "median_ratio_by_kind": by_kind,
                     "max_abs_C": max(r["max_C"] for r in rows), "max_a": max(r["max_a"] for r in rows),
                     "seeds_with_choice_flips": int(sum(r["choice_flips_B"] > 0 for r in rows)),
                     "choice_flips_total": int(sum(r["choice_flips_B"] for r in rows)),
                     "offers_total": int(sum(r["offers"] for r in rows)),
                     "praise_offers_total": int(sum(r["praise_offers"] for r in rows)),
                     "praise_offers_avoided_total": int(sum(r["praise_offers_avoided"] for r in rows)),
                     "C_praise_final_by_kind": {k: float(np.mean([r["C_praise_final"] for r in rows if r["kind"] == k])) for k in KINDS},
                     "head_weights_final_mean": {h: float(np.mean([r["head_weights"][h] for r in rows])) for h in rows[0]["head_weights"]},
                     "n_experiences": rows[0]["n_exp"], "avoided_mean": float(np.mean([r["n_avoided"] for r in rows])),
                     "kinds": [r["kind"] for r in rows]})
        self.assertLess(max(r["max_C"] for r in rows), 10.0)
        self.assertLess(max(r["max_a"] for r in rows), 5.0)
        self.assertLessEqual(float(np.median(ratios)), 1.0)


if __name__ == "__main__":
    unittest.main()
