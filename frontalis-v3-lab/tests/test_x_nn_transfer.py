"""X1〜X4 NN の再検証（SPEC-v3.3 第 1 節）: 類を除いた事前（leave-class-out）と、転移性能での混合。判定基準は実装前に固定されたまま。

  X1 自己参照の排除  既知の類 c の観測だけを書き換えても、c の事前（ヘッドの出力）が変わらない。合格: 変化 = 0（厳密）
  X2 公平な比較      benchmarks/nn_transfer.py を実行し、4 世界 × 2 観測数 × 5 ヘッドの成績を報告する（合否なし）
  X3 NN 固有の役割   事前登録した決定則（benchmarks/nn_transfer.py 冒頭）。結論はどちらでも合格。結論と既定のヘッドの設定が食い違わないことを検査する
  X4 実データでの転移 lexicon の類で、類を除いた交差検証の対数尤度を報告する（R / K / X とランダムな世界の履歴）。X3 と矛盾する場合は理由を書く

測り方（実装前に決めた細部）:
  X1  履歴 = 4 つの類（collaborator, mentor, friend, stranger）の個体が繰り返し現れる 60 経験（種ごとに生成）。双子 = 同じ履歴で、類 c = collaborator が
      行為者の結果の型だけを入れ替えたもの（connection ↔ credit_taken、support ↔ harm。swap）。他の類の観測・経験の順序・文脈は同じ。
      （v3.3 の再開後に足した 2 つ目の双子 erase: c が行為者の結果をすべて消したもの。観測は残り、y が全部 0 になる。判定は同じ: 変化 = 0。
      swap の結果を見る前に足したのではなく、swap が合格した後に、検査が弱くないことを確かめるために足した。）
      全ヘッド（mlp, linear, knn1, knnk, none）を mixture に入れ、20・40・60 経験の後に c の事前（head_outputs）を比べる。差は 0（np.array_equal）。
      c を除いて学び直したモデルの重みも完全に同じ。対照（検査が空でないことの確認）: 双子の間で c の階層の類の率 m_cls は違い、
      他の類（mentor）の事前は（c のデータを使うヘッドで）違う。単一ヘッド（mlp, linear, knn1, knnk）でも同じ。種 0〜4。
  X4  履歴 = generate_history(R / K / X, 種, 36)、generate_random_history(種, 60)。種 0〜19。ヘッドは全部（mlp, linear, knn1, knnk, none）を mixture に入れる。
      交差検証の対数尤度 = 最終状態で全ヘッドを学び直し、リプレイ緩衝の各観測を、その観測の類を除いたヘッドで予測した対数尤度（観測 1 件 = 観測されたセルの和）。
      報告: 観測あたりの平均と、類ごとの平均の平均（類の数がちがうので両方）。ヘッドの混合の重み（ll_ema の softmax）も報告する。
"""
import copy
import tempfile
import unittest

import numpy as np

from helpers import HIST, SEEDS, deep_equal, dump, feed, make, mh, pmap
import nn_transfer as bench          # helpers が benchmarks/ を sys.path に入れる
from npnv3 import NPN, Config, Lexicon
from npnv3.belief import HEAD_NAMES

ALL_HEADS = ("mlp", "linear", "knn1", "knnk", "none")
SWAP = {"connection": "credit_taken", "credit_taken": "connection", "support": "harm", "harm": "support"}


# ----------------------------------------------------------------------
# X1
# ----------------------------------------------------------------------
def x1_history(seed, n=60):
    """4 つの類の個体が繰り返し現れる履歴。類ごとに結果の確率が違う。"""
    rng = np.random.default_rng([seed, 11])
    classes = {"collaborator": (["connection", "credit_taken", "achievement"], [0.7, 0.5, 0.4]),
               "mentor": (["taught", "support", "criticism"], [0.7, 0.3, 0.2]),
               "friend": (["comfort", "support", "connection"], [0.6, 0.5, 0.5]),
               "stranger": (["novelty", "rejection", "harm"], [0.5, 0.3, 0.2])}
    ctxs = ["music_session", "cafe_chat", "workshop", "project_meeting"]
    rows = []
    for i in range(n):
        names = list(classes)
        k = int(rng.integers(1, 3))
        picks = list(rng.choice(len(names), size=k, replace=False))
        parts, outs = [], []
        for j in picks:
            cls = names[j]
            pid = f"x1-{cls}-{int(rng.integers(3))}"
            parts.append({"id": pid, "class": cls})
            types, probs = classes[cls]
            for t, p in zip(types, probs):
                if rng.random() < p:
                    outs.append({"type": t, "actor": pid, "magnitude": round(float(rng.uniform(0.3, 1.0)), 2)})
        rows.append({"id": f"x1-{seed:02d}-{i:03d}", "context": ctxs[int(rng.integers(len(ctxs)))], "concepts": ["collaboration"],
                     "participants": parts, "outcomes": outs, "optional": False, "session": f"x1-{seed}-{i // 3}"})
    return rows


def rewrite_class(rows, cls):
    """類 cls が行為者の結果の型だけを入れ替えた双子（他は同じ）。"""
    twin = copy.deepcopy(rows)
    pid_cls = {p["id"]: p["class"] for e in twin for p in e["participants"]}
    for e in twin:
        for o in e["outcomes"]:
            if pid_cls.get(o["actor"]) == cls and o["type"] in SWAP:
                o["type"] = SWAP[o["type"]]
    return twin


def erase_class(rows, cls):
    """類 cls が行為者の結果をすべて消した双子（観測は残り、その類の y が全部 0 になる）。"""
    twin = copy.deepcopy(rows)
    pid_cls = {p["id"]: p["class"] for e in twin for p in e["participants"]}
    for e in twin:
        e["outcomes"] = [o for o in e["outcomes"] if pid_cls.get(o["actor"]) != cls]
    return twin


def x1_worker(args):
    seed, kind, twin_kind = args
    kw = {"head_kind": "mixture", "mixture_heads": ALL_HEADS} if kind == "mixture" else {"head_kind": kind}
    rows = x1_history(seed)
    twin = rewrite_class(rows, "collaborator") if twin_kind == "swap" else erase_class(rows, "collaborator")
    a, b = make(seed, **kw), make(seed, **kw)
    out = {"seed": seed, "kind": kind, "twin": twin_kind, "checkpoints": []}
    ctx = "music_session"
    names = ALL_HEADS if kind == "mixture" else (kind,)
    for upto in (20, 40, 60):
        feed(a, rows[upto - 20:upto]); feed(b, twin[upto - 20:upto])
        ha = a.belief.head_outputs(["collaborator"], ctx, names)
        hb = b.belief.head_outputs(["collaborator"], ctx, names)
        ma = a.belief.head_outputs(["mentor"], ctx, names)
        mb = b.belief.head_outputs(["mentor"], ctx, names)
        ci = a.belief.rawcls_idx["collaborator"]
        loo_equal = True
        for nm in a.belief.trained_names():
            la, lb = a.belief.loo[nm][ci], b.belief.loo[nm][ci]
            loo_equal &= all(np.array_equal(x, y) for x, y in zip(_arrs(la), _arrs(lb)))
        full_equal = all(np.array_equal(a.belief.heads[nm].W if nm == "linear" else a.belief.heads[nm].W2,
                                        b.belief.heads[nm].W if nm == "linear" else b.belief.heads[nm].W2)
                         for nm in a.belief.trained_names())
        pa = a.belief.predict(ctx, [{"id": "x1-new", "class": "collaborator"}]).m_cls
        pb = b.belief.predict(ctx, [{"id": "x1-new", "class": "collaborator"}]).m_cls
        mix = None
        if kind == "mixture":                        # 報告のみ: 混合した事前 Σ_h w_h prior_h(c) は、重み w を通じて c の観測に依る（SPEC-v3.3 1.2 の交差検証による選択）
            mp = []
            for bel in (a.belief, b.belief):
                agg = bel._class_agg()
                mg, _ = bel._gen_mean(agg)
                mp.append(bel._priors(["collaborator"], ctx, mg, agg)[0])
            mix = float(np.abs(mp[0] - mp[1]).max())
        out["checkpoints"].append({
            "mixture_prior_max_abs_diff_via_weights (reported, not judged)": mix,
            "n": upto,
            "head_outputs_equal": {nm: bool(np.array_equal(ha[nm], hb[nm])) for nm in names},
            "head_output_max_abs_diff": {nm: float(np.abs(ha[nm] - hb[nm]).max()) for nm in names},
            "loo_models_equal": bool(loo_equal),
            "full_models_equal (expected False: they include collaborator)": bool(full_equal),
            "class_rate_m_cls_max_abs_diff (hierarchy uses own data)": float(np.abs(pa - pb).max()),
            "mentor_output_max_abs_diff": {nm: float(np.abs(ma[nm] - mb[nm]).max()) for nm in names},
            "mixture_weights_a": a.belief.head_weights_dict(), "mixture_weights_b": b.belief.head_weights_dict(),
        })
    return out


def _arrs(model):
    return list(model.get_arrays("").values())


class X1SelfReferenceEliminated(unittest.TestCase):
    def test_rewriting_a_class_does_not_change_its_head_outputs(self):
        jobs = [(s, k, tw) for s in range(5) for k in ("mixture", "mlp", "linear", "knn1", "knnk") for tw in ("swap", "erase")]
        rows = pmap(x1_worker, jobs)
        mixd = [c["mixture_prior_max_abs_diff_via_weights (reported, not judged)"] for r in rows if r["kind"] == "mixture" for c in r["checkpoints"]]
        res = {"jobs": [{"seed": r["seed"], "kind": r["kind"], "twin": r["twin"], "checkpoints": r["checkpoints"]} for r in rows[:6]],
               "max_abs_diff_over_all_heads_and_checkpoints": max(
                   max(c["head_output_max_abs_diff"].values()) for r in rows for c in r["checkpoints"]),
               "mixture_prior_via_weights_max_abs_diff (reported, not judged)": {"max": float(max(mixd)), "mean": float(np.mean(mixd))},
               "n_jobs": len(rows)}
        dump("X01", res)
        for r in rows:
            for c in r["checkpoints"]:
                tag = (r["seed"], r["kind"], r["twin"], c["n"])
                for nm, ok in c["head_outputs_equal"].items():
                    self.assertTrue(ok, (tag, nm, c["head_output_max_abs_diff"][nm]))         # 変化 = 0（厳密）
                self.assertTrue(c["loo_models_equal"], tag)
                # 対照: 検査は空でない。c の階層の類の率（自分のデータを使う）は変わり、他の類の事前（c のデータを使うヘッド）も変わる
                self.assertGreater(c["class_rate_m_cls_max_abs_diff (hierarchy uses own data)"], 1e-3, tag)
        mix = [r for r in rows if r["kind"] == "mixture"]
        for r in mix:
            c = r["checkpoints"][-1]
            self.assertGreater(c["mentor_output_max_abs_diff"]["none"], 1e-3)               # none（他の類をまとめた率）は mentor の事前では c を含む
            self.assertGreater(c["mentor_output_max_abs_diff"]["linear"], 1e-6)
            self.assertFalse(c["full_models_equal (expected False: they include collaborator)"])


# ----------------------------------------------------------------------
# X2, X3
# ----------------------------------------------------------------------
_BENCH = None


def bench_result():
    global _BENCH
    if _BENCH is None:
        _BENCH = bench.run_benchmark()
    return _BENCH


class X2FairComparison(unittest.TestCase):
    def test_benchmark_runs_and_reports_all_cells(self):
        res = bench_result()
        self.assertEqual(len(res["cells"]), 8)
        for key, c in res["cells"].items():
            self.assertEqual(set(c["ce_mean"]), set(ALL_HEADS))
            for h in ALL_HEADS:
                self.assertTrue(np.isfinite(c["ce_mean"][h]), (key, h))
                self.assertGreater(c["ce_mean"][h], c["entropy_mean"] - 1e-9)       # 交差エントロピーは真の率のエントロピー以上
            self.assertEqual(sum(c["seeds_best"].values()), res["seeds"])
        sens = {str(s): bench.run_benchmark(steps=s) for s in (100, 1000)}
        dump("X02", {"primary": res, "table_markdown": bench.markdown_table(res), "table_excess_kl_markdown": bench.markdown_table(res, "excess_kl_mean"),
                     "sensitivity_not_judged": {k: {"cells": v["cells"], "x3_rule_applied": v["x3"]} for k, v in sens.items()}})

    def test_benchmark_is_deterministic(self):
        a = bench.run_benchmark(seeds=3)
        b = bench.run_benchmark(seeds=3)
        self.assertTrue(deep_equal(a["cells"], b["cells"]))


class X3NNSpecificRole(unittest.TestCase):
    def test_decision_rule_and_default_heads_agree(self):
        res = bench_result()
        x3 = res["x3"]
        dump("X03", {"conclusion": x3["conclusion"], "mlp_best_conditions": x3["mlp_best_conditions"],
                     "mlp_rows": {k: c["mlp"] for k, c in res["cells"].items()},
                     "default_mixture_heads": list(Config().mixture_heads)})
        default_has_mlp = "mlp" in Config().mixture_heads
        if x3["mlp_best_conditions"]:
            self.assertTrue(default_has_mlp)                  # (i): mlp は既定に残す
        else:
            self.assertFalse(default_has_mlp)                 # (ii): 既定から mlp を外す
        # 設定は残す: mlp を入れた mixture と、単一ヘッド mlp は使える
        n = make(0, mixture_heads=ALL_HEADS)
        self.assertIn("mlp", n.belief.heads)
        self.assertIn("mlp", make(0, head_kind="mlp").belief.heads)


# ----------------------------------------------------------------------
# X4
# ----------------------------------------------------------------------
def x4_worker(seed):
    out = {"seed": seed}
    cfg = {"head_kind": "mixture", "mixture_heads": ALL_HEADS}
    for name, hist in (("R", mh.generate_history("R", seed, 36)), ("K", mh.generate_history("K", seed, 36)),
                       ("X", mh.generate_history("X", seed, 36)), ("random", mh.generate_random_history(seed, 60))):
        n = make(seed, **cfg)
        feed(n, hist)
        w = n.belief.head_weights_dict()
        cv = copy.deepcopy(n).belief.loo_cv_loglik()
        out[name] = {"cv": {h: {"per_observation": cv[h]["per_observation"], "per_class_mean": cv[h]["per_class_mean"]} for h in cv},
                     "n_classes": len(n.belief.rawcls), "classes": list(n.belief.rawcls), "n_obs": n.belief.n_obs, "weights": w}
    return out


class X4RealDataTransfer(unittest.TestCase):
    def test_report_leave_class_out_loglik_on_frontalis_histories(self):
        rows = pmap(x4_worker, SEEDS)
        res = {}
        for name in ("R", "K", "X", "random"):
            res[name] = {
                "n_classes_mean": float(np.mean([r[name]["n_classes"] for r in rows])),
                "cv_loglik_per_observation": {h: float(np.mean([r[name]["cv"][h]["per_observation"] for r in rows])) for h in ALL_HEADS},
                "cv_loglik_per_class_mean": {h: float(np.mean([r[name]["cv"][h]["per_class_mean"] for r in rows])) for h in ALL_HEADS},
                "seeds_where_head_best (per observation)": {h: int(sum(max(r[name]["cv"], key=lambda k: r[name]["cv"][k]["per_observation"]) == h for r in rows)) for h in ALL_HEADS},
                "mixture_weight_mean": {h: float(np.mean([r[name]["weights"][h] for r in rows])) for h in HEAD_NAMES},
            }
            best = max(res[name]["cv_loglik_per_observation"], key=res[name]["cv_loglik_per_observation"].get)
            res[name]["best_head_per_observation"] = best
            res[name]["mlp_minus_best_per_observation"] = res[name]["cv_loglik_per_observation"]["mlp"] - res[name]["cv_loglik_per_observation"][best]
        dump("X04", res)
        for name in res:
            for h in ALL_HEADS:
                self.assertTrue(np.isfinite(res[name]["cv_loglik_per_observation"][h]))
                self.assertLessEqual(res[name]["cv_loglik_per_observation"][h], 0.0)


# ----------------------------------------------------------------------
# 機構の単体検査
# ----------------------------------------------------------------------
def _structured_lexicon(seed, n_cls=12):
    d = Lexicon.load().to_dict()
    rng = np.random.default_rng([seed, 21])
    E = rng.normal(size=(n_cls, 8))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    for i in range(n_cls):
        d["classes"][f"z{i:02d}"] = E[i].tolist()
    return Lexicon(d), E


def _transfer_world_history(seed, E, structured, per_class=8):
    rng = np.random.default_rng([seed, 22])
    n_cls = E.shape[0]
    w = rng.normal(0.0, 3.0, size=8)
    p = 1.0 / (1.0 + np.exp(-(E @ w))) if structured else rng.uniform(0.1, 0.9, size=n_cls)
    rows = []
    order = rng.permutation(n_cls * per_class) % n_cls
    for i, c in enumerate(order):
        pid = f"tw{seed}-{c}-{i}"
        outs = [{"type": "connection", "actor": pid, "magnitude": 0.6}]
        if rng.random() < p[c]:
            outs.append({"type": "credit_taken", "actor": pid, "magnitude": 0.7})
        rows.append({"id": f"tw{seed}-{i:03d}", "context": "music_session", "concepts": ["collaboration"],
                     "participants": [{"id": pid, "class": f"z{c:02d}"}], "outcomes": outs, "optional": False})
    return rows


class XMechanisms(unittest.TestCase):
    def test_models_are_retrained_every_ten_experiences_per_class(self):
        n = make(0, mixture_heads=ALL_HEADS)
        rows = x1_history(0)
        feed(n, rows[:9])
        self.assertEqual(n.belief.loo["linear"], [])                      # まだ学び直していない
        feed(n, rows[9:10])
        self.assertEqual(len(n.belief.loo["linear"]), len(n.belief.rawcls))
        self.assertEqual(len(n.belief.loo["mlp"]), len(n.belief.rawcls))
        k = len(n.belief.rawcls)
        feed(n, rows[10:19])
        self.assertEqual(len(n.belief.loo["linear"]), k)                  # 10 経験ごと（19 経験ではまだ）
        feed(n, rows[19:20])
        self.assertEqual(n.belief.exp_count, 20)

    def test_loo_model_excludes_the_class_and_full_model_includes_it(self):
        n = make(0, head_kind="linear")
        feed(n, x1_history(0)[:40])
        ci = n.belief.rawcls_idx["collaborator"]
        loo, full = n.belief.loo["linear"][ci], n.belief.heads["linear"]
        self.assertFalse(np.array_equal(loo.W, full.W))
        # 未見の類（データがない）には全データのモデルが使われる
        self.assertIs(n.belief._model_for("linear", "never_seen"), full)
        self.assertIs(n.belief._model_for("linear", "collaborator"), loo)

    def test_transfer_loglik_is_head_prediction_for_the_left_out_class(self):
        """ヘッドの重みを決める対数尤度は、その参加者の類を除いたヘッドの出力で、観測されたセルだけを数えたもの（階層の計数は通さない）。"""
        n = make(0, mixture_heads=ALL_HEADS)
        rows = x1_history(1)
        feed(n, rows[:25])
        e = rows[25]
        P = e["participants"]
        y = np.zeros((len(P), 19)); mask = np.ones((len(P), 19))
        from npnv3.innate import TYPE_INDEX
        for o in e["outcomes"]:
            j = [p["id"] for p in P].index(o["actor"])
            y[j, TYPE_INDEX[o["type"]]] = 1.0
        hp = n.belief.head_outputs([p["class"] for p in P], e["context"])
        expected = {nm: float((mask * (y * np.log(hp[nm]) + (1 - y) * np.log(1 - hp[nm]))).sum(axis=1).mean()) for nm in hp}
        before = n.belief.ll_ema.copy(); cnt = n.belief.ll_cnt
        n.belief._update_head_ll(e["context"], P, y, mask)
        rate = max(1.0 / (cnt + 1.0), n.cfg.head_ll_rate)
        for nm, v in expected.items():
            i = HEAD_NAMES.index(nm)
            self.assertAlmostEqual(float(n.belief.ll_ema[i]), float(before[i] + rate * (v - before[i])), places=12)

    def test_weights_are_zero_for_heads_not_in_the_mixture(self):
        n = make(0)
        feed(n, x1_history(0)[:30])
        w = n.belief.head_weights_dict()
        self.assertEqual(w["mlp"], 0.0)
        self.assertEqual(w["knnk"], 0.0)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=12)
        self.assertEqual(set(n.belief.heads), {"linear"})

    def test_structured_world_gives_heads_more_weight_than_unrelated_world(self):
        """転移できる世界（率が埋め込みの線形関数）では、類を見るヘッドの重みが、率が埋め込みと無関係な世界より大きい（20 本の人格の平均）。"""
        def run(structured):
            out = []
            for s in range(6):
                lex, E = _structured_lexicon(s)
                n = NPN(seed=s, lexicon=lex)
                feed(n, _transfer_world_history(s, E, structured))
                out.append(n.belief.rho())
            return float(np.mean(out))
        self.assertGreater(run(True), run(False))

    def test_save_restore_keeps_leave_class_out_models(self):
        n = make(0, mixture_heads=ALL_HEADS)
        feed(n, x1_history(2)[:35])
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        ma, aa = n.get_state(); mb, ab = m.get_state()
        self.assertTrue(deep_equal(ma, mb) and all(np.array_equal(aa[k], ab[k]) for k in aa))
        rest = x1_history(2)[35:50]
        self.assertTrue(deep_equal(feed(n, copy.deepcopy(rest)), feed(m, copy.deepcopy(rest))))
        ma, aa = n.get_state(); mb, ab = m.get_state()
        self.assertTrue(deep_equal(ma, mb) and all(np.array_equal(aa[k], ab[k]) for k in aa))

    def test_save_restore_all_configurations(self):
        """保存 → 復元の状態が完全一致し、同じ経験を続けた結果（返り値と状態）も完全一致する。前任は mixture（全ヘッド）と既定しか検査していなかったので、
        単一ヘッド 5 種、single_cause、no_conditioning も加えた（v3.3 の再開後。判定ではなく機構の検査）。"""
        rows = x1_history(3)
        configs = [{"head_kind": k} for k in ALL_HEADS] + [{"single_cause": True}, {"mixture_heads": ALL_HEADS},
                                                            {"mixture_heads": ALL_HEADS, "single_cause": True}, {"no_conditioning": True}, {}]
        for kw in configs:
            n = make(1, **kw)
            feed(n, rows[:33])
            with tempfile.TemporaryDirectory() as d:
                n.save(d)
                m = NPN.load(d)
            ma, aa = n.get_state(); mb, ab = m.get_state()
            self.assertTrue(deep_equal(ma, mb) and aa.keys() == ab.keys() and all(np.array_equal(aa[k], ab[k]) for k in aa), kw)
            self.assertTrue(deep_equal(feed(n, copy.deepcopy(rows[33:55])), feed(m, copy.deepcopy(rows[33:55]))), kw)
            ma, aa = n.get_state(); mb, ab = m.get_state()
            self.assertTrue(deep_equal(ma, mb) and all(np.array_equal(aa[k], ab[k]) for k in aa), kw)

    def test_ledger_records_every_head_retrain(self):
        """台帳（v3.3）: 学び直しごとに、使った類・観測の数・学習するヘッド・混合の重みを kind = "head_retrain" で残す。実行は経験の台帳の前に入る。"""
        n = make(0, mixture_heads=ALL_HEADS)
        rows = x1_history(0)
        feed(n, rows[:25])
        rt = [e for e in n.ledger.entries if e["kind"] == "head_retrain"]
        self.assertEqual([e["after"] for e in rt], [rows[9]["id"], rows[19]["id"]])
        self.assertEqual(rt[-1]["trained_heads"], ["mlp", "linear"])
        self.assertEqual(sorted(rt[-1]["classes"]), sorted(n.belief.rawcls))
        self.assertAlmostEqual(sum(rt[-1]["weights"].values()), 1.0, places=12)
        self.assertEqual(n.ledger.entries[-1]["kind"], "experience")                     # 最後は常に経験の台帳
        i = [k for k, e in enumerate(n.ledger.entries) if e["kind"] == "head_retrain"][-1]
        nxt = next(e for e in n.ledger.entries[i + 1:] if e["kind"] == "experience")
        self.assertEqual(nxt["id"], rows[19]["id"])                                          # 学び直しの次の経験の台帳が、その経験のもの

    def test_head_kind_none_is_the_hierarchical_general_rate(self):
        """単一ヘッド none（no_head）は、階層の一般の率 m_gen のまま（v3.2 までと同じ）。ヘッドの学習も近傍も使わない。"""
        n = make(0, no_head=True)
        feed(n, x1_history(0)[:30])
        self.assertEqual(n.belief.heads, {})
        p = n.belief.predict("music_session", [{"id": "u", "class": "never_seen"}])
        self.assertFalse(p.head_used)


if __name__ == "__main__":
    unittest.main()
