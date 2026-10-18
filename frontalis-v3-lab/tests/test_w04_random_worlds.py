"""W4 ランダムな世界での用量反応: 人格の型を手で書かない世界でも、経験の量が受け止め方と行動に比例して反映されるか。

SPEC-v3.2 第 2 節: ランダムな世界 40 本で、各人格が経験した credit_taken の率と、曖昧な出来事の「取られた」解釈の重みの相関。
各人格の経験した平均の効用と、申し出への接近率の相関。合格: 前者の相関 ≥ 0.7、後者の相関 ≥ 0.5。

測り方（実装前に固定した細部）:
  世界 = random_world(種)（種 0〜39）。場面ごと・結果の型ごとの起こる確率を Beta(1, 3) から引く（人格の型は書かない）。
  経験史 = generate_random_history(種, n = 60, offer_rate = 0)（60 件、申し出なし = すべて経験される。
    n = 60 は実行前に決めた: 協働が約 30 件あり、率 0.25 程度の事象の頻度を ±0.08 程度で推定できる）。人格の NPN の種も同じ。
  (a) 経験した credit_taken の率 = 協働（類 collaborator）の事象のうち credit_taken が起きた割合。
      「取られた」解釈の重み = 履歴の後、未見の共同作業者（music_session）の曖昧な結果 {credit_taken: 0.5, credit_shared: 0.5} を
      appraise した解釈の重み（interpretations の weights["credit_taken"]。信念で決まる、U4 の機構）。ピアソンの相関。
  (b) 経験した平均の効用 = 経験史の全事象の du_total（Σ_k Δu_k）の平均。接近率 = 同じ世界の共同作業の申し出（generate_random_offers、40 回）に
      履歴の後で接近した割合（その人生が続く閉ループ）。ピアソンの相関。
  参考（判定しない）: スピアマンの相関、協働の事象だけの平均の効用との相関、credit_taken と credit_shared の両方で決まる「取られた比」との相関。
"""
import unittest

import numpy as np

from helpers import dump, make, mh, pmap

N_WORLDS = 40
AMB = {"id": "w4-amb", "context": "music_session", "concepts": ["collaboration"],
       "participants": [{"id": "w4-new", "class": "collaborator"}],
       "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "w4-new", "magnitude": 0.7}],
       "optional": False}


def worker(seed):
    hist = mh.generate_random_history(seed, 60, offer_rate=0.0)
    n = make(seed)
    du, du_collab = [], []
    taken, shared, collab = 0, 0, 0
    for e in hist:
        ap = n.experience(e)
        du.append(ap["du_total"])
        if e["participants"] and e["participants"][0]["class"] == "collaborator":
            collab += 1
            taken += any(o.get("type") == "credit_taken" for o in e["outcomes"])
            shared += any(o.get("type") == "credit_shared" for o in e["outcomes"])
            du_collab.append(ap["du_total"])
    w_taken = n.appraise(AMB)["interpretations"][0]["weights"]["credit_taken"]
    offers = mh.generate_random_offers(seed, 40)
    rate = float(np.mean([bool(n.experience(e)["approached"]) for e in offers]))
    return {"seed": seed, "taken_rate": taken / max(collab, 1), "shared_rate": shared / max(collab, 1), "w_taken": w_taken,
            "mean_du": float(np.mean(du)), "mean_du_collab": float(np.mean(du_collab)) if du_collab else 0.0,
            "approach": rate, "n_collab": collab,
            "true_p_taken": next(p for t, _, p, _ in mh.random_world(seed)["collab"] if t == "credit_taken")}


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


class W4RandomWorlds(unittest.TestCase):
    def test_dose_response_in_random_worlds(self):
        rows = pmap(worker, range(N_WORLDS))
        g = lambda k: np.array([r[k] for r in rows])
        taken_ratio = g("taken_rate") / np.maximum(g("taken_rate") + g("shared_rate"), 1e-9)
        r_a, r_b = corr(g("taken_rate"), g("w_taken")), corr(g("mean_du"), g("approach"))
        res = {"corr_taken_rate_vs_taken_interpretation_weight": r_a, "corr_mean_utility_vs_approach_rate": r_b,
               "spearman_a": corr(rank(g("taken_rate")), rank(g("w_taken"))), "spearman_b": corr(rank(g("mean_du")), rank(g("approach"))),
               "reference_corr_collab_only_mean_utility_vs_approach": corr(g("mean_du_collab"), g("approach")),
               "reference_corr_taken_over_taken_plus_shared_vs_weight": corr(taken_ratio, g("w_taken")),
               "reference_corr_true_p_taken_vs_weight": corr(g("true_p_taken"), g("w_taken")),
               "mean_collab_events": float(g("n_collab").mean()), "mean_taken_rate": float(g("taken_rate").mean()),
               "std_taken_rate": float(g("taken_rate").std()), "mean_w_taken": float(g("w_taken").mean()),
               "std_w_taken": float(g("w_taken").std()), "mean_approach_rate": float(g("approach").mean()),
               "std_approach_rate": float(g("approach").std()), "std_mean_utility": float(g("mean_du").std())}
        dump("W04", res)
        self.assertGreaterEqual(r_a, 0.7, res)
        self.assertGreaterEqual(r_b, 0.5, res)


if __name__ == "__main__":
    unittest.main()
