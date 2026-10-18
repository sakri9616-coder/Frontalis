"""W2 C → 行動: 条件づけられた価値が、接近・回避（選択）を変えるか。

SPEC-v3.2 第 2 節: 経験史 H1（褒められた後に害）と H2（褒められた後に支え）の後に、褒め言葉を伴う申し出を 40 回与える。
合格: 接近率が H1 で H2 より 0.15 以上低い（20 本の平均）。`no_conditioning` では差が 0.05 未満。

測り方（実装前に固定した細部）:
  H1/H2 = generate_praise_history（U2 と同じ。24 エピソード。場面・大きさ・相手・合間の出来事は同じ種なら H1 と H2 で同一）。
  申し出 = generate_praise_offers(seed, 40): optional、announced = ["praise"]。参加者なし、H1/H2 の履歴に現れない文脈（studio）と概念（music）で、
    結果は褒め言葉だけ（生得の効果なし）。H1 と H2 に同じ申し出を同じ順序で与える。接近率 = 40 回のうち接近した割合。
    こうして、H1 と H2 の差が通る経路を、praise という手がかりに条件づけられた価値 C（Q に入る、1.2）にほぼ限る
    （欲求の状態 h や関心の重み a の違いを通る経路は残る。no_conditioning の対照がそれを測る）。
    乱数: H1 と H2 の人格は同じ種で、履歴の長さが同じなので、接近の乱数の流れは対応する（共通乱数）。
  参考（判定しない）: 申し出の相手が H1/H2 と同じ類 friend の新しい個体のとき（類への一般化も通る）。最初の申し出の Q と C(type:praise)。
  申し出を受ける自体が C を消す方向（褒め言葉の後に何も起きない = 消去）に働く。回避は消去を止める（標本の偏りによる有界な正帰還）。
  参考（判定しない）: 最初の 20 回と最後の 20 回の接近率、学習なしで 40 件を評価した p_approach の平均（閉ループの消去を除いた、C → 選択の効果そのもの）。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, feed, make, mh, pmap


def rates(seed, kw, var, with_partner):
    n = make(seed, **kw)
    feed(n, mh.generate_praise_history(var, seed))
    c0 = n.probe({"cue": "type:praise"})["C"]
    offers = mh.generate_praise_offers(seed, 40, with_partner=with_partner)
    first = n.appraise(offers[0])
    open_loop = float(np.mean([n.appraise(e)["p_approach"] for e in offers]))      # 学習なしで 40 件を評価した p_approach の平均（参考）
    app = [bool(n.experience(e)["approached"]) for e in offers]
    return {"rate": float(np.mean(app)), "first20": float(np.mean(app[:20])), "last20": float(np.mean(app[20:])),
            "C0": c0, "Q0": first["Q"], "C_Q0": first["C_Q"], "p0": first["p_approach"], "open_loop": open_loop}


def worker(seed):
    out = {"seed": seed}
    for name, kw in (("full", {}), ("no_conditioning", {"no_conditioning": True}), ("no_cq", {"no_cq": True})):
        for var in ("H1", "H2"):
            out[f"{name}_{var}"] = rates(seed, kw, var, False)
    for var in ("H1", "H2"):
        out[f"partner_{var}"] = rates(seed, {}, var, True)
    return out


class W2ConditioningToAction(unittest.TestCase):
    def test_harmed_after_praise_avoids_praise_bearing_offers(self):
        rows = pmap(worker, SEEDS)
        m = lambda key, f: float(np.mean([r[key][f] for r in rows]))
        diff = {name: m(f"{name}_H2", "rate") - m(f"{name}_H1", "rate") for name in ("full", "no_conditioning", "no_cq")}
        res = {"approach_rate_H1": m("full_H1", "rate"), "approach_rate_H2": m("full_H2", "rate"), "difference_H2_minus_H1": diff["full"],
               "no_conditioning_difference": diff["no_conditioning"],
               "no_cq_difference": diff["no_cq"],
               "no_conditioning_rates": [m("no_conditioning_H1", "rate"), m("no_conditioning_H2", "rate")],
               "no_cq_rates": [m("no_cq_H1", "rate"), m("no_cq_H2", "rate")],
               "first20": [m("full_H1", "first20"), m("full_H2", "first20")], "last20": [m("full_H1", "last20"), m("full_H2", "last20")],
               "first20_difference": m("full_H2", "first20") - m("full_H1", "first20"),
               "last20_difference": m("full_H2", "last20") - m("full_H1", "last20"),
               "open_loop_mean_p_approach (no learning between offers)": [m("full_H1", "open_loop"), m("full_H2", "open_loop")],
               "open_loop_difference": m("full_H2", "open_loop") - m("full_H1", "open_loop"),
               "open_loop_no_conditioning": [m("no_conditioning_H1", "open_loop"), m("no_conditioning_H2", "open_loop")],
               "C_praise_before_offers": [m("full_H1", "C0"), m("full_H2", "C0")],
               "Q_first_offer": [m("full_H1", "Q0"), m("full_H2", "Q0")],
               "p_approach_first_offer": [m("full_H1", "p0"), m("full_H2", "p0")],
               "seeds_with_diff_ge_0.15": int(sum(r["full_H2"]["rate"] - r["full_H1"]["rate"] >= 0.15 for r in rows)),
               "reference_with_friend_partner": {"rates": [m("partner_H1", "rate"), m("partner_H2", "rate")],
                                                 "difference_H2_minus_H1": m("partner_H2", "rate") - m("partner_H1", "rate")},
               "rate_H1_per_seed": [r["full_H1"]["rate"] for r in rows], "rate_H2_per_seed": [r["full_H2"]["rate"] for r in rows]}
        dump("W02", res)
        self.assertGreaterEqual(diff["full"], 0.15, res)
        self.assertLess(abs(diff["no_conditioning"]), 0.05, res)


if __name__ == "__main__":
    unittest.main()
