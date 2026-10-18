"""U7 NN の寄与の正直な比較。

SPEC-v3.1 第 3 節: T8 の課題を mlp、linear、knn1、none で比べる。さらに R/K/X の経験史で、ヘッドが類の事前に与えた重み ρ_h の値を報告する。
判定: mlp の誤差が none の 0.7 倍以下（T8 と同じ）。linear・knn1 との比較は合否にせず、数値を報告する（NN が勝たない場合は「勝たない」と書く）。

測り方（実装前に固定した細部）:
  課題 = T8（data/histories/T8_train.jsonl: 共同作業者は手柄を取る率 0.7、師・見知らぬ人は 0.05。120 経験）。
  誤差 = 未見の類「バンド仲間」の未見の個体と、共同作業者の未見の個体の P(credit_taken | music_session) の差の絶対値（T8 と同じ）。
  head_kind ごとに種 0〜19（mlp の初期値と学習順が種で変わる）。ρ_h = n_head/(n_head+30)。
  R/K/X の ρ_h: 種ごとの生成器の経験史（36 件）の後の、観測数 n_head と ρ_h の平均。

v3.3 の結果（閾値は変えていない）: mlp / none の誤差比 0.772（> 0.7）で **不合格**（v3.2: 0.277）。mlp の誤差 0.0559 → 0.1555。
  原因: 共同作業者（既知の類）の事前が、類を除いたヘッドになり（共同作業者の予測 0.58 → 0.32）、未見の類 band_mate は全データのモデルの値（0.52 → 0.48）のままなので、
  二つの差が縮まらない。v3.2 の「差が小さい」は、既知の類が自分のデータを事前に使っていたことに依っていた。学習の回数を増やすと（head_fit_steps = 3000）差は
  縮まる（比 0.33）が、それは band_mate が下がって共同作業者に近づくためで、転移ではない（band_mate 0.41、共同作業者 0.34）。
  単一ヘッド（mlp / linear / knn1）はどれも比 0.77〜0.89 で、mixture（既定）は 0.165 で合格のまま（T8）。ただし T8 の合格も、重みが none に 0.97 寄り、
  どちらの類も全体の率 0.34〜0.37 になるためで、埋め込みの類似度による転移ではない（W07）。
"""
import unittest

import numpy as np

from helpers import KINDS, SEEDS, dump, feed, load_jsonl, make, persona_gen, pmap

COLLAB = {"entity": "u7_new_collab", "class": "collaborator", "context": "music_session"}
BAND = {"entity": "u7_new_band", "class": "band_mate", "context": "music_session"}
HEADS = ("mlp", "linear", "knn1", "none", "mixture")      # v3.2: mixture（既定）を追加。判定は v3.1 のまま mlp 対 none


def worker(seed):
    train = load_jsonl("T8_train.jsonl")
    out = {"seed": seed}
    for kind in HEADS:
        n = make(seed, head_kind=kind)
        feed(n, train)
        pc, pb = n.probe(COLLAB)["p_person"]["credit_taken"], n.probe(BAND)["p_person"]["credit_taken"]
        out[kind] = {"gap": abs(pc - pb), "collab": pc, "band": pb, "rho": n.belief.rho()}
        if kind == "mixture":
            out[kind]["weights"] = n.belief.head_weights_dict()
    out["rho_gen"] = {}
    for k in KINDS:
        n = persona_gen(k, seed)
        out["rho_gen"][k] = {"n_head": n.belief.n_obs, "rho": n.belief.rho()}
    return out


class U7NNContribution(unittest.TestCase):
    def test_mlp_vs_none_judged_and_baselines_reported(self):
        rows = pmap(worker, SEEDS)
        gaps = {k: float(np.mean([r[k]["gap"] for r in rows])) for k in HEADS}
        ratio = gaps["mlp"] / gaps["none"]
        beats = {k: float(np.mean([r["mlp"]["gap"] < r[k]["gap"] for r in rows])) for k in ("linear", "knn1", "none", "mixture")}
        dump("U07", {"mean_gap": gaps, "mlp_over_none": ratio, "judged_threshold": 0.7,
                     "fraction_of_seeds_mlp_gap_smaller_than": beats,
                     "collab_prediction_mean": {k: float(np.mean([r[k]["collab"] for r in rows])) for k in HEADS},
                     "band_prediction_mean": {k: float(np.mean([r[k]["band"] for r in rows])) for k in HEADS},
                     "rho_T8": float(rows[0]["mlp"]["rho"]),
                     "rho_h_RKX_36_experiences": {k: {"n_head_mean": float(np.mean([r["rho_gen"][k]["n_head"] for r in rows])),
                                                      "rho_mean": float(np.mean([r["rho_gen"][k]["rho"] for r in rows]))}
                                                  for k in KINDS},
                     "mlp_beats_linear_and_knn1": bool(gaps["mlp"] < gaps["linear"] and gaps["mlp"] < gaps["knn1"]),
                     "mixture_over_none (v3.2 default; reported, not judged here; T8 judges it)": gaps["mixture"] / gaps["none"]})
        self.assertLessEqual(ratio, 0.7)


if __name__ == "__main__":
    unittest.main()
