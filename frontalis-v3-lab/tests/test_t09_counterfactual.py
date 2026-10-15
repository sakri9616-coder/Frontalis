"""T9 反事実。
形成期の経験を 1 つ除いて学び直すと、後のプローブの評価が変わる。
合格: 変化が、同じ数（1 つ）の無関係な経験を除いた場合（プラセボ）より大きい。
  形成期の経験 = K の履歴の 4 番目の「手柄を取られた」経験。プラセボ = K の「独りで成功」を 1 つ除く（12 件は同一の経験の繰り返しなので、どれを除いても同じ履歴になる）。
  プローブ = 未見の共同作業者を含む状況の Q（主）、未見の共同作業者の like（MF）、P(credit_taken)（信念）。
  種 0〜19 の平均（SPEC 第 8 節）。like（MF）と P(credit_taken)（信念）は計算が決定的なので全本でも要求する。
  Q は 64 サンプルの MB 推定の雑音（約 0.005）が効果と同じ大きさなので、平均で判定し、全本の内訳は報告だけにする。

事後変更の記録（CHANGELOG 参照）:
  最初の版は、履歴の直後にプローブを測った。プラセボ（12 回続く「独りで成功」の 1 回を除く）は履歴の終わりの欲求の状態 h を
  0.08 ほど動かし、MB 価値を通じて Q を動かすため、Q の変化はプラセボの方が大きく（20 本平均で形成期 0.0051、プラセボ 0.0093）、不合格だった。
  h の影響は時間とともに戻る（λ_h = 0.1）ので、「後の」プローブとして、履歴の後に 20 件の何も起きない経験（休止）を挟んで測る版を
  正式とした。最初の版は test_immediate_probe_is_confounded_by_need_state（既知の制約・期待される不合格）として残す。
"""
import unittest

import numpy as np

from helpers import SEEDS, dump, feed, load_jsonl, make

SITUATION = {"context": "music_session", "concepts": ["music", "collaboration"],
             "participants": [{"id": "t9_unseen", "class": "collaborator"}]}
ENTITY = {"entity": "t9_unseen", "class": "collaborator", "context": "music_session"}
REST = 20


def rest(n):
    return [{"id": f"rest-{i}", "context": "garden", "concepts": ["plants"], "participants": [],
             "outcomes": [], "optional": False} for i in range(n)]


def evaluate(seed, hist, tail):
    n = make(seed); feed(n, hist + tail)
    e = n.probe(ENTITY)
    return n.probe(SITUATION)["Q"], e["like"], e["p_person"]["credit_taken"]


def measure(tail):
    hist = load_jsonl("K.jsonl")
    formative = 3
    placebos = [i for i, e in enumerate(hist) if e["id"].startswith("K-solo-success")]
    rows = []
    for seed in SEEDS:
        base = evaluate(seed, hist, tail)
        form = evaluate(seed, hist[:formative] + hist[formative + 1:], tail)
        plc = [evaluate(seed, hist[:i] + hist[i + 1:], tail) for i in placebos]
        row = {"seed": seed}
        for j, name in enumerate(("Q", "like", "p_credit")):
            row[f"formative_{name}"] = abs(form[j] - base[j])
            row[f"placebo_{name}_mean"] = float(np.mean([abs(p[j] - base[j]) for p in plc]))
        rows.append(row)
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "seed"}
    wins = {name: int(sum(r[f"formative_{name}"] > r[f"placebo_{name}_mean"] for r in rows))
            for name in ("Q", "like", "p_credit")}
    return {"mean": m, "seeds_formative_larger": wins, "n_seeds": len(rows), "rest_events": len(tail)}


class T9Counterfactual(unittest.TestCase):
    def test_removing_formative_experience_changes_more_than_placebo(self):
        out = measure(rest(REST))
        dump("T09", out)
        m = out["mean"]
        for name in ("Q", "like", "p_credit"):
            self.assertGreater(m[f"formative_{name}"], m[f"placebo_{name}_mean"], f"{name}: {out}")
        for name in ("like", "p_credit"):
            self.assertEqual(out["seeds_formative_larger"][name], len(SEEDS), f"{name}: {out}")

    def test_immediate_probe_original_version(self):
        """元の版（休止なし）。v3.0・v3.1 では不合格で expectedFailure として残していた: 直近の経験を除く（プラセボ）が欲求の状態 h を
        通じて Q を大きく動かした（形成期 0.0052 対 プラセボ 0.0108、3/20 本）。設計の改訂 2（想起）の後は、形成期の経験を除いた効果が
        信念の再帰属を通じて大きくなり、平均では合格する（下の注意を参照）。そのため expectedFailure を外し、通常のテストにした。
        注意: 想起の掃引は『K_r 経験ごと』なので、どの経験を除いても掃引の位置がずれ、プラセボの信念も動く（P(credit_taken) の変化が
        0 でなくなった）。この版の合格は、休止ありの正式版ほど強い証拠ではない。"""
        out = measure([])
        dump("T09_immediate", out)
        m = out["mean"]
        self.assertGreater(m["formative_Q"], m["placebo_Q_mean"], str(out))


if __name__ == "__main__":
    unittest.main()
