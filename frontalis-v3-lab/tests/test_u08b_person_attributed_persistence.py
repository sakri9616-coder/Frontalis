"""U8′ 人に帰属する評価の持続（**事後に定義した基準（オーケストレーターが定義。U8 の不合格を見た後）**）。

U8（test_u08_total_evaluation_persistence.py）は不合格のまま残してある（持続 0.567 / 0.411）。U8′ は U8 の不合格を見た後に、
オーケストレーターが、診断（CHANGELOG の U8 の節）に基づいて定義した別の基準である。U8 を置き換えない。U8 と U8′ の両方を報告する。
理由: U8 は「過去の持続」と、設計上の正当な性質二つ（場面の価値の可塑性、欲求の状態への依存）を混ぜて測っていた。
  - 場面の価値の可塑性: N との良い経験は、状況モデル（文脈・概念）の価値も更新する。未見の N′ を同じ文脈で評価すると、その差が縮むのは
    場面が実際に変わった（良い協働が続いた）からで、過去の搾取の忘却ではない。
  - 欲求の状態への依存: Q は h と気分に依存する（MB の g_k）。P1 と P2 で h が違えば、同じ信念でも Q が違う。

手順（オーケストレーターの指定）:
  T4/U8 の課題（generate_t4(種): P1 = 8 人に搾取、P2 = 普通に協働、N との良い経験 32 回）。
  測定の直前に、欲求の状態 h と気分 m を基準値に揃える（測定用の複製の上で。学習した状態は変えない）。
  未見の共同作業者 N′ を、履歴に一度も現れていない文脈（cafe_chat）で評価する（T4 の履歴の文脈は music_session、project_meeting、workshop、studio）。
  測るもの: Q のうち人に帰属する部分 Q_person = ω V_MF,person + (1 − ω) V_MB,person
    V_MF,person = 参加者の価値（person_value）の和、V_MB,person = 人が起こす結果型の MB の寄与（同じサンプルで、全体の価値 − 自分・世界が起こす分だけの価値）。
  持続   = N′ への Q_person の P2 − P1 の差が、N との経験の後にどれだけ残るか（後の差 / 前の差）。
  可塑性 = P1 の N への Q_person が真値に近づいた割合（(後 − 前) / (真値 − 前)）。
           真値 = N との最後の経験と同じ結果（新しい id）を、h・m を揃えた P1 で appraise したときの、N が行為者の結果の
           a で重み付けた寄与の総和（contributions のうち actor = N の Σ_k a_k contrib_k）。
  基準（U8 と同じ数値）: m = 8, 32 で持続 ≥ 0.7、可塑性 ≥ 0.7。20 本の平均。
参考（判定しない）: 全体の Q（C を含む）の持続、MF 部分・MB 部分それぞれの持続と可塑性、cafe_chat でなく music_session で測った場合、h・m を揃えない場合。
"""
import copy
import unittest

import numpy as np

from helpers import SEEDS, align_shell, dump, feed, make, mh, pmap

UNSEEN_CONTEXT = "cafe_chat"


def sit(pid, ctx=UNSEEN_CONTEXT):
    return {"context": ctx, "concepts": ["collaboration", "music"], "participants": [{"id": pid, "class": "collaborator"}]}


def measure(net, pid, ctx=UNSEEN_CONTEXT, align=True):
    c = copy.deepcopy(net)
    if align:
        align_shell(c)
    return c.probe(sit(pid, ctx))


def worker(seed):
    p1_h, p2_h, n_h = mh.generate_t4(seed)
    nprime, nid = f"u8b-nprime-{seed}", f"t4n-{seed}"
    p1, p2 = make(seed, record_ledger=False), make(seed, record_ledger=False)
    feed(p1, p1_h); feed(p2, p2_h)
    def pair(ctx=UNSEEN_CONTEXT, align=True):
        a, b = measure(p1, nprime, ctx, align), measure(p2, nprime, ctx, align)
        return {"person": b["Q_person"] - a["Q_person"], "Q": b["Q"] - a["Q"],
                "mf": b["V_MF_person"] - a["V_MF_person"], "mb": b["V_MB_person"] - a["V_MB_person"]}
    d0 = pair()
    d0_music = pair("music_session")
    d0_noalign = pair(align=False)
    pr0 = measure(p1, nid)
    q_n0 = pr0["Q_person"]
    out = {"seed": seed, "d0": d0["person"], "d0_Q": d0["Q"], "q_n0": q_n0}
    done = 0
    for m in (8, 32):
        feed(p1, n_h[done:m]); feed(p2, n_h[done:m]); done = m
        dm, dm_music, dm_noalign = pair(), pair("music_session"), pair(align=False)
        c = copy.deepcopy(p1)
        align_shell(c)
        last = dict(n_h[m - 1]); last["id"] = f"truth-{seed}-{m}"
        ap = c.appraise(last)
        a = c.concern.a
        truth = float(sum(float(a @ np.array([o["contrib"][k] for k in ("R", "C", "A", "S", "Sa")]))
                          for o in ap["contributions"] if o["actor"] == nid))
        prn = measure(p1, nid)
        q_n = prn["Q_person"]
        frac = lambda x, y: x / y if abs(y) > 1e-12 else 0.0
        out[m] = {"persistence": frac(dm["person"], d0["person"]), "plasticity": frac(q_n - q_n0, truth - q_n0),
                  "dm": dm["person"], "q_n": q_n, "truth": truth,
                  "persistence_total_Q": frac(dm["Q"], d0["Q"]),
                  "persistence_mf_part": frac(dm["mf"], d0["mf"]), "persistence_mb_part": frac(dm["mb"], d0["mb"]),
                  "persistence_music_session": frac(dm_music["person"], d0_music["person"]),
                  "persistence_not_aligned": frac(dm_noalign["person"], d0_noalign["person"]),
                  # 可塑性の内訳（診断。判定しない）: 部品ごとに、N への値が真値のどこまで来たか
                  "plasticity_mf_part": frac(prn["V_MF_person"] - pr0["V_MF_person"], truth - pr0["V_MF_person"]),
                  "plasticity_mb_part": frac(prn["V_MB_person"] - pr0["V_MB_person"], truth - pr0["V_MB_person"]),
                  "V_MF_person_N": prn["V_MF_person"], "V_MB_person_N": prn["V_MB_person"], "omega_N": prn["omega"]}
    return out


class U8bPersonAttributedPersistence(unittest.TestCase):
    def test_person_attributed_evaluation_persists_and_stays_plastic(self):
        """事後に定義した基準（オーケストレーターが定義。U8 の不合格を見た後）。"""
        rows = pmap(worker, SEEDS)
        res = {"label": "事後に定義した基準（オーケストレーターが定義。U8 の不合格を見た後）", "context_of_evaluation": UNSEEN_CONTEXT}
        for m in (8, 32):
            keys = [k for k in rows[0][m]]
            res[f"m{m}"] = {k: float(np.mean([r[m][k] for r in rows])) for k in keys}
            res[f"m{m}"]["persistence_median"] = float(np.median([r[m]["persistence"] for r in rows]))
            res[f"m{m}"]["plasticity_median"] = float(np.median([r[m]["plasticity"] for r in rows]))
        res["d0_person_mean"] = float(np.mean([r["d0"] for r in rows]))
        res["d0_total_Q_mean"] = float(np.mean([r["d0_Q"] for r in rows]))
        res["q_person_N_before_mean"] = float(np.mean([r["q_n0"] for r in rows]))
        dump("U08b", res)
        for m in (8, 32):
            self.assertGreaterEqual(res[f"m{m}"]["persistence"], 0.7, res)
            self.assertGreaterEqual(res[f"m{m}"]["plasticity"], 0.7, res)


if __name__ == "__main__":
    unittest.main()
