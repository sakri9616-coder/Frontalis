"""設計の改訂 2（オーケストレーター承認）: 想起・再帰属（docs/04 第 5 節 手順 7）の検査。

仕様: 信念ネットは参加者が行為者の観測（個体、文脈、型ごとの y・マスク・割り当てた φ）を記録として保持する（保存・復元に含める）。
K_r = 10 経験ごとに全記録を 1 回掃引し、各観測の現在の寄与を計数から引き（retract）、今の信念（π と単位の率）で φ を計算し直し、
足し直す（reassign）。不変量: 各観測の寄与の和は常に 1、計数の総和は変わらない。曖昧な結果は計数に入らない。
価値ネット側の教師の分割は再帰属しない（非対称）。アブレーション no_reattribution。

自己増幅の検査: 想起を繰り返しても π が 0/1 に張り付かないか、張り付くならどの条件でか。
"""
import copy
import tempfile
import unittest

import numpy as np

from helpers import KINDS, SEEDS, deep_equal, dump, feed, load_jsonl, make, mh, persona_gen, pmap
from npnv3 import NPN
from npnv3.innate import TYPE_INDEX

CT = TYPE_INDEX["credit_taken"]


def mass(b):
    return float((b.N1 + b.N0).sum() + (b.CP1 + b.CP0).sum())


def rebuilt(b):
    """記録から計数を独立に数え直す（テスト側の実装）。"""
    R = b.rec_n
    N1 = np.zeros_like(b.N1); N0 = np.zeros_like(b.N0); C1 = np.zeros_like(b.CP1); C0 = np.zeros_like(b.CP0)
    for r in range(R):
        i, c = b.rec_ind[r], b.rec_ctx[r]
        m, y, p = b.rec_m[r], b.rec_y[r], b.rec_phi[r]
        N1[i] += m * p * y; N0[i] += m * p * (1 - y); C1[c] += m * (1 - p) * y; C0[c] += m * (1 - p) * (1 - y)
    return N1, N0, C1, C0


def pi_person(b):
    return b.pi[:, 0] / b.pi.sum(axis=1)


class RecallInvariants(unittest.TestCase):
    def test_counts_mass_and_phi_range_over_many_sweeps(self):
        n = persona_gen("K", 3, 60)
        b = n.belief
        self.assertGreater(b.rec_n, 5)
        total = mass(b)
        self.assertAlmostEqual(total, float(b.rec_m[: b.rec_n].sum()), places=9)        # 1 観測の寄与の和は 1
        pi_sum = b.pi.sum(axis=1).copy()
        for _ in range(100):
            stats = b.reattribute()
            self.assertAlmostEqual(mass(b), total, places=9)                              # 計数の総和は変わらない
            self.assertTrue(np.allclose(b.pi.sum(axis=1), pi_sum, atol=1e-9))
            phi = b.rec_phi[: b.rec_n]
            self.assertTrue(np.all(phi >= 0.0) and np.all(phi <= 1.0))
            self.assertTrue(np.all(b.N1 >= -1e-12) and np.all(b.N0 >= -1e-12) and np.all(b.CP1 >= -1e-12) and np.all(b.CP0 >= -1e-12))
        for a, c in zip((b.N1, b.N0, b.CP1, b.CP0), rebuilt(b)):                         # 計数は記録の和に等しい
            self.assertTrue(np.allclose(a, c, atol=1e-9))

    def test_sweep_schedule_and_ledger_summary(self):
        n = make(0)
        hist = mh.generate_history("R", 1, 35, offer_rate=0.0)
        feed(n, hist)
        recalls = [e for e in n.ledger.entries if e["kind"] == "recall"]
        self.assertEqual([e["after"] for e in recalls], [hist[9]["id"], hist[19]["id"], hist[29]["id"]])      # K_r = 10 経験ごと
        self.assertEqual(n.belief.exp_count, 35)

    def test_ambiguous_types_stay_unobserved_through_sweeps(self):
        n = persona_gen("K", 0)
        amb = {"id": "amb", "context": "project_meeting", "concepts": ["collaboration"],
               "participants": [{"id": "c1", "class": "collaborator"}],
               "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "c1", "magnitude": 0.8}],
               "optional": False}
        n.experience(amb)
        b = n.belief
        i = b.ind_idx["c1"]
        cols = [TYPE_INDEX["credit_taken"], TYPE_INDEX["credit_shared"]]
        for _ in range(20):
            b.reattribute()
        self.assertEqual(float(b.N1[i, cols].sum() + b.N0[i, cols].sum()), 0.0)
        self.assertEqual(float(b.rec_m[b.rec_n - 1, cols].sum()), 0.0)

    def test_no_reattribution_ablation_freezes_phi_and_keeps_no_records(self):
        n = make(0, no_reattribution=True)
        feed(n, mh.generate_history("K", 2, 36))
        self.assertEqual(n.belief.rec_n, 0)
        self.assertEqual([e for e in n.ledger.entries if e["kind"] == "recall"], [])
        m = make(0, reattribute_every=0)                                                  # 周期 0 でも掃引しない
        feed(m, mh.generate_history("K", 2, 36))
        self.assertTrue(np.array_equal(n.belief.N1, m.belief.N1))

    def test_value_teachers_are_not_reattributed(self):
        """非対称: 価値ネットの教師は経験の時点の φ で分け、再帰属しない。想起の有無で人モデル・状況モデルの重みは同じ。"""
        a = persona_gen("K", 4, 36)
        b = persona_gen("K", 4, 36, no_reattribution=True)
        self.assertTrue(np.array_equal(a.value.person.W.shape, b.value.person.W.shape))
        # 教師は φ に依存するので、想起のある信念が後の経験の φ を変え、W も少し違う。ただし過去の経験の教師（台帳）は凍結されたまま
        first = lambda n: [e for e in n.ledger.entries if e["kind"] == "experience"][0]["teachers"]
        self.assertTrue(deep_equal(first(a), first(b)))
        led = [e for e in a.ledger.entries if e["kind"] == "experience"]
        self.assertEqual(len(led), a.n_experiences)

    def test_records_survive_save_and_load_exactly(self):
        n = persona_gen("R", 5, 36)
        with tempfile.TemporaryDirectory() as d:
            n.save(d)
            m = NPN.load(d)
        for k in ("rec_n", "exp_count"):
            self.assertEqual(getattr(n.belief, k), getattr(m.belief, k))
        for k in ("rec_ind", "rec_ctx", "rec_y", "rec_m", "rec_phi"):
            self.assertTrue(np.array_equal(getattr(n.belief, k)[: n.belief.rec_n], getattr(m.belief, k)[: m.belief.rec_n]))
        e = mh.generate_history("R", 5, 12)[-1]
        e = dict(e); e["id"] = "after-load"
        self.assertTrue(deep_equal(n.experience(e), m.experience(e)))


class RecallMechanism(unittest.TestCase):
    """Kelley の共変モデル: 同じ場面で別の人が手柄を取らなければ、『場面のせい』という説明は弱まり、人のせいへ帰属し直される。"""

    def test_exploitation_is_reattributed_to_people_when_the_context_is_exonerated(self):
        def person_share(n):
            b = n.belief
            rec = [r for r in range(b.rec_n) if b.ind_ids[b.rec_ind[r]].startswith("t4p1") and b.rec_m[r, CT] > 0 and b.rec_y[r, CT] > 0.5]
            return float(np.mean([b.rec_phi[r, CT] for r in rec]))
        gains, base = [], []
        for seed in range(10):
            p1, p2, nn = mh.generate_t4(seed)
            a = make(seed); feed(a, p1)
            b = make(seed, no_reattribution=True); feed(b, p1)
            s0 = person_share(a)
            feed(a, nn[:32]); feed(b, nn[:32])
            gains.append(person_share(a) - s0)
            base.append(float(pi_person(a.belief)[CT]) - float(pi_person(b.belief)[CT]))
        dump("RECALL_mechanism", {"person_share_of_exploitation_gain_mean": float(np.mean(gains)),
                                  "pi_person_credit_taken_with_minus_without_recall": float(np.mean(base))})
        self.assertGreater(float(np.mean(gains)), 0.1)           # 搾取の責任が人のほうへ移る
        self.assertGreater(float(np.mean(base)), 0.0)


def _repeat_worker(args):
    kind, seed = args
    n = persona_gen(kind, seed, 36)
    b = n.belief
    traj = []
    last = None
    for _ in range(400):
        st = b.reattribute()
        traj.append(pi_person(b).copy())
        last = st
    traj = np.array(traj)
    active = (b.rec_m[: b.rec_n].sum(axis=0) > 0)
    return {"kind": kind, "seed": seed, "pi_min": float(traj[:, active].min()), "pi_max": float(traj[:, active].max()),
            "final_pi_min": float(traj[-1, active].min()), "final_pi_max": float(traj[-1, active].max()),
            "moved_last_sweep": float(np.abs(traj[-1] - traj[-2]).max()), "mean_abs_dphi_last": last["mean_abs_dphi"],
            "pi_first": traj[0].tolist(), "pi_last": traj[-1].tolist()}


def _collinear_history(kind, n=30):
    ex = lambda i, pid, ctx: {"id": f"col-{kind}-{i}", "context": ctx, "concepts": ["collaboration"], "session": f"s{i}",
                              "participants": [{"id": pid, "class": "collaborator"}], "optional": False,
                              "outcomes": [{"type": "connection", "actor": pid, "magnitude": 0.6},
                                           {"type": "credit_taken", "actor": pid, "magnitude": 0.7}]}
    ctxs = ["music_session", "project_meeting", "workshop", "studio"]
    if kind == "same_partner_same_context":
        return [ex(i, "p0", "music_session") for i in range(n)]
    if kind == "new_partner_same_context":
        return [ex(i, f"p{i}", "music_session") for i in range(n)]
    if kind == "same_partner_new_context":
        return [ex(i, "p0", ctxs[i % 4]) for i in range(n)]
    return [ex(i, f"p{i}", ctxs[i % 4]) for i in range(n)]            # independent


def _collinear_worker(args):
    kind, n_ev = args
    n = make(0)
    feed(n, _collinear_history(kind, n_ev))
    b = n.belief
    start = float(pi_person(b)[CT])
    traj = {}
    for s in range(1, 601):
        b.reattribute()
        if s in (1, 10, 100, 600):
            traj[s] = float(pi_person(b)[CT])
    mean_phi = float(np.mean(b.rec_phi[: b.rec_n, CT]))
    return {"kind": kind, "n_events": n_ev, "pi_person_start": start, "pi_person_by_sweep": traj,
            "pi_person_after_600": float(pi_person(b)[CT]), "mean_phi_person": mean_phi}


class RecallSelfAmplification(unittest.TestCase):
    def test_repeated_recall_does_not_pin_pi_to_0_or_1_on_ordinary_histories(self):
        rows = pmap(_repeat_worker, [(k, s) for k in KINDS for s in range(8)])
        dump("RECALL_repeat", {"worst_pi_min": min(r["pi_min"] for r in rows), "worst_pi_max": max(r["pi_max"] for r in rows),
                               "final_pi_min": min(r["final_pi_min"] for r in rows), "final_pi_max": max(r["final_pi_max"] for r in rows),
                               "max_change_last_sweep": max(r["moved_last_sweep"] for r in rows),
                               "n_histories": len(rows), "sweeps": 400})
        self.assertGreater(min(r["pi_min"] for r in rows), 0.02)
        self.assertLess(max(r["pi_max"] for r in rows), 0.98)
        self.assertLess(max(r["moved_last_sweep"] for r in rows), 1e-3)           # 固定点へ収束する（振動・暴走しない）

    def test_collinear_histories_are_reported(self):
        """相手と場面がいつも一緒の履歴（docs/04 第 4.3 節: 帰属はデータで決まらず事前で決まる）で、想起を 600 回繰り返したときの π。
        張り付くかどうかの報告（条件の特定）。ここでは値を記録し、有界であること（0 < π < 1）だけを要求する。"""
        rows = [_collinear_worker((k, n)) for n in (30, 100) for k in ("same_partner_same_context", "new_partner_same_context",
                                                                       "same_partner_new_context", "independent")]
        dump("RECALL_collinear", rows)
        for r in rows:
            self.assertTrue(0.0 < r["pi_person_after_600"] < 1.0)


if __name__ == "__main__":
    unittest.main()
