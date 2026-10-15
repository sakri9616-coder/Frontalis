"""T6 自己増幅なし。
(a) 価値の状態写像（W → W'）のヤコビアンのスペクトル半径を有限差分で測る（T2 の全経験: R, K, X の履歴と新しい経験）。
    合格: ≤ 1 + 1e-6。W と気分 m をあわせた状態の写像（実装全体を通す）も、新しい経験で確認する。
(b) 同じ良い経験を 5000 回。価値が有界で、経験の真の効用変化に収束する（相対誤差 < 10%）。
"""
import copy
import unittest

import numpy as np

from helpers import dump, feed, load_json, load_jsonl, make
from npnv3 import innate

EPS = 1e-6
GOOD = {"id": "good", "context": "music_session", "concepts": ["music", "collaboration"],
        "participants": [{"id": "friendA", "class": "collaborator"}],
        "outcomes": [{"type": "connection", "actor": "friendA", "magnitude": 1.0},
                     {"type": "support", "actor": "friendA", "magnitude": 0.7},
                     {"type": "achievement", "actor": "self", "magnitude": 0.6}], "optional": False}


def value_map_radius(npn, exp):
    """この経験の価値更新（教師を固定して W → W'）のヤコビアンのスペクトル半径。"""
    ap = npn.appraise(exp)
    P = exp["participants"]
    ptea = [np.array([ap["teachers"]["person"][p["id"]][n] for n in innate.NEEDS]) for p in P]
    stea = np.array([ap["teachers"]["situation"][n] for n in innate.NEEDS])
    base = copy.deepcopy(npn.value)
    base.prepare(P, exp["context"], exp["concepts"])           # 定義域を揃える（新しい特徴は W=0 で追加）
    n_p, n_s = base.person.W.shape[0], base.sit.W.shape[0]

    def F(w):
        v = copy.deepcopy(base)
        v.person.W = w[: n_p * 5].reshape(n_p, 5).copy()
        v.sit.W = w[n_p * 5:].reshape(n_s, 5).copy()
        v.learn(P, exp["context"], exp["concepts"], ptea, stea)
        return np.concatenate([v.person.W.ravel(), v.sit.W.ravel()])

    w0 = np.concatenate([base.person.W.ravel(), base.sit.W.ravel()])
    J = np.zeros((len(w0), len(w0)))
    for j in range(len(w0)):
        d = np.zeros(len(w0)); d[j] = EPS
        J[:, j] = (F(w0 + d) - F(w0 - d)) / (2 * EPS)
    return float(np.max(np.abs(np.linalg.eigvals(J))))


def full_map_radius(npn, exp):
    """W と気分 m をあわせた状態の写像のスペクトル半径（experience() 全体を通す）。"""
    base = copy.deepcopy(npn)
    base.value.prepare(exp["participants"], exp["context"], exp["concepts"])
    n_p, n_s = base.value.person.W.shape[0], base.value.sit.W.shape[0]
    def get(n):
        return np.concatenate([n.value.person.W.ravel(), n.value.sit.W.ravel(), [n.mood]])
    def F(x):
        n = copy.deepcopy(base)
        n.value.person.W = x[: n_p * 5].reshape(n_p, 5).copy()
        n.value.sit.W = x[n_p * 5: n_p * 5 + n_s * 5].reshape(n_s, 5).copy()
        n.mood = float(x[-1])
        n.experience(exp)
        return get(n)
    x0 = get(base)
    J = np.zeros((len(x0), len(x0)))
    for j in range(len(x0)):
        d = np.zeros(len(x0)); d[j] = EPS
        J[:, j] = (F(x0 + d) - F(x0 - d)) / (2 * EPS)
    return float(np.max(np.abs(np.linalg.eigvals(J))))


class T6NoAmplification(unittest.TestCase):
    def test_a_spectral_radius_of_value_map(self):
        new = load_json("new_event.json")
        radii = {}
        worst = 0.0
        for name in ("R", "K", "X"):
            n = make(0)
            r = []
            for e in load_jsonl(f"{name}.jsonl") + [new]:
                r.append(value_map_radius(n, e))
                n.experience(e)
            radii[name] = {"max": max(r), "n_experiences": len(r)}
            worst = max(worst, max(r))
        # 気分も含む全体の写像（新しい経験）
        full = {}
        for name in ("R", "K", "X"):
            n = make(0); feed(n, load_jsonl(f"{name}.jsonl"))
            full[name] = full_map_radius(n, new)
        # ablation: fixed_teacher でも半径は 1 以下
        nf = make(0, fixed_teacher=True); feed(nf, load_jsonl("K.jsonl"))
        fixed = value_map_radius(nf, new)
        dump("T06a", {"value_map_radius": radii, "worst": worst, "W_and_mood_map_radius": full,
                      "fixed_teacher_radius": fixed})
        self.assertLessEqual(worst, 1 + 1e-6)
        self.assertLessEqual(max(full.values()), 1 + 1e-6)
        self.assertLessEqual(fixed, 1 + 1e-6)

    def test_b_good_experience_5000_times(self):
        n = make(0)
        max_w, max_v = 0.0, 0.0
        for i in range(5000):
            e = dict(GOOD); e["id"] = f"good-{i}"
            n.experience(e)
            if i % 250 == 0 or i == 4999:
                max_w = max(max_w, float(np.abs(n.value.person.W).max()), float(np.abs(n.value.sit.W).max()))
        e = dict(GOOD); e["id"] = "good-final"
        ap = n.appraise(e)                                  # 最後の状態での真の効用変化 Δu（appraisal の du_total）
        probe = n.probe({"context": e["context"], "concepts": e["concepts"], "participants": e["participants"]})
        true_du = ap["du_total"]
        rel_err = abs(probe["V_MF"] - true_du) / abs(true_du)
        diag_max = max(float(np.diag(n.value.person.S).max()), float(np.diag(n.value.sit.S).max()))
        dump("T06b", {"V_MF": probe["V_MF"], "true_du": true_du, "relative_error": rel_err,
                      "max_abs_W": max_w, "max_diag_Sigma": diag_max, "h": [float(x) for x in n.h],
                      "mood": n.mood})
        self.assertTrue(np.isfinite(max_w) and max_w < 10.0)
        self.assertLessEqual(diag_max, 0.25 + 1e-12)       # 共分散は事前分散を超えない（有界）
        self.assertLess(rel_err, 0.10)


if __name__ == "__main__":
    unittest.main()
