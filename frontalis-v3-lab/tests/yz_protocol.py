"""Y 系（価値の再帰属）と Z 系（選択の正当化）の共通プロトコル。tests/test_y_correction.py、tests/test_z_choice.py と診断が使う。

SPEC-v3.3 の 2.4（Y1〜Y7）と 3.5（Z1〜Z7）の判定基準は実装前に固定されている（閾値は一度も変えていない）。ここに書くのは、SPEC に細部がなかった
プロトコルの決め（**テストを走らせて数値を見る前に決めた**。スモークテストで動作を確かめたことは CHANGELOG の「v3.3 第2部」の D 節に書く）。

【Y のプロトコル】
  個体: A（類 collaborator）、organizer（類 collaborator。A と同じ類にして、類の水準を通した他の類からの転移で A の値が動かないようにする）、
        B（類 collaborator。手柄とは無関係の相手）。
  誤帰属の履歴 H_mis（16 経験）: j0..j7 は文脈 project_meeting、概念 authorship、参加者 A と organizer、結果は
        {credit_taken, 行為者 A, 大きさ m_i}（誤帰属。実際は organizer）と {achievement, self, 0.4}。j_i の間に f_i（B との協働: connection）を挟む。
        m_i は種ごとに U(0.6, 0.9)。session は経験ごとに別。
  対照の履歴 H_ctrl: 同じ。ただし credit_taken の行為者は最初から organizer（誤帰属なし）。
  訂正: j_i それぞれに kind = "correction"（refers_to = j_i、corrected_outcomes = 訂正後の結果の全体 = {credit_taken, organizer, m_i} と {achievement, self, 0.4}、
        source = self_discovery）を、履歴の後に 8 個続けて与える。
  測り方: A への信念 = entity プローブ（context = project_meeting）の p_person[credit_taken]（SPEC 4.1 の P(y_{e,k}=1 | e, class(e), context)）。
          A への価値 = 同じプローブの like（人モデルの A の予測を a で重み付けた合計 = 価値ネット W の経路）。条件づけ C(ind:A) は別に報告する（訂正は C を変えない）。
  Y1: (P_mis − P_訂正後) / (P_mis − P_ctrl) ≥ 0.8（20 本の平均どうしの比）。P_mis は訂正を与える前の誤帰属の人格。
  Y2: d_pre = like_mis − like_ctrl（訂正前）、d_post = like_訂正後 − like_ctrl。残る割合 d_post / d_pre が 0.3 以上 0.9 以下（20 本の平均どうし）。
  Y3: 訂正の後、A と良い経験（connection 0.7 と support 0.5 の組）を 10 回。誤帰属の人格（訂正後）と対照の両方に同じ 10 回を与える。
      |d_10| / |d_post| ≤ 0.3（d_10 = like_訂正後 − like_ctrl、10 回の後）。
  Y4: retroactive_value で、同じ訂正の後の |d_retro| ≤ 0.1 |d_pre|（90% 以上が消える）。
  Y5: 訂正の前後で、台帳の当時の経験の記録（f, φ, Δu, 感情を含む全項目）が完全一致（訂正を与える前のコピーと、訂正の後で比べる）。
  Y6: 訂正の後、A が行為者の曖昧な結果（credit_taken 0.5 / credit_shared 0.5、大きさ 0.8、LLM の素の読み）を appraise し、「取られた」の解釈の重み w を測る。
      対照との差の縮み 1 − (w_訂正後 − w_ctrl) / (w_mis − w_ctrl) ≥ 0.5（20 本の平均どうし）。
  Y7: 訂正を与えない。誤帰属の人格と対照の両方に、訂正の代わりに同じ長さ（8 経験）の A と無関係な経験（独りの練習。結果は小さな achievement）を与える。
      信念も価値も、対照との差が 70% 以上残る: (P_mis − P_ctrl)_後 / (P_mis − P_ctrl)_前 ≥ 0.7、(like_mis − like_ctrl)_後 / (like_mis − like_ctrl)_前 ≥ 0.7。

【Z のプロトコル】
  Z で使う人格は、先に履歴を与えた人格（ally と foe の経験: 下の warm_persona）か、何も経験していない人格。選択肢は状況（結果を持たない）。
  Z1: 易しい選択（|ΔQ| > 0.3）: 友人 ally（類 friend）の選択肢と敵 foe（類 rival）の選択肢の二択を 20 回。ΔQ は台帳の記録（選ぶ前の Q の差）で確かめる
      （前提: 20 回とも |ΔQ| > 0.3）。迷う選択（|ΔQ| < 0.05）: 20 回、毎回別の新しい 2 つの概念（結果はどちらも空）の二択（前提: 20 回とも |ΔQ| < 0.05）。
      易しい選択での J の総変化 = 20 回の間に動かした J の絶対値の和 < 0.01。迷う選択: 各回の更新の直後の（選んだ側の J の平均 − 選ばなかった側の J の平均）の最小 ≥ 0.02。
  Z2: 同じ迷う二択（概念 hes_a と hes_b、結果は空）を 1000 回。max|J| ≤ J_max（全種・全時刻）。摂動の比 = J に 1e-3 のノルムの摂動を入れ、同じ乱数で 1000 回
      続けた後の ‖ΔJ‖ / ‖ΔJ_0‖（中央値 ≤ 1。U6/W6 と同じ。最大も報告）。J の増分 = ‖J‖ の増加量: 前半 Δ1 = ‖J_500‖ − ‖J_0‖、後半 Δ2 = ‖J_1000‖ − ‖J_500‖。
      Δ2 ≤ 0.1 Δ1（20 本の平均どうし）。
  Z3: 迷う二択（概念 z3_a、z3_b）を 6 回して J を作り、同時に概念 z3_w に achievement を 6 回与えて根拠のある W を作る。その後、これらの手がかりを含まない
      経験（独りの練習など）を 100 回。J の残存 = Σ|J_100| / Σ|J_0|（作った 2 つの手がかり）≤ 0.2。W の残存 = like_concept(z3_w) の 100 回後 / 前 ≥ 0.7。
  Z4: 迷う二択（概念 z4_a、z4_b）を 6 回して J を作る。Q の高い側を「好む側」とし、その側の手がかりを含む経験に害（harm 0.8、行為者 world）を与える。
      害を 1 回与えるごとに状況プローブで Q の差を測り、好む側の Q が他方を下回る（選好が反転する）までの回数が 5 以内。全種で。
  Z5: 同じ人格の写し 2 つに、同じ結果の経験（achievement 0.6 と connection 0.5）を、(1) optional として与えて接近した場合（chosen = 1）、
      (2) 強制（optional = False）で与えた場合。a の変化量のノルムの比 ‖Δa_選んだ‖ / ‖Δa_強制‖ ≥ 1.5（20 本の平均）。接近しなかった場合は乱数の状態を変えて再試行（最大 50 回）。
  Z6: 結果が空の二択（Δu = 0）を 100 回。a の変化（ノルム）< 1e-6。
  Z7: 同じ強制の経験列（R の履歴の optional を外したもの 36 経験）を、J だけを入れ替えた 2 つの人格に与える。W（人・状況の W と Σ）と C（CK, CNT, TK, AL）が
      ビット単位で同一（np.array_equal）。検査が空でないことを、Q が両者で異なることで確かめる。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from helpers import load_jsonl, make
from npnv3 import NPN

A_ID, O_ID, B_ID = "A", "organizer", "B"
A_PART = {"id": A_ID, "class": "collaborator"}
O_PART = {"id": O_ID, "class": "collaborator"}
B_PART = {"id": B_ID, "class": "collaborator"}
CTX = "project_meeting"
N_J = 8
W_REAPP = 0.5


def y_magnitudes(seed: int) -> List[float]:
    rng = np.random.default_rng([seed, 4001])
    return [round(float(rng.uniform(0.6, 0.9)), 2) for _ in range(N_J)]


def y_history(seed: int, misattributed: bool) -> List[Dict[str, Any]]:
    """16 経験: j_i（A と organizer。手柄は誤って A、または対照では最初から organizer）と f_i（B との協働）の交互。"""
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(y_magnitudes(seed)):
        out.append({"id": f"j{i}", "context": CTX, "concepts": ["authorship"], "participants": [dict(A_PART), dict(O_PART)],
                    "outcomes": [{"type": "credit_taken", "actor": A_ID if misattributed else O_ID, "magnitude": m},
                                 {"type": "achievement", "actor": "self", "magnitude": 0.4}], "session": f"sj{i}"})
        out.append({"id": f"f{i}", "context": "music_session", "concepts": ["music"], "participants": [dict(B_PART)],
                    "outcomes": [{"type": "connection", "actor": B_ID, "magnitude": 0.5}], "session": f"sf{i}"})
    return out


def y_corrections(seed: int) -> List[Dict[str, Any]]:
    return [{"id": f"c{i}", "kind": "correction", "refers_to": f"j{i}", "source": "self_discovery",
             "corrected_outcomes": [{"type": "credit_taken", "actor": O_ID, "magnitude": m}, {"type": "achievement", "actor": "self", "magnitude": 0.4}]}
            for i, m in enumerate(y_magnitudes(seed))]


def unrelated_to_A(seed: int, n: int, tag: str = "u") -> List[Dict[str, Any]]:
    """A と無関係な経験（独りの練習。小さな achievement）。"""
    rng = np.random.default_rng([seed, 4002])
    return [{"id": f"{tag}{i}", "context": "solo_practice", "concepts": ["solo_work"], "participants": [],
             "outcomes": [{"type": "achievement", "actor": "self", "magnitude": round(float(rng.uniform(0.2, 0.5)), 2)}], "session": f"{tag}s{i}"}
            for i in range(n)]


def good_with_A(n: int = 10, tag: str = "g") -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        out.append({"id": f"{tag}{i}", "context": "music_session", "concepts": ["music"], "participants": [dict(A_PART)],
                    "outcomes": [{"type": "connection", "actor": A_ID, "magnitude": 0.7}, {"type": "support", "actor": A_ID, "magnitude": 0.5}],
                    "session": f"{tag}s{i}"})
    return out


AMBIGUOUS_A = {"id": "amb", "context": CTX, "concepts": ["authorship"], "participants": [dict(A_PART)],
               "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": A_ID, "magnitude": 0.8}], "optional": False}


def feed(npn: NPN, exps: List[Dict[str, Any]]) -> None:
    for e in exps:
        npn.experience(copy.deepcopy(e))


def belief_A(npn: NPN) -> float:
    """A への信念: P(credit_taken | A, 類, 文脈 project_meeting)。"""
    return float(npn.probe({"entity": A_ID, "class": "collaborator", "context": CTX})["p_person"]["credit_taken"])


def belief_A_individual(npn: NPN) -> float:
    """参考: A 個人の P(credit_taken)（スコープの混合のうち個体の部分 p_ind。文脈の部分を含まない）。"""
    return float(npn.probe({"entity": A_ID, "class": "collaborator", "context": CTX})["p_ind"]["credit_taken"])


def like_A(npn: NPN) -> float:
    return float(npn.probe({"entity": A_ID, "class": "collaborator", "context": CTX})["like"])


def cond_A(npn: NPN) -> float:
    return float(npn.probe({"entity": A_ID, "class": "collaborator", "context": CTX})["conditioned"])


def taken_weight(npn: NPN) -> float:
    """曖昧な出来事（A が行為者）の「取られた」の解釈の重み（appraise。状態は変えない）。"""
    ap = npn.appraise(copy.deepcopy(AMBIGUOUS_A))
    return float(ap["interpretations"][0]["weights"]["credit_taken"])


def y_worker(seed: int) -> Dict[str, Any]:
    """1 つの種で Y1〜Y7 の測定を一通り行う。"""
    hist_mis, hist_ctrl = y_history(seed, True), y_history(seed, False)
    corr = y_corrections(seed)
    mis, ctrl = make(seed), make(seed)
    feed(mis, hist_mis)
    feed(ctrl, hist_ctrl)
    out: Dict[str, Any] = {"seed": seed}
    out["pre"] = {"P_mis": belief_A(mis), "P_ctrl": belief_A(ctrl), "like_mis": like_A(mis), "like_ctrl": like_A(ctrl),
                  "C_mis": cond_A(mis), "C_ctrl": cond_A(ctrl), "w_mis": taken_weight(mis), "w_ctrl": taken_weight(ctrl),
                  "Pind_mis": belief_A_individual(mis), "Pind_ctrl": belief_A_individual(ctrl)}
    # 分岐: 訂正を与える（既定）、retroactive_value、no_reappraisal、訂正を与えない（Y7）
    led_before = copy.deepcopy(mis.ledger.entries)
    cor = copy.deepcopy(mis)
    n_ledger_before = len(cor.ledger.entries)
    feed(cor, corr)
    out["Y5_ledger_unchanged"] = bool(cor.ledger.entries[:n_ledger_before] == led_before)
    out["post"] = {"P": belief_A(cor), "like": like_A(cor), "C": cond_A(cor), "w": taken_weight(cor), "Pind": belief_A_individual(cor)}
    rec = make(seed, recall_after_correction=True)               # 診断（SPEC にない実験設定）: 訂正の後に想起の掃引を 1 回
    feed(rec, hist_mis)
    feed(rec, corr)
    out["post_recall_option"] = {"P": belief_A(rec), "Pind": belief_A_individual(rec)}
    retro = make(seed, retroactive_value=True)
    feed(retro, hist_mis)
    feed(retro, corr)
    out["retro"] = {"P": belief_A(retro), "like": like_A(retro), "w": taken_weight(retro)}
    noreap = make(seed, no_reappraisal=True)
    feed(noreap, hist_mis)
    feed(noreap, corr)
    out["no_reappraisal"] = {"P": belief_A(noreap), "like": like_A(noreap)}
    # Y3: 訂正の後、A と良い経験 10 回（訂正後の人格と対照の両方）
    c3, k3 = copy.deepcopy(cor), copy.deepcopy(ctrl)
    g = good_with_A(10)
    feed(c3, g)
    feed(k3, g)
    out["Y3"] = {"like_corrected": like_A(c3), "like_ctrl": like_A(k3)}
    r3 = copy.deepcopy(retro)
    feed(r3, g)
    out["Y3_retro"] = {"like": like_A(r3)}
    # Y7: 訂正を与えず、同じ長さの無関係な経験
    m7, k7 = copy.deepcopy(mis), copy.deepcopy(ctrl)
    u = unrelated_to_A(seed, N_J)
    feed(m7, u)
    feed(k7, u)
    out["Y7"] = {"P_mis": belief_A(m7), "P_ctrl": belief_A(k7), "like_mis": like_A(m7), "like_ctrl": like_A(k7),
                 "w_mis": taken_weight(m7), "w_ctrl": taken_weight(k7)}
    return out


# ======================================================================
# Z 系
# ======================================================================
ALLY = {"id": "ally", "class": "friend"}
FOE = {"id": "foe", "class": "rival"}
Z_CTX = "music_session"


def warm_persona(seed: int, **kw: Any) -> NPN:
    """ally（類 friend）には connection と support、foe（類 rival）には harm と betrayal の経験を 6 回ずつ。易しい選択（ally と foe の二択）の前提を作る。"""
    n = make(seed, **kw)
    for i in range(6):
        n.experience({"id": f"wa{i}", "context": Z_CTX, "concepts": ["project_x"], "participants": [dict(ALLY)],
                      "outcomes": [{"type": "connection", "actor": "ally", "magnitude": 0.8}, {"type": "support", "actor": "ally", "magnitude": 0.6}], "session": f"wa{i}"})
        n.experience({"id": f"wb{i}", "context": Z_CTX, "concepts": ["project_x"], "participants": [dict(FOE)],
                      "outcomes": [{"type": "harm", "actor": "foe", "magnitude": 0.8}, {"type": "betrayal", "actor": "foe", "magnitude": 0.6}], "session": f"wb{i}"})
    return n


def easy_choice(i: int) -> Dict[str, Any]:
    return {"id": f"easy{i}", "kind": "choice",
            "options": [{"context": Z_CTX, "concepts": ["project_x"], "participants": [dict(ALLY)]},
                        {"context": Z_CTX, "concepts": ["project_x"], "participants": [dict(FOE)]}],
            "outcomes_by_option": {"0": [{"type": "connection", "actor": "ally", "magnitude": 0.6}], "1": [{"type": "harm", "actor": "foe", "magnitude": 0.6}]}}


def hesitant_choice(cid: str, a: str, b: str, ctx: str = "studio") -> Dict[str, Any]:
    """迷う二択: 結果が空で、手がかり（概念 a と b）以外は同じ。信念が同じなら Q の差は 0。"""
    return {"id": cid, "kind": "choice", "options": [{"context": ctx, "concepts": [a]}, {"context": ctx, "concepts": [b]}],
            "outcomes_by_option": {"0": [], "1": []}}


def jnorm(d: Dict[str, float], keys: Optional[List[str]] = None) -> float:
    ks = list(d) if keys is None else keys
    return float(np.sqrt(sum(d.get(k, 0.0) ** 2 for k in ks)))


def z1_worker(seed: int) -> Dict[str, Any]:
    n = warm_persona(seed)
    j0 = dict(n.just)
    easy_dq, total_update = [], 0.0
    for i in range(20):
        r = n.experience(easy_choice(i))
        easy_dq.append(abs(r["choice"]["dQ"]))
        total_update += float(sum(abs(v) for v in r["choice"]["dJ"].values()))
    net = float(sum(abs(n.just.get(k, 0.0) - j0.get(k, 0.0)) for k in set(n.just) | set(j0)))
    hes_dq, hes_gap = [], []
    for i in range(20):
        r = n.experience(hesitant_choice(f"hes{i}", f"z1_{i}_a", f"z1_{i}_b"))
        c = r["choice"]
        hes_dq.append(abs(c["dQ"]))
        ch = c["chosen"]
        kc, ku = f"concept:z1_{i}_{'ab'[ch]}", f"concept:z1_{i}_{'ab'[1 - ch]}"
        hes_gap.append(float(c["J_after"][kc] - c["J_after"][ku]))
    return {"seed": seed, "easy_min_abs_dQ": min(easy_dq), "easy_total_update": total_update, "easy_net_change": net,
            "hes_max_abs_dQ": max(hes_dq), "hes_min_gap": min(hes_gap), "hes_mean_gap": float(np.mean(hes_gap))}


def z2_worker(seed: int) -> Dict[str, Any]:
    keys = ["concept:hes_a", "concept:hes_b"]
    base = make(seed, record_ledger=False)
    pert = copy.deepcopy(base)
    rng = np.random.default_rng([seed, 4003])
    noise = rng.normal(size=2)
    noise *= 1e-3 / np.linalg.norm(noise)
    pert.just = {k: float(v) for k, v in zip(keys, noise)}
    traj, maxj = [], 0.0
    flips = 0
    for i in range(1000):
        cid = {"id": f"hz{i}", "kind": "choice", "options": [{"context": "studio", "concepts": ["hes_a"]}, {"context": "studio", "concepts": ["hes_b"]}],
               "outcomes_by_option": {"0": [], "1": []}}
        r1 = base.experience(copy.deepcopy(cid))
        r2 = pert.experience(copy.deepcopy(cid))
        flips += int(r1["choice"]["chosen"] != r2["choice"]["chosen"])
        maxj = max(maxj, max((abs(v) for v in base.just.values()), default=0.0), max((abs(v) for v in pert.just.values()), default=0.0))
        traj.append(jnorm(base.just, keys))
    d_end = float(np.sqrt(sum((base.just.get(k, 0.0) - pert.just.get(k, 0.0)) ** 2 for k in keys)))
    d1 = traj[499] - 0.0
    d2 = traj[999] - traj[499]
    return {"seed": seed, "max_abs_J": maxj, "ratio": d_end / 1e-3, "flips": flips, "norm_500": traj[499], "norm_1000": traj[999],
            "delta1": d1, "delta2": d2, "J_end": [float(base.just.get(k, 0.0)) for k in keys], "norm_traj": traj[::50]}


def z3_worker(seed: int) -> Dict[str, Any]:
    n = make(seed)
    for i in range(6):
        n.experience(hesitant_choice(f"z3c{i}", "z3_a", "z3_b"))
        n.experience({"id": f"z3w{i}", "context": "workshop", "concepts": ["z3_w"], "participants": [], "outcomes": [{"type": "achievement", "actor": "self", "magnitude": 0.6}], "session": f"z3w{i}"})
    keys = ["concept:z3_a", "concept:z3_b"]
    j0 = sum(abs(n.just.get(k, 0.0)) for k in keys)
    w0 = float(n.probe({"concept": "z3_w"})["like"])
    rng = np.random.default_rng([seed, 4004])
    kinds = [("solo_practice", ["music", "solo_work"], "achievement"), ("project_meeting", ["authorship"], "novelty"),
             ("solo_practice", ["routine"], "comfort"), ("project_meeting", ["learning"], None)]
    for i in range(100):
        ctx, concepts, t = kinds[int(rng.integers(len(kinds)))]
        out = [] if t is None else [{"type": t, "actor": "self" if t == "achievement" else "world", "magnitude": round(float(rng.uniform(0.2, 0.6)), 2)}]
        n.experience({"id": f"z3u{i}", "context": ctx, "concepts": concepts, "participants": [], "outcomes": out, "session": f"z3us{i}"})
    j1 = sum(abs(n.just.get(k, 0.0)) for k in keys)
    w1 = float(n.probe({"concept": "z3_w"})["like"])
    return {"seed": seed, "J_before": j0, "J_after": j1, "J_remaining": j1 / j0, "W_before": w0, "W_after": w1, "W_remaining": w1 / w0}


def z4_worker(seed: int, **kw: Any) -> Dict[str, Any]:
    n = make(seed, **kw)
    for i in range(6):
        n.experience(hesitant_choice(f"z4c{i}", "z4_a", "z4_b"))
    probe = lambda c: n.probe({"context": "studio", "concepts": [c]})
    qa, qb = probe("z4_a")["Q"], probe("z4_b")["Q"]
    pref, other = ("z4_a", "z4_b") if qa >= qb else ("z4_b", "z4_a")
    gap0 = abs(qa - qb)
    reversed_at = None
    gaps = []
    for k in range(1, 11):
        n.experience({"id": f"z4h{k}", "context": "studio", "concepts": [pref], "participants": [],
                      "outcomes": [{"type": "harm", "actor": "world", "magnitude": 0.8}], "session": f"z4hs{k}"})
        g = probe(pref)["Q"] - probe(other)["Q"]
        gaps.append(float(g))
        if g < 0.0 and reversed_at is None:
            reversed_at = k
    return {"seed": seed, "gap_before": gap0, "reversed_at": reversed_at, "gaps": gaps, "J_pref": float(n.just.get(f"concept:{pref}", 0.0)), "J_other": float(n.just.get(f"concept:{other}", 0.0))}


Z5_EXP = {"id": "z5", "context": "workshop", "concepts": ["z5_task"], "participants": [],
          "outcomes": [{"type": "achievement", "actor": "self", "magnitude": 0.6}, {"type": "novelty", "actor": "world", "magnitude": 0.5}]}


def z5_worker(seed: int) -> Dict[str, Any]:
    base = warm_persona(seed)
    a0 = base.concern.a.copy()
    forced = copy.deepcopy(base)
    forced.experience({**Z5_EXP, "optional": False})
    d_forced = float(np.linalg.norm(forced.concern.a - a0))
    d_chosen, attempts = None, 0
    for attempt in range(50):
        attempts += 1
        c = copy.deepcopy(base)
        c.rng = np.random.default_rng([seed, 5000 + attempt])
        r = c.experience({**Z5_EXP, "optional": True, "announced": []})
        if r["approached"]:
            d_chosen = float(np.linalg.norm(c.concern.a - a0))
            break
    return {"seed": seed, "d_forced": d_forced, "d_chosen": d_chosen, "ratio": (d_chosen / d_forced) if d_chosen is not None else None, "attempts": attempts}


def z6_worker(seed: int) -> Dict[str, Any]:
    n = warm_persona(seed)
    a0 = n.concern.a.copy()
    du_max = 0.0
    for i in range(100):
        r = n.experience(hesitant_choice(f"z6c{i}", "z6_a", "z6_b"))
        du_max = max(du_max, abs(r["du_total"]))
    return {"seed": seed, "a_change": float(np.linalg.norm(n.concern.a - a0)), "max_abs_du_total": du_max, "n_learned": n.n_experiences}


def forced_history(name: str = "R") -> List[Dict[str, Any]]:
    out = []
    for e in load_jsonl(f"{name}.jsonl"):
        e = copy.deepcopy(e)
        e["optional"] = False
        out.append(e)
    return out


def z7_worker(seed: int) -> Dict[str, Any]:
    hist = forced_history("R")
    a, b = make(seed), make(seed)
    rng = np.random.default_rng([seed, 4005])
    cues = ["concept:music", "concept:collaboration", "concept:authorship", "cls:collaborator", "cls:band_mate", "ctx:music_session", "ctx:project_meeting"]
    a.just = {k: float(rng.uniform(-0.15, 0.15)) for k in cues}
    b.just = {k: float(rng.uniform(-0.15, 0.15)) for k in cues}
    qa, qb = [], []
    for e in hist:
        qa.append(a.experience(copy.deepcopy(e))["Q"])
        qb.append(b.experience(copy.deepcopy(e))["Q"])
    same = {}
    for tag, x, y in (("person_W", a.value.person.W, b.value.person.W), ("sit_W", a.value.sit.W, b.value.sit.W),
                      ("person_S", a.value.person.S, b.value.person.S), ("sit_S", a.value.sit.S, b.value.sit.S),
                      ("CK", a.cond.CK, b.cond.CK), ("CNT", a.cond.CNT, b.cond.CNT), ("TK", a.cond.TK, b.cond.TK), ("AL", a.cond.AL, b.cond.AL),
                      ("a", a.concern.a, b.concern.a)):
        same[tag] = bool(np.array_equal(x, y))
    return {"seed": seed, "identical": same, "max_abs_Q_difference": float(np.max(np.abs(np.array(qa) - np.array(qb)))), "n": len(hist)}


# ======================================================================
# Z8（追加。SPEC-v3.3 7.3 の表の Z8、8.2）: 正当化の追跡
# ======================================================================
def _state_pair(n: NPN):
    m, a = n.get_state()
    return copy.deepcopy(m), {k: np.array(v, copy=True) for k, v in a.items()}


def _states_equal(x: NPN, y: NPN) -> bool:
    from helpers import deep_equal
    mx, ax = x.get_state()
    my, ay = y.get_state()
    return deep_equal(mx, my) and ax.keys() == ay.keys() and all(np.array_equal(ax[k], ay[k]) for k in ax)


def z8_choices(seed: int) -> List[Dict[str, Any]]:
    """Z8 の二択の列: 迷う二択 12（概念を 3 組で巡る）、易しい二択 4、結果つきの二択 4（選んだ側の結果が違う）。"""
    out: List[Dict[str, Any]] = []
    for i in range(12):
        out.append(hesitant_choice(f"z8h{i}", f"z8_{i % 3}_a", f"z8_{i % 3}_b"))
    for i in range(4):
        out.append(easy_choice(100 + i))
    for i in range(4):
        out.append({"id": f"z8o{i}", "kind": "choice", "options": [{"context": "workshop", "concepts": ["z8_t1"]}, {"context": "workshop", "concepts": ["z8_t2"]}],
                    "outcomes_by_option": {"0": [{"type": "achievement", "actor": "self", "magnitude": 0.4}], "1": [{"type": "harm", "actor": "world", "magnitude": 0.3}]}})
    return out


def z8_worker(args) -> Dict[str, Any]:
    """二択ごとの台帳（c, ΔQ, ΔJ）と explain_preference（Q = 根拠 + 正当化）、保存・復元の完全一致を、1 つの種・1 つの条件（orig / persist）で測る。"""
    import tempfile
    from helpers import deep_equal
    seed, mode = args
    n = warm_persona(seed, justification_mode=mode)
    cfg = n.cfg
    err = {"conflict": 0.0, "dQ": 0.0, "p_choose": 0.0, "Q_vs_appraise": 0.0, "Q_split": 0.0, "dJ_chosen_side": 0.0, "J_before": 0.0, "J_after": 0.0, "explain_Q_vs_appraise": 0.0,
           "explain_Q_split": 0.0, "explain_delta_split": 0.0, "explain_sources_sum": 0.0}
    n_ch, n_dec, explain_pure = 0, 0, True
    for ch in z8_choices(seed):
        pre_state = _state_pair(n) if n_ch == 0 else None
        pre_rng = copy.deepcopy(n.rng.bit_generator.state)
        j_before = dict(n.just)
        ap = n.appraise(copy.deepcopy(ch))
        ex = n.explain_preference([{k: o[k] for k in ("context", "concepts", "participants") if k in o} for o in ch["options"]], key=ch["id"])
        if pre_state is not None:
            post = _state_pair(n)
            explain_pure = bool(deep_equal(pre_state[0], post[0]) and all(np.array_equal(pre_state[1][k], post[1][k]) for k in post[1]) and n.rng.bit_generator.state == pre_rng)
        for i in (0, 1):
            e_o, a_o = ex["options"][i], ap["options"][i]
            err["explain_Q_vs_appraise"] = max(err["explain_Q_vs_appraise"], abs(e_o["Q"] - a_o["Q"]), abs(e_o["Q_val"] - a_o["Q_val"]), abs(e_o["C_Q"] - a_o["C_Q"]), abs(e_o["J_Q"] - a_o["J_Q"]))
            err["explain_Q_split"] = max(err["explain_Q_split"], abs(e_o["Q"] - (e_o["Q_grounded"] + e_o["J_Q"])), abs(e_o["Q_grounded"] - (e_o["Q_val"] + e_o["C_Q"])))
            for c, d in e_o["justification"].items():
                err["explain_sources_sum"] = max(err["explain_sources_sum"], abs(sum(d["sources"].values()) - d["J"]))
        err["explain_delta_split"] = max(err["explain_delta_split"], abs(ex["delta"]["Q"] - (ex["delta"]["grounded"] + ex["delta"]["justification"])))
        r = n.experience(copy.deepcopy(ch))
        ent = [e for e in n.ledger.entries if e["kind"] == "choice"][-1]
        n_ch += 1
        c = ent
        err["conflict"] = max(err["conflict"], abs(c["conflict"] - float(np.exp(-((c["dQ"] / cfg.sigma_conflict) ** 2)))))
        err["dQ"] = max(err["dQ"], abs(c["dQ"] - (c["Q"][0] - c["Q"][1])))
        z = cfg.beta_choice * np.array(c["Q"])
        p = np.exp(z - z.max()); p /= p.sum()
        err["p_choose"] = max(err["p_choose"], float(np.abs(np.array(c["p_choose"]) - p).max()))
        err["Q_vs_appraise"] = max(err["Q_vs_appraise"], abs(c["Q"][0] - ap["options"][0]["Q"]), abs(c["Q"][1] - ap["options"][1]["Q"]))
        for o in c["options"]:
            err["Q_split"] = max(err["Q_split"], abs(o["Q_val"] + o["C_Q"] + o["J_Q"] - (c["Q"][c["options"].index(o)])))
        # ΔJ: 選んだ側 +η_J c、選ばなかった側 −η_J c（両方に共通の手がかりは相殺）
        for cue, d in c["dJ"].items():
            err["dJ_chosen_side"] = max(err["dJ_chosen_side"], abs(abs(d) - cfg.eta_j * c["conflict"]))
        # J の前後: 二択の前の値と、二択の後の状態
        for cue, v in c["J_before"].items():
            err["J_before"] = max(err["J_before"], abs(v - j_before.get(cue, 0.0)))
        post_ev: Dict[str, float] = {}
        for en in n.ledger.entries[-3:]:
            if en["kind"] == "justification" and en["after"] == ch["id"]:
                for ev in en["events"]:
                    if mode != "orig" and ev["source"] in ("overwhelm", "decay"):          # J_after は overwhelm と（変種の）減衰の前の値。orig の J_after は有界化と減衰を含む
                        post_ev[ev["cue"]] = post_ev.get(ev["cue"], 0.0) + ev["delta"]
        for cue, v in c["J_after"].items():
            expect = v + post_ev.get(cue, 0.0)
            err["J_after"] = max(err["J_after"], abs(n.just.get(cue, 0.0) - expect))
        n_dec += int(c["chosen"] in (0, 1))
    n_ledger_choice = sum(1 for e in n.ledger.entries if e["kind"] == "choice")
    # 保存・復元: 完全一致。説明も、続きの選択も
    final_pair = [{"context": "studio", "concepts": ["z8_0_a"]}, {"context": "studio", "concepts": ["z8_0_b"]}]
    with tempfile.TemporaryDirectory() as d:
        n.save(d)
        m = NPN.load(d)
    same_state = _states_equal(n, m)
    same_explain = bool(deep_equal(n.explain_preference(final_pair, key="z8-final"), m.explain_preference(final_pair, key="z8-final")))
    same_future = True
    for i in range(6):
        ch = hesitant_choice(f"z8f{i}", f"z8_{i % 3}_a", f"z8_{i % 3}_b")
        same_future = same_future and bool(deep_equal(n.experience(copy.deepcopy(ch)), m.experience(copy.deepcopy(ch))))
    same_future = same_future and _states_equal(n, m)
    ex = n.explain_preference(final_pair, key="z8-final")
    # 正当化が選好を決めている（Q と根拠のみで向きが違う）二択が作れるか: 根拠が同じ（概念だけが違う）で J だけが違う → Q の差は J の差
    n2 = warm_persona(seed, justification_mode=mode)
    for i in range(6):
        n2.experience(hesitant_choice(f"z8d{i}", "z8_d_a", "z8_d_b"))
    ex2 = n2.explain_preference([{"context": "studio", "concepts": ["z8_d_a"]}, {"context": "studio", "concepts": ["z8_d_b"]}], key="z8-d")
    return {"seed": seed, "mode": mode, "n_choices": n_ch, "errors": err, "explain_is_pure": explain_pure, "restore_same_state": same_state, "restore_same_explain": same_explain,
            "restore_same_future": same_future, "example_explain": ex, "example_J_decides": ex2, "ledger_choice_entries": n_ledger_choice}
