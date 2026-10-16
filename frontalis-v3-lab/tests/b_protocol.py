"""B1〜B3 数値的な健全性（SPEC-v3.3 第7.2節。**W6 の微小摂動の収縮性は数学的な安定性の指標で、人格形成の健全性そのものではない**）の共通プロトコル。
tests/test_b_health.py と診断が使う。

故障とみなすのは、(1) 状態が有界でない、(2) 学習規則が自分の定義どおりに動いていない、(3) 分岐や変化の原因を記録から説明できない、の三つだけ。
選択の分岐が起きた結果として比が大きくなっても、正当化が持続・増幅しても、それだけでは故障ではない（SPEC 7.1、8.1）。

【共通の実行（W6 と同じ。全機構を有効にして 2000 経験・乱数種 0〜19）】
  経験史 = generate_extended_history(型 = R/K/X を種で巡回, 種, 2000, praise_rate = 0.25)（W6 と同じ）。with_choices = True のときは、さらに
  8 経験ごとに二択（5 種類の鋳型を巡る。迷う二択・易しい二択・参加者つき）、100 経験ごとに訂正（それ以前の、参加者のいる非 optional の経験の結果の大きさを半分に直す）を挟む
  （二択と訂正は、基準の実行と摂動の実行に同じ順序で与える。訂正は非 optional の経験だけを参照するので、両方の実行で学習済み）。
  最初の 220 イベントを基準の人格に与え（ウォームアップ）、その写しに全体のノルム 1e-3 の摂動（W、C、a、m、h、ll_ema、J の値。さらに想起の記録の φ に別の 1e-3）を入れて、
  残りを 2 つの人格に同じ順序で与える（共通乱数）。比 = ‖Δ最終‖/‖Δ初期‖（W6 の摂動 B と同じ ‖Δ‖。J の差を加えた）。

【B1 分岐の因果的説明可能性】
  離散的な決定 = 経験ごとに record_decisions が残す（optional の接近・回避、二択の選択、潜在原因の割当（原因を作るか・MAP の原因）、復位（呼ぶか・更新した原因））。
  最初の分岐点 = 2 つの実行の決定が最初に異なったイベント。
  比 > 1 の実行について、(a) 分岐点より前の比（ウォームアップ直後から分岐点の直前までの最大。毎イベント測る。分岐点より前は特徴の集合が同じなので、配列の差で測れる）≤ 1 + 1e-6、
  (b) 摂動した実行を同じ初期状態から、離散的な決定をすべて基準の実行に合わせて強制し（乱数は同じだけ引く）再生したときの最終の比 ≤ 1。
  分岐が見つからない（決定が最後まで同じ）のに比 > 1 の実行は、離れた分が記録された分岐で説明されないので、(a)(b) を満たさないとして数える。
  合格は、比 > 1 の実行の 95% 以上で (a) と (b) の両方が成り立つこと（比 > 1 の実行がなければ対象なしで合格）。

【B2 状態の有界性】
  毎イベント（基準・摂動の両方）に監視: 有限、|W| ≤ 10、|C| ≤ 2（全ての原因の価値）、0 ≤ a ≤ 5、|J| ≤ J の上限（orig は J_max、persist は J_num）、計数 ≥ −1e-9、原因数 ≤ K。
  Σ（欲求ごと・人モデルと状況モデル）は 20 イベントごとと最終に、対称（最大の非対称 ≤ 1e-9）で半正定値（最小固有値 ≥ −1e-9）を確かめる（毎イベントだと固有値分解が重いため）。

【B3 学習規則の整合性】（基準の実行は台帳を記録する）
  (1) 台帳の ΔW の総和が人モデル・状況モデルの W に一致（≤ 1e-9）、台帳の ΔC_k の総和が全ての原因の価値に一致（≤ 1e-9）
  (2) φ の和が 1: 信念の計数の質量（Σ個体の(N1+N0) + Σ文脈の(CP1+CP0)）が、観測の記録の質量 Σ m に型ごとに一致（≤ 1e-8）、π の計数 Σ も同じ
  (3) W の教師が Δu を φ で分けたものに等しい: 訂正を受けていない各経験について、経験の記録（欲求の状態 h、結果、解釈の重み、φ_o）から教師を計算し直した値が、
      台帳の教師に一致（≤ 1e-12。Z-persist は当時の割り引きの割合 q を同じに適用）。全ての経験で、教師の合計（人 + 状況）が Δu に一致（Δh_k ≠ 0 の欲求。≤ 1e-12）
  (4) a の目標が |Δu| に等しい: 台帳の Δu と chosen だけから a を規則どおりに再生した値が、台帳の a に一致（≤ 1e-12）
  (5) J の変化の経路: 台帳の J の出来事（kind = "justification"）の源ごとの和が J に一致（≤ 1e-9）。spread・reaffirm・counter・overwhelm・bound の出来事は、台帳の二択の経験でだけ起きる。
      J の状態の源ごとの和 just_src の合計が J に一致
  (6) 価値ネット自身の出力が、どの教師にも入らない: 200 イベントごとに、基準の写しと、価値ネットの W を勝手に動かした写しに次のイベントを与え、教師・Δu・φ・割り引きが同一
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from helpers import KINDS, make, mh
from npnv3 import NPN, innate
from npnv3.innate import N_NEEDS, N_TYPES, OUTCOME_TYPES
from npnv3.ledger import Ledger
from npnv3.value import Concern

WARMUP = 220
TOTAL = 2000
EPS = 1e-3
PRAISE_RATE = 0.25
PSD_EVERY = 20
INDEP_EVERY = 200


# ----------------------------------------------------------------------
# 経験史
# ----------------------------------------------------------------------
def _choice_template(i: int) -> Dict[str, Any]:
    t = i % 5
    cid = f"b-ch{i:04d}"
    if t == 0:        # 迷う二択（概念だけが違う）
        return {"id": cid, "kind": "choice", "options": [{"context": "solo_practice", "concepts": ["music"]}, {"context": "solo_practice", "concepts": ["exploration"]}],
                "outcomes_by_option": {"0": [{"type": "achievement", "actor": "self", "magnitude": 0.3}], "1": [{"type": "novelty", "actor": "world", "magnitude": 0.3}]}}
    if t == 1:
        return {"id": cid, "kind": "choice", "options": [{"context": "workshop", "concepts": ["routine"]}, {"context": "workshop", "concepts": ["learning"]}],
                "outcomes_by_option": {"0": [{"type": "comfort", "actor": "world", "magnitude": 0.3}], "1": [{"type": "novelty", "actor": "world", "magnitude": 0.3}]}}
    if t == 2:        # 易しい二択（友人と敵）
        return {"id": cid, "kind": "choice",
                "options": [{"context": "music_session", "concepts": ["collaboration"], "participants": [{"id": "b_ally", "class": "friend"}]},
                            {"context": "music_session", "concepts": ["collaboration"], "participants": [{"id": "b_foe", "class": "rival"}]}],
                "outcomes_by_option": {"0": [{"type": "connection", "actor": "b_ally", "magnitude": 0.5}], "1": [{"type": "harm", "actor": "b_foe", "magnitude": 0.5}]}}
    if t == 3:
        return {"id": cid, "kind": "choice", "options": [{"context": "project_meeting", "concepts": ["authorship"]}, {"context": "project_meeting", "concepts": ["collaboration"]}],
                "outcomes_by_option": {"0": [{"type": "novelty", "actor": "world", "magnitude": 0.2}], "1": [{"type": "comfort", "actor": "world", "magnitude": 0.2}]}}
    return {"id": cid, "kind": "choice",      # 参加者つき。選んだ側に害がありうる（反証・抵抗の経路）
            "options": [{"context": "music_session", "concepts": ["music"], "participants": [{"id": "b_pal1", "class": "friend"}]},
                        {"context": "music_session", "concepts": ["exploration"], "participants": [{"id": "b_pal2", "class": "friend"}]}],
            "outcomes_by_option": {"0": [{"type": "harm", "actor": "b_pal1", "magnitude": 0.5}], "1": [{"type": "connection", "actor": "b_pal2", "magnitude": 0.5}]}}


def b_events(seed: int, with_choices: bool) -> List[Dict[str, Any]]:
    kind = KINDS[seed % 3]
    hist = mh.generate_extended_history(kind, seed, TOTAL, praise_rate=PRAISE_RATE)
    if not with_choices:
        return hist
    rng = np.random.default_rng([seed, 8801])
    out: List[Dict[str, Any]] = []
    n_ch = 0
    n_corr = 0
    for idx, e in enumerate(hist):
        out.append(e)
        if idx % 8 == 7:
            out.append(_choice_template(n_ch)); n_ch += 1
        if idx % 100 == 99:
            cands = [h for h in hist[:idx - 20] if not h.get("optional") and h["participants"] and h["outcomes"]]
            if cands:
                ref = cands[int(rng.integers(len(cands)))]
                outs = []
                for o in ref["outcomes"]:
                    o2 = copy.deepcopy(o)
                    o2["magnitude"] = round(o2["magnitude"] * 0.5, 3)
                    outs.append(o2)
                out.append({"id": f"b-corr{n_corr:03d}", "kind": "correction", "refers_to": ref["id"], "corrected_outcomes": outs, "source": "self_discovery"})
                n_corr += 1
    return out


# ----------------------------------------------------------------------
# 状態の距離（W6 の摂動 B の ‖Δ‖ に J の差を加えた）
# ----------------------------------------------------------------------
def state_vec(n: NPN) -> Dict[Any, float]:
    d: Dict[Any, float] = {}
    for tag, kal in (("pW", n.value.person), ("sW", n.value.sit)):
        for i, name in enumerate(kal.names):
            for k in range(N_NEEDS):
                d[(tag, name, k)] = float(kal.W[i, k])
    d.update(n.cond.state_items())
    for k in range(N_NEEDS):
        d[("a", k)] = float(n.concern.a[k]); d[("h", k)] = float(n.h[k])
    d[("m",)] = float(n.mood)
    for cue, v in n.just.items():
        d[("J", cue)] = float(v)
    return d


def phi_sq(n1: NPN, n2: NPN) -> float:
    a, b = n1.belief, n2.belief
    R = max(a.rec_n, b.rec_n)
    pa, pb = np.zeros((R, N_TYPES)), np.zeros((R, N_TYPES))
    pa[: a.rec_n] = a.rec_phi[: a.rec_n] * (a.rec_m[: a.rec_n] > 0.5)
    pb[: b.rec_n] = b.rec_phi[: b.rec_n] * (b.rec_m[: b.rec_n] > 0.5)
    return float(np.sum((pa - pb) ** 2))


def full_dist(n1: NPN, n2: NPN) -> float:
    a, b = state_vec(n1), state_vec(n2)
    d2 = sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in set(a) | set(b))
    d2 += phi_sq(n1, n2) + float(np.sum((n1.belief.ll_ema - n2.belief.ll_ema) ** 2))
    return float(np.sqrt(d2))


class ShapeMismatch(Exception):
    pass


def fast_dist(n1: NPN, n2: NPN) -> float:
    """分岐点より前に使う、配列の差による距離（full_dist と同じ量。特徴の集合・原因の構造・J の手がかり・記録の行数が同じときだけ）。"""
    if (n1.value.person.W.shape != n2.value.person.W.shape or n1.value.sit.W.shape != n2.value.sit.W.shape or n1.cond.CK.shape != n2.cond.CK.shape
            or n1.belief.rec_n != n2.belief.rec_n or set(n1.just) != set(n2.just)):
        raise ShapeMismatch()
    s = float(np.sum((n1.value.person.W - n2.value.person.W) ** 2) + np.sum((n1.value.sit.W - n2.value.sit.W) ** 2))
    alive1, alive2 = n1.cond.AL > 0.5, n2.cond.AL > 0.5
    if not np.array_equal(alive1, alive2):
        raise ShapeMismatch()
    s += float(np.sum(((n1.cond.CK - n2.cond.CK) * alive1) ** 2))
    s += float(np.sum((n1.concern.a - n2.concern.a) ** 2) + np.sum((n1.h - n2.h) ** 2) + (n1.mood - n2.mood) ** 2)
    for cue in n1.just:
        s += (n1.just[cue] - n2.just[cue]) ** 2
    s += phi_sq(n1, n2) + float(np.sum((n1.belief.ll_ema - n2.belief.ll_ema) ** 2))
    return float(np.sqrt(s))


def perturb(n: NPN, rng: np.random.Generator) -> None:
    """W、C、a、m、h、ll_ema、J の値に、全体のノルムが EPS のガウス雑音を入れ、続いて想起の記録の φ に別の EPS を入れて計数を作り直す（W6 の摂動 B + J）。"""
    jc = sorted(n.just)
    shapes = [n.value.person.W.shape, n.value.sit.W.shape, n.cond.C.shape, (5,), (), (5,), n.belief.ll_ema.shape, (len(jc),)]
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
    n.belief.ll_ema = n.belief.ll_ema + noise[6]
    for c, v in zip(jc, noise[7]):
        n.just[c] = n.just[c] + float(v)
    b = n.belief
    R = b.rec_n
    mask = b.rec_m[:R] > 0.5
    nz = rng.normal(size=int(mask.sum()))
    nz *= EPS / np.linalg.norm(nz)
    phi = b.rec_phi[:R].copy()
    phi[mask] = np.clip(phi[mask] + nz, 1e-6, 1.0 - 1e-6)
    b.rec_phi[:R] = phi
    b._rebuild_counts()


# ----------------------------------------------------------------------
# B2: 状態の有界性
# ----------------------------------------------------------------------
class Monitor:
    def __init__(self, jbound: float, K: int):
        self.jbound, self.K = jbound, K
        self.max_abs_W = 0.0
        self.max_abs_C = 0.0
        self.min_a, self.max_a = 1e9, -1e9
        self.max_abs_J = 0.0
        self.min_count = 0.0
        self.max_causes = 0
        self.max_asym = 0.0
        self.min_eig = 1e9
        self.violations: Dict[str, int] = {}
        self.n_checks = 0

    def _bad(self, name: str) -> None:
        self.violations[name] = self.violations.get(name, 0) + 1

    def cheap(self, n: NPN) -> None:
        self.n_checks += 1
        W1, W2, CK = n.value.person.W, n.value.sit.W, n.cond.CK
        w = max(float(np.abs(W1).max()) if W1.size else 0.0, float(np.abs(W2).max()) if W2.size else 0.0)
        c = float(np.abs(CK).max()) if CK.size else 0.0
        self.max_abs_W, self.max_abs_C = max(self.max_abs_W, w), max(self.max_abs_C, c)
        a = n.concern.a
        self.min_a, self.max_a = min(self.min_a, float(a.min())), max(self.max_a, float(a.max()))
        j = max((abs(v) for v in n.just.values()), default=0.0)
        self.max_abs_J = max(self.max_abs_J, j)
        b = n.belief
        cnt = min(float(x.min()) if x.size else 0.0 for x in (b.N1, b.N0, b.CP1, b.CP0, b.CS1, b.CS0, n.cond.CNT))
        self.min_count = min(self.min_count, cnt)
        nc = int(n.cond.AL.sum(axis=1).max()) if n.cond.AL.size else 0
        self.max_causes = max(self.max_causes, nc)
        if not (np.isfinite(W1).all() and np.isfinite(W2).all() and np.isfinite(CK).all() and np.isfinite(a).all() and np.isfinite(n.h).all() and np.isfinite(n.mood)):
            self._bad("non_finite")
        if not all(np.isfinite(v) for v in n.just.values()):
            self._bad("non_finite_J")
        if w > 10.0:
            self._bad("W_range")
        if c > 2.0:
            self._bad("C_range")
        if float(a.min()) < 0.0 or float(a.max()) > 5.0:
            self._bad("a_range")
        if j > self.jbound + 1e-12:
            self._bad("J_range")
        if cnt < -1e-9:
            self._bad("count_negative")
        if nc > self.K:
            self._bad("causes_over_K")

    def psd(self, n: NPN) -> None:
        for kal in (n.value.person, n.value.sit):
            S = kal.S
            if S.size == 0:                                              # まだ特徴がない（参加者のいない履歴の人モデル）
                continue
            if not np.isfinite(S).all():
                self._bad("S_non_finite")
                continue
            self.max_asym = max(self.max_asym, float(np.abs(S - S.transpose(0, 2, 1)).max()))
            if float(np.abs(S - S.transpose(0, 2, 1)).max()) > 1e-9:
                self._bad("S_asymmetric")
            for k in range(S.shape[0]):
                ev = float(np.linalg.eigvalsh(0.5 * (S[k] + S[k].T))[0])
                self.min_eig = min(self.min_eig, ev)
                if ev < -1e-9:
                    self._bad("S_not_psd")

    def summary(self) -> Dict[str, Any]:
        return {"max_abs_W": self.max_abs_W, "max_abs_C": self.max_abs_C, "a_range": [self.min_a, self.max_a], "max_abs_J": self.max_abs_J, "J_bound": self.jbound,
                "min_count": self.min_count, "max_causes": self.max_causes, "K": self.K, "max_S_asymmetry": self.max_asym, "min_S_eigenvalue": self.min_eig,
                "violations": dict(self.violations), "n_checks": self.n_checks}


# ----------------------------------------------------------------------
# B3: 学習規則の整合性
# ----------------------------------------------------------------------
def check_rules(n: NPN) -> Dict[str, Any]:
    """基準の実行（台帳あり）の最終状態に対する、学習規則の整合性。違反の数を返す。"""
    cfg = n.cfg
    v: Dict[str, Any] = {}
    # (1) 台帳の和
    err_w = 0.0
    for model, kal in (("person", n.value.person), ("situation", n.value.sit)):
        tot = n.ledger.totals(model)
        for nm, row in tot.items():
            err_w = max(err_w, float(np.abs(kal.W[kal.idx[nm]] - row).max()))
        for nm, w in zip(kal.names, kal.W):
            if nm not in tot:
                err_w = max(err_w, float(np.abs(w).max()))
    v["ledger_W_max_error"] = err_w
    cd = n.cond
    sums: Dict[str, np.ndarray] = {}
    for en in n.ledger.entries:
        if en["kind"] == "experience":
            for cue, d in en["td"]["lc"].items():
                x = np.array(d["dCk"])
                sums[cue] = sums[cue] + x if cue in sums else x
    err_c = 0.0
    for cue, x in sums.items():
        err_c = max(err_c, float(np.abs(cd.CK[cd.idx[cue]] - x).max()))
    v["ledger_C_max_error"] = err_c
    # (2) φ の和（質量）
    b = n.belief
    R = b.rec_n
    mass_rec = (b.rec_m[:R]).sum(axis=0)
    mass_cnt = (b.N1 + b.N0).sum(axis=0) + (b.CP1 + b.CP0).sum(axis=0)
    mass_pi = b.pi.sum(axis=1) - 2.0
    v["phi_mass_max_error"] = float(np.abs(mass_rec - mass_cnt).max())
    v["pi_mass_max_error"] = float(np.abs(mass_rec - mass_pi).max())
    # (3) 教師
    led_exps = [e for e in n.ledger.entries if e["kind"] == "experience"]
    err_t, err_sum, n_checked, n_skipped = 0.0, 0.0, 0, 0
    for rec, en in zip(n.exp_log, led_exps):
        du, dh = np.array(en["du"]), np.array(en["dh"])
        tot = np.zeros(N_NEEDS)
        for pid, t in en["teachers"]["person"].items():
            tot += np.array(t)
        tot += np.array(en["teachers"]["situation"])
        nz = np.abs(dh) > 1e-12
        if nz.any():
            err_sum = max(err_sum, float(np.abs(tot[nz] - du[nz]).max()))
        if rec["corrections"]:
            n_skipped += 1
            continue
        P = rec["participants"]
        n_o = len(rec["outcomes"])
        Wout = np.array(rec["w_out"], dtype=np.float64).reshape(n_o, N_TYPES)
        rows = innate.mixed_rows(Wout, [o["magnitude"] for o in rec["outcomes"]], n.M)
        dh2, du2, contrib = innate.contributions(np.array(rec["h_before"]), rows)
        actor_j = np.array(rec["actor_j"], dtype=int)
        pt, st = n._split_teachers(contrib, np.array(rec["phi_o"]), (actor_j >= 0).astype(np.float64), actor_j, len(P))
        if rec.get("discount_q", 0.0) > 0.0:
            pt, st, _ = n._apply_discount(pt, st, rec["discount_q"])
        for j, p in enumerate(P):
            err_t = max(err_t, float(np.abs(np.array(en["teachers"]["person"][p["id"]]) - pt[j]).max()))
        err_t = max(err_t, float(np.abs(np.array(en["teachers"]["situation"]) - st).max()))
        err_t = max(err_t, float(np.abs(du2 - du).max()))
        n_checked += 1
    v["teacher_max_error"] = err_t
    v["teacher_sum_vs_du_max_error"] = err_sum
    v["teacher_checked"] = n_checked
    v["teacher_skipped_corrected"] = n_skipped
    # (4) a の再生
    c = Concern(cfg)
    err_a = 0.0
    for en in led_exps:
        c.update(np.array(en["du"]), bool(en["chosen"]))
        err_a = max(err_a, float(np.abs(c.a - np.array(en["a"][1])).max()))
    v["a_replay_max_error"] = err_a
    # (5) J の経路
    choice_ids = {e["id"] for e in n.ledger.entries if e["kind"] == "choice"}
    jsum: Dict[str, float] = {}
    bad_path = 0
    for en in n.ledger.entries:
        if en["kind"] != "justification":
            continue
        for ev in en["events"]:
            jsum[ev["cue"]] = jsum.get(ev["cue"], 0.0) + ev["delta"]
            if ev["source"] in ("spread", "reaffirm", "counter", "overwhelm", "bound") and en["after"] not in choice_ids:
                bad_path += 1
    err_j = max([abs(n.just.get(k, 0.0) - jsum.get(k, 0.0)) for k in set(n.just) | set(jsum)] or [0.0])
    err_src = max([abs(sum(row.values()) - n.just.get(k, 0.0)) for k, row in n.just_src.items()] or [0.0])
    v["J_ledger_sum_max_error"] = err_j
    v["J_sources_sum_max_error"] = err_src
    v["J_events_outside_choice"] = bad_path
    v["n_choice_entries"] = len(choice_ids)
    return v


def violations_of(v: Dict[str, Any]) -> List[str]:
    out = []
    for key, tol in (("ledger_W_max_error", 1e-9), ("ledger_C_max_error", 1e-9), ("phi_mass_max_error", 1e-8), ("pi_mass_max_error", 1e-8),
                     ("teacher_max_error", 1e-12), ("teacher_sum_vs_du_max_error", 1e-12), ("a_replay_max_error", 1e-12),
                     ("J_ledger_sum_max_error", 1e-9), ("J_sources_sum_max_error", 1e-9)):
        if not v[key] <= tol:
            out.append(key)
    if v["J_events_outside_choice"] != 0:
        out.append("J_events_outside_choice")
    return out


def value_independence_check(n: NPN, e_next: Dict[str, Any], seed: int) -> bool:
    """価値ネット自身の出力が教師に入らない: n の 2 つの写し（片方は価値ネットの W を勝手に動かした）に次のイベントを与えて、教師・Δu・φ・割り引きが同一。"""
    led, elog, dec = n.ledger, n.exp_log, n.decisions
    n.ledger, n.exp_log, n.decisions = Ledger(), [], []
    try:
        a, b = copy.deepcopy(n), copy.deepcopy(n)
    finally:
        n.ledger, n.exp_log, n.decisions = led, elog, dec
    a.cfg = a.cfg.replace(record_ledger=False, record_decisions=True); b.cfg = b.cfg.replace(record_ledger=False)
    a.cond.trace = b.cond.trace = False
    a.decisions = []
    rng = np.random.default_rng([seed, 31337])
    b.value.person.W = b.value.person.W + rng.normal(0, 0.5, b.value.person.W.shape)
    b.value.sit.W = b.value.sit.W + rng.normal(0, 0.5, b.value.sit.W.shape)
    ra = a.experience(copy.deepcopy(e_next))
    # 接近・回避と二択の選択は、価値ネットの出力が決める離散的な行動（教師ではない）。ここでは a の決定に合わせて、教師の経路だけを比べる
    b.force_decisions = [{"approach": a.decisions[0]["approach"], "choice": a.decisions[0]["choice"]}]
    b._force_i = 0
    rb = b.experience(copy.deepcopy(e_next))
    ok = (ra["teachers"] == rb["teachers"] and ra["du"] == rb["du"] and ra["phi"] == rb["phi"] and ra.get("discount") == rb.get("discount"))
    return bool(ok)


# ----------------------------------------------------------------------
# ワーカー
# ----------------------------------------------------------------------
def b_worker(args: Tuple[int, str, str, bool]) -> Dict[str, Any]:
    """1 つの種・1 つの条件（lc_design、justification_mode、二択と訂正を含めるか）について B1〜B3 を測る。"""
    seed, design, jmode, with_choices = args
    events = b_events(seed, with_choices)
    jbound = (make(0, justification_mode=jmode).cfg.j_max if jmode == "orig" else make(0, justification_mode=jmode).cfg.j_num)
    base = make(seed, lc_design=design, justification_mode=jmode, record_decisions=True)
    for e in events[:WARMUP]:
        base.experience(copy.deepcopy(e))
    n_dec0 = len(base.decisions)
    pert = copy.deepcopy(base)
    pert.ledger, pert.exp_log = Ledger(), copy.deepcopy(base.exp_log)
    pert.cfg = pert.cfg.replace(record_ledger=False)
    pert.cond.trace = False
    perturb(pert, np.random.default_rng([seed, 77]))
    pert.rng.bit_generator.state = base.rng.bit_generator.state
    pert0 = copy.deepcopy(pert)                                     # 強制再生の出発点
    pert.decisions = []
    d0 = full_dist(base, pert)
    mon_b, mon_p = Monitor(jbound, base.cond.K), Monitor(jbound, base.cond.K)
    branch: Optional[int] = None
    branch_info: Optional[Dict[str, Any]] = None
    pre_max, pre_n = 0.0, 0
    indep_ok, indep_n = 0, 0
    rest = events[WARMUP:]
    for i, e in enumerate(rest):
        if i % INDEP_EVERY == 0 and e.get("kind") != "correction":
            indep_n += 1
            indep_ok += int(value_independence_check(base, e, seed + i))
        base.experience(copy.deepcopy(e))
        pert.experience(copy.deepcopy(e))
        mon_b.cheap(base); mon_p.cheap(pert)
        if i % PSD_EVERY == 0:
            mon_b.psd(base); mon_p.psd(pert)
        if branch is None:
            if e.get("kind") != "correction":
                k = len(pert.decisions) - 1
                db, dp = base.decisions[n_dec0 + k], pert.decisions[k]
                if db != dp:
                    branch = i
                    branch_info = {"event_id": e["id"], "kind": e.get("kind", "experience"), "differs": sorted(x for x in db if db[x] != dp[x]), "base": db, "perturbed": dp,
                                   "decisions_before": k}
            if branch is None:
                try:
                    pre_max = max(pre_max, fast_dist(base, pert) / d0)
                    pre_n += 1
                except ShapeMismatch:
                    branch = i
                    branch_info = {"event_id": e["id"], "kind": e.get("kind", "experience"), "differs": ["structure (decisions identical)"], "decisions_before": len(pert.decisions)}
    mon_b.psd(base); mon_p.psd(pert)
    ratio = full_dist(base, pert) / d0
    out: Dict[str, Any] = {"seed": seed, "design": design, "mode": jmode, "with_choices": with_choices, "n_events": len(events), "ratio": ratio, "d0": d0,
                           "branch_index": branch, "branch_info": branch_info, "pre_branch_max_ratio": pre_max, "pre_branch_steps": pre_n,
                           "n_choices": sum(1 for e in events if e.get("kind") == "choice"), "n_corrections": sum(1 for e in events if e.get("kind") == "correction"),
                           "n_avoided": base.n_avoided, "n_experiences": base.n_experiences}
    # B1 (b): 強制再生
    if ratio > 1.0:
        if branch is None:
            out["B1_a_ok"] = bool(pre_max <= 1.0 + 1e-6)
            out["B1_b_ok"] = False                                  # 決定は最後まで同じなのに離れた: 記録された分岐で説明されない
            out["forced_ratio"] = None
        else:
            pert0.force_decisions = base.decisions[n_dec0:]
            pert0._force_i = 0
            pert0.cfg = pert0.cfg.replace(record_ledger=False)
            for e in rest:
                pert0.experience(copy.deepcopy(e))
            fr = full_dist(base, pert0) / d0
            out["forced_ratio"] = fr
            out["B1_a_ok"] = bool(pre_max <= 1.0 + 1e-6)
            out["B1_b_ok"] = bool(fr <= 1.0)
    # B2
    out["B2"] = {"base": mon_b.summary(), "perturbed": mon_p.summary()}
    out["B2_ok"] = not mon_b.violations and not mon_p.violations
    # B3
    rules = check_rules(base)
    rules["value_independence_checks"] = indep_n
    rules["value_independence_ok"] = indep_ok
    viol = violations_of(rules)
    if indep_ok != indep_n:
        viol.append("value_output_in_teacher")
    rules["violations"] = viol
    out["B3"] = rules
    out["B3_ok"] = not viol
    # J の例（源ごとの分解）
    out["J_final"] = {k: float(v) for k, v in base.just.items()}
    out["J_sources_final"] = {k: {s: float(d) for s, d in row.items()} for k, row in base.just_src.items()}
    return out
