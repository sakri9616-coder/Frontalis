"""E 系（SPEC-v3.3 第 4 節）の共通プロトコル。tests/test_e_extinction.py と tests/diagnostics_e.py が使う。

【プロトコル（SPEC-v3.3 4.3 のとおり。細部は結果を見る前に決めた）】
  手がかり x = 概念 "praise_from_X"（手がかり名 concept:praise_from_X）。文脈 A = studio、B = cafe_chat、未見の C = garden。
  1 回の試行 = 2 つの経験（同じ文脈・同じ session。試行ごとに別の session）。
    獲得の試行 : [手がかり x の経験（結果なし）] → [害の経験（harm、行為者 world、大きさ U(0.6, 1.0)）]
    消去の試行 : [手がかり x の経験] → [何も起きない経験（結果なし）]
  獲得: 文脈 A で 12 試行。消去: 文脈 B で 20 試行（E9 では 3 つの文脈 cafe_chat / workshop / travel に 7 + 7 + 6 試行）。
  無関係な経験: x を含まない経験（独りの練習・会議・習慣などの小さな結果。大きさは種ごとに抽出）。1 経験ごとに別の session。
  テスト: 学習しない probe {"cue": "concept:praise_from_X", "context": 文脈} の C。獲得直後（12 試行目の後）の文脈 A の C を 1 とした比
  （比 = C / C_獲得直後。どちらも負なので、比 1 = 獲得直後と同じ、0 = 消えた）。
  分岐: 各テストは「消去の直後の状態の写し」から分岐させる（deepcopy）。
  乱数種: 種ごとに NPN の種、害の大きさ、無関係な経験の中身を変える（人格は同じ初期状態・同じ係数）。20 本の平均で判定する。

  E8 の別の手がかり: 一度も x と対にならない手がかり concept:tea_time を、文脈 A で 6 回（結果なしの経験と対にして）提示してから（C = 0）、
  プロトコルの間は提示しない。判定するのはその C の変化（漏れ）。値を持つ別の手がかり（concept:evening_walk を comfort と対に条件づけたもの）の変化は
  参考として報告する（時間減衰と区別できないので判定しない）。
"""
from __future__ import annotations

import copy
import tempfile
from typing import Any, Dict, List, Sequence

import numpy as np

from helpers import deep_equal, make
from npnv3 import NPN

CUE = "praise_from_X"
CUE_NAME = f"concept:{CUE}"
CTX_A, CTX_B, CTX_C = "studio", "cafe_chat", "garden"
EXT_CONTEXTS_3 = ("cafe_chat", "workshop", "travel")
N_ACQ, N_EXT, N_UNRELATED, N_REINSTATE = 12, 20, 50, 2
RE_ACQ_MAX = 40


class Counter:
    def __init__(self, tag: str):
        self.tag, self.n = tag, 0

    def next(self, kind: str) -> str:
        self.n += 1
        return f"{self.tag}-{kind}{self.n:04d}"


def _ev(eid: str, ctx: str, concepts: Sequence[str], outcomes: List[Dict[str, Any]], sess: str) -> Dict[str, Any]:
    return {"id": eid, "context": ctx, "concepts": list(concepts), "participants": [], "outcomes": outcomes,
            "optional": False, "session": sess}


def trial(npn: NPN, c: Counter, rng: np.random.Generator, kind: str, ctx: str) -> None:
    """kind = "acq"（害が続く）| "ext"（何も起きない）。"""
    sess = c.next("s")
    npn.experience(_ev(c.next("c"), ctx, [CUE], [], sess))
    if kind == "acq":
        m = round(float(rng.uniform(0.6, 1.0)), 2)
        npn.experience(_ev(c.next("u"), ctx, [], [{"type": "harm", "actor": "world", "magnitude": m}], sess))
    else:
        npn.experience(_ev(c.next("n"), ctx, [], [], sess))


def unrelated(npn: NPN, c: Counter, rng: np.random.Generator, n: int) -> None:
    """x を含まない経験 n 件。"""
    kinds = [("solo_practice", ["music", "solo_work"], "achievement"), ("project_meeting", ["authorship"], "novelty"),
             ("solo_practice", ["routine"], "comfort"), ("project_meeting", ["learning"], None)]
    for _ in range(n):
        ctx, concepts, t = kinds[int(rng.integers(len(kinds)))]
        out = [] if t is None else [{"type": t, "actor": "self" if t == "achievement" else "world",
                                     "magnitude": round(float(rng.uniform(0.2, 0.6)), 2)}]
        npn.experience(_ev(c.next("f"), ctx, concepts, out, c.next("s")))


def lone_harm(npn: NPN, c: Counter, rng: np.random.Generator, ctx: str, n: int) -> None:
    """手がかりなしの害 n 回（復位の刺激）。"""
    for _ in range(n):
        m = round(float(rng.uniform(0.6, 1.0)), 2)
        npn.experience(_ev(c.next("h"), ctx, [], [{"type": "harm", "actor": "world", "magnitude": m}], c.next("s")))


def pretrain_controls(npn: NPN, c: Counter, rng: np.random.Generator) -> None:
    """別の手がかり: tea_time（結果なしと対。C = 0）と evening_walk（comfort と対。正の値）。文脈 A。"""
    for _ in range(6):
        sess = c.next("s")
        npn.experience(_ev(c.next("t"), CTX_A, ["tea_time"], [], sess))
        npn.experience(_ev(c.next("t"), CTX_A, [], [], sess))
    for _ in range(6):
        sess = c.next("s")
        npn.experience(_ev(c.next("w"), CTX_A, ["evening_walk"], [], sess))
        npn.experience(_ev(c.next("w"), CTX_A, [], [{"type": "comfort", "actor": "world", "magnitude": 0.6}], sess))


def read(npn: NPN, ctx: str, cue: str = CUE_NAME) -> float:
    return float(npn.probe({"cue": cue, "context": ctx})["C"])


def check_bounds(npn: NPN, acc: Dict[str, float]) -> None:
    """E8(a): 全ての原因の価値の絶対値の最大、原因数の最大、応答の絶対値の最大を記録する。"""
    cd = npn.cond
    if cd.CK.size:
        acc["max_abs_cause_value"] = max(acc["max_abs_cause_value"], float(np.abs(cd.CK).max()))
        acc["max_causes"] = max(acc["max_causes"], float(cd.AL.sum(axis=1).max()))
    acc["max_abs_response"] = max(acc["max_abs_response"], abs(read(npn, CTX_A)), abs(read(npn, CTX_B)), abs(read(npn, CTX_C)))


def states_equal(x: NPN, y: NPN) -> bool:
    mx, ax = x.get_state()
    my, ay = y.get_state()
    return deep_equal(mx, my) and ax.keys() == ay.keys() and all(np.array_equal(ax[k], ay[k]) for k in ax)


def first_reaching(values: Sequence[float], target_ratio: float, ref: float) -> int:
    """比 = 値 / ref が target_ratio に初めて達した試行数（1 始まり）。達しなければ len + 1（打ち切り）。"""
    for i, v in enumerate(values):
        if v / ref >= target_ratio:
            return i + 1
    return len(values) + 1


def run_variant(seed: int, single_cause: bool = False, ext_contexts: Sequence[str] = (CTX_B,), full: bool = True,
                post_make: Any = None, lone_harm_fn: Any = None, **kw: Any) -> Dict[str, Any]:
    """1 つの人格（種 seed、single_cause かどうか）について、E1〜E8 の測定を一通り行う。
    full=False のときは、獲得 → 消去 → ABC の測定だけ（E9 用）。
    post_make（NPN を受け取る）と lone_harm_fn（lone_harm と同じ引数）は診断用の差し替え（判定のテストでは使わない）。
    post_make を使うときは、保存・復元（E8c）の検査は意味を持たないので full=False で使う。"""
    rng = np.random.default_rng([seed, 303])
    c = Counter(f"E{seed}")
    npn = make(seed, single_cause=single_cause, **kw)
    if post_make is not None:
        post_make(npn)
    acc = {"max_abs_cause_value": 0.0, "max_causes": 0.0, "max_abs_response": 0.0}
    out: Dict[str, Any] = {"seed": seed, "single_cause": single_cause}
    if full:
        pretrain_controls(npn, c, rng)
        out["ctl_before"] = {"tea_time": read(npn, CTX_A, "concept:tea_time"), "evening_walk": read(npn, CTX_A, "concept:evening_walk")}
    # --- 獲得 ---
    acq_curve: List[float] = []
    for _ in range(N_ACQ):
        trial(npn, c, rng, "acq", CTX_A)
        acq_curve.append(read(npn, CTX_A))
    c_acq = acq_curve[-1]
    out["C_acq"] = c_acq
    out["acq_curve_ratio"] = [v / c_acq for v in acq_curve]
    out["n_causes_after_acq"] = npn.cond.n_causes(CUE_NAME)
    check_bounds(npn, acc)
    if full:
        # E1: 消去なしで無関係な経験 50 回
        m1 = copy.deepcopy(npn)
        unrelated(m1, c, np.random.default_rng([seed, 304]), N_UNRELATED)
        out["E1"] = read(m1, CTX_A) / c_acq
        check_bounds(m1, acc)
    # --- 消去（途中で保存・復元の検査 E8(c)）---
    per_ctx = [N_EXT] if len(ext_contexts) == 1 else [7, 7, 6]
    ext_plan = [ctx for ctx, k in zip(ext_contexts, per_ctx) for _ in range(k)]
    half = len(ext_plan) // 2
    for ctx in ext_plan[:half]:
        trial(npn, c, rng, "ext", ctx)
    saved = None
    if full:
        with tempfile.TemporaryDirectory() as d:
            npn.save(d)
            saved = NPN.load(d)
        out["E8c_restore_state_equal_at_save"] = states_equal(npn, saved)
    rng_rest = np.random.default_rng([seed, 305])
    rng_rest2 = np.random.default_rng([seed, 305])
    c2 = copy.deepcopy(c)
    for ctx in ext_plan[half:]:
        trial(npn, c, rng_rest, "ext", ctx)
        if saved is not None:
            trial(saved, c2, rng_rest2, "ext", ctx)
    check_bounds(npn, acc)
    out["E2_B"] = read(npn, ext_contexts[0]) / c_acq
    out["ABA"] = read(npn, CTX_A) / c_acq
    out["ABC"] = read(npn, CTX_C) / c_acq
    out["n_causes_after_ext"] = npn.cond.n_causes(CUE_NAME)
    if full:
        out["E8c_restore_continue_equal"] = states_equal(npn, saved) and read(npn, CTX_B) == read(saved, CTX_B)
        # E5: 消去の後、無関係な経験 50 回、その後 B でテスト
        m5 = copy.deepcopy(npn)
        unrelated(m5, c, np.random.default_rng([seed, 306]), N_UNRELATED)
        out["E5_after"] = read(m5, CTX_B) / c_acq
        check_bounds(m5, acc)
        # E6: 手がかりなしの害 2 回 → B でテスト。対照は同じ経験数の無関係な経験
        m6 = copy.deepcopy(npn)
        (lone_harm_fn or lone_harm)(m6, c, np.random.default_rng([seed, 307]), CTX_B, N_REINSTATE)
        out["E6_reinstated"] = read(m6, CTX_B) / c_acq
        m6c = copy.deepcopy(npn)
        unrelated(m6c, c, np.random.default_rng([seed, 308]), N_REINSTATE)
        out["E6_control"] = read(m6c, CTX_B) / c_acq
        check_bounds(m6, acc)
        # E7: 消去の後、B で再獲得（手がかり → 害）。獲得直後の 70% に達するまでの試行数
        m7 = copy.deepcopy(npn)
        re_curve: List[float] = []
        for _ in range(RE_ACQ_MAX):
            trial(m7, c, rng, "acq", CTX_B)
            re_curve.append(read(m7, CTX_B))
        out["E7_first"] = first_reaching(acq_curve, 0.7, c_acq)
        out["E7_re"] = first_reaching(re_curve, 0.7, c_acq)
        out["re_curve_ratio"] = [v / c_acq for v in re_curve[:12]]
        check_bounds(m7, acc)
        # E8(b): 別の手がかり
        out["ctl_after"] = {"tea_time": read(npn, CTX_A, "concept:tea_time"), "evening_walk": read(npn, CTX_A, "concept:evening_walk")}
        out["ctl_after_unrelated"] = {"evening_walk": read(m5, CTX_A, "concept:evening_walk")}
        out["bounds"] = acc
    return out
