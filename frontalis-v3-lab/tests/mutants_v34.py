"""v3.4 RA11 の変異体 MU1〜MU8（SPEC-v3.4 第 8 節）。検査の信頼性の前提（検出力）を測るための、意図的に壊した実装。

各変異体は NPN / ValueNet のクラスの属性を一時的に差し替えるコンテキスト（apply_mutant）として実装する。npnv3/ のファイルは変えない。
差し替えは with 句の中だけで有効で、抜けると元に戻る（unittest が確かめる）。ワーカープロセス（spawn）では、仕事の関数がその中で with を開く。

  MU1 吸収率を状態によらない定数にする（w_reapp 型）: M1 の g_i ≡ 0.5。台帳には z・LLR・κ̄ をそのまま記録し、g だけが定数になる。
  MU2 罪悪感・後悔・自己正当化を計算・記録するが τ に渡さない: τ の 1 回の更新量 `_tau_step` が常に 0。台帳には tau_raw（元の押し）が残り、tau_step が 0 になる。
  MU3 v3.3 型の相対的なずらしを積み重ねる: M1 の吸収を、v3.3 の再評価（観測を「現在の予測 + ずらし」とするカルマン更新。観測雑音 R/(a w_reapp)、Σ が縮む、
      状況モデルには触れない）に置き換える。ずらしは、意図した変化 g_i D_i をそのまま使う。台帳の D・g・delta はそのまま記録される。
  MU4 吸収のゲインを 3 倍にし、台帳には一貫して記録する: 実際に W に足す量が 3 × g_i D_i。台帳の dW・測った予測の変化（pred_change）は実際の量で一貫するが、
      台帳の g・D・delta は元の g_i D_i のまま（批評の検算 D と同じ型: 実装の誤りが W の台帳の和と整合する形で入る）。
  MU5 訂正で当時の f を書き換える: 訂正の後に、参照先の経験の台帳の項（kind = "experience"）の f を、訂正時の f_c の読み出しで上書きする。
  MU6 τ の境界を外し、罪悪感の訂正を 100 回続ける: tau_max を 1e6 にする。100 回の訂正は stress_guilt（tests/ra_checks_v34.py）が与える。
  MU7 気分を解釈の重みに漏らす: 曖昧な結果の解釈の重みの指数を τ · exp(気分) にする。
  MU8 τ を下げる向きにだけ動かす: `_tau_step` が正の押し（自己正当化の優勢）を 0 に切る（草稿 1 の設計）。
"""
from __future__ import annotations

import contextlib
import math
from typing import Any, Callable, Dict, List

import numpy as np

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from npnv3 import NPN  # noqa: E402
from npnv3.innate import N_TYPES, TYPE_INDEX  # noqa: E402
from npnv3.value import ValueNet  # noqa: E402

G_CONST = 0.5
GAIN = 3.0
TAU_MAX_UNBOUNDED = 1.0e6
MOOD_LEAK = 1.0

Undo = Callable[[], None]


def _patch(cls: Any, attr: str, new: Any) -> Undo:
    old = cls.__dict__[attr]                     # staticmethod はラッパーごと保存して戻す
    setattr(cls, attr, new)

    def undo() -> None:
        setattr(cls, attr, old)
    return undo


def _mu1() -> Undo:
    orig = NPN._m1_absorb

    def m1_absorb(self, *a, **k):
        old = self.cfg.g_const
        self.cfg.g_const = G_CONST
        try:
            return orig(self, *a, **k)
        finally:
            self.cfg.g_const = old
    return _patch(NPN, "_m1_absorb", m1_absorb)


def _mu2() -> Undo:
    return _patch(NPN, "_tau_step", staticmethod(lambda defend, guilt, regret, eta: 0.0))


def _mu3() -> Undo:
    def absorb(self, participants, ctx, concepts, delta_person, delta_sit):
        prec = np.ones(len(delta_sit)) * self.cfg.w_reapp              # v3.3: 観測雑音 R / (a w_reapp)。a は呼び出し側にないので 1 とする（a の和は 5）
        res = self.reappraise(participants, [np.asarray(d, dtype=np.float64) for d in delta_person], prec)
        return {"dW_person": res["dW_person"], "dW_situation": {}}       # v3.3 は状況モデルに触れない
    return _patch(ValueNet, "absorb", absorb)


def _mu4() -> Undo:
    orig = ValueNet.absorb

    def absorb(self, participants, ctx, concepts, delta_person, delta_sit):
        return orig(self, participants, ctx, concepts, [GAIN * np.asarray(d, dtype=np.float64) for d in delta_person], GAIN * np.asarray(delta_sit, dtype=np.float64))
    return _patch(ValueNet, "absorb", absorb)


def _mu5() -> Undo:
    orig = NPN._correct

    def correct(self, c):
        ap = orig(self, c)
        for e in self.ledger.entries:
            if e.get("kind") == "experience" and e.get("id") == c["refers_to"]:
                e["f"] = float(ap["f"])                                 # 当時の f を、訂正時の f_c の読み出しで上書きする
        return ap
    return _patch(NPN, "_correct", correct)


def _mu6() -> Undo:
    orig = NPN._m3_evaluate

    def m3(self, *a, **k):
        old = self.cfg.tau_max
        self.cfg.tau_max = TAU_MAX_UNBOUNDED
        try:
            return orig(self, *a, **k)
        finally:
            self.cfg.tau_max = old
    return _patch(NPN, "_m3_evaluate", m3)


def _mu7() -> Undo:
    def interp_weights(self, interp, pvec, tau=1.0):
        w = np.zeros(N_TYPES)
        tau_eff = tau * math.exp(MOOD_LEAK * float(self.mood))         # 気分が解釈の重みの指数に漏れる
        for t, q in interp.items():
            k = TYPE_INDEX[t]
            w[k] = q if self.cfg.no_interpretation else q * pvec[k] ** tau_eff
        return w / w.sum()
    return _patch(NPN, "_interp_weights", interp_weights)


def _mu8() -> Undo:
    orig = NPN._tau_step                          # 関数（staticmethod の中身）

    def step(defend, guilt, regret, eta):
        return min(float(orig(defend, guilt, regret, eta)), 0.0)        # 上げる向き（自己正当化の優勢）を 0 に切る
    return _patch(NPN, "_tau_step", staticmethod(step))


MUTANTS: Dict[str, Dict[str, Any]] = {
    "MU1": {"apply": _mu1, "description": "吸収率を状態によらない定数にする（w_reapp 型。g ≡ 0.5。台帳の z・LLR・κ̄ は記録される）"},
    "MU2": {"apply": _mu2, "description": "罪悪感・後悔・自己正当化を計算・記録するが τ に渡さない（τ の更新量が常に 0）"},
    "MU3": {"apply": _mu3, "description": "v3.3 型の相対的なずらしを積み重ねる（吸収を、意図した変化 g D を観測とするカルマン更新に置き換える。Σ が縮む）"},
    "MU4": {"apply": _mu4, "description": "吸収のゲインを 3 倍にし、台帳には一貫して記録する（W の和・測った予測の変化は実際の量。g・D・delta は元のまま）"},
    "MU5": {"apply": _mu5, "description": "訂正で当時の f を書き換える（参照先の経験の台帳の f を訂正時の f_c で上書き）"},
    "MU6": {"apply": _mu6, "description": "τ の境界を外す（tau_max = 1e6）。罪悪感の訂正を 100 回続ける stress_guilt で検査する"},
    "MU7": {"apply": _mu7, "description": "気分を解釈の重みに漏らす（指数 τ·exp(m)）"},
    "MU8": {"apply": _mu8, "description": "τ を下げる向きにだけ動かす（正の押しを 0 に切る。草稿 1 の設計）"},
}
MUTANT_IDS = tuple(MUTANTS)
MUST_DETECT = ("MU2", "MU3", "MU4", "MU5", "MU7")             # SPEC RA11: MU2〜MU5 と MU7 はすべて検出されなければならない
MIN_DETECTED = 7                                               # 8 体中 7 体以上


@contextlib.contextmanager
def apply_mutant(name):
    """name が None / "baseline" / "none" なら何もしない。それ以外は変異体を適用し、with を抜けると元に戻す。"""
    if name in (None, "baseline", "none"):
        yield
        return
    undo = MUTANTS[name]["apply"]()
    try:
        yield
    finally:
        undo()
