"""v3.4 Y 系の測定と条件の枠組み（SPEC-v3.4 第 5〜7 節）。Stage 1 の版に Stage 3 の入口（C-v34・C-OR・clamp・A-τ-off・A-clock-off・τ-yoke・G5 の入れ替え）を足した版。
Stage 3 の変更は CHANGELOG-v3.4.md に旧・新の SHA-256 と理由を書いた（Stage 4 が freeze-protocol.json の第 2 版を書く）。

この版で動く条件（新しい機構 M1〜M3 を要さないもの）:
  C-v33（v3.3 の既定。w_reapp = 0.5 のカルマン更新 1 回、時計を進めない）、C-nocorr（訂正なしの枝）、C-retro（retroactive_value）、
  C-none（no_reappraisal）、diag-q0（C-v33 で、再評価の更新のときだけ過程雑音 q = 0。診断）。
  訂正の直前の状態の入れ替え（val-swap・hm-eq・swap-G）の道具（swap_groups、run_swap）は、C-v33 などの任意の動く条件の上で動く（G5 = κ・τ だけは未実装）。
Stage 3 で動く条件: C-v34、C-OR、C-nocorr-v34（v34 の訂正なしの枝: 空の刻みを N 進める）、A-τ-off、A-clock-off、clamp-τ、clamp-mood、clamp-both、τ-yoke（run_tau_yoke）。
M3 の変種（A-τ-off、A-clock-off、clamp-*、τ-yoke）は **COR の直前から**効かせる（cor_overrides）。COR の直前の状態を全条件で同じにするため（SPEC 第 7.3 節）。
動かない条件（A-g-const、A-κ-fixed、local-reattr）は、取り出すと NotImplementedError を起こす。

乱数種の守り: 走らせてよいのは、開発 3490〜3499（phase = "dev"）、診断 3460〜3479（phase = "diag"）、変異 3450〜3459（phase = "mut"。Stage 4 で許可）だけ。
評価 3400〜3439 と予約 3500〜3539 は、どの条件でも実行できない（guard_seeds が拒否する）。Stage 5 が許可を広げるときは STAGE_ALLOWED と PHASES を変え、
CHANGELOG-v3.4.md に書く（守りだけの変更。CHANGELOG-v3.4 S4-13）。

Stage 4 の追加（CHANGELOG-v3.4.md の Stage 4 の節に旧・新の SHA-256 と理由を書いた）: 条件 A-g-const・A-κ-fixed・local-reattr（・local-reattr-v33）・C-none-v34、
Shapley 分解（run_shapley、shapley_values）、run_swap の nocorr オプション（入れ替えた人格の訂正なしの枝）、phase = "mut"。

SPEC に細部がなく、Stage 1 で決めたこと（CHANGELOG-v3.4.md に日付つきで記録）:
  - 空の刻み（kind = "tick"）: 時計を進める出来事の印。Stage 1 の条件はどれも「訂正は時計を進めない」（v3.3 の T.2）なので刻みを無視する。
    v34 の条件では advance_clock が要る（Stage 2。この版では NotImplementedError）。
  - 状態の入れ替え（swap_groups）は、状態の群の中身を、入れ替え元の人格の深い写しで置き換える。名前の並びの一致は aligned() で確かめる。
  - 読み出し R の r6・r7・r9 は appraise（状態を変えない）で測る。
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_histories_v34 as H  # noqa: E402
import npnv3.value as V  # noqa: E402
from npnv3 import NPN  # noqa: E402
from npnv3.innate import N_NEEDS  # noqa: E402

# ----------------------------------------------------------------------
# 乱数種（SPEC 第 5.1 節）
# ----------------------------------------------------------------------
SEEDS_EVAL = tuple(range(3400, 3440))
SEEDS_MUTATION = tuple(range(3450, 3460))
SEEDS_DIAG = tuple(range(3460, 3480))
SEEDS_DEV = tuple(range(3490, 3500))
SEEDS_RESERVED = tuple(range(3500, 3540))
PHASES = {"dev": SEEDS_DEV, "diag": SEEDS_DIAG, "mut": SEEDS_MUTATION, "stage5": SEEDS_EVAL}
STAGE_ALLOWED = frozenset(SEEDS_DEV) | frozenset(SEEDS_DIAG) | frozenset(SEEDS_MUTATION) | frozenset(SEEDS_EVAL)           # Stage 5
BOOTSTRAP_SEED = [34990]
N_BOOT = 10000


def guard_seeds(seeds: Iterable[int], phase: str) -> None:
    ss = list(seeds)
    if phase not in PHASES:
        raise ValueError(f"phase は {sorted(PHASES)} のどれか: {phase!r}")
    bad = [s for s in ss if s not in STAGE_ALLOWED or s not in PHASES[phase]]
    if bad:
        raise PermissionError(f"この段階・この phase（{phase}）で実行できない乱数種: {bad[:5]}（評価 3400〜3439、変異 3450〜3459、予約 3500〜3539 は実行しない）")


# ----------------------------------------------------------------------
# 読み出しベクトル R（SPEC 第 7.1 節）
# ----------------------------------------------------------------------
R_NAMES = ["r1_like_A", "r2_like_A2", "r3_like_O", "r4_P_taken_A", "r5_P_taken_A2", "r6_w_taken_A", "r7_w_taken_A2", "r8_Q_situation", "r9_w_taken_A2_music"]
IDX_V = [0, 1, 2, 7]            # 価値の部分 V = {r1, r2, r3, r8}
IDX_P = [3, 4, 5, 6, 8]         # 予測と受け止め方の部分 P = {r4, r5, r6, r7, r9}
CTX = H.CTX_MIS
CTX_OTHER = H.CTX_INS


def _amb_event(actor: str, ctx: str, concept: str) -> Dict[str, Any]:
    return {"id": "amb_probe", "context": ctx, "concepts": [concept], "participants": [{"id": actor, "class": "collaborator"}],
            "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": actor, "magnitude": 0.8}], "optional": False}


def _w_taken(npn: NPN, actor: str, ctx: str, concept: str) -> float:
    ap = npn.appraise(_amb_event(actor, ctx, concept))
    return float(ap["interpretations"][0]["weights"]["credit_taken"])


def read_R(npn: NPN, o_class: str) -> np.ndarray:
    """読み出しベクトル R（9 成分。すべて学習しない評価）。"""
    pa = npn.probe({"entity": H.A, "class": "collaborator", "context": CTX})
    pa2 = npn.probe({"entity": H.A2, "class": "collaborator", "context": CTX})
    po = npn.probe({"entity": H.O, "class": o_class, "context": CTX})
    sit = npn.probe({"context": CTX, "concepts": [H.CON_MIS], "participants": [{"id": H.A, "class": "collaborator"}, {"id": H.O, "class": o_class}]})
    return np.array([pa["like"], pa2["like"], po["like"], pa["p_person"]["credit_taken"], pa2["p_person"]["credit_taken"],
                     _w_taken(npn, H.A, CTX, H.CON_MIS), _w_taken(npn, H.A2, CTX, H.CON_MIS), sit["Q"],
                     _w_taken(npn, H.A2, CTX_OTHER, H.CON_INS)], dtype=np.float64)


def state_digest(npn: NPN) -> str:
    """保存形式のハッシュ（読み出しが状態を変えないことの確認用）。"""
    meta, arrays = npn.get_state()
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode("utf-8"))
    for k in sorted(arrays):
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------
# 条件（SPEC 第 6 節）
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    name: str
    implemented: bool = True
    note: str = ""
    prefix_overrides: Tuple[Tuple[str, Any], ...] = ()      # 最初から人格に与える設定
    skip_cor: bool = False                                  # 訂正を与えない枝
    q0: bool = False                                        # 再評価の更新のときだけ過程雑音 q = 0（診断）
    cor_overrides: Tuple[Tuple[str, Any], ...] = ()         # COR の直前から人格の設定に与える（M3 の変種。通常の経験・背景・情報源の履歴には効かせない）
    yoke: bool = False                                      # τ-yoke: 他の人格の τ の軌跡が要る（run_tau_yoke）
    local_reattr: bool = False                              # local-reattr: 各訂正の後に、参照先と参加者を共有する記録だけ φ を推定し直す（対照にも同じ処理）


_STUB = "{} は M1〜M3（reappraisal_mode='v34'。Stage 2・3 で実装）を要する条件で、Stage 1 では実装されていない。"   # Stage 4 で全条件が動く（未使用）
_IMPL = [
    Condition("C-v33", note="v3.3 の既定"),
    Condition("C-nocorr", skip_cor=True, note="訂正なしの枝（時計は進めない: v3.3 の T.2 と揃える）"),
    Condition("C-retro", prefix_overrides=(("retroactive_value", True),), note="自動消去の対照"),
    Condition("C-none", prefix_overrides=(("no_reappraisal", True),), note="価値を変えない対照"),
    Condition("diag-q0", q0=True, note="C-v33 で、再評価のカルマン更新のときだけ q = 0（RC1 の診断）"),
]
_V34 = (("reappraisal_mode", "v34"),)
_IMPL3 = [
    Condition("C-v34", prefix_overrides=_V34, note="M1 + M2 + M3 + 訂正を時間の出来事にする（主条件）"),
    Condition("C-OR", prefix_overrides=(("reappraisal_mode", "v34_or"),), note="上限のない吸収（過剰な反省）"),
    Condition("C-nocorr-v34", prefix_overrides=_V34, skip_cor=True, note="v34 の訂正なしの枝: COR の代わりに空の刻みを N 進める"),
    Condition("A-τ-off", prefix_overrides=_V34, cor_overrides=(("m3_variant", "tau_off"),), note="τ ≡ 1"),
    Condition("A-clock-off", prefix_overrides=_V34, cor_overrides=(("m3_variant", "clock_off"),), note="訂正が時計と気分を進めない（T.2 のまま）"),
    Condition("clamp-τ", prefix_overrides=_V34, cor_overrides=(("m3_variant", "clamp_tau"),), note="罪悪感・後悔・自己正当化を τ に渡さない（気分の経路は残す）"),
    Condition("clamp-mood", prefix_overrides=_V34, cor_overrides=(("m3_variant", "clamp_mood"),), note="f_c・安堵を気分に渡さない（τ の経路は残す）"),
    Condition("clamp-both", prefix_overrides=_V34, cor_overrides=(("m3_variant", "clamp_both"),), note="両方を切る"),
    Condition("τ-yoke", prefix_overrides=_V34, cor_overrides=(("m3_variant", "tau_yoke"),), yoke=True, note="別の背景の人格の τ の軌跡を与える（run_tau_yoke）"),
]
_IMPL4 = [      # Stage 4
    Condition("A-g-const", prefix_overrides=_V34, cor_overrides=(("g_const", 0.5),), note="g_i ≡ 0.5（LLR と κ̄ を g に使わない）。COR の直前から（S3-9 と同じ読み。CHANGELOG-v3.4 S4-1）"),
    Condition("A-κ-fixed", prefix_overrides=_V34, cor_overrides=(("kappa_fixed", True),), note="κ̄ ≡ 0.5、検証なし。COR の直前から（S4-1）"),
    Condition("local-reattr", prefix_overrides=_V34, local_reattr=True, note="C-v34 の上で、各訂正の後に局所の φ の推定し直し。対照にも同じ処理（S4-2）"),
    Condition("local-reattr-v33", local_reattr=True, note="C-v33 の上で、各訂正の後に局所の φ の推定し直し。対照にも同じ処理（RA7(c) の旧設計版。S4-2）"),
    Condition("C-none-v34", prefix_overrides=_V34 + (("no_reappraisal", True),), note="v34 の上で価値を変えない対照（H7: r4・r5 が C-v34 と一致するか）"),
]
CONDITIONS: Dict[str, Condition] = {c.name: c for c in _IMPL + _IMPL3 + _IMPL4}
ALIASES = {"clamp-tau": "clamp-τ", "tau-yoke": "τ-yoke", "A-tau-off": "A-τ-off", "A-kappa-fixed": "A-κ-fixed"}
_STUBS: List[str] = []          # Stage 4 で、動かない条件はなくなった


def get_condition(name: str) -> Condition:
    name = ALIASES.get(name, name)
    if name not in CONDITIONS:
        raise KeyError(f"未知の条件: {name!r}")
    c = CONDITIONS[name]
    if not c.implemented:
        raise NotImplementedError(c.note)
    return c


def advance_clock(npn: NPN, k: int = 1) -> None:
    """空の刻み。v34 の人格では時計を進める（npn.advance_clock）。v33 の人格では何もしない（v3.3 の T.2: 刻みも訂正も時計を進めない）。"""
    if npn.cfg.reappraisal_mode != "v33":
        npn.advance_clock(k)


# ----------------------------------------------------------------------
# 経験史の投入
# ----------------------------------------------------------------------
def make_persona(seed: int, overrides: Tuple[Tuple[str, Any], ...] = ()) -> NPN:
    return NPN(seed=seed, **dict(overrides))


def feed_items(npn: NPN, items: Sequence[Dict[str, Any]], after_correction: Optional[Callable[[NPN, Dict[str, Any]], None]] = None) -> None:
    """経験と訂正を与える。空の刻みは、v33 の人格では時計を進めないので無視し、v34 の人格では時計を 1 進める。
    after_correction: 訂正を与えた直後に呼ぶ関数（local-reattr）。"""
    for it in items:
        if it.get("kind") == "tick":
            advance_clock(npn, 1)
            continue
        npn.experience(copy.deepcopy(it))
        if after_correction is not None and it.get("kind") == "correction":
            after_correction(npn, it)


def local_reattribute(npn: NPN, correction: Dict[str, Any]) -> None:
    """local-reattr（SPEC 第 6 節）: 訂正の後に、参照先の経験と参加者を共有する記録（参照先の参加者が個体である観測の行）だけ、責任 φ を想起の 1 回の掃引で推定し直す。
    他の行の φ は変えない。掃引は全行を同時に（ヤコビ型で）計算するので、選んだ行の値は全体の想起の掃引と同じになる（計算した後で、選ばなかった行を元に戻す）。
    価値・条件づけ・関心の重みは変えない（信念の φ と計数だけ）。"""
    bel = npn.belief
    rec = npn.exp_log[npn._exp_pos[correction["refers_to"]]]
    pids = {p["id"] for p in rec["participants"]}
    R = bel.rec_n
    if R == 0:
        return
    sel = np.array([bel.ind_ids[int(i)] in pids for i in bel.rec_ind[:R]], dtype=bool)
    phi_old = bel.rec_phi[:R].copy()
    bel.reattribute()
    phi_new = bel.rec_phi[:R].copy()
    phi_new[~sel] = phi_old[~sel]
    bel.rec_phi[:R] = phi_new
    bel._rebuild_counts()


# ----------------------------------------------------------------------
# 観測（再評価の利得、特徴の層ごとの変化）。npnv3 は変えない。
# ----------------------------------------------------------------------
@contextlib.contextmanager
def reappraise_probe(log: List[Dict[str, Any]], q0: bool = False):
    """ValueNet.reappraise を一時的に包む。更新の直前に、参加者ごとの予測空間の利得 k = zᵀΣ_k z / (zᵀΣ_k z + R/prec_k) と shift を log に足す。
    q0 = True なら、その更新のあいだだけ人モデルの過程雑音 q を 0 にする（診断 diag-q0）。"""
    orig = V.ValueNet.reappraise

    def wrapped(self, participants, shifts, prec):
        model = self.person
        pr = np.maximum(np.asarray(prec, dtype=np.float64), V.A_FLOOR)
        rec = {"parts": []}
        for p, d in zip(participants, shifts):
            if not np.any(np.abs(d) > 1e-12):
                continue
            ii, z = model._prepare(self.person_feats(p["id"], p["class"]))
            Sz = np.einsum("kij,j->ki", model.S[:, :, ii], z)
            zSz = Sz[:, ii] @ z
            gain = zSz / (zSz + model.R / pr)
            rec["parts"].append({"id": p["id"], "gain": gain.copy(), "shift": np.asarray(d, dtype=np.float64).copy()})
        log.append(rec)
        saved = model.q
        if q0:
            model.q = {k: 0.0 for k in saved}
        try:
            return orig(self, participants, shifts, prec)
        finally:
            model.q = saved

    V.ValueNet.reappraise = wrapped
    try:
        yield
    finally:
        V.ValueNet.reappraise = orig


def gain_summary(log: List[Dict[str, Any]]) -> Dict[str, float]:
    """訂正ごとの利得 gain_i = Σ_e Σ_k |Δ_ek| g_ek / Σ_e Σ_k |Δ_ek|（含意された変化の絶対値で重み付けた予測空間の利得）の和 S_all。
    S_A は参加者 A だけ。"""
    s_all = s_a = 0.0
    for rec in log:
        num = den = na = da = 0.0
        for p in rec["parts"]:
            w = np.abs(p["shift"])
            num += float((w * p["gain"]).sum())
            den += float(w.sum())
            if p["id"] == H.A:
                na += float((w * p["gain"]).sum())
                da += float(w.sum())
        s_all += num / den if den > 0 else 0.0
        s_a += na / da if da > 0 else 0.0
    return {"S_all": s_all, "S_A": s_a, "n_updates": len(log)}


def person_W(npn: NPN) -> Dict[str, np.ndarray]:
    m = npn.value.person
    return {nm: m.W[i].copy() for i, nm in enumerate(m.names)}


def layer_split(npn_post: NPN, W_pre: Dict[str, np.ndarray], a: np.ndarray, pid: str, cls: str) -> Dict[str, float]:
    """参加者 (pid, cls) の好悪の変化 a·Σ z ΔW を、特徴の層（ind、people、cls、emb）ごとに分ける。合計は好悪の変化に一致する。"""
    m = npn_post.value.person
    out: Dict[str, float] = {"ind": 0.0, "people": 0.0, "cls": 0.0, "emb": 0.0}
    for name, layer, v in npn_post.value.person_feats(pid, cls):
        if v == 0.0 or name not in m.idx:
            continue
        d = m.W[m.idx[name]] - W_pre.get(name, np.zeros(N_NEEDS))
        out[layer] = out.get(layer, 0.0) + float(v * (a @ d))
    return out


# ----------------------------------------------------------------------
# 1 個体の実行
# ----------------------------------------------------------------------
def _prefix(seed: int, h: Dict[str, Any], overrides: Tuple[Tuple[str, Any], ...], arm: str) -> NPN:
    npn = make_persona(seed, overrides)
    feed_items(npn, h["pre"])
    feed_items(npn, h[arm])
    feed_items(npn, h["extra"])
    return npn


def run_individual(seed: int, bg: str, src: str, err: str, n: int, m_extra: int, conds: Sequence[str], phase: str) -> Dict[str, Any]:
    """1 個体（種 × BG × SRC × ERR × N × M）を、指定した各条件で実行する。
    返すもの: 各条件の R_pre（訂正の直前）・R_cor（訂正の直後）・R_fol（後続の後）と観測、対照の R_pre・R_fol。"""
    guard_seeds([seed], phase)
    cs = [get_condition(c) for c in conds]
    h = H.make_history(seed, bg, src, err, n, m_extra)
    oc = h["meta"]["o_class"]
    out: Dict[str, Any] = {"meta": h["meta"], "cond": {}, "ctrl": {}}
    prefixes: Dict[Tuple, Dict[str, Any]] = {}
    for c in cs:
        key = c.prefix_overrides
        if key not in prefixes:
            mis = _prefix(seed, h, key, "mis")
            ctrl = _prefix(seed, h, key, "ctrl")
            prefixes[key] = {"mis": mis, "ctrl": ctrl, "R_pre": read_R(mis, oc), "R_ctrl_pre": read_R(ctrl, oc),
                             "W_pre": person_W(mis), "a_pre": mis.concern.a.copy()}
            ctrl2 = copy.deepcopy(ctrl)
            feed_items(ctrl2, h["fol"])
            out["ctrl"][json.dumps(key, default=str)] = {"R_pre": prefixes[key]["R_ctrl_pre"], "R_fol": read_R(ctrl2, oc)}
    for c in cs:
        pf = prefixes[c.prefix_overrides]
        if c.yoke:
            raise ValueError("τ-yoke は他の人格の τ の軌跡が要る。run_tau_yoke を使う")
        npn = copy.deepcopy(pf["mis"])
        for k_, v_ in c.cor_overrides:
            setattr(npn.cfg, k_, v_)
        log: List[Dict[str, Any]] = []
        hook = local_reattribute if c.local_reattr else None
        if not c.skip_cor:
            with reappraise_probe(log, q0=c.q0):
                feed_items(npn, h["cor"], hook)
        else:
            advance_clock(npn, len(h["cor"]))            # v34 の訂正なしの枝: 訂正の数だけ空の刻み（v33 では何もしない）
        R_cor = read_R(npn, oc)
        split = {pid: layer_split(npn, pf["W_pre"], pf["a_pre"], pid, "collaborator") for pid in (H.A, H.A2)}
        feed_items(npn, h["fol"])
        out["cond"][c.name] = {"R_pre": pf["R_pre"], "R_cor": R_cor, "R_fol": read_R(npn, oc), "R_ctrl_pre": pf["R_ctrl_pre"],
                               "gain": gain_summary(log), "layer_split": split}
        if c.local_reattr:                               # 対照にも同じ時点（訂正の列の後）で同じ処理を、訂正と同じ参照先の列で行う
            ctrl_lr = copy.deepcopy(pf["ctrl"])
            for cor in h["cor"]:
                if cor.get("kind") == "correction":
                    local_reattribute(ctrl_lr, cor)
            out["cond"][c.name]["R_ctrl_cor"] = read_R(ctrl_lr, oc)
            feed_items(ctrl_lr, h["fol"])
            out["cond"][c.name]["R_ctrl_fol"] = read_R(ctrl_lr, oc)
    return out


def individual_worker(args: Tuple) -> Dict[str, Any]:
    return run_individual(*args)


def pmap(fn: Callable, items: Sequence, workers: Optional[int] = None) -> List[Any]:
    """プロセス並列（Windows では spawn。結果は決定的で、順序を保つ）。"""
    items = list(items)
    workers = workers if workers is not None else max(1, min(9, (os.cpu_count() or 2) - 2))
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=1))


# ----------------------------------------------------------------------
# 状態の入れ替え（val-swap、hm-eq、swap-G。SPEC 第 6 節・第 7.4 節）
# ----------------------------------------------------------------------
GROUPS = {"G1": "価値（人・状況の W と Σ）", "G2": "信念と経験記録", "G3": "関心の重み a", "G4": "条件づけ C", "G5": "v3.4 の状態（κ、τ）", "G6": "欲求の状態 h と気分 m"}
VAL_SWAP = ("G1", "G3", "G4")            # val-swap: 価値の評価状態 = W・Σ・a・C
HM_EQ = ("G6",)


def aligned(a: NPN, b: NPN) -> bool:
    """入れ替えの条件（第 6 節「特徴の名前の並びが完全に一致」）の、Stage 1 での読み: 価値ネットの特徴（人モデル・状況モデル）の名前と並びが一致し、
    経験記録の id の並びが一致する。条件づけ C の手がかりの並びは求めない（C は 1 つの対象として丸ごと入れ替える。手がかりは結果の型ごとに BG で
    違うので、並びまで求めると全ての組が除かれる。CHANGELOG-v3.4.md の曖昧さ A1）。"""
    return (a.value.person.names == b.value.person.names and a.value.sit.names == b.value.sit.names
            and [r["id"] for r in a.exp_log] == [r["id"] for r in b.exp_log])


def aligned_literal(a: NPN, b: NPN) -> bool:
    """文面どおりの読み: aligned に加えて、条件づけの手がかりの並びも一致する。除かれる数の報告用。"""
    return aligned(a, b) and list(a.cond.idx) == list(b.cond.idx)


def swap_groups(target: NPN, source: NPN, groups: Sequence[str]) -> NPN:
    """target の指定した状態の群を、source の深い写しで置き換えた新しい人格を返す（元の 2 つは変えない）。"""
    t = copy.deepcopy(target)
    s = copy.deepcopy(source)
    for g in groups:
        if g == "G1":
            t.value = s.value
        elif g == "G2":
            s.belief.rng = t.rng if hasattr(s.belief, "rng") else None
            t.belief = s.belief
            t.exp_log = s.exp_log
            t._exp_pos = s._exp_pos
        elif g == "G3":
            t.concern = s.concern
        elif g == "G4":
            t.cond = s.cond
        elif g == "G5":
            t.src_beta = s.src_beta                 # κ の Beta
            t.log_tau = s.log_tau                   # τ
            t.G = s.G                               # C-OR の累積（M3 の評価から作られる）
            t.last_ev = s.last_ev
            t.n_verifications = s.n_verifications
        elif g == "G5k":                            # G5 の κ だけ（Shapley の速さ／中身の分類用の細分。S4-4）
            t.src_beta = s.src_beta
            t.n_verifications = s.n_verifications
        elif g == "G5t":                            # G5 の τ だけ（τ、C-OR の累積 G、出来事の参照）
            t.log_tau = s.log_tau
            t.G = s.G
            t.last_ev = s.last_ev
        elif g == "G6":
            t.h = s.h.copy()
            t.mood = float(s.mood)
        else:
            raise KeyError(g)
    return t


def run_swap(seed: int, src: str, err: str, bg_target: str, bg_source: str, groups: Sequence[str], cond: str, phase: str,
             n: int = 8, m_extra: int = 0, literal: bool = False, nocorr: bool = False) -> Optional[Dict[str, Any]]:
    """同じ種・同じ SRC・同じ ERR の 2 個体（背景だけ違う）の間で、訂正の直前に状態の群を入れ替えて訂正を与える。並びが一致しなければ None（除く）。
    literal = True なら aligned_literal（条件づけの手がかりの並びまで一致を求める文面どおりの読み。感度分析。CHANGELOG-v3.4 A1・L1）。
    nocorr = True なら、入れ替えた同じ人格に、訂正の代わりに空の刻みを N 進めた枝（訂正なしの枝）も実行し、R_cor_nc・R_fol_nc に返す（入れ替えた人格の CE = 訂正あり − 訂正なしの差を作る。S4-3）。
    val-swap = groups VAL_SWAP、hm-eq = HM_EQ、swap-G = 任意の群（G5 = κ・τ を含む）。cond の cor_overrides（clamp など）は入れ替えの後に効かせる。"""
    guard_seeds([seed], phase)
    c = get_condition(cond)
    ht = H.make_history(seed, bg_target, src, err, n, m_extra)
    hs = H.make_history(seed, bg_source, src, err, n, m_extra)
    oc = ht["meta"]["o_class"]
    t = _prefix(seed, ht, c.prefix_overrides, "mis")
    s = _prefix(seed, hs, c.prefix_overrides, "mis")
    if not (aligned_literal(t, s) if literal else aligned(t, s)):
        return None
    sw = swap_groups(t, s, groups)
    for k_, v_ in c.cor_overrides:
        setattr(sw.cfg, k_, v_)
    sw_nc = copy.deepcopy(sw) if nocorr else None
    log: List[Dict[str, Any]] = []
    if not c.skip_cor:
        with reappraise_probe(log, q0=c.q0):
            feed_items(sw, ht["cor"])
    R_cor = read_R(sw, oc)
    feed_items(sw, ht["fol"])
    out = {"R_cor": R_cor, "R_fol": read_R(sw, oc)}
    if sw_nc is not None:
        advance_clock(sw_nc, len(ht["cor"]))
        out["R_cor_nc"] = read_R(sw_nc, oc)
        feed_items(sw_nc, ht["fol"])
        out["R_fol_nc"] = read_R(sw_nc, oc)
    return out


def run_val_swap(seed: int, src: str, err: str, bg_target: str, bg_source: str, phase: str, cond: str = "C-v34", n: int = 8, m_extra: int = 0,
                 literal: bool = False) -> Optional[Dict[str, Any]]:
    """val-swap（SPEC 第 6 節）: 信念・経験記録・h・m・κ・τ は自分のまま、価値の評価状態（G1 の W・Σ、G3 の a、G4 の C）だけを入れ替える。"""
    return run_swap(seed, src, err, bg_target, bg_source, VAL_SWAP, cond, phase, n, m_extra, literal)


def run_hm_eq(seed: int, src: str, err: str, bg_target: str, bg_source: str, phase: str, cond: str = "C-v34", n: int = 8, m_extra: int = 0,
              literal: bool = False) -> Optional[Dict[str, Any]]:
    """hm-eq（SPEC 第 6 節）: 欲求の状態 h と気分 m だけを相手の値に揃える。"""
    return run_swap(seed, src, err, bg_target, bg_source, HM_EQ, cond, phase, n, m_extra, literal)


SHAPLEY_PLAYERS = ("G1", "G2", "G3", "G4", "G5", "G6")
SHAPLEY_PLAYERS_SPLIT_G5 = ("G1", "G2", "G3", "G4", "G5k", "G5t", "G6")


def shapley_values(v: Dict[frozenset, float], players: Sequence[str]) -> Dict[str, float]:
    """Shapley 値 φ_g = Σ_{S ⊆ N∖{g}} |S|! (n − |S| − 1)! / n! · (v(S ∪ {g}) − v(S))。v は全ての部分集合（frozenset）の値を持つ。効率性 Σ φ = v(N) − v(∅)。"""
    pl = list(players)
    n = len(pl)
    fact = [math.factorial(i) for i in range(n + 1)]
    out: Dict[str, float] = {}
    for g in pl:
        rest = [x for x in pl if x != g]
        tot = 0.0
        for mask in range(1 << len(rest)):
            S = frozenset(rest[i] for i in range(len(rest)) if mask >> i & 1)
            w = fact[len(S)] * fact[n - len(S) - 1] / fact[n]
            tot += w * (v[S | {g}] - v[S])
        out[g] = tot
    return out


def run_shapley(seed: int, src: str, err: str, bg_target: str, bg_source: str, phase: str, cond: str = "C-v34", n: int = 8, m_extra: int = 0,
                players: Sequence[str] = SHAPLEY_PLAYERS, literal: bool = False) -> Optional[Dict[str, Any]]:
    """RA2(b) の Shapley 分解の素材（SPEC 第 6 節 swap-G、第 7.4 節）。同じ種・同じ SRC・同じ ERR の 2 個体（背景だけ違う）で、訂正の直前に、
    players の部分集合 S のそれぞれ（2^|players| 通り）の状態の群を target から source のものに入れ替え、入れ替えた人格に COR → FOL（訂正あり）と
    空の刻み → FOL（訂正なし）を与える。各 S の CE_S = R_fol(訂正あり) − R_fol(訂正なし)。target・source の入れ替えなしの CE（CE_b、CE_b′）も返す。
    プレイヤーの価値関数は呼び出し側が正規化した CE から作る（run_shapley_values）。並びが一致しなければ None（除く）。
    prefix は 1 回だけ作る（部分集合ごとに作らない）。"""
    guard_seeds([seed], phase)
    c = get_condition(cond)
    ht = H.make_history(seed, bg_target, src, err, n, m_extra)
    hs = H.make_history(seed, bg_source, src, err, n, m_extra)
    oc = ht["meta"]["o_class"]
    t = _prefix(seed, ht, c.prefix_overrides, "mis")
    s = _prefix(seed, hs, c.prefix_overrides, "mis")
    if not (aligned_literal(t, s) if literal else aligned(t, s)):
        return None
    pl = list(players)

    def ce_of(person_cor: NPN, hcor: Sequence[Dict[str, Any]], hfol: Sequence[Dict[str, Any]]) -> np.ndarray:
        a = copy.deepcopy(person_cor)
        b = copy.deepcopy(person_cor)
        for k_, v_ in c.cor_overrides:
            setattr(a.cfg, k_, v_)
        feed_items(a, hcor)
        feed_items(a, hfol)
        advance_clock(b, len(hcor))
        feed_items(b, hfol)
        return read_R(a, oc) - read_R(b, oc)

    ce: Dict[str, List[float]] = {}
    for mask in range(1 << len(pl)):
        S = [pl[i] for i in range(len(pl)) if mask >> i & 1]
        sw = swap_groups(t, s, S) if S else copy.deepcopy(t)
        ce["|".join(sorted(S))] = [float(x) for x in ce_of(sw, ht["cor"], ht["fol"])]
    own_t = ce_of(t, ht["cor"], ht["fol"])
    own_s = ce_of(s, hs["cor"], hs["fol"])
    return {"players": pl, "CE_subsets": ce, "CE_target": [float(x) for x in own_t], "CE_source": [float(x) for x in own_s]}


def shapley_from_run(run: Dict[str, Any], scale: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """run_shapley の出力から、1 種・1 組の Shapley 分担を作る。価値関数 v(S) = 1 − ‖CE_S − CE_b′‖ / ‖CE_b − CE_b′‖（正規化した CE。
    ギャップのうち、群 S の入れ替えで閉じた割合。v(∅) = 0）。効率性: Σ φ = v(全群) − v(∅)。ギャップが 0 なら None。"""
    pl = run["players"]
    ceb = normalize(np.array(run["CE_target"]), scale, mask)
    cebp = normalize(np.array(run["CE_source"]), scale, mask)
    gap = float(np.linalg.norm(ceb - cebp))
    if gap < 1e-12:
        return {"gap": gap, "phi": None, "v_all": None}
    v: Dict[frozenset, float] = {}
    for key, vec in run["CE_subsets"].items():
        S = frozenset(key.split("|")) if key else frozenset()
        v[S] = 1.0 - float(np.linalg.norm(normalize(np.array(vec), scale, mask) - cebp)) / gap
    phi = shapley_values(v, pl)
    return {"gap": gap, "phi": phi, "v_all": v[frozenset(pl)], "v_empty": v[frozenset()]}


def run_tau_yoke(seed: int, src: str, err: str, bg_target: str, bg_donor: str, phase: str, n: int = 8, m_extra: int = 0) -> Dict[str, Any]:
    """τ-yoke（プラセボ τ。SPEC 第 6 節）: 同じ種・同じ SRC・同じ ERR で別の背景の人格（donor）に同じ訂正の列を与えて生じた τ の軌跡を、
    target に（自分の評価で動かす代わりに）そのまま与える。donor は C-v34、target の他の状態は自分のまま。"""
    guard_seeds([seed], phase)
    from npnv3.trace import tau_trajectory
    ht = H.make_history(seed, bg_target, src, err, n, m_extra)
    hd = H.make_history(seed, bg_donor, src, err, n, m_extra)
    oc = ht["meta"]["o_class"]
    donor = _prefix(seed, hd, _V34, "mis")
    feed_items(donor, hd["cor"])
    traj = tau_trajectory(donor.ledger.entries)
    c = get_condition("τ-yoke")
    t = _prefix(seed, ht, c.prefix_overrides, "mis")
    for k_, v_ in c.cor_overrides:
        setattr(t.cfg, k_, v_)
    t.set_tau_yoke(traj)
    feed_items(t, ht["cor"])
    R_cor = read_R(t, oc)
    feed_items(t, ht["fol"])
    return {"R_cor": R_cor, "R_fol": read_R(t, oc), "log_tau": dict(t.log_tau), "donor_log_tau": dict(donor.log_tau)}


# ----------------------------------------------------------------------
# 効果と統計（SPEC 第 7.2・7.3 節）
# ----------------------------------------------------------------------
def scale_of(R_pre_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """正規化の尺度: 主要因計画の全個体の COR の直前の R の標準偏差。1e-9 未満の成分は除く（mask = False）。"""
    sd = np.asarray(R_pre_all, dtype=np.float64).std(axis=0, ddof=1)
    mask = sd >= 1e-9
    return np.where(mask, sd, 1.0), mask


def normalize(X: np.ndarray, scale: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)[..., mask] / scale[mask]


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def _boot_idx(n: int, nboot: int = N_BOOT) -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(nboot, n))


def mean_ci(X: np.ndarray, level: float = 0.95, nboot: int = N_BOOT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """種を単位にしたブートストラップのパーセンタイル区間（成分ごと）。X は (種, 成分)。"""
    X = np.asarray(X, dtype=np.float64)
    idx = _boot_idx(X.shape[0], nboot)
    bm = X[idx].mean(axis=1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return X.mean(axis=0), np.percentile(bm, lo, axis=0), np.percentile(bm, hi, axis=0)


def cos_ci(X1: np.ndarray, X2: np.ndarray, level: float = 0.95, nboot: int = N_BOOT) -> Tuple[float, float, float]:
    """2 つの平均ベクトルの余弦と、種を単位にした対のブートストラップ区間。X1, X2 は (種, 成分)で行が対応する。"""
    X1, X2 = np.asarray(X1, dtype=np.float64), np.asarray(X2, dtype=np.float64)
    idx = _boot_idx(X1.shape[0], nboot)
    m1, m2 = X1[idx].mean(axis=1), X2[idx].mean(axis=1)
    num = (m1 * m2).sum(axis=1)
    den = np.linalg.norm(m1, axis=1) * np.linalg.norm(m2, axis=1)
    cs = num / np.where(den > 0, den, np.nan)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return cosine(X1.mean(axis=0), X2.mean(axis=0)), float(np.nanpercentile(cs, lo)), float(np.nanpercentile(cs, hi))


def directions_differ(X1: np.ndarray, X2: np.ndarray, level: float = 0.95) -> Dict[str, Any]:
    """「向きが違う」（SPEC 第 7.3 節）: (a) ある成分で平均の符号が逆で、両方の区間が 0 を含まない、または (b) cos の区間の上限が 0.5 未満。"""
    m1, l1, h1 = mean_ci(X1, level)
    m2, l2, h2 = mean_ci(X2, level)
    nz1 = (l1 > 0) | (h1 < 0)
    nz2 = (l2 > 0) | (h2 < 0)
    opp = nz1 & nz2 & (np.sign(m1) != np.sign(m2))
    c, clo, chi = cos_ci(X1, X2, level)
    return {"differ": bool(opp.any() or chi < 0.5), "a_opposite_sign": bool(opp.any()), "opposite_components": [int(i) for i in np.where(opp)[0]],
            "b_cos_upper_lt_0.5": bool(chi < 0.5), "cos": c, "cos_lo": clo, "cos_hi": chi}


def wilson(k: int, n: int, level: float = 0.95) -> Tuple[float, float, float]:
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + level / 2)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - r, c + r


LEVEL_MAIN = 1 - 0.05 / 3            # 主判定 RA1〜RA3 は Bonferroni で 98.33% 区間
LEVEL_OTHER = 0.95


def remaining_fraction(like_mis_pre: float, like_ctrl_pre: float, like_post: float) -> Tuple[float, float, float]:
    """残る割合 r = d_post / d_pre。d_pre = like(A)_誤帰属 − like(A)_対照（COR の直前）、d_post = like(A)_直後 − like(A)_対照（同じ対照）。"""
    d_pre = like_mis_pre - like_ctrl_pre
    d_post = like_post - like_ctrl_pre
    return d_pre, d_post, (d_post / d_pre if abs(d_pre) > 1e-12 else float("nan"))


def file_sha256(path: Path) -> str:
    """SHA-256（改行は LF に揃えてから。git の autocrlf で作業ツリーの改行が変わっても同じ値になる）。"""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
