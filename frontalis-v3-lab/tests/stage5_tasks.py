"""v3.4 Stage 5（評価）の仕事の関数（ワーカーで実行）。凍結済みのプロトコル（ra_protocol、ra_checks_v34、ra_parallel）は変えず、その上に
(1) 主要因計画の 1 個体を全条件で実行する main_task（凍結した run_individual の呼び出しを、観察者が包んで RA6・RA7(a)・H1・H2 を全条件で検査する）、
(2) N の掃引 sweep_task（5 条件 × N 5 × M 2。対照を超えて反対側へ行った個体は台帳から再構成する: RA5）、
(3) extras_task（RA12 の転移の報告、RA13、読み出しの補助）、
(4) old_y_task（v3.3 の Y1〜Y7 のプロトコルそのまま。RA8 の「Y3 の比」と C-v33 の比較）
を足す。観察者は状態を変えない（読むだけ）。結果はすべて JSON にできる形（to_jsonable）で返す。

乱数種の守りは ra_protocol.guard_seeds（phase = "dev" | "stage5"）。評価の種 3400〜3439 は phase = "stage5" でだけ通る。
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_histories_v34 as H  # noqa: E402
import ra_checks_v34 as CK  # noqa: E402
import ra_protocol as RP  # noqa: E402
from npnv3.innate import N_NEEDS  # noqa: E402

# 主要因計画の 1 個体で実行する条件（stage4_timing.MAIN_CONDS に、RA7(c) の旧設計版 local-reattr-v33 と、H7 の全セル検査用の C-none-v34 を足したもの）
MAIN_CONDS = ["C-v34", "C-OR", "C-v33", "C-retro", "C-none", "C-nocorr", "C-nocorr-v34", "A-g-const", "A-κ-fixed", "A-τ-off", "A-clock-off",
              "clamp-τ", "clamp-mood", "clamp-both", "local-reattr", "local-reattr-v33", "C-none-v34"]
SWEEP_CONDS = ["C-v34", "C-OR", "C-v33", "C-retro", "C-nocorr"]
N_GRID = (2, 4, 8, 12, 16)
M_GRID = (0, 4)
OVERSHOOT = -0.05                     # SPEC RA5: 対照を超えて反対側へ行った個体は r < −0.05


def to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def _led(npn, kind: str) -> List[Dict[str, Any]]:
    return [e for e in npn.ledger.entries if e["kind"] == kind]


# ======================================================================
# 観察者: run_individual の中の feed_items を包んで、訂正の列・後続の列の前後に検査を当てる（状態は変えない）
# ======================================================================
def _label(npn) -> str:
    c = npn.cfg
    return "|".join(str(getattr(c, k, None)) for k in ("reappraisal_mode", "no_reappraisal", "retroactive_value", "m3_variant", "g_const", "kappa_fixed"))


def _classify(items: Sequence[Dict[str, Any]]) -> Optional[str]:
    if not items:
        return None
    if all(it.get("kind") == "correction" for it in items):
        return "cor"
    if all(str(it.get("id", "")).startswith("fol") for it in items):
        return "fol"
    return None


class Observer:
    """全条件に当てる検査: RA6（当時の台帳と original_outcomes の不変、訂正の記録の数）、RA7(a)、H1、H2。1 つの検査は何度当てても 1 つでも破れたら破れ。"""

    def __init__(self) -> None:
        self.inv = CK.Inv()
        self.n_events = 0
        self.ce_counts: Dict[str, List[int]] = {}

    def check_after(self, npn, kind: str, snap: Optional[Dict[str, Any]], n_items: int, label: str) -> None:
        self.n_events += 1
        mode = "v34" if npn.cfg.reappraisal_mode == "v34" else "other"
        CK._check_h1(npn, self.inv, f"{label}:{kind}", mode)
        # H2: 台帳の和 = 状態（W。v34 の κ と log τ も）
        ok2 = True
        for model, net in (("person", npn.value.person), ("situation", npn.value.sit)):
            tot = npn.ledger.totals(model)
            for nm in net.names:
                row = tot.get(nm, np.zeros(N_NEEDS))
                ok2 &= bool(np.abs(row - net.W[net.idx[nm]]).max() <= CK.TOL)
        vers = _led(npn, "verification")
        for s_, (al, be) in npn.src_beta.items():
            ok2 &= abs(al - 1.0 - sum(e["d_alpha"] for e in vers if e["source"] == s_)) <= 1e-12 and abs(be - 1.0 - sum(e["d_beta"] for e in vers if e["source"] == s_)) <= 1e-12
        self.inv.add("H2_W_kappa", ok2, f"{label}:{kind}: 台帳の和 ≠ 状態（W または κ）")
        lt_sum: Dict[str, float] = {}
        for ce in _led(npn, "correction_experience"):
            for k, (b_, a_) in ce["log_tau"].items():
                lt_sum[k] = lt_sum.get(k, 0.0) + (a_ - b_)
        ok_t = set(lt_sum) == set(npn.log_tau) and all(abs(lt_sum[k] - npn.log_tau[k]) <= 1e-12 for k in lt_sum)
        self.inv.add("H2_logtau", bool(ok_t), f"{label}:{kind}: 台帳の log τ の和 ≠ 状態")
        if kind != "cor" or snap is None:
            return
        exp1 = json.dumps(_led(npn, "experience")[: snap["n_exp"]], sort_keys=True, default=str)
        orig1 = json.dumps([r["original_outcomes"] for r in npn.exp_log][: snap["n_orig"]], sort_keys=True)
        n_cor_added = len(_led(npn, "correction")) - snap["n_cor"]
        n_ce_added = len(_led(npn, "correction_experience")) - snap["n_ce"]
        self.inv.add("RA6", exp1 == snap["exp"] and orig1 == snap["orig"] and n_cor_added == n_items, f"{label}: 当時の台帳/original_outcomes の変化、または訂正の数 {n_cor_added} ≠ {n_items}")
        self.ce_counts.setdefault(label, []).append(int(n_ce_added))
        b2 = copy.deepcopy(npn.belief, {id(npn.emb): npn.emb})
        b2._rebuild_counts()
        self.inv.add("RA7a", all(np.abs(getattr(npn.belief, k) - getattr(b2, k)).max() <= 1e-12 for k in ("N1", "N0", "CP1", "CP0", "pi")), f"{label}: 信念の計数が記録から作り直した値と不一致")

    @contextlib.contextmanager
    def patch(self):
        orig = RP.feed_items
        me = self

        def wrapped(npn, items, *a, **k):
            kind = _classify(items)
            snap = None
            if kind == "cor":
                snap = {"exp": json.dumps(_led(npn, "experience"), sort_keys=True, default=str), "n_exp": len(_led(npn, "experience")),
                        "orig": json.dumps([r["original_outcomes"] for r in npn.exp_log], sort_keys=True), "n_orig": len(npn.exp_log),
                        "n_cor": len(_led(npn, "correction")), "n_ce": len(_led(npn, "correction_experience"))}
            r = orig(npn, items, *a, **k)
            if kind is not None:
                me.check_after(npn, kind, snap, len(items), _label(npn))
            return r

        RP.feed_items = wrapped
        try:
            yield self
        finally:
            RP.feed_items = orig

    def summary(self) -> Dict[str, Any]:
        d = self.inv.dump()
        return {"n_events": self.n_events, "inv": d, "ce_counts": {k: sorted(set(v)) for k, v in self.ce_counts.items()}}


def _reduce_individual(out: Dict[str, Any]) -> Dict[str, Any]:
    return to_jsonable({"meta": out["meta"], "cond": out["cond"], "ctrl": out["ctrl"]})


# ======================================================================
# 仕事
# ======================================================================
def main_task(seed: int, bg: str, src: str, err: str, phase: str, n: int = 8) -> Dict[str, Any]:
    """主要因計画の 1 個体（種 × BG × SRC × ERR、N = 8、M = 0）を全条件で実行する。"""
    RP.guard_seeds([seed], phase)
    obs = Observer()
    with obs.patch():
        out = RP.run_individual(seed, bg, src, err, n, 0, MAIN_CONDS, phase)
    return {"run": _reduce_individual(out), "observer": to_jsonable(obs.summary())}


def ledger_recon(seed: int, bg: str, src: str, err: str, n: int, m: int, mode: str, phase: str) -> Dict[str, Any]:
    """対照を超えて反対側へ行った個体の再構成（RA5）: COR の前後の like(A) の変化が、(i) 台帳の ΔW、(ii) 台帳の Σ g_i D_i（a · Σ_i g_e[A] D_i[A]）から、1e-9 以内で作れるか。"""
    RP.guard_seeds([seed], phase)
    h = H.make_history(seed, bg, src, err, n, m)
    mis = RP._prefix(seed, h, (("reappraisal_mode", mode),), "mis")
    a0 = mis.concern.a.copy()
    n_cor0 = len(_led(mis, "correction"))
    l0 = float(mis.value.like_entity(H.A, "collaborator", a0)[0])
    RP.feed_items(mis, h["cor"])
    l1 = float(mis.value.like_entity(H.A, "collaborator", a0)[0])
    z = {f: v for f, _l, v in mis.value.person_feats(H.A, "collaborator")}
    dl_led = 0.0
    sum_gd = 0.0
    for c in _led(mis, "correction")[n_cor0:]:
        dW = c["dW"]["person"]
        dl_led += sum(z[f] * float(np.dot(a0, np.array(dW[f]))) for f in z if f in dW)
        m1 = c["m1"]
        sum_gd += float(np.dot(a0, m1["g_e"][H.A] * CK.need(m1["D"][H.A])))
    return {"like_change": l1 - l0, "ledger_dW": dl_led, "sum_gD": sum_gd, "err_dW": abs(dl_led - (l1 - l0)), "err_gD": abs(sum_gd - (l1 - l0))}


def sweep_task(seed: int, bg: str, err: str, phase: str) -> Dict[str, Any]:
    """N の掃引（SRC-none）の 1 つの (BG, ERR) を、N 5 × M 2 の全個体で。5 条件。反対側へ行った C-v34・C-OR の個体は台帳から再構成する。"""
    RP.guard_seeds([seed], phase)
    recs: List[Dict[str, Any]] = []
    for n in N_GRID:
        for m in M_GRID:
            r = RP.run_individual(seed, bg, "none", err, n, m, SWEEP_CONDS, phase)
            rec: Dict[str, Any] = {"N": n, "M": m, "run": _reduce_individual(r), "recon": {}}
            for c, mode in (("C-v34", "v34"), ("C-OR", "v34_or")):
                x = r["cond"][c]
                d_pre = float(x["R_pre"][0] - x["R_ctrl_pre"][0])
                if abs(d_pre) > 1e-12:
                    rr = float((x["R_cor"][0] - x["R_ctrl_pre"][0]) / d_pre)
                    if rr < OVERSHOOT:
                        rec["recon"][c] = ledger_recon(seed, bg, "none", err, n, m, mode, phase)
            recs.append(rec)
    return {"bg": bg, "err": err, "records": recs}


PROBE_ENTITIES = [("B", H.CTX_INS), ("A2", H.CTX_MIS), ("A", H.CTX_MIS)] + [(f"P{k:02d}", H.CTX_BG) for k in range(1, 13)]


def _likes(npn) -> Dict[str, float]:
    return {pid: float(npn.probe({"entity": pid, "class": "collaborator", "context": ctx})["like"]) for pid, ctx in PROBE_ENTITIES}


def extras_task(seed: int, bg: str, src: str, err: str, phase: str, n: int = 8) -> Dict[str, Any]:
    """RA12 の転移の報告（B・A2・P01〜P12 の好悪の変化）、RA13（C-v34 / C-nocorr-v34 / 対照 / FOL だけの白紙の人格）の素材。"""
    RP.guard_seeds([seed], phase)
    h = H.make_history(seed, bg, src, err, n, 0)
    oc = h["meta"]["o_class"]
    ov = RP._V34
    mis = RP._prefix(seed, h, ov, "mis")
    ctrl = RP._prefix(seed, h, ov, "ctrl")
    out: Dict[str, Any] = {"pre": _likes(mis), "ctrl_pre": _likes(ctrl)}
    a = copy.deepcopy(mis)
    RP.feed_items(a, h["cor"])
    out["cor"] = _likes(a)
    RP.feed_items(a, h["fol"])
    out["fol"] = _likes(a)
    nc = copy.deepcopy(mis)
    RP.advance_clock(nc, len(h["cor"]))
    out["nocorr_cor"] = _likes(nc)
    RP.feed_items(nc, h["fol"])
    out["nocorr_fol"] = _likes(nc)
    RP.feed_items(ctrl, h["fol"])
    out["ctrl_fol"] = _likes(ctrl)
    blank = RP.make_persona(seed, ov)
    RP.feed_items(blank, h["fol"])
    out["R_blank_fol"] = RP.read_R(blank, oc)
    out["likes_blank_fol"] = _likes(blank)
    return to_jsonable(out)


def old_y_task(seed: int, phase: str) -> Dict[str, Any]:
    """v3.3 の Y1〜Y7 のプロトコルそのまま（C-v33。tests/yz_protocol.y_worker）。評価の種で、RA8 の「v3.3 の Y3 の比」と C-v33 の比較に使う。"""
    RP.guard_seeds([seed], phase)
    import yz_protocol as Y
    return to_jsonable(Y.y_worker(seed))


def align_task(seed: int, err: str, phase: str) -> Dict[str, Any]:
    """入れ替えの条件（SPEC 第 6 節の「特徴の名前の並びが完全に一致」）の 2 通りの読みで、背景の組が揃うか: aligned（Stage 1 の読み。リードの判断 L1）と
    aligned_literal（条件づけ C の手がかりの並びまで求める文面どおりの読み。感度分析）。除かれる数を報告するための素材。"""
    RP.guard_seeds([seed], phase)
    hs = {bg: H.make_history(seed, bg, "none", err, 8, 0) for bg in H.BGS}
    pre = {bg: RP._prefix(seed, hs[bg], RP._V34, "mis") for bg in H.BGS}
    out: Dict[str, Any] = {}
    for bt in H.BGS:
        for bs in H.BGS:
            if bt != bs:
                out[f"{bt}<-{bs}"] = {"aligned": bool(RP.aligned(pre[bt], pre[bs])), "aligned_literal": bool(RP.aligned_literal(pre[bt], pre[bs]))}
    return out
