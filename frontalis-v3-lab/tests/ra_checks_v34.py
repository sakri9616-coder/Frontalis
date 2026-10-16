"""v3.4 の故障の検査（SPEC-v3.4 第 8 節の不変量）と、変異テスト（RA11）のための検出の行列の計算。Stage 4 で凍結するプロトコルの一部。

1 つの個体（種 × BG × SRC × ERR）を、変異体（mutants_v34.apply_mutant）の下で実行し、次の検査を当てる（以下 検査 ID）。1 つでも破れたら、その検査は「検出」。
  RA3_3  clamp-τ は COR の直後に τ 以外の学習状態を C-v34 と変えない（ビット単位）          RA3_4  clamp-mood は FOL の後の学習状態・読み出し R をビット単位で変えない
  RA3_5  ERR-report / report-x では τ が動かず、clamp-τ との差が厳密に 0
  RA5_range 各訂正で g ∈ (0, 1)（C-v34）、LLR 有限、g^OR 有限（C-OR）  RA5_formula g = σ(logit κ̄ + LLR)、g^OR = σ(·)(1 + λ_g G_e)、κ̄ = α/(α+β)
  RA5_pred 吸収の後の予測の変化 = g_i D_i（欲求ごと 1e-9、外から測る。参加者と状況）  RA5_sigma 吸収で Σ・C・a がビット単位で不変
  RA5_remainder 台帳の delta + remainder = D、delta = g D  RA5_reconstruct 台帳の ΔW から like(A) の変化が 1e-9 以内で再構成できる
  RA6    当時の台帳（kind = "experience"）と exp_log の original_outcomes が不変、訂正の経験の追記はちょうど N 件   RA7a 訂正後の信念の計数が観測の記録から作り直した値に一致
  RA8a   FOL で A との直接の経験をするたび、A の観測だけの更新（更新前の人モデルの写しに、台帳の教師を与える）で、ind:A の更新の符号が sign(教師 − 予測)。
         台帳の ΔW の合計は O の更新が共分散を通じて ind:A を動かすので符号が逆になりうる（初版はこれと混同した。CHANGELOG-v3.4 S4-8）。合計の不一致の数は報告に残す  RA8b 同じ経験で予測空間の利得 ≥ 1e-3
  RA8c   τ の境界（強い押しでも ±log 3 を出ない）と、境界から反対向きの押しで動く、規則の両方向  RA8d 反対向きの検証で κ̄ が 0.01 以上動く
  H1 有限・範囲  H2 台帳の和 = 状態（W、κ、log τ）  H3 κ・τ は検証・訂正の出来事でだけ変わる  H4 保存・復元がビット単位・決定的  H5 τ ≡ 1 / v34 の通常の経験の経路が v33 とビット単位で一致
  H7 C-v34 と C-none の r4・r5 が一致  H8 気分の循環の有界性  H9 τ は訂正の出来事でだけ、|Δ log τ| ≤ η_τ  RA12 無関係な価値の崩壊なし

判定の閾値や定義は SPEC の値のまま（1e-9、(0, 1)、|W| ≤ 10、η_τ、50%/0.02 など）。この版の検査の集合が「初期の集合」で、Stage 4 の変異テストの結果を見て足した検査は
CANDIDATE_CHECKS に別に置く（CHANGELOG-v3.4.md に、足す前・後の行列を書く）。

主判定の基準（RA0〜RA4）の判定関数は、変異テストの「判定の反転」列として使う（criteria_verdicts）。合否の判定そのもの（Stage 5）の結果の書き出しではない。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_histories_v34 as H  # noqa: E402
import ra_protocol as RP  # noqa: E402
from mutants_v34 import apply_mutant  # noqa: E402
from npnv3 import NPN  # noqa: E402
from npnv3.config import Config  # noqa: E402
from npnv3.innate import N_NEEDS, NEEDS  # noqa: E402
from npnv3.value import A_FLOOR  # noqa: E402

CFG = Config()
TOL = 1e-9
W_LIMIT = 10.0
ETA = CFG.eta_tau
LIM = math.log(CFG.tau_max)
LAMBDA_G = CFG.lambda_g
V34 = RP._V34
KEY_TAU = "collaborator|project_meeting"          # A の曖昧な出来事の τ の鍵
LEARN_META = ("belief", "value", "concern", "conditioning", "justification", "justification_sources")

# 検査の ID。invariant は「1 つでも破れたら検出」、criteria は「判定が基準（変異なし）から反転したら検出」
INVARIANT_CHECKS = ["RA3_3", "RA3_4", "RA3_5", "RA5_range", "RA5_formula", "RA5_pred", "RA5_sigma", "RA5_remainder", "RA5_reconstruct",
                    "RA6", "RA7a", "RA8a", "RA8b", "RA8c", "RA8d", "H1", "H2", "H3", "H4", "H5", "H7", "H8", "H9", "RA12"]
CRITERIA_CHECKS = ["RA0", "RA1", "RA2", "RA2_hm", "RA3_1", "RA3_2", "RA4a", "RA4b", "RA4c"]
# 初期の集合の外（結果を見た後に足してよい検査。CHANGELOG-v3.4 に前後の行列を書く）
CANDIDATE_CHECKS = ["C_tau_rule", "C_D_recompute"]
NOT_APPLICABLE = {"RA-Y4": "対照 C-retro の検査。変異体は対照の経路を変えない", "H6": "既定モードでの v3.3 の全テスト（別の手順。Stage 2・3 で実行済み）",
                  "RA9": "報告のみ", "RA13": "報告のみ", "RA14": "報告のみ（C-OR の性質）"}


def need(d: Dict[str, float]) -> np.ndarray:
    return np.array([d[k] for k in NEEDS], dtype=np.float64)


def led(npn: NPN, kind: str) -> List[Dict[str, Any]]:
    return [e for e in npn.ledger.entries if e["kind"] == kind]


def fl(x: Any) -> Any:
    return np.asarray(x, dtype=np.float64).tolist()


def _sigma(z: float) -> float:
    z = max(-CFG.g_logit_clip, min(CFG.g_logit_clip, float(z)))
    return 1.0 / (1.0 + math.exp(-z))


def learn_digest(npn: NPN, with_kappa_tau: bool) -> str:
    """学習の状態（W・Σ・信念・C・a・h・J）のハッシュ。with_kappa_tau なら κ の Beta と τ も含める。気分・台帳・時計は含めない。"""
    meta, arrays = npn.get_state()
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(arrays[k]).tobytes())
    for k in LEARN_META:
        h.update(json.dumps(meta[k], sort_keys=True, default=str).encode("utf-8"))
    h.update(json.dumps(npn.src_beta, sort_keys=True).encode("utf-8"))
    if with_kappa_tau:
        h.update(json.dumps(npn.log_tau, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def same_state(a: NPN, b: NPN) -> bool:
    return RP.state_digest(a) == RP.state_digest(b)


def u_vector(npn: NPN) -> List[float]:
    """RA12 の無関係な価値のベクトル: Q1〜Q4（類 peer）の好悪、概念 music と solo_work の好悪。"""
    out = [float(npn.probe({"entity": f"Q{k}", "class": "peer", "context": H.CTX_SRC})["like"]) for k in range(1, 5)]
    out += [float(npn.probe({"concept": c})["like"]) for c in (H.CON_INS, H.CON_SOLO)]
    return out


def ra12_ok(u0: Sequence[float], u1: Sequence[float]) -> Tuple[bool, str]:
    """SPEC RA12: 直前の |成分| > 0.02 の成分について、ノルムが 50% 未満に縮む、または符号が 2 つ以上反転すれば「崩れる」。"""
    idx = [i for i, x in enumerate(u0) if abs(x) > 0.02]
    if not idx:
        return True, ""
    a = np.array([u0[i] for i in idx])
    b = np.array([u1[i] for i in idx])
    shrink = np.linalg.norm(b) < 0.5 * np.linalg.norm(a)
    flips = int(np.sum(np.sign(a) * np.sign(b) < 0))
    return (not shrink and flips < 2), f"norm {np.linalg.norm(a):.4g}->{np.linalg.norm(b):.4g}, flips {flips}"


class Inv:
    """検査の結果の集まり。1 つの検査は、何度当てても 1 つでも破れたら破れ。"""

    def __init__(self) -> None:
        self.r: Dict[str, Dict[str, Any]] = {}

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        e = self.r.setdefault(name, {"ok": True, "fails": []})
        if not ok:
            e["ok"] = False
            if len(e["fails"]) < 3:
                e["fails"].append(detail)

    def dump(self) -> Dict[str, Any]:
        return self.r


def _check_h1(npn: NPN, inv: Inv, tag: str, mode: str) -> None:
    ok = True
    for net in (npn.value.person, npn.value.sit):
        ok &= bool(np.isfinite(net.W).all() and np.isfinite(net.S).all() and np.abs(net.W).max() <= W_LIMIT)
    for s, (al, be) in npn.src_beta.items():
        ok &= bool(0.0 < al / (al + be) < 1.0)
    for k, v in npn.log_tau.items():
        ok &= bool(math.isfinite(v) and 1.0 / 3.0 - 1e-12 <= math.exp(v) <= 3.0 + 1e-12)
    ok &= bool(abs(npn.mood) < 2.0 and np.isfinite(npn.h).all())
    for c in led(npn, "correction"):
        m1 = c.get("m1")
        if m1 is None:
            continue
        ok &= bool(math.isfinite(m1["llr"]) and math.isfinite(m1["z"]))
        if mode == "v34":
            ok &= bool(0.0 < m1["g"] < 1.0)
    inv.add("H1", ok, f"{tag}: W/S/κ/τ/mood/g の範囲")


# ======================================================================
# 1 つの個体の実行と検査
# ======================================================================
def run_checked(seed: int, bg: str, src: str, err: str, n: int, phase: str, mode: str = "v34", det: bool = False) -> Dict[str, Any]:
    """mode = "v34"（C-v34）| "v34_or"（C-OR）で、COR を 1 件ずつ与えて測りながら実行する。返すもの: inv（検査の結果）、ev（測定値・候補の検査の素材）、npn（実行後の状態。呼び出し側が使う）。"""
    RP.guard_seeds([seed], phase)
    h = H.make_history(seed, bg, src, err, n, 0)
    oc = h["meta"]["o_class"]
    ov = (("reappraisal_mode", mode),)
    inv = Inv()
    mis = RP._prefix(seed, h, ov, "mis")
    pre = copy.deepcopy(mis)
    R_pre = RP.read_R(mis, oc)
    ctrl = RP._prefix(seed, h, ov, "ctrl")
    R_ctrl_pre = RP.read_R(ctrl, oc)
    S0 = (mis.value.person.S.copy(), mis.value.sit.S.copy())
    names0 = (list(mis.value.person.names), list(mis.value.sit.names))
    C0, a0 = mis.cond.C.copy(), mis.concern.a.copy()
    exp0 = json.dumps(led(mis, "experience"), sort_keys=True, default=str)
    orig0 = json.dumps([r["original_outcomes"] for r in mis.exp_log], sort_keys=True)
    n_ce0 = len(led(mis, "correction_experience"))
    n_cor0 = len(led(mis, "correction"))
    u0 = u_vector(mis)
    like_pre = RP.read_R(mis, oc)[0]
    parts = [{"id": H.A, "class": "collaborator"}, {"id": H.O, "class": oc}]
    ctx, con = H.CTX_MIS, [H.CON_MIS]
    # ---- COR（1 件ずつ）----
    cor_recs: List[Dict[str, Any]] = []
    like_chk: List[float] = []
    for cor in h["cor"]:
        kb = json.dumps(mis.src_beta, sort_keys=True)
        lt0 = dict(mis.log_tau)
        nv0 = mis._replay_values()
        b_p = [mis.value.predict_person(p) for p in parts]
        b_s = mis.value.predict_situation(ctx, con)
        r_b = [nv0.predict_person(p) for p in parts]
        r_bs = nv0.predict_situation(ctx, con)
        nver0 = len(led(mis, "verification"))
        a_now = mis.concern.a.copy()
        l0 = float(mis.value.like_entity(H.A, "collaborator", a_now)[0])
        RP.feed_items(mis, [cor])
        l1 = float(mis.value.like_entity(H.A, "collaborator", a_now)[0])
        a_p = [mis.value.predict_person(p) for p in parts]
        a_s = mis.value.predict_situation(ctx, con)
        nv1 = mis._replay_values()
        r_a = [nv1.predict_person(p) for p in parts]
        r_as = nv1.predict_situation(ctx, con)
        ent = led(mis, "correction")[-1]
        m1 = ent["m1"]
        ce = led(mis, "correction_experience")[-1]
        pids = [p["id"] for p in parts]
        ok_pred = True
        det_pred = ""
        for j, pid in enumerate(pids):
            err_j = float(np.abs((a_p[j] - b_p[j]) - m1["g_e"][pid] * need(m1["D"][pid])).max())
            if err_j > TOL:
                ok_pred = False
                det_pred = f"{cor['id']} {pid} err {err_j:.3g}"
        err_s = float(np.abs((a_s - b_s) - m1["g"] * need(m1["D_sit"])).max())
        if err_s > TOL:
            ok_pred = False
            det_pred = f"{cor['id']} sit err {err_s:.3g}"
        inv.add("RA5_pred", ok_pred, det_pred)
        like_chk.append(l1 - l0)
        # H3/H9: κ は検証の出来事でだけ、τ は訂正の出来事でだけ、|Δ log τ| ≤ η
        inv.add("H3", (json.dumps(mis.src_beta, sort_keys=True) == kb) or (len(led(mis, "verification")) > nver0), f"κ が検証なしで変化 {cor['id']}")
        dl = max([abs(mis.log_tau.get(k, 0.0) - lt0.get(k, 0.0)) for k in set(mis.log_tau) | set(lt0)] + [0.0])
        inv.add("H9", dl <= ETA + 1e-15, f"|Δlog τ| = {dl:.3g} {cor['id']}")
        cor_recs.append({
            "id": cor["id"], "g": m1["g"], "g_e": m1["g_e"], "G_e": m1["G_e"], "llr": m1["llr"], "z": m1["z"], "kappa_bar": m1["kappa_bar"],
            "kab": m1["kappa_alpha_beta"], "or": m1["or"], "tau_step": ce["tau_step"], "tau_raw": ce["tau_raw"],
            "own": ce["own"], "guilt": ce["guilt"], "regret": ce["regret"], "defend": ce["defend"],
            "D_max_abs_diff_replay": float(max([np.abs(need(m1["D"][pid]) - (r_a[j] - r_b[j])).max() for j, pid in enumerate(pids)]
                                               + [np.abs(need(m1["D_sit"]) - (r_as - r_bs)).max()]))})
    post_cor = copy.deepcopy(mis)
    R_cor = RP.read_R(mis, oc)
    # ---- COR の後の検査 ----
    corr_all = led(mis, "correction")
    ok_range = True
    for c in corr_all:
        m1 = c["m1"]
        if mode == "v34":
            ok_range &= bool(0.0 < m1["g"] < 1.0 and math.isfinite(m1["llr"]))
            ok_range &= all(0.0 < x < 1.0 for x in m1["g_e"].values())
        else:
            ok_range &= bool(math.isfinite(m1["llr"]) and all(math.isfinite(x) for x in m1["g_e"].values()))
    inv.add("RA5_range", ok_range, f"{mode}: g の範囲/有限")
    ok_f, ok_rem, df = True, True, ""
    for c in corr_all:
        m1 = c["m1"]
        al, be = m1["kappa_alpha_beta"]
        if abs(m1["g"] - _sigma(math.log(al / be) + m1["llr"])) > 1e-12 or abs(m1["kappa_bar"] - al / (al + be)) > 1e-12:
            ok_f, df = False, f"{c['id']} g={m1['g']:.6g} σ={_sigma(math.log(al / be) + m1['llr']):.6g}"
        for pid, ge in m1["g_e"].items():
            want = m1["g"] * (1.0 + LAMBDA_G * m1["G_e"][pid]) if mode == "v34_or" else m1["g"]
            if pid in m1["G_e"] and abs(ge - want) > 1e-12:
                ok_f, df = False, f"{c['id']} g_e[{pid}]={ge:.6g} want {want:.6g}"
        for pid in m1["D"]:
            d, dl, rm = need(m1["D"][pid]), need(m1["delta"][pid]), need(m1["remainder"][pid])
            if np.abs(d - dl - rm).max() > 1e-12 or np.abs(dl - m1["g_e"][pid] * d).max() > 1e-12:
                ok_rem, dr = False, f"{c['id']} {pid}"
        if np.abs(need(m1["D_sit"]) - need(m1["delta_sit"]) - need(m1["remainder_sit"])).max() > 1e-12 or np.abs(need(m1["delta_sit"]) - m1["g"] * need(m1["D_sit"])).max() > 1e-12:
            ok_rem = False
        for pid in m1["D"]:                                              # 台帳に自分で測って書いた予測の変化も g_e D に一致
            if np.abs(need(m1["pred_change"][pid]) - m1["g_e"][pid] * need(m1["D"][pid])).max() > TOL:
                ok_rem = False
    inv.add("RA5_formula", ok_f, df)
    inv.add("RA5_remainder", ok_rem, "delta/remainder/pred_change の台帳の整合")
    same_sigma = (np.array_equal(S0[0], mis.value.person.S) and np.array_equal(S0[1], mis.value.sit.S) and names0 == (list(mis.value.person.names), list(mis.value.sit.names))
                  and np.array_equal(C0, mis.cond.C) and np.array_equal(a0, mis.concern.a))
    inv.add("RA5_sigma", same_sigma, "Σ・C・a が訂正で変化")
    # 台帳の ΔW から like(A) の変化を再構成
    a_use = a0
    z = {f: v for f, _l, v in mis.value.person_feats(H.A, "collaborator")}
    dl_led = 0.0
    for c in led(mis, "correction")[n_cor0:]:
        dW = c["dW"]["person"]
        dl_led += sum(z[f] * float(np.dot(a_use, np.array(dW[f]))) for f in z if f in dW)
    inv.add("RA5_reconstruct", abs(dl_led - sum(like_chk)) <= TOL, f"ΔW 再構成 {dl_led:.6g} vs {sum(like_chk):.6g}")
    exp1 = json.dumps(led(mis, "experience")[: len(json.loads(exp0))], sort_keys=True, default=str)
    orig1 = json.dumps([r["original_outcomes"] for r in mis.exp_log][: len(json.loads(orig0))], sort_keys=True)
    inv.add("RA6", exp1 == exp0 and orig1 == orig0 and len(led(mis, "correction_experience")) - n_ce0 == len(h["cor"]) and len(led(mis, "correction")) - n_cor0 == len(h["cor"]),
            "当時の台帳/original_outcomes の変化、または訂正の記録の数")
    b2 = copy.deepcopy(mis.belief, {id(mis.emb): mis.emb})
    b2._rebuild_counts()
    inv.add("RA7a", all(np.abs(getattr(mis.belief, k) - getattr(b2, k)).max() <= 1e-12 for k in ("N1", "N0", "CP1", "CP0", "pi")), "信念の計数が記録から作り直した値と不一致")
    _check_h1(mis, inv, "COR 後", mode)
    # ---- FOL（1 件ずつ。RA8a/b）----
    moods: List[float] = []
    ledger_sign_disagree = 0
    lt_fol0 = dict(mis.log_tau)
    for it in h["fol"]:
        direct = any(("type" in o and o["actor"] == H.A) for o in it["outcomes"]) and any(p["id"] == H.A for p in it["participants"])
        if direct:
            a_now = mis.concern.a.copy()
            model = mis.value.person
            pre_model = copy.deepcopy(model)                               # A の観測の更新だけを取り出す（O の更新も ind:A を共分散で動かすので、台帳の合計とは別）
            ii, zz = model._prepare(mis.value.person_feats(H.A, "collaborator"))
            Sz = np.einsum("kij,j->ki", model.S[:, :, ii], zz)
            zSz = Sz[:, ii] @ zz
            prec = np.maximum(a_now, A_FLOOR)
            gain = zSz / (zSz + model.R / prec)
        nver0 = len(led(mis, "verification"))
        kb = json.dumps(mis.src_beta, sort_keys=True)
        lt0 = dict(mis.log_tau)
        resp = mis.experience(copy.deepcopy(it))
        inv.add("H3", (json.dumps(mis.src_beta, sort_keys=True) == kb) or (len(led(mis, "verification")) > nver0), f"FOL {it['id']} κ が検証なしで変化")
        inv.add("H9", mis.log_tau == lt0, f"FOL {it['id']} τ が訂正なしで変化")
        moods.append(float(mis.mood))
        if direct:
            e_ = need(resp["value_change"]["prediction_error"]["person"][H.A])
            ent = led(mis, "experience")[-1]
            teacher = np.array(ent["teachers"]["person"][H.A], dtype=np.float64)
            r_ = pre_model.update(mis.value.person_feats(H.A, "collaborator"), teacher, a_now)       # A の観測だけの更新（実装の更新規則そのもの）
            dW_ind = r_["dW"][pre_model.idx[f"ind:{H.A}"]]
            nz = np.abs(e_) > 1e-12
            inv.add("RA8a", bool(np.all(np.sign(dW_ind[nz]) == np.sign(e_[nz]))), f"{it['id']} sign(dW_ind)={np.sign(dW_ind)} sign(err)={np.sign(e_)}")
            dW_led = np.array(ent["dW"]["person"].get(f"ind:{H.A}", np.zeros(N_NEEDS)))
            ledger_sign_disagree += int(np.any(np.sign(dW_led[nz]) != np.sign(e_[nz])))
            sel = (a_now > 0.05) & (np.abs(e_) > 0.01)
            inv.add("RA8b", bool(np.all(gain[sel] >= 1e-3)), f"{it['id']} gain {gain[sel]}")
    R_fol = RP.read_R(mis, oc)
    u1 = u_vector(mis)
    digest_fol, learn_fol = RP.state_digest(mis), learn_digest(mis, True)
    _check_h1(mis, inv, "FOL 後", mode)
    # H2: 台帳の和 = 状態
    ok2 = True
    for model, net in (("person", mis.value.person), ("situation", mis.value.sit)):
        tot = mis.ledger.totals(model)
        for nm in net.names:
            row = tot.get(nm, np.zeros(N_NEEDS))
            ok2 &= bool(np.abs(row - net.W[net.idx[nm]]).max() <= TOL)
    vers = led(mis, "verification")
    for s_, (al, be) in mis.src_beta.items():
        ok2 &= abs(al - 1.0 - sum(e["d_alpha"] for e in vers if e["source"] == s_)) <= 1e-12 and abs(be - 1.0 - sum(e["d_beta"] for e in vers if e["source"] == s_)) <= 1e-12
    lt_sum: Dict[str, float] = {}
    for ce in led(mis, "correction_experience"):
        for k, (b_, a_) in ce["log_tau"].items():
            lt_sum[k] = lt_sum.get(k, 0.0) + (a_ - b_)
    ok2 &= set(lt_sum) == set(mis.log_tau) and all(abs(lt_sum[k] - mis.log_tau[k]) <= 1e-12 for k in lt_sum)
    inv.add("H2", ok2, "台帳の和 ≠ 状態")
    inv.add("H3", mis.n_verifications == len(vers), "n_verifications ≠ 台帳の検証の数")
    # RA8d: 反対向きの検証で κ̄ が動く（写しの上で）
    k_before = mis.kappa_bar(H.SOURCE)
    n2, n3 = copy.deepcopy(mis), copy.deepcopy(mis)
    e_dn = n2.verify_source(H.SOURCE, -5.0)
    e_up = n3.verify_source(H.SOURCE, +5.0)
    d_dn, d_up = e_dn["kappa"][1] - e_dn["kappa"][0], e_up["kappa"][1] - e_up["kappa"][0]
    inv.add("RA8d", (d_dn <= 0.0 and d_up >= 0.0 and (d_dn <= -0.01 if k_before >= 0.5 else d_up >= 0.01)), f"κ̄ の動き 下 {d_dn:.4g} 上 {d_up:.4g}（κ̄ = {k_before:.3g}）")
    ok12, d12 = ra12_ok(u0, u1)
    inv.add("RA12", ok12, d12)
    # H8: 気分の摂動（COR の直後の写しで）
    pm = copy.deepcopy(post_cor)
    pm.mood += 1e-3
    dprev = 1e-3
    ok8 = True
    for i, it in enumerate(h["fol"]):
        pm.experience(copy.deepcopy(it))
        dd = abs(pm.mood - moods[i])
        ok8 &= (dd / dprev <= 0.8 + 0.2 * CFG.beta_mood + 1e-9 and dd / dprev <= 1.0 + 1e-12)
        dprev = dd if dd > 0 else dprev
    ok8 &= learn_digest(pm, True) == learn_fol
    inv.add("H8", ok8, "気分の摂動の比、または学習状態の不一致")
    # H4: 保存・復元（写しの上で。続きの経験の応答と状態が一致）
    x = copy.deepcopy(mis)
    with tempfile.TemporaryDirectory() as d:
        x.save(d)
        m = NPN.load(d)
    ok4 = same_state(x, m) and m.src_beta == x.src_beta and m.log_tau == x.log_tau
    ra = x.experience(copy.deepcopy(H._insert("zz9")))
    rb = m.experience(copy.deepcopy(H._insert("zz9")))
    ok4 &= json.dumps(ra, sort_keys=True, default=str) == json.dumps(rb, sort_keys=True, default=str) and same_state(x, m)
    inv.add("H4", ok4, "保存・復元の不一致")
    if det:
        again = RP._prefix(seed, h, ov, "mis")
        RP.feed_items(again, h["cor"])
        RP.feed_items(again, h["fol"])
        inv.add("H4", RP.state_digest(again) == digest_fol, "同じ入力で状態が一致しない（決定性）")
    taus = {k: float(v) for k, v in mis.log_tau.items()}
    ev = {
        "meta": h["meta"], "mode": mode, "R_pre": fl(R_pre), "R_ctrl_pre": fl(R_ctrl_pre), "R_cor": fl(R_cor), "R_fol": fl(R_fol),
        "cor_recs": cor_recs, "log_tau": taus, "tau_boundary": {k: bool(abs(abs(v) - LIM) < 1e-9) for k, v in taus.items()},
        "kappa": {k: float(v[0] / (v[0] + v[1])) for k, v in mis.src_beta.items()}, "alpha_beta_sum": {k: float(v[0] + v[1]) for k, v in mis.src_beta.items()},
        "n_verifications": int(mis.n_verifications), "ra8a_ledger_total_sign_disagreements": ledger_sign_disagree, "mood_end": float(mis.mood), "u_pre": u0, "u_post": u1, "like_pre": float(like_pre),
        "kappa_move": [float(d_dn), float(d_up)], "state_digest": digest_fol,
        "learn_digest_cor": learn_digest(post_cor, False), "learn_digest_fol": learn_fol,
    }
    return {"inv": inv, "ev": ev, "pre": pre, "post_cor": post_cor, "final": mis, "oc": oc, "h": h}


def cell_task(mutant: Optional[str], seed: int, bg: str, src: str, err: str, n: int, phase: str, cross: bool, det: bool) -> Dict[str, Any]:
    """1 つのセルの仕事（ワーカーで実行）。変異体を適用して run_checked を C-v34 で行い、cross なら clamp-τ・clamp-mood・訂正なし・C-none-v34・C-OR も行う。"""
    with apply_mutant(mutant):
        main = run_checked(seed, bg, src, err, n, phase, "v34", det)
        inv: Inv = main["inv"]
        ev = main["ev"]
        h, oc, pre = main["h"], main["oc"], main["pre"]
        out_ev: Dict[str, Any] = {"v34": ev}
        if cross:
            nc = copy.deepcopy(pre)
            RP.advance_clock(nc, len(h["cor"]))
            R_cor_nc = RP.read_R(nc, oc)
            RP.feed_items(nc, h["fol"])
            out_ev["nocorr"] = {"R_cor": fl(R_cor_nc), "R_fol": fl(RP.read_R(nc, oc))}
            ct = copy.deepcopy(pre)
            ct.cfg.m3_variant = "clamp_tau"
            RP.feed_items(ct, h["cor"])
            r_cor_ct = RP.read_R(ct, oc)
            dg_ct = learn_digest(ct, False)
            RP.feed_items(ct, h["fol"])
            out_ev["clamp_tau"] = {"R_cor": fl(r_cor_ct), "R_fol": fl(RP.read_R(ct, oc)), "log_tau": dict(ct.log_tau)}
            inv.add("RA3_3", dg_ct == ev["learn_digest_cor"], "clamp-τ が COR 直後に τ 以外の学習状態を変えた")
            cm = copy.deepcopy(pre)
            cm.cfg.m3_variant = "clamp_mood"
            RP.feed_items(cm, h["cor"])
            RP.feed_items(cm, h["fol"])
            r_fol_cm = RP.read_R(cm, oc)
            inv.add("RA3_4", learn_digest(cm, True) == ev["learn_digest_fol"] and np.array_equal(r_fol_cm, np.array(ev["R_fol"])), "clamp-mood が FOL の後の学習状態・R を変えた")
            out_ev["clamp_mood"] = {"R_fol": fl(r_fol_cm)}
            if err != "interp":
                inv.add("RA3_5", ev["log_tau"] == {} and np.array_equal(np.array(out_ev["clamp_tau"]["R_fol"]), np.array(ev["R_fol"]))
                        and np.array_equal(np.array(out_ev["clamp_tau"]["R_cor"]), np.array(ev["R_cor"])), "報告の誤りで τ が動いた/clamp-τ との差が 0 でない")
            cn = RP._prefix(seed, h, V34 + (("no_reappraisal", True),), "mis")
            RP.feed_items(cn, h["cor"])
            r_cn = RP.read_R(cn, oc)
            inv.add("H7", bool(r_cn[3] == ev["R_cor"][3] and r_cn[4] == ev["R_cor"][4]), "COR 直後の r4・r5 が C-v34 と C-none で不一致")
            out_ev["none_v34"] = {"R_cor": fl(r_cn)}
            if err == "interp":            # ERR-report / report-x は own = 0 で G_e ≡ 0 なので、C-OR は C-v34 とビット単位で同じ（Stage 2 の単体テスト）。計算を省く
                orr = run_checked(seed, bg, src, err, n, phase, "v34_or", False)
                for k_, v_ in orr["inv"].dump().items():                   # C-OR の検査は同じ ID に畳み込む（C-OR の RA5 の条項など）
                    inv.add(k_, v_["ok"], "C-OR: " + (v_["fails"][0] if v_["fails"] else ""))
                out_ev["or"] = {"R_fol": orr["ev"]["R_fol"], "cor_recs": orr["ev"]["cor_recs"]}
    return {"inv": inv.dump(), "ev": out_ev, "meta": {"mutant": mutant, "seed": seed, "bg": bg, "src": src, "err": err, "n": n, "cross": cross}}


# ======================================================================
# 種ごとの仕事: H5、RA8c（規則と境界）、stress_guilt（MU6 の 100 回）
# ======================================================================
def stress_history(seed: int, n: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """罪悪感の訂正を n 回続ける履歴: ERR-interp の誤帰属 n 経験（大きさと q は種から）と、その訂正 n 件。"""
    rng = np.random.default_rng([int(seed), 3403])
    mags = np.round(rng.uniform(0.6, 0.9, size=n), 2)
    qs = np.round(rng.uniform(0.4, 0.6, size=n), 2)
    mis, cor = [], []
    for i in range(n):
        mis.append(H._mis_exp(f"j{i}", "interp", "wrong", mags[i], qs[i]))
        truth = H._mis_exp(f"j{i}", "interp", "truth", mags[i], qs[i])["outcomes"]
        cor.append({"id": f"c{i}", "kind": "correction", "refers_to": f"j{i}", "source": H.SOURCE, "corrected_outcomes": truth})
    return mis, cor


def seed_task(mutant: Optional[str], seed: int, phase: str, stress_n: int = 0) -> Dict[str, Any]:
    RP.guard_seeds([seed], phase)
    inv = Inv()
    ev: Dict[str, Any] = {}
    with apply_mutant(mutant):
        # --- H5: 訂正も空の刻みもない背景 + 誤帰属の列で、v34 の通常の経験の経路が v33 とビット単位で一致 ---
        h = H.make_history(seed, "betray", "none", "report", 4, 0)
        seq = h["pre"][:H.N_BG] + h["mis"] + h["extra"]
        a, b = NPN(seed=seed, reappraisal_mode="v33"), NPN(seed=seed, reappraisal_mode="v34")
        RP.feed_items(a, seq)
        RP.feed_items(b, seq)
        ma, aa = a.get_state()
        mb, ab = b.get_state()
        ca, cb = dict(ma["config"]), dict(mb["config"])
        ca.pop("reappraisal_mode"), cb.pop("reappraisal_mode")
        ok5 = ca == cb and set(aa) == set(ab) and all(np.array_equal(aa[k], ab[k]) for k in aa)
        ok5 &= json.dumps(ma["exp_log"], sort_keys=True, default=str) == json.dumps(mb["exp_log"], sort_keys=True, default=str)
        inv.add("H5", ok5, "v34 の通常の経験の経路が v33 とビット単位で不一致")
        # --- RA8c: 規則の両方向、境界、境界から反対向きの押しで動く ---
        f = NPN._tau_step
        up, dn = f(1.0, 0.0, 0.0, ETA), f(0.0, 0.5, 0.5, ETA)
        ok_rule = up > 0.0 and dn < 0.0 and abs(f(0.0, 0.99, 0.99, ETA)) <= ETA + 1e-15 and abs(f(0.99, 0.0, 0.0, ETA)) <= ETA + 1e-15
        inv.add("RA8c", bool(ok_rule), f"規則の両方向: 上 {up} 下 {dn}")
        hh = H.make_history(seed, "betray", "none", "interp", 4, 0)
        for sign in (+1, -1):                                              # 強く押し続けても境界を出ない
            n_ = NPN(seed=seed, reappraisal_mode="v34", eta_tau=50.0)
            RP.feed_items(n_, hh["pre"] + hh["mis"])
            n_._tau_step = lambda d, g, r, e, s=sign: s * 50.0
            RP.feed_items(n_, hh["cor"])
            v_ = n_.log_tau.get(KEY_TAU, 0.0)
            inv.add("RA8c", abs(v_) <= LIM + 1e-12, f"境界: log τ = {v_:.4g}（|·| ≤ {LIM:.4g}）")
            inv.add("H1", abs(v_) <= LIM + 1e-12, "τ が範囲外")
        base = NPN(seed=seed, reappraisal_mode="v34")
        RP.feed_items(base, hh["pre"] + hh["mis"])
        for pos in (+1, -1):                                               # 境界にある τ を反対向きに押すと動く。同じ向きなら境界に留まる
            for push, moves in ((-pos * 0.5, True), (pos * 0.5, False)):
                n_ = copy.deepcopy(base)
                n_.log_tau[KEY_TAU] = pos * LIM
                n_._tau_step = lambda d, g, r, e, p=push: p
                RP.feed_items(n_, hh["cor"][:1])
                inv.add("RA8c", (n_.log_tau[KEY_TAU] != pos * LIM) == moves, f"境界 {pos:+d} で push {push:+.1f}: 動いた = {n_.log_tau[KEY_TAU] != pos * LIM}")
        # --- stress_guilt: 罪悪感の訂正を stress_n 回（MU6）。押しを罪悪感の飽和（−η_τ）に固定して、H1・H9・RA8c の境界を検査する ---
        if stress_n > 0:
            smis, scor = stress_history(seed, stress_n)
            n_ = NPN(seed=seed, reappraisal_mode="v34")
            RP.feed_items(n_, h["pre"][:H.N_BG] + smis)
            n_._tau_step = lambda d, g, r, e: -e
            lt_prev = 0.0
            ok9 = True
            for c in scor:
                RP.feed_items(n_, [c])
                v_ = n_.log_tau.get(KEY_TAU, 0.0)
                ok9 &= abs(v_ - lt_prev) <= ETA + 1e-15
                lt_prev = v_
            ev["stress"] = {"n": stress_n, "log_tau_end": float(n_.log_tau.get(KEY_TAU, 0.0)), "tau_end": float(math.exp(n_.log_tau.get(KEY_TAU, 0.0)))}
            inv.add("H9", ok9, "stress: 1 回の |Δ log τ| > η")
            inv.add("H1", 1.0 / 3.0 - 1e-12 <= math.exp(n_.log_tau.get(KEY_TAU, 0.0)) <= 3.0 + 1e-12, f"stress: τ = {ev['stress']['tau_end']:.4g} が [1/3, 3] の外")
            inv.add("RA8c", abs(n_.log_tau.get(KEY_TAU, 0.0)) <= LIM + 1e-12, f"stress: log τ = {ev['stress']['log_tau_end']:.4g}")
    return {"inv": inv.dump(), "ev": ev, "meta": {"mutant": mutant, "seed": seed, "stress_n": stress_n}}


# ======================================================================
# RA2: val-swap・hm-eq（SRC-none、ERR ごと、背景の 6 通りの向き）
# ======================================================================
def swap_task(mutant: Optional[str], seed: int, err: str, phase: str, n: int = 8) -> Dict[str, Any]:
    RP.guard_seeds([seed], phase)
    with apply_mutant(mutant):
        hs = {bg: H.make_history(seed, bg, "none", err, n, 0) for bg in H.BGS}
        oc = hs["trust"]["meta"]["o_class"]
        pre = {bg: RP._prefix(seed, hs[bg], V34, "mis") for bg in H.BGS}

        def ce_of(person: NPN, hcor: Sequence[Dict[str, Any]], hfol: Sequence[Dict[str, Any]]) -> List[float]:
            a, b = copy.deepcopy(person), copy.deepcopy(person)
            RP.feed_items(a, hcor)
            RP.feed_items(a, hfol)
            RP.advance_clock(b, len(hcor))
            RP.feed_items(b, hfol)
            return fl(RP.read_R(a, oc) - RP.read_R(b, oc))

        out: Dict[str, Any] = {"CE": {bg: ce_of(pre[bg], hs[bg]["cor"], hs[bg]["fol"]) for bg in H.BGS}, "val": {}, "hm": {}, "excluded": []}
        for bt in H.BGS:
            for bs in H.BGS:
                if bt == bs:
                    continue
                key = f"{bt}<-{bs}"
                if not RP.aligned(pre[bt], pre[bs]):
                    out["excluded"].append(key)
                    continue
                out["val"][key] = ce_of(RP.swap_groups(pre[bt], pre[bs], RP.VAL_SWAP), hs[bt]["cor"], hs[bt]["fol"])
                out["hm"][key] = ce_of(RP.swap_groups(pre[bt], pre[bs], RP.HM_EQ), hs[bt]["cor"], hs[bt]["fol"])
    return {"ev": out, "meta": {"mutant": mutant, "seed": seed, "err": err}}


# ======================================================================
# 判定の反転（RA0〜RA4）。変異体の「判定の列」。基準（変異なし）と比べて反転したら検出
# ======================================================================
def criteria_verdicts(cells: List[Dict[str, Any]], swaps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """cells: cell_task の結果の列（cross の有無が混ざる）、swaps: swap_task の結果の列。判定は変異テスト用の簡易版（SPEC 第 7.3・8 節の定義に従う）。
    返す verdict の値は True（合格）/ False（不合格）/ None（判定できない）。"""
    by: Dict[Tuple[str, str, str], Dict[int, Dict[str, Any]]] = {}
    for c in cells:
        m = c["meta"]
        by.setdefault((m["err"], m["bg"], m["src"]), {})[m["seed"]] = c["ev"]
    seeds = sorted({m for d in by.values() for m in d})
    R_pre_all = np.array([ev["v34"]["R_pre"] for d in by.values() for ev in d.values()])
    scale, mask = RP.scale_of(R_pre_all)
    out: Dict[str, Any] = {}
    detail: Dict[str, Any] = {}

    def vec(ev_list: List[Dict[str, Any]], f) -> np.ndarray:
        return np.array([RP.normalize(np.array(f(e)), scale, mask) for e in ev_list])

    # RA0
    ok0 = True
    for key, d in by.items():
        k = 0
        for ev in d.values():
            v = ev["v34"]
            dpre = abs(v["R_pre"][0] - v["R_ctrl_pre"][0])
            imm = True
            if "nocorr" in ev:
                imm = bool(np.any(np.array(v["R_cor"]) != np.array(ev["nocorr"]["R_cor"])))
            k += int(dpre > 1e-6 and imm)
        ok0 &= k >= 0.95 * len(d)
    out["RA0"] = bool(ok0)
    # RA1（SRC-none、ERR ごと）
    ra1: Dict[str, bool] = {}
    for err in H.ERRS:
        X = {}
        for bg in H.BGS:
            d = by.get((err, bg, "none"), {})
            if not d or not all("nocorr" in e for e in d.values()):
                X = None
                break
            sd = sorted(d)
            X[bg] = vec([d[s] for s in sd], lambda e: np.array(e["v34"]["R_fol"]) - np.array(e["nocorr"]["R_fol"]))
        if X is None:
            continue
        nd = sum(int(RP.directions_differ(X[a], X[b], RP.LEVEL_MAIN)["differ"]) for a, b in (("trust", "betray"), ("trust", "neutral"), ("betray", "neutral")))
        ra1[err] = nd >= 2
    out["RA1"] = bool(all(ra1.values())) if len(ra1) == 3 else None
    detail["RA1_by_err"] = ra1
    # RA2（val-swap 6 通り、hm-eq）
    ra2: Dict[str, bool] = {}
    ra2hm: Dict[str, bool] = {}
    for err in H.ERRS:
        sw = sorted([s for s in swaps if s["meta"]["err"] == err], key=lambda s: s["meta"]["seed"])
        if not sw:
            continue
        ndiff = 0
        npair = 0
        for bt in H.BGS:
            for bs in H.BGS:
                if bt == bs:
                    continue
                key = f"{bt}<-{bs}"
                if not all(key in s["ev"]["val"] for s in sw):
                    continue
                own = np.array([RP.normalize(np.array(s["ev"]["CE"][bt]), scale, mask) for s in sw])
                swp = np.array([RP.normalize(np.array(s["ev"]["val"][key]), scale, mask) for s in sw])
                ndiff += int(RP.directions_differ(own, swp, RP.LEVEL_MAIN)["differ"])
                npair += 1
        ra2[err] = (ndiff >= 4) if npair == 6 else None
        okp = 0
        for bt, bs in (("trust", "betray"), ("trust", "neutral"), ("betray", "neutral")):
            key = f"{bt}<-{bs}"
            if not all(key in s["ev"]["hm"] for s in sw):
                continue
            rho = []
            for s in sw:
                ceb = RP.normalize(np.array(s["ev"]["CE"][bt]), scale, mask)
                cebp = RP.normalize(np.array(s["ev"]["CE"][bs]), scale, mask)
                gap = np.linalg.norm(ceb - cebp)
                if gap > 1e-12:
                    rho.append(np.linalg.norm(RP.normalize(np.array(s["ev"]["hm"][key]), scale, mask) - cebp) / gap)
            okp += int(bool(rho) and float(np.median(rho)) >= 0.5)
        ra2hm[err] = okp >= 2
    out["RA2"] = bool(all(v for v in ra2.values())) if len(ra2) == 3 and all(v is not None for v in ra2.values()) else None
    out["RA2_hm"] = bool(all(ra2hm.values())) if len(ra2hm) == 3 else None
    detail["RA2_by_err"] = ra2
    detail["RA2hm_by_err"] = ra2hm
    # RA3(1)(2)（ERR-interp、SRC 3 × BG 3。Δτ_b = L_fol(C-v34) − L_fol(clamp-τ)。先に種ごとに SRC を平均）
    dtau: Dict[str, Dict[str, np.ndarray]] = {}
    ok = True
    for bg in H.BGS:
        for src in H.SRCS:
            d = by.get(("interp", bg, src), {})
            if not d or not all("clamp_tau" in e for e in d.values()):
                ok = False
                break
            sd = sorted(d)
            dtau.setdefault(src, {})[bg] = vec([d[s] for s in sd], lambda e: (np.array(e["v34"]["R_fol"]) - np.array(e["v34"]["R_cor"]))
                                               - (np.array(e["clamp_tau"]["R_fol"]) - np.array(e["clamp_tau"]["R_cor"])))
    if ok and len(dtau) == 3:
        r31 = True
        for bg in H.BGS:
            pooled = np.mean([dtau[s][bg] for s in H.SRCS], axis=0)
            m_, lo, hi = RP.mean_ci(pooled, RP.LEVEL_MAIN)
            r31 &= bool(np.any((lo > 0) | (hi < 0)))
        out["RA3_1"] = bool(r31)
        n_src = 0
        for s_ in H.SRCS:
            nd = sum(int(RP.directions_differ(dtau[s_][a], dtau[s_][b], RP.LEVEL_MAIN)["differ"]) for a, b in (("trust", "betray"), ("trust", "neutral"), ("betray", "neutral")))
            n_src += int(nd >= 2)
        out["RA3_2"] = n_src >= 2
    else:
        out["RA3_1"] = out["RA3_2"] = None
    # RA4（27 セルの r）
    r_cell: Dict[Tuple[str, str, str], np.ndarray] = {}
    for key, d in by.items():
        rr = []
        for s in sorted(d):
            v = d[s]["v34"]
            dpre, dpost, r = RP.remaining_fraction(v["R_pre"][0], v["R_ctrl_pre"][0], v["R_cor"][0])
            rr.append(r)
        r_cell[key] = np.array(rr)
    if len(r_cell) == 27:
        means = np.array([np.nanmean(v) for v in r_cell.values()])
        out["RA4a"] = bool(means.max() - means.min() >= 0.15)
        diffs = []
        for s_i, s in enumerate(seeds):
            rel = np.nanmean([r_cell[(e, b, "reliable")][s_i] for e in H.ERRS for b in H.BGS])
            unr = np.nanmean([r_cell[(e, b, "unreliable")][s_i] for e in H.ERRS for b in H.BGS])
            diffs.append(rel - unr)
        m_, lo, hi = RP.mean_ci(np.array(diffs).reshape(-1, 1), RP.LEVEL_OTHER)
        out["RA4b"] = bool(lo[0] > 0 or hi[0] < 0)
        detail["RA4b_mean_diff"] = float(m_[0])
        out["RA4c"] = bool(sum(int(np.nanstd(v, ddof=1) > 0.0134) for v in r_cell.values()) >= 18)
        detail["RA4_range"] = float(means.max() - means.min())
        detail["RA4c_cells_over"] = int(sum(int(np.nanstd(v, ddof=1) > 0.0134) for v in r_cell.values()))
    else:
        out["RA4a"] = out["RA4b"] = out["RA4c"] = None
    out["_detail"] = detail
    return out


# ======================================================================
# 候補の検査（初期の集合の外。結果を見た後に足すかを判断する）
# ======================================================================
def candidate_results(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """C_tau_rule: 台帳の τ_step が η_τ · clip(defend − guilt − regret, −1, 1) に一致（τ の規則の適合）。
    C_D_recompute: 台帳の D_i が、訂正の前後の価値の再生の差（独立に再計算）に 1e-9 以内で一致（D_i の定義の適合）。"""
    res = {k: {"ok": True, "fails": []} for k in CANDIDATE_CHECKS}
    for c in cells:
        for mode in ("v34", "or"):
            ev = c["ev"].get(mode)
            if ev is None:
                continue
            recs = ev["cor_recs"] if mode in ("or",) else ev["cor_recs"]
            for r in recs:
                want = ETA * max(-1.0, min(1.0, r["defend"] - r["guilt"] - r["regret"]))
                if abs(r["tau_step"] - want) > 1e-12 or abs(r["tau_raw"] - (r["defend"] - r["guilt"] - r["regret"])) > 1e-12:
                    res["C_tau_rule"]["ok"] = False
                    if len(res["C_tau_rule"]["fails"]) < 3:
                        res["C_tau_rule"]["fails"].append(f"{c['meta']['seed']} {r['id']} step {r['tau_step']} want {want}")
                if "D_max_abs_diff_replay" in r and r["D_max_abs_diff_replay"] > TOL:
                    res["C_D_recompute"]["ok"] = False
                    if len(res["C_D_recompute"]["fails"]) < 3:
                        res["C_D_recompute"]["fails"].append(f"{c['meta']['seed']} {r['id']} diff {r['D_max_abs_diff_replay']:.3g}")
    return res


def merge_invariants(results: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """cell_task / seed_task の inv を検査 ID ごとにまとめる。1 つでも破れたら破れ。"""
    out = {k: {"ok": True, "n_fail": 0, "n_run": 0, "first_fails": []} for k in INVARIANT_CHECKS}
    for r in results:
        for k, v in r["inv"].items():
            e = out.setdefault(k, {"ok": True, "n_fail": 0, "n_run": 0, "first_fails": []})
            e["n_run"] += 1
            if not v["ok"]:
                e["ok"] = False
                e["n_fail"] += 1
                if len(e["first_fails"]) < 3:
                    e["first_fails"].append({"meta": r["meta"], "fails": v["fails"][:1]})
    return out
