"""v3.4 の経験史の生成器（SPEC-v3.4 第 5.2 節）。既存の data/make_histories.py と data/histories/*.jsonl は変えない。

使い方（ライブラリ）:
    from make_histories_v34 import make_history
    h = make_history(seed, bg="trust", src="none", err="interp", n=8, m_extra=0)
    h["pre"]    # BG + SRC（経験・訂正・空の刻み）
    h["mis"]    # MIS: 誤帰属の履歴（N 経験 + 各経験の後の B との挟み込み）
    h["ctrl"]   # MIS を最初から真実の結果にした同じ履歴（対照）
    h["extra"]  # 訂正されない誤帰属 M 本（誤帰属の人格にも対照にも同じものを与える）
    h["cor"]    # 訂正 N 件（j_0..j_{N-1} を順に訂正。source = self_discovery）
    h["fol"]    # 後続 10 経験（全人格で同一）
    h["meta"]   # 識別子と類（O の類など）
  CLI（開発用の確認）:  python data/make_histories_v34.py --seed 3490 [--bg trust --src none --err interp --n 8 --m 0] [--out DIR]

項目は 3 種類の dict: 経験（kind なし）、訂正（kind = "correction"）、空の刻み（kind = "tick"。経験でも訂正でもない。時計だけを 1 進める出来事の印で、
どう扱うかは条件が決める。NPN にそのまま渡してはならない）。

乱数の流れ（SPEC 第 5.1 節）: default_rng([seed, 3401]) 背景、[seed, 3402] 情報源の履歴、[seed, 3403] 誤帰属の大きさと q_LLM、[seed, 3404] 挟み込みの数、
[seed, 3405] 後続の経験。どの流れも、BG・SRC・ERR・N・M の選択に依らず同じ順序で同じ個数を引く。したがって同じ種の中では、条件を比べるときに
経験史の乱数が揃う（N = 2 の履歴は N = 16 の履歴の先頭と一致する）。

SPEC に細部がなく、ここで決めたこと（CHANGELOG-v3.4.md に日付つきで記録）:
  - BG の各項目は、相手 Pk（k = 1..12）ごとに 1 つの経験で、{...}、{...} は同じ経験の 2 つの結果と読む（12 経験 = 12 人）。順序は種で並べ替え、
    m1・m2 は経験ごとに引く（並べ替えた順に割り当てる）。BG の 3 種は同じ並べ替えと同じ m1・m2 を使う。
  - SRC の (3)(4) の直接の経験 2 回は、それぞれに同じ型の結果を持ち、m′ は経験ごとに引く。
  - SRC-none は 4 エピソードそれぞれで、独りの練習 + 空の刻み + 独りの練習 2 回（経験 12 + 刻み 4）。
  - 訂正されない誤帰属 M 本は、対照にも同じ（誤った）ものを与える。対照との差は、訂正した N 本だけから生じる。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SALT_BG, SALT_SRC, SALT_MIS, SALT_INSERT, SALT_FOL = 3401, 3402, 3403, 3404, 3405
N_BG = 12
N_SRC_EP = 4
MAX_SLOTS = 20            # 引く誤帰属の枠の数（N + M ≤ 20。N の掃引の最大は 16 + 4）
N_FOL = 10
BGS = ("trust", "betray", "neutral")
SRCS = ("reliable", "unreliable", "none")
ERRS = ("interp", "report", "report-x")
SOURCE = "self_discovery"

A, O, B, A2 = "A", "O", "B", "A2"
CTX_BG, CON_BG = "studio", "joint_work"
CTX_SRC, CON_SRC = "workshop", "review"
CTX_MIS, CON_MIS = "project_meeting", "authorship"
CTX_INS, CON_INS = "music_session", "music"
CTX_SOLO, CON_SOLO = "solo_practice", "solo_work"
FOL_ORDER = "abacabacba"      # (a)(b)(a)(c)(a)(b)(a)(c)(b)(a)
assert FOL_ORDER == "abacabacba" and FOL_ORDER.count("a") == 5 and FOL_ORDER.count("b") == 3 and FOL_ORDER.count("c") == 2


def _o(t: str, actor: str, m: float) -> Dict[str, Any]:
    return {"type": t, "actor": actor, "magnitude": round(float(m), 2)}


def _amb(actor: str, q: float, m: float) -> Dict[str, Any]:
    """曖昧な結果: credit_taken が q、credit_shared が 1 − q。"""
    q = round(float(q), 2)
    return {"interpretations": {"credit_taken": q, "credit_shared": round(1.0 - q, 2)}, "actor": actor, "magnitude": round(float(m), 2)}


def _p(pid: str, cls: str) -> Dict[str, str]:
    return {"id": pid, "class": cls}


def _exp(eid: str, ctx: str, concept: str, parts: List[Dict[str, str]], outs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"id": eid, "context": ctx, "concepts": [concept], "participants": parts, "outcomes": outs, "session": "s-" + eid}


def _rng(seed: int, salt: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), salt])


def o_class(err: str) -> str:
    return "mentor" if err == "report-x" else "collaborator"


# ----------------------------------------------------------------------
def make_bg(seed: int, bg: str) -> List[Dict[str, Any]]:
    """BG 背景（12 経験）。順序は種で並べ替え。m1・m2 ~ U(0.5, 0.9)（0.01 で丸め）。"""
    assert bg in BGS
    rng = _rng(seed, SALT_BG)
    perm = rng.permutation(N_BG)
    m1 = np.round(rng.uniform(0.5, 0.9, size=N_BG), 2)
    m2 = np.round(rng.uniform(0.5, 0.9, size=N_BG), 2)
    out = []
    for i in range(N_BG):
        pk = f"P{int(perm[i]) + 1:02d}"
        part = [_p(pk, "collaborator")]
        if bg == "trust":
            outs = [_o("credit_shared", pk, m1[i]), _o("support", pk, m2[i])]
        elif bg == "betray":
            outs = [_o("credit_taken", pk, m1[i]), _o("achievement", "self", 0.3)]
        else:
            outs = [_o("connection", pk, 0.2 * m1[i])]
        out.append(_exp(f"bg{i:02d}", CTX_BG, CON_BG, part, outs))
    return out


def make_src(seed: int, src: str) -> List[Dict[str, Any]]:
    """SRC 情報源の履歴（4 エピソード）。reliable / unreliable は報告 → 訂正 → 直接の経験 2 回。none は経験 12 + 空の刻み 4。"""
    assert src in SRCS
    rng = _rng(seed, SALT_SRC)
    m = np.round(rng.uniform(0.6, 0.9, size=N_SRC_EP), 2)
    m2 = np.round(rng.uniform(0.6, 0.9, size=(N_SRC_EP, 2)), 2)
    solo = np.round(rng.uniform(0.2, 0.5, size=(N_SRC_EP, 3)), 2)
    out: List[Dict[str, Any]] = []
    for k in range(N_SRC_EP):
        q = f"Q{k + 1}"
        part = [_p(q, "peer")]
        if src == "none":
            out.append(_exp(f"src{k}a", CTX_SOLO, CON_SOLO, [], [_o("achievement", "self", solo[k, 0])]))
            out.append({"id": f"src{k}t", "kind": "tick"})
            out.append(_exp(f"src{k}c", CTX_SOLO, CON_SOLO, [], [_o("achievement", "self", solo[k, 1])]))
            out.append(_exp(f"src{k}d", CTX_SOLO, CON_SOLO, [], [_o("achievement", "self", solo[k, 2])]))
            continue
        out.append(_exp(f"src{k}r", CTX_SRC, CON_SRC, part, [_o("credit_taken", q, m[k])]))
        out.append({"id": f"src{k}x", "kind": "correction", "refers_to": f"src{k}r", "source": SOURCE,
                    "corrected_outcomes": [_o("credit_shared", q, m[k])]})
        for j in range(2):
            if src == "reliable":
                outs = [_o("credit_shared", q, m2[k, j]), _o("support", q, 0.4)]
            else:
                outs = [_o("credit_taken", q, m2[k, j]), _o("achievement", "self", 0.3)]
            out.append(_exp(f"src{k}{'cd'[j]}", CTX_SRC, CON_SRC, part, outs))
    return out


def _mis_exp(eid: str, err: str, version: str, m: float, q: float) -> Dict[str, Any]:
    """誤帰属の経験 1 つ。version = "wrong"（誤り）| "truth"（真実）。"""
    parts = [_p(A, "collaborator"), _p(O, o_class(err))]
    ach = _o("achievement", "self", 0.4)
    if err == "interp":
        first = _amb(A, q, m) if version == "wrong" else _o("credit_shared", A, m)
    else:
        first = _o("credit_taken", A if version == "wrong" else O, m)
    return _exp(eid, CTX_MIS, CON_MIS, parts, [first, ach])


def _insert(eid: str) -> Dict[str, Any]:
    return _exp(eid, CTX_INS, CON_INS, [_p(B, "collaborator")], [_o("connection", B, 0.5)])


def make_mis(seed: int, err: str, n: int, m_extra: int) -> Dict[str, List[Dict[str, Any]]]:
    """MIS（N 経験。各経験の後に B との挟み込みを 1〜2 回）、対照、訂正されない誤帰属 M 本、訂正 N 件。"""
    assert err in ERRS and 1 <= n <= 16 and 0 <= m_extra <= 4 and n + m_extra <= MAX_SLOTS
    rng = _rng(seed, SALT_MIS)
    mags = np.round(rng.uniform(0.6, 0.9, size=MAX_SLOTS), 2)
    qs = np.round(rng.uniform(0.4, 0.6, size=MAX_SLOTS), 2)
    ins = _rng(seed, SALT_INSERT).integers(1, 3, size=MAX_SLOTS)
    mis: List[Dict[str, Any]] = []
    ctrl: List[Dict[str, Any]] = []
    cor: List[Dict[str, Any]] = []
    for i in range(n):
        mis.append(_mis_exp(f"j{i}", err, "wrong", mags[i], qs[i]))
        ctrl.append(_mis_exp(f"j{i}", err, "truth", mags[i], qs[i]))
        for r in range(int(ins[i])):
            e = _insert(f"f{i}_{r}")
            mis.append(e)
            ctrl.append(dict(e))
        truth = _mis_exp(f"j{i}", err, "truth", mags[i], qs[i])["outcomes"]
        cor.append({"id": f"c{i}", "kind": "correction", "refers_to": f"j{i}", "source": SOURCE, "corrected_outcomes": truth})
    extra: List[Dict[str, Any]] = []
    for t in range(m_extra):
        s = 16 + t                                   # 固定の枠（N に依らない）
        extra.append(_mis_exp(f"x{t}", err, "wrong", mags[s], qs[s]))
        for r in range(int(ins[s])):
            extra.append(_insert(f"fx{t}_{r}"))
    return {"mis": mis, "ctrl": ctrl, "extra": extra, "cor": cor}


def make_fol(seed: int, err: str) -> List[Dict[str, Any]]:
    """FOL 後続（10 経験、全人格で同一）。(a) A と O の協働、(b) A2 の曖昧な出来事（project_meeting）、(c) A2 の曖昧な出来事（music_session）。"""
    rng = _rng(seed, SALT_FOL)
    u = rng.uniform(size=N_FOL)
    out = []
    for i, ch in enumerate(FOL_ORDER):
        if ch == "a":
            m = 0.4 + 0.3 * u[i]
            out.append(_exp(f"fol{i}", CTX_MIS, CON_MIS, [_p(A, "collaborator"), _p(O, o_class(err))],
                            [_o("credit_shared", A, m), _o("connection", A, 0.5)]))
        else:
            m = 0.6 + 0.2 * u[i]
            ctx, con = (CTX_MIS, CON_MIS) if ch == "b" else (CTX_INS, CON_INS)
            out.append(_exp(f"fol{i}", ctx, con, [_p(A2, "collaborator")], [_amb(A2, 0.5, m)]))
    return out


def make_history(seed: int, bg: str = "trust", src: str = "none", err: str = "interp", n: int = 8, m_extra: int = 0) -> Dict[str, Any]:
    mm = make_mis(seed, err, n, m_extra)
    return {"pre": make_bg(seed, bg) + make_src(seed, src), "mis": mm["mis"], "ctrl": mm["ctrl"], "extra": mm["extra"], "cor": mm["cor"],
            "fol": make_fol(seed, err),
            "meta": {"seed": int(seed), "bg": bg, "src": src, "err": err, "n": n, "m_extra": m_extra, "o_class": o_class(err)}}


def flat(h: Dict[str, Any], arm: str = "mis", with_cor: bool = True, with_fol: bool = True) -> List[Dict[str, Any]]:
    """全体を 1 つの列にする（確認用）。arm = "mis" | "ctrl"。対照は訂正を持たない。"""
    xs = list(h["pre"]) + list(h[arm]) + list(h["extra"])
    if with_cor and arm == "mis":
        xs += list(h["cor"])
    if with_fol:
        xs += list(h["fol"])
    return xs


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bg", default="trust", choices=BGS)
    ap.add_argument("--src", default="none", choices=SRCS)
    ap.add_argument("--err", default="interp", choices=ERRS)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--m", type=int, default=0)
    ap.add_argument("--out", default=None, help="指定すると <out>/v34_<seed>_<bg>_<src>_<err>_<arm>.jsonl に書く")
    a = ap.parse_args()
    h = make_history(a.seed, a.bg, a.src, a.err, a.n, a.m)
    for arm in ("mis", "ctrl"):
        rows = flat(h, arm)
        if a.out:
            d = Path(a.out)
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"v34_{a.seed}_{a.bg}_{a.src}_{a.err}_{arm}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            print(arm, len(rows), "items;", sum(1 for r in rows if "kind" not in r), "experiences;",
                  sum(1 for r in rows if r.get("kind") == "correction"), "corrections;", sum(1 for r in rows if r.get("kind") == "tick"), "ticks")


if __name__ == "__main__":
    _main()
