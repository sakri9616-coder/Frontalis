"""v3.4 Stage 4: 変異テスト（RA11）の実行と検出の行列（SPEC-v3.4 第 8 節 RA11、PLAN の Stage 4）。

  python -B tests/stage4_mutation.py --phase mut --out tests/results/v34/stage4_mutation.json --cache <scratch>/mu_cache      （変異の種 3450〜3459。本番）
  python -B tests/stage4_mutation.py --phase dev --seeds 3490,3491 --design small --stress-n 12 --out <scratch>/dev.json ...  （開発の種での予行）

変異体ごとに、(1) 27 セル（ERR 3 × BG 3 × SRC 3）× 種で C-v34 を測りながら実行（cell_task。SRC-none の 9 セルと ERR-interp の 15 セルでは cross: clamp-τ・clamp-mood・訂正なし・
C-none-v34・C-OR も実行）、(2) 種ごとの仕事（H5、RA8c、先頭 2 種で罪悪感の訂正の続行 stress）、(3) val-swap・hm-eq（SRC-none、ERR ごと、背景の 6 通り）を行い、
検査の結果を変異なしの基準と比べる。変異体ごとの生の結果は cache に pickle で置く（途中で止まっても再開できる。検査を足したときの再評価にも使う）。
Windows の spawn では `if __name__ == "__main__"` が必須。自分が起動したワーカーの PID だけを扱う（ra_parallel）。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_histories_v34 as H  # noqa: E402
import ra_checks_v34 as C  # noqa: E402
import ra_protocol as RP  # noqa: E402
from mutants_v34 import MUST_DETECT, MUTANT_IDS, MUTANTS, MIN_DETECTED  # noqa: E402
from ra_parallel import run_tasks  # noqa: E402

DET_CELL = ("interp", "betray", "unreliable")


def cells_of(design: str) -> List[Tuple[str, str, str]]:
    allc = [(e, b, s) for e in H.ERRS for b in H.BGS for s in H.SRCS]
    if design == "full":
        return allc
    return [c for c in allc if c[0] == "interp" or c[2] == "none"]            # small: ERR-interp 全部 + SRC-none


def build_tasks(mutant: Optional[str], seeds: List[int], phase: str, design: str, stress_n: int, swaps: bool) -> List[Tuple[str, Tuple]]:
    cross_tasks, plain_tasks = [], []
    for seed in seeds:
        for (err, bg, src) in cells_of(design):
            cross = (src == "none") or (err == "interp")
            t = ("ra_checks_v34:cell_task", (mutant, seed, bg, src, err, 8, phase, cross, (err, bg, src) == DET_CELL))
            (cross_tasks if cross else plain_tasks).append(t)
    seed_tasks = [("ra_checks_v34:seed_task", (mutant, seed, phase, stress_n if i < 2 else 0)) for i, seed in enumerate(seeds)]
    swap_tasks = [("ra_checks_v34:swap_task", (mutant, seed, err, phase, 8)) for seed in seeds for err in H.ERRS] if swaps else []
    return seed_tasks + swap_tasks + cross_tasks + plain_tasks          # 重い仕事から先に


def run_mutant(mutant: Optional[str], args, cache: Optional[Path]) -> Dict[str, Any]:
    name = mutant or "baseline"
    f = cache / f"{name}.pkl" if cache else None
    if f is not None and f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    tasks = build_tasks(mutant, args.seeds, args.phase, args.design, args.stress_n, not args.no_swaps)
    t0 = time.time()
    res, info = run_tasks(tasks, workers=args.workers)
    out = {"mutant": name, "tasks": [t[0].split(":")[1] for t in tasks], "results": res, "wall": time.time() - t0,
           "cpu_seconds": float(sum(info["seconds"])), "exit_codes": info["exit_codes"], "pids": info["pids"]}
    print(f"[{name}] {len(tasks)} 仕事, 壁時計 {out['wall']:.0f} s, CPU {out['cpu_seconds']:.0f} s, ワーカー {len(info['pids'])} 本, 終了コード {info['exit_codes']}", flush=True)
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(out, fh)
    return out


def evaluate(raw: Dict[str, Any]) -> Dict[str, Any]:
    kinds = raw["tasks"]
    cells = [r for k, r in zip(kinds, raw["results"]) if k == "cell_task"]
    seedr = [r for k, r in zip(kinds, raw["results"]) if k == "seed_task"]
    swaps = [r for k, r in zip(kinds, raw["results"]) if k == "swap_task"]
    inv = C.merge_invariants(cells + seedr)
    crit = C.criteria_verdicts(cells, swaps)
    cand = C.candidate_results(cells)
    stress = [r["ev"].get("stress") for r in seedr if r["ev"].get("stress")]
    tb = {}
    for r in cells:
        for k, v in r["ev"]["v34"]["tau_boundary"].items():
            tb.setdefault(k, [0, 0])
            tb[k][1] += 1
            tb[k][0] += int(v)
    led_dis = int(sum(r["ev"]["v34"]["ra8a_ledger_total_sign_disagreements"] for r in cells))
    return {"ra8a_ledger_total_sign_disagreements": led_dis, "inv": inv, "crit": crit, "cand": cand, "stress": stress, "n_cells": len(cells), "n_swaps": len(swaps), "tau_boundary_counts": tb,
            "wall": raw["wall"], "cpu_seconds": raw["cpu_seconds"]}


def detection_matrix(ev: Dict[str, Dict[str, Any]], with_candidates: bool = False) -> Dict[str, Any]:
    base = ev["baseline"]
    checks_inv = [k for k in C.INVARIANT_CHECKS]
    checks_crit = [k for k in C.CRITERIA_CHECKS]
    extra = list(C.CANDIDATE_CHECKS) if with_candidates else []
    matrix: Dict[str, Dict[str, str]] = {}
    summary: Dict[str, Any] = {}
    base_clean = [k for k in checks_inv if not base["inv"][k]["ok"]]
    for m in MUTANT_IDS:
        if m not in ev:
            continue
        e = ev[m]
        row: Dict[str, str] = {}
        det_inv, det_crit, det_cand = [], [], []
        for k in checks_inv:
            ok = e["inv"][k]["ok"]
            row[k] = "pass" if ok else "FAIL"
            if not ok and k not in base_clean:
                det_inv.append(k)
        for k in checks_crit:
            vb, vm = base["crit"].get(k), e["crit"].get(k)
            if vb is None or vm is None:
                row[k] = "n/a"
            else:
                row[k] = "same" if vb == vm else "FLIP"
                if vb != vm:
                    det_crit.append(k)
        for k in extra:
            ok = e["cand"][k]["ok"]
            row[k] = "pass" if ok else "FAIL"
            if not ok and base["cand"][k]["ok"]:
                det_cand.append(k)
        matrix[m] = row
        summary[m] = {"detected": bool(det_inv or det_crit or det_cand), "by_invariant": det_inv, "by_verdict_flip": det_crit, "by_candidate": det_cand,
                      "kind": "invariant" if det_inv else ("verdict-flip only" if det_crit else ("candidate only" if det_cand else "none"))}
    n_det = sum(int(s["detected"]) for s in summary.values())
    must = {m: summary[m]["detected"] for m in MUST_DETECT if m in summary}
    passed = bool(must) and all(must.values()) and n_det >= MIN_DETECTED and len(summary) == len(MUTANT_IDS)
    return {"matrix": matrix, "summary": summary, "n_detected": n_det, "must_detect": must, "baseline_invariant_failures": base_clean,
            "rule": f"MU2〜MU5 と MU7 がすべて検出され、{len(MUTANT_IDS)} 体中 {MIN_DETECTED} 体以上が検出されれば合格", "passed": passed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="mut", choices=["mut", "dev"])
    ap.add_argument("--seeds", default=None, help="カンマ区切り。省略すると phase の全種（mut: 3450〜3459）")
    ap.add_argument("--design", default="full", choices=["full", "small"])
    ap.add_argument("--mutants", default="baseline," + ",".join(MUTANT_IDS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stress-n", type=int, default=100)
    ap.add_argument("--no-swaps", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    args.seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(RP.PHASES[args.phase])
    RP.guard_seeds(args.seeds, args.phase)
    cache = Path(args.cache) if args.cache else None
    ev: Dict[str, Dict[str, Any]] = {}
    for m in args.mutants.split(","):
        raw = run_mutant(None if m == "baseline" else m, args, cache)
        ev[m] = evaluate(raw)
    out = {"stage": "4", "phase": args.phase, "seeds": args.seeds, "design": args.design, "stress_n": args.stress_n,
           "mutants": {m: MUTANTS[m]["description"] for m in MUTANT_IDS}, "not_applicable": C.NOT_APPLICABLE,
           "invariant_checks": C.INVARIANT_CHECKS, "criteria_checks": C.CRITERIA_CHECKS,
           "baseline": {"verdicts": ev["baseline"]["crit"], "invariants": {k: v["ok"] for k, v in ev["baseline"]["inv"].items()}},
           "per_mutant": {m: {"invariants": {k: {"ok": v["ok"], "n_run": v["n_run"], "n_fail": v["n_fail"], "first_fails": v["first_fails"]} for k, v in e["inv"].items()},
                              "verdicts": e["crit"], "candidates": e["cand"], "ra8a_ledger_total_sign_disagreements": e["ra8a_ledger_total_sign_disagreements"], "stress": e["stress"], "tau_boundary_counts": e["tau_boundary_counts"],
                              "wall": e["wall"], "cpu_seconds": e["cpu_seconds"]} for m, e in ev.items()},
           "initial_set": detection_matrix(ev, with_candidates=False), "with_candidate_checks": detection_matrix(ev, with_candidates=True)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    s = out["initial_set"]
    print(json.dumps({m: (v["detected"], v["kind"], v["by_invariant"], v["by_verdict_flip"]) for m, v in s["summary"].items()}, ensure_ascii=False, indent=1))
    print("検出", s["n_detected"], "/", len(s["summary"]), "必須", s["must_detect"], "合格", s["passed"])


if __name__ == "__main__":
    main()
