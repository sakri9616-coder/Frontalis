"""v3.4 Stage 4: 1 個体あたりの所要時間の実測と、Stage 5（評価）の壁時計の見積もり（PLAN: 約 14 CPU 時間、10 並列）。開発用の種 3490〜3491 だけで測る。

  python -B tests/stage4_timing.py --out tests/results/v34/stage4_timing.json

測るもの（ワーカー 10 本で同時に走らせた状態での各仕事の所要時間。直列より遅くなる分も含む）:
  main        主要因計画の 1 個体（27 セルの代表 8 つ）に、Stage 5 が必要とする全条件（MAIN_CONDS）を run_individual で実行
  sweep       N の掃引の 1 個体（N ∈ {2, 4, 8, 12, 16} × M ∈ {0, 4}）を 5 条件で
  shapley     Shapley 分解 1 組（6 群、64 通りの入れ替え）
  swap        val-swap・hm-eq（SRC-none、ERR 1 つ、背景の 6 通りの向き）の 1 種分
  yoke        τ-yoke 1 個体
  checks      故障の検査（ra_checks_v34.cell_task）1 セル（plain と cross）
見積もりは、各仕事の数 × 所要時間の平均。並列の効率（総 CPU / (ワーカー数 × 壁時計)）も実測する。Stage 5 の仕事の構成は Stage 5 が決める。ここでの構成は PLAN の記述に沿った見積もり用の想定。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_histories_v34 as H  # noqa: E402
import ra_parallel as PAR  # noqa: E402
import ra_protocol as RP  # noqa: E402

MAIN_CONDS = ["C-v34", "C-OR", "C-v33", "C-retro", "C-none", "C-nocorr", "C-nocorr-v34", "A-g-const", "A-κ-fixed", "A-τ-off", "A-clock-off",
              "clamp-τ", "clamp-mood", "clamp-both", "local-reattr"]
SWEEP_CONDS = ["C-v34", "C-OR", "C-v33", "C-retro", "C-nocorr"]
N_GRID = (2, 4, 8, 12, 16)
M_GRID = (0, 4)
SEEDS = (3490, 3491)

# Stage 5 の仕事の数（SPEC 第 7.3 節・PLAN の Stage 5）。評価の種は 40 本、Shapley は 20 本。
N_EVAL, N_SHAP = 40, 20


def build_samples() -> Dict[str, List]:
    s: Dict[str, List] = {"main": [], "sweep": [], "shapley": [], "swap": [], "yoke": [], "checks_plain": [], "checks_cross": []}
    cells = [("interp", "betray", "none"), ("interp", "trust", "unreliable"), ("report", "neutral", "reliable"), ("report-x", "betray", "none"),
             ("interp", "neutral", "none"), ("report", "trust", "unreliable"), ("report-x", "trust", "reliable"), ("interp", "betray", "reliable")]
    for (err, bg, src) in cells:
        s["main"].append(("ra_protocol:run_individual", (SEEDS[0], bg, src, err, 8, 0, MAIN_CONDS, "dev")))
    for N in N_GRID:
        for M in M_GRID:
            s["sweep"].append(("ra_protocol:run_individual", (SEEDS[1], "betray", "none", "interp", N, M, SWEEP_CONDS, "dev")))
    s["shapley"].append(("ra_protocol:run_shapley", (SEEDS[0], "none", "interp", "betray", "trust", "dev")))
    s["swap"].append(("ra_checks_v34:swap_task", (None, SEEDS[0], "interp", "dev", 8)))
    s["yoke"].append(("ra_protocol:run_tau_yoke", (SEEDS[0], "none", "interp", "neutral", "betray", "dev")))
    s["checks_plain"].append(("ra_checks_v34:cell_task", (None, SEEDS[0], "trust", "reliable", "report", 8, "dev", False, False)))
    s["checks_cross"].append(("ra_checks_v34:cell_task", (None, SEEDS[0], "betray", "none", "interp", 8, "dev", True, False)))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    samples = build_samples()
    order = ["shapley", "swap", "main", "sweep", "checks_cross", "checks_plain", "yoke"]      # 重い仕事から先に
    tasks: List = []
    kinds: List[str] = []
    for k in order:
        tasks += samples[k]
        kinds += [k] * len(samples[k])
    t0 = time.perf_counter()
    res, info = PAR.run_tasks(tasks, workers=a.workers)
    wall = time.perf_counter() - t0
    by: Dict[str, List[float]] = {}
    for k, sec in zip(kinds, info["seconds"]):
        by.setdefault(k, []).append(sec)
    mean = {k: sum(v) / len(v) for k, v in by.items()}
    cpu = float(sum(info["seconds"]))
    eff = cpu / (info["workers"] * wall)
    # Stage 5 の仕事の数
    n_main = 27 * N_EVAL                                    # 1,080 個体
    n_sweep = len(N_GRID) * len(M_GRID) * 3 * 3 * N_EVAL    # SRC-none、BG 3 × ERR 3 × 40 種 × N 5 × M 2 = 3,600 個体
    n_shap = 3 * 3 * N_SHAP                                 # 3 組 × 3 ERR × 20 種
    n_swap = 3 * N_EVAL                                     # (種, ERR) ごとに val-swap・hm-eq の 6 通りの向き
    n_yoke = 9 * N_EVAL                                     # ERR-interp の SRC 3 × BG 3 × 40 種（donor は別の背景 1 つ）
    n_chk_plain = 12 * N_EVAL                               # 27 セルのうち cross でない 12 セル
    n_chk_cross = 15 * N_EVAL                               # SRC-none 9 + ERR-interp の他 SRC 6
    lines = {
        "main": (n_main, mean["main"]), "sweep": (n_sweep, mean["sweep"]), "shapley": (n_shap, mean["shapley"]), "swap": (n_swap, mean["swap"]),
        "yoke": (n_yoke, mean["yoke"]), "checks_plain": (n_chk_plain, mean["checks_plain"]), "checks_cross": (n_chk_cross, mean["checks_cross"])}
    est_cpu = {k: n * m for k, (n, m) in lines.items()}
    total_cpu = float(sum(est_cpu.values()))
    # 壁時計: 各仕事の所要時間（実測の平均）を並べた貪欲な割り当て。実測の並列効率で割った見積もりと、理想（効率 1）の両方
    all_secs: List[float] = []
    for k, (n, m) in lines.items():
        all_secs += [m] * n
    wall_ideal = PAR.estimate_wall(all_secs, 10)
    out: Dict[str, Any] = {
        "stage": 4, "seeds": list(SEEDS), "workers": info["workers"], "sample_wall_s": wall, "sample_cpu_s": cpu, "parallel_efficiency": eff,
        "exit_codes": info["exit_codes"], "pids_started": info["pids"],
        "mean_seconds_per_task_under_10_workers": mean, "all_seconds": by,
        "stage5_task_counts": {k: v[0] for k, v in lines.items()},
        "stage5_cpu_seconds_estimate": est_cpu, "stage5_cpu_hours_estimate": total_cpu / 3600.0,
        "stage5_wall_hours_10_workers_ideal": wall_ideal / 3600.0, "stage5_wall_hours_10_workers_at_measured_efficiency": wall_ideal / 3600.0 / min(1.0, max(eff, 1e-9)),
        "n_conditions_main": len(MAIN_CONDS), "note": "見積もり用の想定の仕事の構成（PLAN-v3.4 の Stage 5 の記述に沿う）。実測は開発用の種 3490〜3491 の代表的な個体。所要時間は 10 本の同時実行下の値。",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("sample_wall_s", "sample_cpu_s", "parallel_efficiency", "mean_seconds_per_task_under_10_workers", "stage5_task_counts",
                                          "stage5_cpu_hours_estimate", "stage5_wall_hours_10_workers_ideal", "stage5_wall_hours_10_workers_at_measured_efficiency")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
