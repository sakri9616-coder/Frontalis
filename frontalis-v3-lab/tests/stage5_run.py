"""v3.4 Stage 5 の実行器（評価の種 3400〜3439 を 1 回だけ。予行は開発用の種 3490〜3499）。

  python -B tests/stage5_run.py --phase dev    --seeds 3490,3491 --out <dir> --cache <dir> --log <file>           （予行）
  python -B tests/stage5_run.py --phase stage5 --out tests/results/v34 --cache <scratch>/cache --log <scratch>/log   （本番。freeze-protocol-v3.json の hash と一致しなければ拒否）

仕事（1 つの種あたり）: main 27（主要因計画 27 セル × 全条件）、checks 27（故障の検査 cell_task）、extras 27、sweep 9（N の掃引。BG 3 × ERR 3）、swap 3（val-swap・hm-eq）、align 3（入れ替えの並びの一致。2 通りの読み）、
yoke 18（τ-yoke。ERR-interp の SRC 3 × 背景の向き 6）、shapley 9（種 3400〜3419 だけ。予行は先頭 2 種。3 組 × ERR 3、7 群）、seed 1（H5・RA8c）、oldy 1（v3.3 の Y1〜Y7）。
各仕事の結果は cache に pickle で置き（途中で止まっても再開できる）、1 つの種の仕事がすべて揃ったら <out>/stage5_seed_<seed>.json を書く。
ワーカーは spawn のプロセス（最大 6 本。リードの決定）。自分が起動したプロセスの PID だけを pid ファイルに書き、イメージ名での一括終了はしない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import platform
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(ROOT / "data"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_histories_v34 as H  # noqa: E402
import ra_parallel as PAR  # noqa: E402
import ra_protocol as RP  # noqa: E402
from stage5_tasks import to_jsonable  # noqa: E402

MAX_WORKERS = 6                                   # リードの決定（機械は共有。PLAN の 10 ではなく 6）
DET_CELL = ("interp", "betray", "unreliable")
PAIRS = (("trust", "betray"), ("trust", "neutral"), ("betray", "neutral"))      # 無向の 3 組。入れ替えの target は先の側（RA2-hm と同じ。S4-10）
DONORS = [(t, d) for t in H.BGS for d in H.BGS if t != d]
EVAL_CODE = ["tests/ra_protocol.py", "data/make_histories_v34.py", "tests/mutants_v34.py", "tests/ra_parallel.py", "tests/ra_checks_v34.py", "tests/stage4_mutation.py",
             "tests/stage5_tasks.py", "tests/stage5_run.py", "tests/test_ra_criteria.py"]
SYSTEM_DIR = "npnv3"


def cells() -> List[Tuple[str, str, str]]:
    return [(e, b, s) for e in H.ERRS for b in H.BGS for s in H.SRCS]


def build_seed_tasks(seed: int, phase: str, with_shapley: bool) -> List[Tuple[str, str, str, Tuple]]:
    """(種類, キー, 関数名, 引数) の列。重い仕事から先に。"""
    T: List[Tuple[str, str, str, Tuple]] = []
    if with_shapley:
        for err in H.ERRS:
            for bt, bs in PAIRS:
                T.append(("shapley", f"{err}|{bt}<-{bs}", "ra_protocol:run_shapley", (seed, "none", err, bt, bs, phase, "C-v34", 8, 0, RP.SHAPLEY_PLAYERS_SPLIT_G5)))
    for bg in H.BGS:
        for err in H.ERRS:
            T.append(("sweep", f"{bg}|{err}", "stage5_tasks:sweep_task", (seed, bg, err, phase)))
    for (err, bg, src) in cells():
        T.append(("main", f"{err}|{bg}|{src}", "stage5_tasks:main_task", (seed, bg, src, err, phase)))
    for err in H.ERRS:
        T.append(("swap", err, "ra_checks_v34:swap_task", (None, seed, err, phase, 8)))
    for err in H.ERRS:
        T.append(("align", err, "stage5_tasks:align_task", (seed, err, phase)))
    for (err, bg, src) in cells():
        cross = (src == "none") or (err == "interp")
        T.append(("checks", f"{err}|{bg}|{src}", "ra_checks_v34:cell_task", (None, seed, bg, src, err, 8, phase, cross, (err, bg, src) == DET_CELL)))
    for (err, bg, src) in cells():
        T.append(("extras", f"{err}|{bg}|{src}", "stage5_tasks:extras_task", (seed, bg, src, err, phase)))
    for src in H.SRCS:
        for (t, d) in DONORS:
            T.append(("yoke", f"{src}|{t}<-{d}", "ra_protocol:run_tau_yoke", (seed, src, "interp", t, d, phase)))
    T.append(("seed", "", "ra_checks_v34:seed_task", (None, seed, phase, 0)))
    T.append(("oldy", "", "stage5_tasks:old_y_task", (seed, phase)))
    return T


def task_id(seed: int, kind: str, key: str) -> str:
    return f"{seed}__{kind}__{key.replace('|', '_').replace('<-', '_from_')}"


def file_hashes() -> Dict[str, str]:
    d = {f"frontalis-v3-lab/{rel}": RP.file_sha256(ROOT / rel) for rel in EVAL_CODE if (ROOT / rel).exists()}
    for p in sorted((ROOT / SYSTEM_DIR).glob("*.py")):
        d[f"frontalis-v3-lab/{SYSTEM_DIR}/{p.name}"] = RP.file_sha256(p)
    return d


def verify_freeze() -> None:
    """本番（phase = stage5）の前に、freeze-protocol-v3.json の hash が現在のファイルと一致することを確かめる。"""
    f = ROOT / "tests" / "results" / "v34" / "freeze-protocol-v3.json"
    if not f.exists():
        sys.exit("freeze-protocol-v3.json がない: 本番の前に評価コードの hash を凍結すること")
    fz = json.loads(f.read_text(encoding="utf-8"))
    bad = []
    for rel, h in fz["evaluation_code_sha256"].items():
        if RP.file_sha256(ROOT.parent / rel) != h:
            bad.append(rel)
    for rel, h in fz["system_files_sha256"].items():
        if RP.file_sha256(ROOT.parent / rel) != h:
            bad.append(rel)
    if bad:
        sys.exit(f"凍結した hash と一致しないファイル: {bad}")


def log(fh, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def assemble(seed: int, tasks: List[Tuple[str, str, str, Tuple]], cache: Path, phase: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"seed": seed, "phase": phase, "main": {}, "checks": {}, "extras": {}, "sweep": {}, "swap": {}, "align": {}, "yoke": {}, "shapley": {}, "seed_task": None, "old_y": None}
    timing: Dict[str, float] = {}
    for (kind, key, _fn, _args) in tasks:
        with open(cache / (task_id(seed, kind, key) + ".pkl"), "rb") as fh:
            rec = pickle.load(fh)
        res, secs = to_jsonable(rec["result"]), rec["seconds"]
        timing[f"{kind}:{key}"] = secs
        if kind == "seed":
            out["seed_task"] = res
        elif kind == "oldy":
            out["old_y"] = res
        else:
            out[kind][key] = res
    out["timing"] = {"seconds": timing, "total_cpu_seconds": float(sum(timing.values()))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["dev", "stage5"])
    ap.add_argument("--seeds", default=None, help="カンマ区切り。省略すると phase の全種（stage5: 3400〜3439）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pidfile", default=None)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--shapley-seeds", type=int, default=None, help="先頭から何本の種で Shapley を行うか（既定: stage5 は 20、dev は 2）")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else list(RP.PHASES[a.phase])
    RP.guard_seeds(seeds, a.phase)
    if a.phase == "stage5":
        verify_freeze()
    workers = max(1, min(a.workers, MAX_WORKERS))
    n_shap = a.shapley_seeds if a.shapley_seeds is not None else (20 if a.phase == "stage5" else 2)
    out_dir, cache = Path(a.out), Path(a.cache)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.log, "a", encoding="utf-8")
    log(fh, f"start phase={a.phase} seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) workers={workers} shapley_seeds={n_shap} python={platform.python_version()} pid={os.getpid()}")
    plan: Dict[int, List[Tuple[str, str, str, Tuple]]] = {}
    todo: List[Tuple[int, str, str, str, Tuple]] = []
    for i, s in enumerate(seeds):
        if (out_dir / f"stage5_seed_{s}.json").exists():
            log(fh, f"seed {s}: 既に完了（{out_dir / f'stage5_seed_{s}.json'}）。飛ばす")
            continue
        tl = build_seed_tasks(s, a.phase, i < n_shap)
        plan[s] = tl
        for (k, key, fn, args) in tl:
            if not (cache / (task_id(s, k, key) + ".pkl")).exists():
                todo.append((s, k, key, fn, args))
    total = len(todo)
    log(fh, f"残りの仕事 {total}（キャッシュ済みを除く）")
    remaining: Dict[int, int] = {s: sum(1 for t in todo if t[0] == s) for s in plan}
    finished_seeds: List[int] = []

    def finish_seed(s: int) -> None:
        d = assemble(s, plan[s], cache, a.phase)
        d["environment"] = {"python": platform.python_version(), "platform": platform.platform(), "workers": workers}
        d["code_sha256"] = file_hashes()
        tmp = out_dir / f"stage5_seed_{s}.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, out_dir / f"stage5_seed_{s}.json")
        finished_seeds.append(s)
        log(fh, f"seed {s} 完了: CPU {d['timing']['total_cpu_seconds']:.0f} s（{len(finished_seeds)}/{len(plan)}）")

    for s in list(plan):
        if remaining[s] == 0:
            finish_seed(s)
    if not todo:
        log(fh, "仕事なし。終了")
        fh.close()
        return
    saved_env = {k: os.environ.get(k) for k in PAR.THREAD_ENV}
    for k in PAR.THREAD_ENV:
        os.environ[k] = "1"
    t0 = time.time()
    failed: List[str] = []
    done = 0
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=PAR._init_worker) as ex:
            pids = []
            futs = {}
            for (s, k, key, fn, args) in todo:
                futs[ex.submit(PAR._run_one, (fn, args))] = (s, k, key)
            pids = [p.pid for p in getattr(ex, "_processes", {}).values()]
            if a.pidfile:
                Path(a.pidfile).write_text(json.dumps({"orchestrator": os.getpid(), "workers": pids}), encoding="utf-8")
            log(fh, f"提出 {len(futs)} 仕事。ワーカー PID {pids}")
            pending = set(futs)
            while pending:
                dn, pending = wait(pending, return_when=FIRST_COMPLETED)
                cur = sorted(p.pid for p in getattr(ex, "_processes", {}).values())
                if a.pidfile and cur != sorted(pids):
                    pids = cur
                    Path(a.pidfile).write_text(json.dumps({"orchestrator": os.getpid(), "workers": pids}), encoding="utf-8")
                for f in dn:
                    s, k, key = futs[f]
                    try:
                        res, pid, secs = f.result()
                    except Exception:
                        failed.append(f"{s}:{k}:{key}")
                        log(fh, f"失敗 {s} {k} {key}\n{traceback.format_exc()}")
                        continue
                    tmp = cache / (task_id(s, k, key) + ".pkl.tmp")
                    with open(tmp, "wb") as g:
                        pickle.dump({"result": res, "seconds": secs, "pid": pid}, g)
                    os.replace(tmp, cache / (task_id(s, k, key) + ".pkl"))
                    done += 1
                    remaining[s] -= 1
                    el = time.time() - t0
                    if done % 25 == 0 or remaining[s] == 0:
                        log(fh, f"進捗 {done}/{total}  経過 {el / 3600:.2f} h  残り見込み {el / done * (total - done) / 3600:.2f} h（直近 {k}:{key} {secs:.0f} s）")
                    if remaining[s] == 0 and s in plan:
                        try:
                            finish_seed(s)
                        except Exception:
                            failed.append(f"{s}:assemble")
                            log(fh, f"種 {s} の書き出しに失敗 {traceback.format_exc()}")
            codes = [p.exitcode for p in getattr(ex, "_processes", {}).values()]
        log(fh, f"プールを閉じた。ワーカーの終了コード {codes}")
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    log(fh, f"終了: 完了した種 {len(finished_seeds)}、失敗した仕事 {len(failed)} {failed[:10]}、壁時計 {(time.time() - t0) / 3600:.2f} h")
    fh.close()
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
