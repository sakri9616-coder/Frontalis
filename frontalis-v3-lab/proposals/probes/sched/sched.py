"""Minimal multi-cloud experiment scheduler (stdlib only; one file).

  python sched/sched.py plan   <queue.txt>   expand the queue, show which provider each job would get, change nothing
  python sched/sched.py run    <queue.txt>   dispatch jobs to free capacity until the queue is done (Ctrl+C is safe; rerun resumes)
  python sched/sched.py status               providers (free budget left, slots) and job states
  python sched/sched.py approve <provider> <usd>   allow paid spend (beyond the free budget) on that provider, up to <usd>
  python sched/sched.py reset  [--failed | "<job line>"]   put failed / given jobs back to pending
  --only=<name>[,<name>]  (plan/run) use only these providers, e.g. --only=modal to keep one machine type and the local GPU free

Queue: one job per line, the same form as queue4.txt / queue_cloud.txt ("script.py args"). Brace lists split a line into
jobs automatically:  p15_prod.py cells {set,krr} {IH,I0} {C-off,C-SLp}  ->  8 jobs.
Providers live in providers.json. A new service needs no code: give it a "cmd" template that runs one job line on it and
leaves the result files in the probes directory (see modal_runner.py). Money rule: a job goes to a provider on its own only
if its worst-case cost (price_per_hour * max_job_hours) fits in that provider's remaining free budget; otherwise the job
waits as BLOCKED until `approve` grants the excess. Evaluation seeds (77100-77139) are refused without --allow-eval."""
import atexit, json, os, re, subprocess, sys, threading, time, uuid
from pathlib import Path

HERE = Path(__file__).parent
PROBES = HERE.parent
CFG, STATE, LOGS = HERE / "providers.json", HERE / "sched_state.json", HERE / "sched_logs"
LOCK = threading.Lock()


def expand(line):
    m = re.search(r"\{([^{}]*,[^{}]*)\}", line)
    if not m:
        return [line]
    return [x for opt in m.group(1).split(",") for x in expand(line[:m.start()] + opt + line[m.end():])]


def load_queue(path, allow_eval=False):
    jobs = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        for line in expand(raw):
            if re.search(r"[;&|<>$`\"'\\%^()!:]", line) or ".." in line or re.search(r"(^| )/", line) or not re.match(r"^[\w.-]+\.py( |$)", line) or not (PROBES / line.split()[0]).exists():
                sys.exit(f"refused (unsafe or unknown script): {line}")
            if not allow_eval and re.search(r"\b771[0-3]\d\b", line):
                sys.exit(f"refused (evaluation seed; use --allow-eval only for the one-time evaluation): {line}")
            if line not in jobs:
                jobs.append(line)
    return jobs


def load():
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    st.setdefault("jobs", {}); st.setdefault("spend", {}); st.setdefault("approved", {}); st.setdefault("reserved", {})
    return cfg["providers"], st


def save(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)  # atomic: a crash cannot leave half a file


def free_left(p, st):
    return max(0.0, p.get("free_usd", 0.0) - st["spend"].get(p["name"], 0.0) - st["reserved"].get(p["name"], 0.0))


def gate(p, st):
    """-> (ok, excess_needing_approval). Free providers (price 0) and jobs that fit the free budget need no approval."""
    worst = p.get("price_per_hour", 0.0) * p.get("max_job_hours", 1.0)
    if worst == 0:
        return True, 0.0
    excess = worst if p.get("paid") else max(0.0, worst - free_left(p, st))
    return (excess == 0 or st["approved"].get(p["name"], 0.0) >= excess), excess


def busy_slots(p, st):
    return sum(1 for j in st["jobs"].values() if j["status"] == "running" and j["provider"] == p["name"])


def vram_ok(p):
    need = p.get("min_free_vram_mb")
    if not need:
        return True
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0]) >= need
    except Exception:
        return False


ONLY = None   # set from --only=<names>


def pick(providers, st):
    """First enabled provider with a free slot whose spend gate is open. Returns (provider|None, blocked_excess|None)."""
    blocked = None
    for p in providers:
        if (ONLY and p["name"] not in ONLY) or not p.get("enabled", True) or busy_slots(p, st) >= p.get("concurrency", 1) or not vram_ok(p):
            continue
        ok, excess = gate(p, st)
        if ok:
            return p, None
        blocked = (p["name"], excess)
    return None, blocked


def run_job(p, line, st):
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"{time.strftime('%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_{p['name']}.log"
    cmd = p["cmd"].replace("{python}", sys.executable).replace("{line}", line)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"$ {cmd}\n")
        try:
            rc = subprocess.run(cmd, shell=True, cwd=PROBES, stdout=f, stderr=subprocess.STDOUT, timeout=p.get("max_job_hours", 1.0) * 3600,
                                env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}).returncode
        except subprocess.TimeoutExpired:
            rc = -9
        except Exception as e:  # never leave the job 'running' with its reservation held
            f.write(f"runner error: {e!r}\n"); rc = -1
    secs = time.time() - t0
    cost = p.get("price_per_hour", 0.0) * secs / 3600
    with LOCK:
        _, st2 = load()
        old, new = st2["spend"].get(p["name"], 0.0), st2["spend"].get(p["name"], 0.0) + cost
        st2["spend"][p["name"]] = new
        over = max(0.0, new - p.get("free_usd", 0.0)) - max(0.0, old - p.get("free_usd", 0.0))
        st2["approved"][p["name"]] = max(0.0, st2["approved"].get(p["name"], 0.0) - over)
        st2["reserved"][p["name"]] = max(0.0, st2["reserved"].get(p["name"], 0.0) - p.get("price_per_hour", 0.0) * p.get("max_job_hours", 1.0))
        st2["jobs"][line].update(status="done" if rc == 0 else "failed", rc=rc, secs=round(secs), cost=round(cost, 4), log=log.name)
        save(st2)
    print(f"{time.strftime('%T')} {'DONE' if rc == 0 else 'FAIL'} rc={rc} {round(secs)}s ${cost:.3f} [{p['name']}] {line}")


def single_instance():
    """One scheduler at a time: an OS file lock that disappears with the process (no stale lock after a crash)."""
    f = open(HERE / ".run.lock", "w")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another sched.py run is already active; wait for it or stop it first")
    return f


def cmd_run(queue, dry=False, allow_eval=False):
    lock = single_instance()  # keep the reference alive for the whole run
    providers, st = load()
    jobs = load_queue(queue, allow_eval)
    for j in st["jobs"].values():  # leftovers of a crashed run, from any queue: nothing is running now
        if j["status"] in ("running", "blocked"):
            j["status"] = "pending"
    st["reserved"] = {}
    for line in jobs:
        st["jobs"].setdefault(line, {"status": "pending"})
    save(st)
    threads, warned = [], set()
    while True:
        started = 0
        with LOCK:
            providers, st = load()
            pend = [l for l in jobs if st["jobs"][l]["status"] == "pending"]
            for line in pend:
                p, blocked = pick(providers, st)
                if p is None:
                    if blocked and (blocked, line) not in warned:
                        warned.add((blocked, line))
                        print(f"BLOCKED needs approval: `python sched/sched.py approve {blocked[0]} {blocked[1]:.2f}`  ({line})")
                    continue
                st["jobs"][line].update(status="running", provider=p["name"])
                st["reserved"][p["name"]] = st["reserved"].get(p["name"], 0.0) + p.get("price_per_hour", 0.0) * p.get("max_job_hours", 1.0)
                save(st)
                started += 1
                print(f"{time.strftime('%T')} START [{p['name']}] {line}")
                if dry:
                    st["reserved"][p["name"]] = 0.0
                    st["jobs"][line].update(status="done", rc=0, secs=0, cost=0); save(st)
                else:
                    t = threading.Thread(target=run_job, args=(p, line, st), daemon=True); t.start(); threads.append(t)
        alive = any(t.is_alive() for t in threads)
        if not alive and (not pend or not started):
            if pend and not started:
                print("Nothing can run now (waiting for approval or a free provider); rerun after approving.")
            break
        time.sleep(0.2 if dry else 5)
    for t in threads:
        t.join()
    _, st = load()
    print("queue finished:", {s: sum(1 for l in jobs if st["jobs"][l]["status"] == s) for s in ("done", "failed", "pending")})


def cmd_plan(queue, allow_eval=False):
    providers, st = load()
    jobs = load_queue(queue, allow_eval)
    print(f"{len(jobs)} jobs")
    for l in jobs:
        s = st["jobs"].get(l, {}).get("status", "new")
        print(f"  [{s}] {l}")
    for p in providers:
        ok, excess = gate(p, st)
        print(f"provider {p['name']}: {'enabled' if p.get('enabled', True) else 'off'}, {p.get('gpu')}, slots {p.get('concurrency', 1)}, "
              f"free left ${free_left(p, st):.2f}, worst job ${p.get('price_per_hour', 0) * p.get('max_job_hours', 1):.2f}, "
              f"{'auto-OK' if ok else f'needs approval of ${excess:.2f}'}")


def cmd_status():
    providers, st = load()
    for p in providers:
        print(f"{p['name']}: free left ${free_left(p, st):.2f} / ${p.get('free_usd', 0):.2f}, spent ${st['spend'].get(p['name'], 0):.3f}, "
              f"approved ${st['approved'].get(p['name'], 0):.2f}, running {busy_slots(p, st)}/{p.get('concurrency', 1)}")
    c = {}
    for j in st["jobs"].values():
        c[j["status"]] = c.get(j["status"], 0) + 1
    print("jobs:", c)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    ev = "--allow-eval" in sys.argv
    ONLY = next((x.split("=", 1)[1].split(",") for x in sys.argv if x.startswith("--only=")), None)
    if not a:
        sys.exit(__doc__)
    if a[0] == "plan": cmd_plan(a[1], ev)
    elif a[0] == "run": cmd_run(a[1], "--dry" in sys.argv, ev)
    elif a[0] == "status": cmd_status()
    elif a[0] == "approve":
        _, st = load(); st["approved"][a[1]] = st["approved"].get(a[1], 0.0) + float(a[2]); save(st); print(f"approved ${a[2]} more on {a[1]}")
    elif a[0] == "reset":
        _, st = load()
        for l, j in st["jobs"].items():
            if (len(a) > 1 and l == a[1]) or ("--failed" in sys.argv and j["status"] == "failed"):
                j["status"] = "pending"
        save(st)
    else:
        sys.exit(__doc__)
