"""Run ONE long job line on Colab without holding a connection open for the whole run (colab_runner.py's single long
`exec` lost its connection after a while on 2026-10-23). The job is started in the background on the VM; short `exec`
calls check it every few minutes; when it is done its new top-level files are downloaded into probes/ and the VM is
always released.  python sched/colab_detached.py "<job line>" [max_hours] [machine]
machine: "cpu-high" (default; CPU runtime, high-RAM shape, more cores, no GPU units) or a GPU name such as "T4".
Like colab_runner.py it runs a fresh clone of the pushed branch, so only committed and pushed code runs there.
The CLI output can contain a runtime proxy URL with a token: this script prints no URLs."""
import re, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBES = HERE.parent
BRANCH = "claude/value-representation-design"
START = '''import subprocess, os
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
if not os.path.exists("/content/F"):
    r = sh("git clone -q --branch %(branch)s --single-branch https://github.com/sakri9616-coder/Frontalis.git /content/F")
    assert r.returncode == 0, r.stderr
D = "/content/F/frontalis-v3-lab/proposals/probes"
open("/content/snap.txt", "w").write("\\n".join(f for f in os.listdir(D) if os.path.isfile(os.path.join(D, f))))
subprocess.Popen("cd %%s && (PYTHONUTF8=1 python -B %(line)s > /content/job.log 2>&1; echo $? > /content/job.rc)" %% D,
                 shell=True, start_new_session=True)
print("STARTED", os.cpu_count(), sh("cd %%s && git rev-parse --short HEAD" %% D).stdout.strip())
'''
CHECK = '''import os
print("DONE" if os.path.exists("/content/job.rc") else "RUNNING")
print(open("/content/job.log").read()[-1500:] if os.path.exists("/content/job.log") else "")
'''
PACK = '''import os, tarfile
D = "/content/F/frontalis-v3-lab/proposals/probes"
old = set(open("/content/snap.txt").read().split("\\n"))
with tarfile.open("/content/out.tar.gz", "w:gz") as t:
    for f in os.listdir(D):
        p = os.path.join(D, f)
        if os.path.isfile(p) and f not in old and not f.endswith(".pyc"):
            t.add(p, arcname=f)
print("RC", open("/content/job.rc").read().strip())
'''


def colab(*a, timeout=900):
    """On timeout kill the whole process tree (cmd -> uvx -> colab.exe -> python); killing only cmd left the call hanging
    for two hours on 2026-10-23 because the grandchildren kept the output pipes open."""
    p = subprocess.Popen(["cmd", "/c", str(HERE / "colab.cmd"), *a], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf-8", errors="replace")
    try:
        out, err = p.communicate(timeout=timeout)
        return subprocess.CompletedProcess(a, p.returncode, out, err)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        try:
            p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        return subprocess.CompletedProcess(a, 124, "", "timeout")


def clean(s):
    return re.sub(r"https?://\S+", "<url>", s)


def run_py(name, code, tmp, tag):
    f = tmp / f"{tag}.py"
    f.write_text(code, encoding="utf-8")
    return colab("exec", "-s", name, "--timeout", "600", "-f", str(f))


def main(line, max_hours, machine):
    name = f"sched{int(time.time()) % 100000}"
    shape = ["--high-mem"] if machine == "cpu-high" else ["--gpu", machine]
    r = colab("new", "-s", name, *shape)
    print(clean(r.stdout[-300:]), clean(r.stderr[-300:]), flush=True)
    if "READY" not in r.stdout + r.stderr:
        sys.exit(75)
    rc, tmp = 1, Path(tempfile.mkdtemp())
    try:
        r = run_py(name, START % dict(branch=BRANCH, line=line), tmp, "start")
        print(clean(r.stdout[-500:]), clean(r.stderr[-300:]), flush=True)
        if "STARTED" not in r.stdout:
            sys.exit(1)
        t0, misses = time.time(), 0
        while time.time() - t0 < max_hours * 3600:
            time.sleep(300)
            r = run_py(name, CHECK, tmp, "check")
            out = r.stdout
            print(time.strftime("%H:%M"), "DONE" if "DONE" in out else ("RUNNING" if "RUNNING" in out else "no answer"),
                  clean(out[-300:]).strip().replace("\n", " | "), flush=True)
            misses = 0 if ("DONE" in out or "RUNNING" in out) else misses + 1
            if "DONE" in out or misses >= 6:
                break
        r = run_py(name, PACK, tmp, "pack")
        print(clean(r.stdout[-300:]), flush=True)
        m = re.findall(r"^RC (-?\d+)\s*$", r.stdout, re.M)
        rc = int(m[-1]) if m else 1
        d = colab("download", "-s", name, "/content/out.tar.gz", str(tmp / "out.tar.gz"))
        if (tmp / "out.tar.gz").exists():
            with tarfile.open(tmp / "out.tar.gz") as t:
                for mem in t.getmembers():
                    if mem.isfile() and "/" not in mem.name and "\\" not in mem.name and not mem.name.startswith("."):
                        (PROBES / mem.name).write_bytes(t.extractfile(mem).read())
                print("files:", [mem.name for mem in t.getmembers()], flush=True)
        else:
            print("download failed", clean(d.stdout[-300:]), clean(d.stderr[-300:]), flush=True)
            rc = rc or 2
    finally:
        print(clean(colab("stop", "-s", name).stdout[-200:]), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 10.0, sys.argv[3] if len(sys.argv) > 3 else "cpu-high")
