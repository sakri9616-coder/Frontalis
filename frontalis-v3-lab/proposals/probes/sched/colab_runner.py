"""Run ONE job line on a free Colab GPU through the official Colab CLI (sched/colab.cmd), copy the files it wrote back into the
probes directory, and always release the VM. Used by sched.py (providers.json "colab"):  python sched/colab_runner.py "<job line>"
Like modal_runner.py it runs a fresh clone of the pushed branch, so only committed and pushed code runs there."""
import re, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBES = HERE.parent
BRANCH = "claude/value-representation-design"
BOOT = '''import subprocess, os, tarfile, time
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
sh("pip -q install numpy 2>&1 | tail -1")
if not os.path.exists("/content/F"):
    r = sh("git clone -q --branch %(branch)s --single-branch https://github.com/sakri9616-coder/Frontalis.git /content/F")
    assert r.returncode == 0, r.stderr
D = "/content/F/frontalis-v3-lab/proposals/probes"
snap = {f: os.stat(os.path.join(D, f)).st_mtime_ns for f in os.listdir(D) if os.path.isfile(os.path.join(D, f))}
r = sh("cd %%s && PYTHONUTF8=1 python -B %(line)s" %% D)
print("COMMIT", sh("cd %%s && git rev-parse --short HEAD" %% D).stdout.strip())
print(r.stdout[-2000:]); print(r.stderr[-2000:])
with tarfile.open("/content/out.tar.gz", "w:gz") as t:
    for f in os.listdir(D):
        p = os.path.join(D, f)
        if os.path.isfile(p) and snap.get(f) != os.stat(p).st_mtime_ns and not f.endswith(".pyc"):
            t.add(p, arcname=f)
print("RC", r.returncode)
'''


def colab(*a):
    return subprocess.run(["cmd", "/c", str(HERE / "colab.cmd"), *a], capture_output=True, text=True, encoding="utf-8", errors="replace")


def main(line):
    name = f"sched{int(time.time()) % 100000}"
    r = colab("new", "-s", name, "--gpu", "T4")
    print(r.stdout[-400:], r.stderr[-400:])
    if "READY" not in r.stdout + r.stderr:
        sys.exit(75)  # no free runtime right now; the scheduler records a failure and the job can be reset and retried
    rc = 1
    try:
        tmp = Path(tempfile.mkdtemp())
        (tmp / "boot.py").write_text(BOOT % dict(branch=BRANCH, line=line), encoding="utf-8")
        r = colab("exec", "-s", name, "--timeout", "10200", "-f", str(tmp / "boot.py"))
        print(r.stdout[-3000:], r.stderr[-1000:])
        m = re.findall(r"^RC (-?\d+)\s*$", r.stdout, re.M)
        rc = int(m[-1]) if m else 1
        d = colab("download", "-s", name, "/content/out.tar.gz", str(tmp / "out.tar.gz"))
        if (tmp / "out.tar.gz").exists():
            with tarfile.open(tmp / "out.tar.gz") as t:
                for m in t.getmembers():
                    if m.isfile() and "/" not in m.name and "\\" not in m.name and not m.name.startswith("."):
                        (PROBES / m.name).write_bytes(t.extractfile(m).read())
                print("files:", [m.name for m in t.getmembers()])
        else:
            print("download failed", d.stdout[-300:], d.stderr[-300:])
            rc = rc or 2
    finally:
        print(colab("stop", "-s", name).stdout[-200:])
    sys.exit(rc)


if __name__ == "__main__":
    main(sys.argv[1])
