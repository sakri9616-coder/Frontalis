"""Run ONE job line on a free Kaggle GPU kernel (private) through the official kaggle CLI, copy the files it wrote back into the
probes directory. Used by sched.py (providers.json "kaggle"):  python sched/kaggle_runner.py "<job line>"
Needs KAGGLE_API_TOKEN (user environment variable; never printed or stored here) and a phone-verified Kaggle account for
internet access inside the kernel (git clone). Free quota is about 30 GPU-hours per week; sched.py does not track it.
Like modal_runner.py it runs a fresh clone of the pushed branch, so only committed and pushed code runs there."""
import json, os, re, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

PROBES = Path(__file__).resolve().parent.parent
BRANCH = "claude/value-representation-design"
USER = "sakuri9616"
SCRIPT = '''import subprocess, os, tarfile
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True)
r = sh("git clone -q --branch %(branch)s --single-branch https://github.com/sakri9616-coder/Frontalis.git /kaggle/working/F")
assert r.returncode == 0, r.stderr
D = "/kaggle/working/F/frontalis-v3-lab/proposals/probes"
snap = {f: os.stat(os.path.join(D, f)).st_mtime_ns for f in os.listdir(D) if os.path.isfile(os.path.join(D, f))}
r = sh("cd %%s && PYTHONUTF8=1 python -B %(line)s" %% D)
print("COMMIT", sh("cd %%s && git rev-parse --short HEAD" %% D).stdout.strip())
print(r.stdout[-2000:]); print(r.stderr[-2000:])
with tarfile.open("/kaggle/working/out.tar.gz", "w:gz") as t:
    for f in os.listdir(D):
        p = os.path.join(D, f)
        if os.path.isfile(p) and snap.get(f) != os.stat(p).st_mtime_ns and not f.endswith(".pyc"):
            t.add(p, arcname=f)
sh("rm -rf /kaggle/working/F")
print("RC", r.returncode)
'''


def kg(*a):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if "KAGGLE_API_TOKEN" not in env:  # a session started before the variable was registered does not have it yet
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            env["KAGGLE_API_TOKEN"] = winreg.QueryValueEx(k, "KAGGLE_API_TOKEN")[0]
    try:
        return subprocess.run(["uvx", "kaggle", *a], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=300)
    except subprocess.TimeoutExpired:  # a hung network call must not block the job forever
        return subprocess.CompletedProcess(a, 124, "", "timeout")


def status(ref):
    m = re.search(r'kernelworkerstatus\.(\w+)', kg("kernels", "status", ref).stdout.lower())
    return m.group(1) if m else "unknown"


def main(line):
    slug = f"frontalis-sched-{int(time.time()) % 1000000}"
    ref = f"{USER}/{slug}"
    d = Path(tempfile.mkdtemp())
    (d / "run.py").write_text(SCRIPT % dict(branch=BRANCH, line=line), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps(dict(
        id=ref, title=slug, code_file="run.py", language="python", kernel_type="script",
        is_private="true", enable_gpu="true", enable_internet="true", dataset_sources=[], competition_sources=[], kernel_sources=[])), encoding="utf-8")
    r = kg("kernels", "push", "-p", str(d))
    print(r.stdout[-500:], r.stderr[-500:])
    if r.returncode != 0:
        sys.exit(75)
    rc = 1
    try:
        t0, s = time.time(), "unknown"
        while time.time() - t0 < 3 * 3600:
            time.sleep(20)
            s = status(ref)
            if s in ("complete", "error", "cancelacknowledged"):
                break
        print("status:", s)
        out = Path(tempfile.mkdtemp())
        kg("kernels", "output", ref, "-p", str(out))
        log = next(iter(out.glob("*.log")), None)
        if log:
            print(log.read_text(encoding="utf-8", errors="replace")[-1500:])
        tar = out / "out.tar.gz"
        if tar.exists():
            with tarfile.open(tar) as t:
                for m in t.getmembers():
                    if m.isfile() and "/" not in m.name and "\\" not in m.name and not m.name.startswith("."):
                        (PROBES / m.name).write_bytes(t.extractfile(m).read())
                print("files:", [m.name for m in t.getmembers()])
            rc = 0 if s == "complete" else 1
    finally:
        kg("kernels", "delete", ref, "-y")  # the kernel is private either way; delete is best effort
    sys.exit(rc)


if __name__ == "__main__":
    main(sys.argv[1])
