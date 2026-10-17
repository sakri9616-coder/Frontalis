"""Run ONE job line on a Modal T4 and copy the files it wrote back into the probes directory.
Used by sched.py (providers.json "modal"); also usable by hand:  uvx modal run sched/modal_runner.py --line "p21_cloud_equiv.py run x"
The job runs on a fresh clone of the pushed branch, so only committed and pushed code runs there."""
import sys
from pathlib import Path
import modal

app = modal.App("frontalis-sched", image=modal.Image.debian_slim(python_version="3.12").apt_install("git").pip_install("numpy", "torch"))
REPO, DIR = "https://github.com/sakri9616-coder/Frontalis.git", "/root/F/frontalis-v3-lab/proposals/probes"
MAX_BYTES = 40_000_000


@app.function(gpu="T4", timeout=7200)
def run(line: str, branch: str):
    import os, subprocess
    subprocess.run(f"git clone -q --branch {branch} --single-branch {REPO} /root/F", shell=True, check=True)
    commit = subprocess.run(f"cd {DIR} && git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    snap = {p: p.stat().st_mtime_ns for p in Path(DIR).glob("*") if p.is_file()}
    r = subprocess.run(f"cd {DIR} && PYTHONUTF8=1 python -B {line}", shell=True, capture_output=True, text=True)
    out, total = {}, 0
    for p in Path(DIR).glob("*"):
        if p.is_file() and snap.get(p) != p.stat().st_mtime_ns and p.suffix != ".pyc":
            b = p.read_bytes(); total += len(b)
            if total <= MAX_BYTES:
                out[p.name] = b
    return dict(rc=r.returncode, commit=commit, stdout=r.stdout[-3000:], stderr=r.stderr[-3000:], files=out)


@app.local_entrypoint()
def main(line: str, branch: str = "claude/value-representation-design"):
    here = Path(__file__).resolve().parent.parent
    res = run.remote(line, branch)
    for name, b in res["files"].items():
        (here / name).write_bytes(b)
    print(f"commit {res['commit']} rc={res['rc']} files={sorted(res['files'])}")
    print(res["stdout"]); print(res["stderr"], file=sys.stderr)
    sys.exit(res["rc"])
