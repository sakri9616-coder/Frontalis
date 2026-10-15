"""Create an empty, isolated rerun lab; never copy previous answers or weights."""
import argparse
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('destination',type=Path)
args=parser.parse_args()
target=args.destination.resolve()
if target==ROOT or target.exists() or not target.is_relative_to(ROOT.parent):
    raise SystemExit('Choose a new, nonexistent directory inside the workspace.')
target.mkdir(parents=True)
for name in ('npnd','tools','scripts','experiments','tests','docs','examples'):
    shutil.copytree(ROOT/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('npn','source-models'):
    shutil.copytree(ROOT/'reference'/name,target/'reference'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ('phase3c-artifact-sha256.json','prior-sha256.json'):
    shutil.copy2(ROOT/'reference'/name,target/'reference'/name)
for name in ('README.md','REPORT.md','CLAUDE_HANDOFF.md','.gitignore'):
    if (ROOT/name).is_file():shutil.copy2(ROOT/name,target/name)
for name in ('reports','models','data','runtime'):(target/name).mkdir()
print(str(target))
