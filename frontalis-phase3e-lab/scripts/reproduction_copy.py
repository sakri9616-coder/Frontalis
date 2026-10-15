"""Create a fresh adjacent source copy without old outputs or personalities."""
import hashlib,json,os,re,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name=sys.argv[1]
if not re.fullmatch(r'frontalis-phase3e-repro-[a-z0-9-]+',name):raise ValueError('Use an adjacent frontalis-phase3e-repro-<name> directory.')
dest=(ROOT.parent/name).resolve()
if dest.parent!=ROOT.parent.resolve() or dest.exists():raise ValueError('New adjacent directory required; no overwrite.')
reg=json.loads((ROOT/'reference/preregistration.json').read_text(encoding='utf8'))
for p,h in reg['sources'].items():
    if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source changed.')
paths=set(reg['sources'])
for folder in ['npne','experiments','scripts','tools','tests','examples','playground','docs']:
    for directory,dirs,files in os.walk(ROOT/folder):
        dirs[:]=[d for d in dirs if d not in ['__pycache__','states','demo-output']]
        for f in files:
            p=Path(directory)/f
            if p.suffix in ['.py','.mjs','.html','.md','.cjs']:paths.add(p.relative_to(ROOT).as_posix())
paths.update(['reference/preregistration.json','reference/prior-sha256.json','start-workbench.ps1'])
dest.mkdir()
for relative in sorted(paths):
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
(dest/'models').mkdir();(dest/'reports').mkdir()
(dest/'README.md').write_text('Fresh Phase 3-E reproduction copy. No old models/results/materialized language plans. Run main/stress/diagnostic/performance, then prepare_language.py materialize. See original CLAUDE_HANDOFF.md.\n',encoding='utf8')
print(json.dumps(dict(directory=str(dest),sourceFiles=len(paths),copiedPriorResults=False)))
