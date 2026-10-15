import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.meaning import digest
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    reg=json.loads((ROOT/'reference/preregistration.json').read_text(encoding='utf8'))
    for p,h in reg['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError('Preregistered source changed: '+p)
    return digest(reg)
if __name__=='__main__':
    files=[p for folder in ['npnf','experiments','data','tests'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    files += [ROOT/'PREREGISTRATION.md',ROOT/'LITERATURE.md',ROOT/'scripts/prepare.py',ROOT/'scripts/freeze.py']
    reg=dict(frozenAt=datetime.now(timezone.utc).isoformat(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)})
    with (ROOT/'reference/preregistration.json').open('x',encoding='utf8') as f:json.dump(reg,f,ensure_ascii=False,indent=2)
    print('Frozen',len(files),'files',verify())
