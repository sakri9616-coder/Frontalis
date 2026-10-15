"""Freeze source/data and verify existing phases without writing there."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for folder in ['npne','data','experiments','tools','tests','docs','scripts'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.json','.txt','.mjs','.md','.cjs']}
def verify():
    rows=json.loads((ROOT/'reference/prior-sha256.json').read_text(encoding='utf8'));changed=[];missing=[]
    for r in rows:
        p=Path(r['path'])
        if not p.is_file():missing.append(str(p))
        elif sha(p)!=r['sha256']:changed.append(str(p))
    return dict(files=len(rows),changed=changed,missing=missing,success=not changed and not missing)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='freeze':
        result=dict(frozenAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),preservation=verify(),resultsInspected=False)
        if not result['preservation']['success']:raise ValueError('Old sources changed.')
        with (ROOT/'reference/preregistration.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps({'frozenFiles':len(result['sources']),'preserved':result['preservation']['files']}))
    elif mode=='verify':
        frozen=json.loads((ROOT/'reference/preregistration.json').read_text(encoding='utf8'))
        changed=[p for p,h in frozen['sources'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
        result=dict(preregisteredSourceChanges=changed,preservation=verify())
        (ROOT/'reports/preservation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=False));sys.exit(bool(changed or not result['preservation']['success']))
