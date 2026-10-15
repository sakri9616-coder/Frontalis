"""Verify immutable old artifacts; never repair or overwrite originals."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
checks=[]
for manifest, base in [('phase3c-artifact-sha256.json',ROOT.parent/'frontalis-phase3c-lab'),('prior-sha256.json',None)]:
    records=json.loads((ROOT/'reference'/manifest).read_text(encoding='utf8'))
    changed=[];missing=[]
    for row in records:
        p=(base/row['path']) if base else Path(row['path'])
        if not p.is_file():missing.append(str(p))
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:changed.append(str(p))
    checks.append({'manifest':manifest,'files':len(records),'changed':changed,'missing':missing})
result={'success':all(not r['changed'] and not r['missing'] for r in checks),'checks':checks}
(ROOT/'reports/preservation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
if not result['success']:raise SystemExit(1)
