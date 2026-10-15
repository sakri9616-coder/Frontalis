"""Hash fixed deliverables; interactive user state is explicitly outside the seal."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];target=ROOT/'artifact-sha256.json';rows=[]
if target.exists():raise FileExistsError('Manifest already exists; do not silently reseal artifacts.')
key=os.environ.get('ANTHROPIC_API_KEY','').encode()
for directory,dirs,files in os.walk(ROOT):
    rel=Path(directory).relative_to(ROOT).as_posix()
    dirs[:]=[d for d in dirs if d not in ['__pycache__','.git'] and not (rel=='playground' and d=='states')]
    for name in sorted(files):
        p=Path(directory)/name
        if p==target:continue
        data=p.read_bytes()
        if key and key in data:raise ValueError('Secret detected; manifest refused.')
        rows.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
rows.sort(key=lambda r:r['path'])
target.write_text(json.dumps(dict(schema='frontalis-3e-artifact-v1',mutableExcluded=['playground/states/'],files=rows),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(sealedFiles=len(rows),bytes=sum(r['bytes'] for r in rows),secretMatches=0)))
