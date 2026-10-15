import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    name=p.relative_to(ROOT).as_posix()
    if name=='artifact-sha256.json' or name.startswith('playground/states/') or '__pycache__' in name:continue
    files.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
with (ROOT/'artifact-sha256.json').open('w',encoding='utf8') as f:
    json.dump(dict(schema='frontalis-3f-artifact-v1',mutableExcluded=['playground/states/'],files=files),f,ensure_ascii=False,indent=2)
print('Sealed',len(files),'files;',sum(p['bytes'] for p in files),'bytes.')
