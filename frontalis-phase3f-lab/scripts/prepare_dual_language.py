import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'experiments/live-future.mjs').read_text(encoding='utf8').replace("['dialogue','pipeline','future']","['dialogue','pipeline','future','dual']")
(ROOT/'experiments/live-dual.mjs').write_text(source,encoding='utf8')
paths=['experiments/live-dual.mjs','scripts/prepare_dual_language.py']
with (ROOT/'reference/dual-language-freeze.json').open('x',encoding='utf8') as f:
    json.dump({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},f,indent=2)
