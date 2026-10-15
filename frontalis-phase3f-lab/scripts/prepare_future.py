import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'experiments/live.mjs').read_text(encoding='utf8')
source=source.replace("['dialogue','pipeline']","['dialogue','pipeline','future']").replace('`pipeline-${engine}`','`${mode}-${engine}`')
(ROOT/'experiments/live-future.mjs').write_text(source,encoding='utf8')
paths=['experiments/future_choice.py','experiments/live-future.mjs','reference/future-choice.md','scripts/prepare_future.py']
with (ROOT/'reference/future-choice-freeze.json').open('x',encoding='utf8') as f:
    json.dump({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},f,indent=2)
