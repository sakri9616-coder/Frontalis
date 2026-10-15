"""Small executable self-learning demo: no LLM, API, or source-state writes."""
import argparse,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npne.core import Personality,experience
from npne.data import WIDTHS,training,context_pairs
from npne.meaning import encode
torch.set_num_threads(1)
p=argparse.ArgumentParser();p.add_argument('--capacity',type=int,choices=range(5),default=0);p.add_argument('--output');a=p.parse_args()
models=[];results=[];probe=context_pairs(count=1)[0][0]
for series in range(4):
    n=Personality(5311,WIDTHS['semantic44'][a.capacity],'semantic44','bounded_logit');initial=n.evaluate(encode(probe))
    for e in training(5311,series,12):n.update(experience(e,'semantic44'))
    formed=n.evaluate(encode(probe))
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'personality';n.save(path);restored=Personality.load(path);exact=n.parameter_hash()==restored.parameter_hash()
    models.append(n);results.append(dict(series=series,parameters=n.count(),experiences=n.generation,initial=initial['channels'],formed=formed['channels'],savedAndRestoredExactly=exact,parameterHash=n.parameter_hash()))
if a.output:
    target=Path(a.output).resolve()
    if target.exists():raise FileExistsError('Use a new output directory.')
    target.mkdir(parents=True)
    for i,n in enumerate(models):n.save(target/f'personality-{i}')
print(json.dumps(dict(mode='offline-demo-not-primary-experiment',results=results),ensure_ascii=False,indent=2))
