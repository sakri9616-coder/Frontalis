"""Small offline personality formation, save/restore and rollback example."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from npnd.core import Personality
from npnd.data import training,probes
from tools.evaluate import decision
torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'models/demo-1')
args=parser.parse_args();destination=args.output.resolve()
if destination.exists() or not destination.is_relative_to(ROOT/'models'):
    raise SystemExit('Choose a new directory under this lab models; existing data is never overwritten.')
query=next(p for p in probes() if p['id']=='promise-change')
request={'query_id':'demo','space':query['space'],'features':query['features'],'basis':'affinity'}
initial=Personality(width=16,loss='bounded_logit');initial.save(destination/'initial')
out={'question':query['user'],'initial':decision(initial,request),'formed':[]}
for person in range(4):
    model=initial.clone()
    for event in training(series=person,count=24):model.update(event)
    prefix=destination/f'p{person}';model.save(prefix)
    loaded=Personality.load(prefix);rolled_back=Personality.load(destination/'initial')
    out['formed'].append({'series':person,'snapshot':str(prefix),'readout':decision(loaded,request),
        'restoredHashExact':loaded.parameter_hash()==model.parameter_hash(),'rollbackExact':rolled_back.parameter_hash()==initial.parameter_hash()})
with (destination/'readouts.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'output':str(destination),'states':4,'allRestoreAndRollbackExact':all(x['restoredHashExact'] and x['rollbackExact'] for x in out['formed'])},ensure_ascii=False))
