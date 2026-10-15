import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnf.core import *
from npnf.dual_view import DualViewAgent
from npnf.data import *
from experiments.language_plan import pack
torch.set_num_threads(1)

if __name__=='__main__':
    rows=[];cases=[]
    for seed in SEEDS:
     for cap in range(5):
      for past in range(4):
        n=DualViewAgent(seed=seed,capacity=cap);control=TemporalNPN(seed=seed,capacity=cap)
        if seed==SEEDS[0] and past==0:
            n.save(ROOT/'models'/f'dual-c{cap}-initial');cases.append(pack(n,f'dual-c{cap}-initial',NOW[0],present_event(0)['interpretation']))
        for i in range(8):n.update(acoustic(i));control.update(acoustic(i))
        for i in range(24):
            e=past_event(past,i,.08,seed);n.update(e);control.update(e)
        n.advance(8);control.advance(8)
        identical=n.n.parameter_hash()==control.parameter_hash()
        if seed==SEEDS[0] and past in [0,2]:
            state='cooperation' if past==0 else 'misunderstanding';n.save(ROOT/'models'/f'dual-c{cap}-{state}');cases.append(pack(n,f'dual-c{cap}-{state}',NOW[0],present_event(0)['interpretation']))
        branches=[]
        for current in range(4):
            e=present_event(current,0,f'past-{past}-0' if current==2 else None);before=n.evaluate(e['interpretation']);b=n.clone()
            for i in range(8):b.update(present_event(current,i,f'past-{past}-0' if current==2 else None))
            branches.append(dict(current=current,before=before,after=b.evaluate(e['interpretation'])))
            if seed==SEEDS[0] and past==2 and current==2:
                b.save(ROOT/'models'/f'dual-c{cap}-corrected');cases.append(pack(b,f'dual-c{cap}-corrected',NOW[0],present_event(0)['interpretation']))
        rows.append(dict(seed=seed,capacity=cap,past=past,branches=branches,weightsMatchObservedControl=identical,activeCount=n.active_count()))
        print(seed,cap,past,flush=True)
    json_write(ROOT/'reports/dual-view.json',dict(type='Post hoc exploratory architecture',rows=rows))
    for engine in ['haiku','deepseek']:json_write(ROOT/f'data/dual-{engine}.json',dict(cases=cases))
