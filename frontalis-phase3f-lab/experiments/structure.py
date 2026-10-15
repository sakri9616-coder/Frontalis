import copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'frontalis-phase3e-lab'))
import numpy as np
import torch
from npne.core import Personality,experience
from npnf.core import TemporalNPN,SEEDS,json_write
from npnf.data import past_event,GOOD,unrelated
from npnf.meaning import encode
from scripts.freeze import verify
torch.set_num_threads(1)

if __name__=='__main__':
    verify();rows=[]
    for seed in SEEDS:
     for cap,width in enumerate([5,19,96,962,9615]):
      for view in ['full','facts8-neutral-context']:
        name=f's{seed}-c{cap}-{view}';p=ROOT/'reports/cache/structure'/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():rows.append(json.loads(p.read_text(encoding='utf8')));continue
        n=Personality(seed=seed,width=width,representation='semantic44');n.base=n.t(TemporalNPN(seed=seed).base.cpu().numpy());n._hash=None
        def update(i,good=False):
            row=past_event(1,i);row=dict(id=row['id'],text=row['text'],meaning=row['interpretation'],space='relational',confirmed=True,origin='synthetic')
            if good:row['meaning']['outcomeFacts8']=GOOD
            e=experience(row,'semantic44')
            if view!='full':e['observed'][8:]=[.5]*36
            return n.update(e)
        for i in range(32):update(i)
        q=encode(past_event(1)['interpretation'],observed=False);u=encode(unrelated());before=n.evaluate_batch([q,u])
        for i in range(64):update(i+1000,True)
        after=n.evaluate_batch([q,u]);change=np.sqrt(np.mean((after-before)**2,axis=1))
        r=dict(seed=seed,capacity=cap,parameters=n.count(),view=view,relearningChange=float(change[0]),unrelatedChange=float(change[1]),spillRatio=float(change[1]/(change[0]+1e-12)),before=before.tolist(),after=after.tolist())
        n.save(ROOT/'models'/('structure-'+name));json_write(p,r);rows.append(r);print(name,flush=True)
    json_write(ROOT/'reports/structure.json',dict(sourceHash=verify(),rows=rows,scope='Additional structure-plus-learning comparison; not an isolated architecture effect.'))
