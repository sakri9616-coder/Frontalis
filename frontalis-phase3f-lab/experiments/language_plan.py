import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnf.core import *
from npnf.data import *
from npnf.meaning import scored_fields
from scripts.freeze import verify
torch.set_num_threads(1)

def options():
    return [dict(id='repair',text='相手と修復に取り組む。',meaning=present_event(0)['interpretation']),
            dict(id='distance',text='相手から距離を置き、自分で方法を選んで探索する。',meaning=unrelated())]
def pack(n,id,text,meaning,language='ja'):
    opts=options();return dict(id=id,text=text,meaning=meaning,options=[{k:v for k,v in x.items() if k!='meaning'} for x in opts],language=language,toolResult=tool_evaluate(n,meaning,opts),history=[r['episode'] for r in n.episodes],modelConfig=n.config)

def dialogue():
    cases=[]
    for cap in range(5):
      for state in ['initial','cooperation','misunderstanding','corrected']:
        n=TemporalNPN.load(ROOT/'models'/f'dialogue-c{cap}-{state}')
        e=present_event(0);cases.append(pack(n,f'c{cap}-{state}',e['text'],e['interpretation']))
    n=TemporalNPN.load(ROOT/'models'/'dialogue-c2-corrected')
    for lang,text in [('ja','友人が謝罪し修復を手伝った。'),('en','My friend apologized and helped repair the damage.'),('fr','Mon ami a présenté ses excuses et aidé à réparer les dégâts.'),('es','Mi amigo se disculpó y ayudó a reparar el daño.')]:
        cases.append(pack(n,'language-'+lang,text,present_event(0)['interpretation'],lang))
    json_write(ROOT/'data/dialogue-plan.json',dict(sourceHash=verify(),cases=cases))

def extraction(engine):
    raw=json.loads((ROOT/f'reports/extract-{engine}.json').read_text(encoding='utf8'));out={}
    for b in raw['batches']:
        if b['data']:
            for r in b['data']['records']:
                try:
                    if set(r)!={'id','facts','interpretation'}:raise ValueError('Unexpected extraction fields.')
                    validate(r['facts']);validate(r['interpretation']);out[r['id']]=r
                except (ValueError,KeyError,TypeError):pass
    return out

def pipeline(engine):
    gold=json.loads((ROOT/'data/natural.json').read_text(encoding='utf8'));extracted=extraction(engine);metrics=[];cases=[]
    for cap in range(5):
     for profile in [0,2]:
      for reverse in [False,True]:
        a=TemporalNPN(capacity=cap,profile=profile);b=TemporalNPN(capacity=cap,profile=profile);count=0
        for e in gold[::-1] if reverse else gold:
            a.update(e)
            if e['id'] in extracted:
                q=copy.deepcopy(e);q.update({k:extracted[e['id']][k] for k in ['facts','interpretation']});b.update(q);count+=1
            else:b.advance()
        errors=[]
        for i in range(4):
            m=present_event(i)['interpretation'];errors.append(float(np.mean(np.abs(np.asarray(a.evaluate(m)['longTerm'])-b.evaluate(m)['longTerm']))))
        metrics.append(dict(capacity=cap,profile=profile,reverse=reverse,learned=count,heldoutMAE=float(np.mean(errors))))
        save=ROOT/'models'/f'natural-{engine}-c{cap}-p{profile}-r{int(reverse)}';b.save(save)
        if not reverse:
            e=present_event(profile//2);cases.append(pack(b,f'natural-c{cap}-p{profile}',e['text'],e['interpretation']))
    json_write(ROOT/f'reports/natural-{engine}.json',dict(sourceHash=verify(),metrics=metrics))
    json_write(ROOT/f'data/pipeline-{engine}.json',dict(sourceHash=verify(),cases=cases))

if __name__=='__main__':
    verify()
    if sys.argv[1]=='dialogue':dialogue()
    else:pipeline(sys.argv[1])
