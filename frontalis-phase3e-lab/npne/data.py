"""Predeclared synthetic conditions, separate from authored natural language."""
import copy
import numpy as np
from .meaning import meaning,event,encode

SEEDS=(5311,5312,5313)
WIDTHS={'raw8':(16,64,312,3125,31250),'semantic44':(5,19,96,962,9615)}

def training(seed,series,count=48):
    rng=np.random.default_rng(seed+8000)
    rows=[]
    for i in range(count):
        # Same cues across all four histories; only outcome distributions differ.
        cue=rng.uniform(.15,.85,8)
        e=event('help' if i%2==0 else 'explore',actor='close_other' if i%3 else 'self',target='self' if i%3 else 'object')
        consent='yes' if i%4<2 else 'no'
        out=cue.copy()
        success=(series%2==0) if (series<2 or i%2) else (series%2!=0)
        out[3]=.05 if success else .95;out[4]=.95 if success else .05
        out[5]=.9 if success and i%2 else .1
        out=np.clip(out+rng.normal(0,.025,8),0,1)
        m=meaning([e],cue.tolist(),out.tolist(),consent=consent,trust='high' if i%3 else 'unknown',causal='causes',temporal='after')
        rows.append({'id':f's{seed}-h{series}-e{i}','text':f'架空の確認済み経験 {i}：行為={e["action"]}、同意={consent}。結果の事実ベクトルを別途記録。','origin':'synthetic','confirmed':True,'space':'relational','meaning':m})
    return rows

def random_meanings(seed,count):
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(count):
        events=[event(action=str(rng.choice(['help','restrict','deceive','explore','create','other'])),actor=str(rng.choice(['self','close_other','stranger','institution'])),target=str(rng.choice(['self','other','object'])),negated=bool(rng.integers(2)),uncertain=bool(rng.integers(2)),intent=str(rng.choice(['benefit','harm','unknown']))) for _ in range(1+i%2)]
        rows.append(meaning(events,rng.uniform(0,1,8).tolist(),rng.uniform(0,1,8).tolist(),consent=str(rng.choice(['yes','no','unknown'])),trust=str(rng.choice(['high','low','unknown'])),causal=str(rng.choice(['causes','prevents','unrelated','unknown'])),temporal=str(rng.choice(['after','simultaneous','unknown'])),delayed=bool(rng.integers(2))))
    return rows

def context_pairs(seed=8701,count=32):
    rows=[]
    for m in random_meanings(seed,count):
        a=copy.deepcopy(m);b=copy.deepcopy(m)
        a['consent']='yes';b['consent']='no'
        a['events'][0]['intent']='benefit';b['events'][0]['intent']='harm'
        rows.append((a,b))
    return rows

def stress_event(seed,index,contrary=False):
    cue=[.8,.2,.7,.2,.6,.7,.3,.8]
    outcome=[.8,.1,.7,.05,.95,.9,.2,.8] if not contrary else [.1,.9,.7,.95,.05,.1,.8,.8]
    m=meaning([event('help')],cue,outcome,consent='yes',trust='high',causal='causes',temporal='after')
    return dict(id=f'stress-{seed}-{index}-{contrary}',text='架空の協力の確認済み結果。反対結果の場合も主観教師を付けない。',origin='synthetic',confirmed=True,space='relational',meaning=m)

def teacher(x,kind,seed):
    rng=np.random.default_rng(seed+94000)
    z=2*np.asarray(x)-1
    w=rng.normal(0,.11,(44,7))
    logits=z@w
    if kind=='quadratic':
        logits += .35*(z[:,[0,3,8,16,22,30,36]]*z[:,[4,5,22,30,37,40,43]])
    elif kind=='interaction':
        # Explicit nonlinear interaction task; favors representations containing these facts.
        logits += .7*np.tanh(2*z[:,[8,9,10,14,22,36,40]]*z[:,[22,23,24,28,8,37,41]])
    return np.tanh(logits)
