import sys,itertools,numpy as np
sys.path.insert(0,sys.argv[1]+'/shim');sys.path.insert(0,sys.argv[1])
from npnf.core import TemporalNPN
from npnf.data import past_event,present_event,acoustic
from npnf.meaning import encode
def rms(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
res=[]
for seed,cap,prof in [(6311,0,0),(6312,2,1),(7901,4,3)]:
    hs=[]
    for p in range(4):
        n=TemporalNPN(seed=seed,capacity=cap,profile=prof)
        for i in range(8):n.update(acoustic(i))
        rows=[n.update(past_event(p,i,.08,seed)) for i in range(24)]
        n.advance(8);hs.append((n,rows))
    print(f'== seed{seed} cap{cap} profile{prof}')
    for p,(n,rows) in enumerate(hs):
        # learned target vs fixed linear map of outcome facts
        e=past_event(p,0)['interpretation'];o=encode(e);a=2*o[:8]-1
        lin=1.1*(a@n.base.numpy())
        tg=np.array([r['targetLogit'] for r in rows])
        print(f' past{p}: active={n.active_count()} target drift first->last RMS={rms(tg[0],tg[-1]):.2e}  target vs 1.1*base*facts RMS={rms(tg[-1],lin):.2e}')
    # kernel overlap between past cue centers and current cue
    for c in range(4):
        e=present_event(c,0,'past-0-0' if c==2 else None);cue=encode(e['interpretation'],observed=False)
        tg=[];ks=[]
        for p,(n,_) in enumerate(hs):
            m=n.clone()
            r=m.update(dict(e,links=[] ,kind='experience') if c==2 else e)
            tg.append(r['targetLogit'])
            idx,h=n._features(n.t(2*cue[None,:]-1),0);ks.append(float(np.max(h)) if len(idx) else 0)
        d=np.mean([rms(a,b) for a,b in itertools.combinations(tg,2)])
        print(f' current{c}: max RBF activation of current cue at past-learned centers={max(ks):.3f}; target difference across 4 histories RMS={d:.2e}')
