"""Prespecified numerical comparisons. Each condition has an immutable cache."""
import argparse,hashlib,json,os,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnf.core import TemporalNPN,SEEDS,WIDTHS,json_write
from npnf.data import *
from npnf.meaning import encode,digest
from scripts.freeze import verify
torch.set_num_threads(1)

def rms(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def probe(n,m,space='relational',cue=False):return np.tanh(n.logits(encode(m,observed=not cue),space)[0])
def cached(group,name,fn):
    p=ROOT/'reports/cache'/group/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        out=json.loads(p.read_text(encoding='utf8'))
        if out['sourceHash']!=verify():raise ValueError('Cache provenance mismatch.')
        return out['result']
    start=time.perf_counter();out=fn();json_write(p,dict(sourceHash=verify(),elapsedSeconds=time.perf_counter()-start,result=out))
    print(group,name,round(time.perf_counter()-start,2),flush=True);return out
def save(n,name):
    p=ROOT/'models'/name
    if not Path(str(p)+'.json').exists():n.save(p)
def blank(n,k):n.advance(k);return n

def timelines():
    rows=[]
    for seed in SEEDS:
      for cap in range(5):
       for profile in range(4):
        for past in range(4):
         def condition():
            cfg=dict(seed=seed,capacity=cap,profile=profile)
            n=TemporalNPN(**cfg)
            if seed==SEEDS[0] and profile==0 and past==0:save(n,f'dialogue-c{cap}-initial')
            for i in range(8):n.update(acoustic(i))
            acoustic_before=probe(n,acoustic()['interpretation'],'acoustic')
            for i in range(24):n.update(past_event(past,i,.08,seed))
            immediate_short=np.linalg.norm(n.short);immediate_mood=np.linalg.norm(n.mood)
            relation=present_event(0)['interpretation'];initial=TemporalNPN(**cfg)
            formed=probe(n,relation,cue=True);start=probe(initial,relation,cue=True)
            idle=n.clone();idle.advance(16)
            formed_effect=rms(formed,start)
            retained=rms(probe(idle,relation,cue=True),start)/(formed_effect+1e-12)
            n.advance(8)
            if seed==SEEDS[0] and profile==0 and past in [0,2]:save(n,f'dialogue-c{cap}-'+('cooperation' if past==0 else 'misunderstanding'))
            branches=[]
            for now in range(4):
                e=present_event(now,0,'past-'+str(past)+'-0' if now==2 else None)
                b=n.clone();pre=b.evaluate(e['interpretation']);pre_cue=probe(b,e['interpretation'],cue=True).tolist()
                for i in range(8):b.update(present_event(now,i,'past-'+str(past)+'-0' if now==2 else None))
                reverse=TemporalNPN(**cfg)
                for i in range(8):reverse.update(acoustic(i))
                for i in range(8):reverse.update(present_event(now,i))
                for i in range(24):reverse.update(past_event(past,i,.08,seed))
                reverse.advance(8)
                post=b.evaluate(e['interpretation']);rev=reverse.evaluate(e['interpretation'])
                if seed==SEEDS[0] and profile==0 and past==2 and now==2:save(b,f'dialogue-c{cap}-corrected')
                branches.append(dict(current=now,before=pre,preCue=pre_cue,after=post,reverse=rev,orderDifference=rms(post['longTerm'],rev['longTerm'])))
            return dict(seed=seed,capacity=cap,profile=profile,past=past,parameterCount=n.count(),activeCount=n.active_count(),
                branches=branches,shortRatio16=float(np.linalg.norm(idle.short)/(immediate_short+1e-12)),moodRatio16=float(np.linalg.norm(idle.mood)/(immediate_mood+1e-12)),
                longTermRetention16=retained,formedEffect=formed_effect,acousticDrift=rms(acoustic_before,probe(n,acoustic()['interpretation'],'acoustic')))
         rows.append(cached('timelines',f's{seed}-c{cap}-p{profile}-h{past}',condition))
    return rows

def trajectory(seed,cap,decay,rate,test=False):
    n=TemporalNPN(seed=seed,capacity=cap,decay=decay,rate=rate)
    p=1 if test else 0;q=0 if test else 1
    m=past_event(p)['interpretation'];u=unrelated();origin=probe(n,m,cue=True)
    for i in range(24):n.update(past_event(p,i,.06,seed))
    formed=probe(n,m,cue=True);formed_effect=rms(formed,origin)
    n.advance(64 if test else 32);idle=probe(n,m,cue=True)
    before=probe(n,m,cue=True);outside=probe(n,u)
    for i in range(32 if test else 24):
        e=past_event(p,i+100,.06,seed);e['interpretation']['outcomeFacts8']=GOOD if q==0 else LOSS;e['facts']=copy.deepcopy(e['interpretation']);n.update(e)
    change=rms(probe(n,m,cue=True),before);unrelated_change=rms(probe(n,u),outside)
    return dict(retention=rms(idle,origin)/(formed_effect+1e-12),formation=formed_effect,relearningChange=change,unrelatedChange=unrelated_change,
                spillRatio=unrelated_change/(change+1e-12),finite=True,activeCount=n.active_count()),n

def decay_search():
    rows=[]
    for seed in SEEDS:
      for cap in range(5):
       def condition():
        validation=[]
        for mode in ['uniform','selective','consolidation']:
         for rate in [.002,.01,.05]:
            r,_=trajectory(seed,cap,mode,rate);validation.append(dict(mode=mode,rate=rate,**r))
        eligible=[v for v in validation if v['relearningChange']>.01 and v['spillRatio']<.5]
        selected=max(eligible,key=lambda v:(v['retention'],-v['rate'])) if eligible else next(v for v in validation if v['mode']=='consolidation' and v['rate']==.01)
        tests=[]
        for mode in ['uniform','selective','consolidation']:
            r,n=trajectory(seed,cap,mode,selected['rate'],True);tests.append(dict(mode=mode,rate=selected['rate'],**r))
            if mode==selected['mode']:save(n,f'decay-s{seed}-c{cap}')
        return dict(seed=seed,capacity=cap,validation=validation,selected=selected,selectionEligible=bool(eligible),test=tests)
       rows.append(cached('decay',f's{seed}-c{cap}',condition))
    return rows

def stress():
    rows=[]
    for seed in SEEDS:
     for cap in range(5):
      for learner in ['local','dense']:
       for repeats in [32,512]:
        def condition():
            n=TemporalNPN(seed=seed,capacity=cap,learner=learner);m=past_event(1)['interpretation'];u=unrelated()
            for i in range(8):n.update(acoustic(i))
            for i in range(repeats):n.update(past_event(1,i))
            before=probe(n,m,cue=True);away=probe(n,u);a=probe(n,acoustic()['interpretation'],'acoustic');snapshots=[]
            for i in range(64):
                e=past_event(1,i+10000);e['interpretation']['outcomeFacts8']=GOOD;e['facts']=copy.deepcopy(e['interpretation']);r=n.update(e)
                if i in [0,7,15,31,63]:snapshots.append(dict(at=i+1,evaluation=probe(n,m,cue=True).tolist(),training=r['training']))
            change=rms(probe(n,m,cue=True),before);spill=rms(probe(n,u),away);acoustic_drift=rms(probe(n,acoustic()['interpretation'],'acoustic'),a)
            for i in range(16):n.update(past_event(1,i+20000))
            reacquired=probe(n,m,cue=True).tolist();n.advance(64)
            save(n,f'stress-s{seed}-c{cap}-{learner}-{repeats}')
            return dict(seed=seed,capacity=cap,learner=learner,repeats=repeats,before=before.tolist(),snapshots=snapshots,
                        relearningChange=change,unrelatedChange=spill,spillRatio=spill/(change+1e-12),acousticDrift=acoustic_drift,
                        reacquired=reacquired,afterIdle=probe(n,m,cue=True).tolist(),activeCount=n.active_count(),finite=True)
        rows.append(cached('stress',f's{seed}-c{cap}-{learner}-{repeats}',condition))
    return rows

def mechanisms():
    rows=[]
    for seed in SEEDS:
     for cap in range(5):
      for mode in ['idle','rumination','correction','extinction','regulation']:
       def condition():
        n=TemporalNPN(seed=seed,capacity=cap);m=past_event(2)['interpretation'];u=unrelated()
        for i in range(24):n.update(past_event(2,i))
        old_record=copy.deepcopy(n.episodes[0]);before=probe(n,m,cue=True);timeline=[]
        for i in range(32):
            if mode=='idle':n.advance()
            elif mode=='rumination':n.reactivate('past-2-0','recall-'+str(i))
            elif mode=='correction':n.update(present_event(2,i,'past-2-0'))
            elif mode=='extinction':
                e=past_event(2,i+100);e['interpretation']['outcomeFacts8']=GOOD;e['interpretation']['events'][0]['actor']='stranger';e['facts']=copy.deepcopy(e['interpretation']);n.update(e)
            else:n.advance();n.short*=.5;n.mood*=.5
            timeline.append(dict(at=i+1,short=float(np.linalg.norm(n.short)),mood=float(np.linalg.norm(n.mood)),originalContext=probe(n,m,cue=True).tolist()))
        original=probe(n,m,cue=True);alternate=copy.deepcopy(m);alternate['events'][0]['actor']='stranger'
        return dict(seed=seed,capacity=cap,mode=mode,before=before.tolist(),after=original.tolist(),alternate=probe(n,alternate,cue=True).tolist(),
                    unrelated=probe(n,u).tolist(),timeline=timeline,recordPreserved=n.episodes[0]==old_record,change=rms(original,before),episodeCount=len(n.episodes))
       rows.append(cached('mechanisms',f's{seed}-c{cap}-{mode}',condition))
    return rows

def diagnostic():
    rows=[]
    for seed in SEEDS:
      def condition():
        rng=np.random.default_rng(seed+9000);x=rng.uniform(.02,.98,(272,44))
        z=.4*(2*x[:,:7]-1)+np.column_stack([np.sin(4*x[:,j])*np.cos(4*x[:,j+8])+.6*(2*x[:,j+16]-1)*(2*x[:,j+24]-1) for j in range(7)])
        y=np.tanh(z);train=slice(0,128);val=slice(128,176);test=slice(176,None)
        out=[]
        for cap in range(5):
            n=TemporalNPN(seed=seed,capacity=cap,decay='none')
            for epoch in range(6):
                for i in range(128):n._learn(x[i],z[i],1.,0)
            pred=np.tanh(n.logits(x));out.append(dict(kind='npn',capacity=cap,parameters=n.count(),activeCount=n.active_count(),validationMSE=float(np.mean((pred[val]-y[val])**2)),testMSE=float(np.mean((pred[test]-y[test])**2))))
            save(n,f'diagnostic-s{seed}-c{cap}')
        for kind in ['linear','quadratic']:
            f=np.column_stack([np.ones(len(x)),2*x-1]);
            if kind=='quadratic':
                a,b=np.triu_indices(44);f=np.column_stack([f,(2*x[:,a]-1)*(2*x[:,b]-1)])
            k=f[train]@f[train].T;choices=[]
            for alpha in [.001,.1,10]:
                coeff=np.linalg.solve(k+alpha*np.eye(128),y[train]);prediction=f@f[train].T@coeff
                choices.append((float(np.mean((prediction[val]-y[val])**2)),alpha,prediction))
            score,alpha,pred=min(choices,key=lambda r:r[0]);out.append(dict(kind=kind,alpha=alpha,parameters=int(f.shape[1]*7),validationMSE=score,testMSE=float(np.mean((pred[test]-y[test])**2))))
        distance=((x[:,None,:]-x[None,:128,:])**2).mean(2);pred=y[:128][distance.argmin(1)]
        out.append(dict(kind='retrieval',storedExamples=128,validationMSE=float(np.mean((pred[val]-y[val])**2)),testMSE=float(np.mean((pred[test]-y[test])**2))))
        return dict(seed=seed,rows=out)
      rows.append(cached('diagnostic',f's{seed}',condition))
    return rows

def controls():
    rows=[]
    for seed in SEEDS:
     for cap in range(5):
      for mode in ['learning-off','affect-off']:
       def condition():
        n=TemporalNPN(seed=seed,capacity=cap,learning=mode!='learning-off',emotional=mode!='affect-off');before=n.parameter_hash()
        for i in range(24):n.update(past_event(1,i))
        emotion=np.linalg.norm(n.short)+np.linalg.norm(n.mood);n.advance(32)
        return dict(seed=seed,capacity=cap,mode=mode,weightsUnchanged=before==n.parameter_hash(),emotionNormBeforeIdle=float(emotion),emotionNormAfterIdle=float(np.linalg.norm(n.short)+np.linalg.norm(n.mood)))
       rows.append(cached('controls',f's{seed}-c{cap}-{mode}',condition))
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('group',choices=['timelines','decay','stress','mechanisms','diagnostic','controls','all']);args=parser.parse_args()
    verify()
    for group,fn in [('timelines',timelines),('decay',decay_search),('stress',stress),('mechanisms',mechanisms),('diagnostic',diagnostic),('controls',controls)]:
        if args.group in [group,'all']:
            p=ROOT/'reports'/(group+'.json')
            if not p.exists():json_write(p,dict(sourceHash=verify(),rows=fn()))
