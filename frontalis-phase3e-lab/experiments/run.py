"""Reproducible, resumable predeclared experiment. New lab outputs only."""
import argparse,copy,datetime,gc,hashlib,json,os,platform,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from npne.core import Personality,experience
from npne.meaning import encode,digest
from npne.data import SEEDS,WIDTHS,training,random_meanings,context_pairs,stress_event,teacher

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def frozen():
    reg=ROOT/'reference/preregistration.json';r=json.loads(reg.read_text(encoding='utf8'))
    for p,h in r['sources'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Preregistered source changed: '+p)
    return hashlib.sha256(reg.read_bytes()).hexdigest()
def synchronize(device):
    if device=='cuda':torch.cuda.synchronize()
def model(seed,rep,cap,loss,device):return Personality(seed,WIDTHS[rep][cap],rep,loss,device)
def features(ms,rep,observed=True):return np.asarray([encode(m,rep,observed) for m in ms])
def stats(y):
    cov=np.cov(y,rowvar=False);e=np.maximum(np.linalg.eigvalsh(cov),0);p=e/e.sum() if e.sum()>1e-20 else np.ones(7)/7
    corr=np.corrcoef(y,rowvar=False)
    return dict(mixedEmotionFraction=float(np.mean(((y[:,4:]> .2).sum(axis=1)>=2))),effectiveRank=float(np.exp(-np.sum(p*np.log(p+1e-30)))) if e.sum()>1e-20 else 0.,outputDiversity=float(np.mean(np.std(y,axis=0))),saturation=float(np.mean(np.abs(y)>.98)),affinityCuriosityCorrelation=float(np.nan_to_num(corr[0,1])),affinityValueCorrelation=float(np.nan_to_num(corr[0,2])))
def cache(name,fn):
    p=ROOT/'reports/cache'/f'{name}.json'
    if p.exists():
        r=json.loads(p.read_text(encoding='utf8'))
        if r['preregistrationSha256']!=REG:raise ValueError('Cache provenance mismatch.')
        return r
    start=time.perf_counter();r=fn();r.update(preregistrationSha256=REG,elapsedSeconds=time.perf_counter()-start)
    write(p,r);print(json.dumps(dict(condition=name,elapsedSeconds=round(r['elapsedSeconds'],2))),flush=True)
    return r

def run_main(device):
    rows=[];ms=random_meanings(8801,128);pairs=context_pairs()
    for seed in SEEDS:
      for rep in WIDTHS:
       x=features(ms,rep);pa=features([a for a,b in pairs],rep);pb=features([b for a,b in pairs],rep)
       for cap in range(5):
        for loss in ['bounded_signal','bounded_logit']:
         for series in range(4):
          name=f'main-{seed}-{rep}-c{cap}-{loss}-h{series}'
          def run_one():
            n=model(seed,rep,cap,loss,device);initial=n.evaluate_batch(x);other=n.branch_hash('acoustic');before=n.parameter_hash()
            for e in training(seed,series):n.update(experience(e,rep))
            y=n.evaluate_batch(x);prefix=ROOT/'models'/name;n.save(prefix)
            restored=Personality.load(prefix,device='cpu');restore=float(np.max(np.abs(restored.evaluate_batch(x)-y)))
            neutral=dict(id='neutral-control',cue=[.5]*n.dim,observed=[.5]*n.dim,confirmed=True,origin='synthetic',space='relational',sourceHash=digest('neutral'))
            prior=n.parameter_hash();n.update(neutral);neutral_ok=prior==n.parameter_hash()
            out=dict(seed=seed,representation=rep,capacityIndex=cap,parameterCount=n.count(),loss=loss,series=series,modelPrefix=str(prefix.relative_to(ROOT)).replace('\\','/'),initialHash=before,formedHash=prior,change=float(np.mean(np.abs(y-initial))),contextSensitivity=float(np.mean(np.abs(n.evaluate_batch(pa)-n.evaluate_batch(pb)))),unrelatedBranchExact=other==n.branch_hash('acoustic'),neutralExact=neutral_ok,restoreMaxError=restore,outputs=y.tolist(),metrics=stats(y),lastUpdate=n.history[-2]['training'])
            del n,restored;gc.collect();return out
          rows.append(cache(name,run_one))
    # Learning-off controls receive the exact same 48 experience stream.
    controls=[]
    for seed in SEEDS:
      for rep in WIDTHS:
       for cap in range(5):
        def control():
            n=model(seed,rep,cap,'bounded_logit',device);n.config['learning']=False;before=n.parameter_hash()
            for e in training(seed,0):n.update(experience(e,rep))
            return dict(seed=seed,representation=rep,capacityIndex=cap,exact=before==n.parameter_hash(),events=n.generation)
        controls.append(cache(f'off-{seed}-{rep}-c{cap}',control))
    summary=[]
    for seed in SEEDS:
      for rep in WIDTHS:
       for cap in range(5):
        for loss in ['bounded_signal','bounded_logit']:
            group=[r for r in rows if (r['seed'],r['representation'],r['capacityIndex'],r['loss'])==(seed,rep,cap,loss)]
            y=np.asarray([r['outputs'] for r in group]);d=[float(np.mean(np.abs(y[i]-y[j]))) for i in range(4) for j in range(i+1,4)]
            summary.append(dict(seed=seed,representation=rep,capacityIndex=cap,loss=loss,historyDifferentiation=float(np.mean(d))))
    write(ROOT/'reports/main.json',dict(conditions=rows,learningOff=controls,differentiation=summary))

def run_stress(device):
    rows=[]
    for seed in SEEDS:
     for rep in WIDTHS:
      anchor=stress_event(seed,0)['meaning'];rng=np.random.default_rng(seed+765)
      related=[]
      for _ in range(32):
        m=copy.deepcopy(anchor);m['cueFacts8']=np.clip(np.asarray(anchor['cueFacts8'])+rng.normal(0,.04,8),0,1).tolist();m['outcomeFacts8']=m['cueFacts8'];related.append(m)
      distant=random_meanings(seed+786,32);x=features(related,rep);far=features(distant,rep);cue=encode(anchor,rep,False)
      for cap in range(5):
       for loss in ['bounded_signal','bounded_logit']:
        name=f'stress-{seed}-{rep}-c{cap}-{loss}'
        def run_one():
            n=model(seed,rep,cap,loss,device);other=n.branch_hash('acoustic');checkpoints=[];strong=None;strongfar=None;strongcue=None
            for i in range(1097):
                if i in [0,64,256,1000,1012,1048,1096]:
                    y=n.evaluate_batch(x);f=n.evaluate_batch(far);q=n.evaluate_batch([cue])[0]
                    record=dict(step=i,parameterHash=n.parameter_hash(),metrics=stats(y),cue=q.tolist(),related=y.tolist(),distant=f.tolist(),lastUpdate=n.history[-1]['training'] if n.history else None)
                    checkpoints.append(record)
                    if i==1000:strong=y.copy();strongfar=f.copy();strongcue=q.copy()
                if i<1096:n.update(experience(stress_event(seed,i,i>=1000),rep))
            n.save(ROOT/'models'/name)
            return dict(seed=seed,representation=rep,capacityIndex=cap,parameterCount=n.count(),loss=loss,checkpoints=checkpoints,oppositeCueChange=float(np.mean(np.abs(q-strongcue))),relatedChange=float(np.mean(np.abs(y-strong))),distantChange=float(np.mean(np.abs(f-strongfar))),unrelatedBranchExact=other==n.branch_hash('acoustic'),finalNorm=n.stable())
        rows.append(cache(name,run_one))
    write(ROOT/'reports/stress.json',dict(conditions=rows))

def basis(x,quadratic=False):
    f=2*x-1
    return np.column_stack([np.ones(len(f)),f]+[f[:,i]*f[:,j] for i in range(f.shape[1]) for j in range(i,f.shape[1])]) if quadratic else np.column_stack([np.ones(len(f)),f])
def baseline(x,y,test,kind):
    z=np.arctanh(np.clip(y,-.999999,.999999))
    if kind=='retrieval':
        dist=((test[:,None,:]-x[None,:,:])**2).mean(axis=2);w=np.exp(-dist/(2*.2**2));w/=w.sum(axis=1,keepdims=True)
        return np.tanh(w@z)
    a=basis(x,kind=='quadratic');b=basis(test,kind=='quadratic')
    # Dual ridge avoids a large 1035x1035 inverse, identical regularization.
    return np.tanh(b@a.T@np.linalg.solve(a@a.T+.01*np.eye(len(a)),z))
def run_diagnostic(device):
    rows=[]
    for seed in SEEDS:
      ms=random_meanings(seed+92000,384);full=features(ms,'semantic44')
      for kind in ['linear','quadratic','interaction']:
       y=teacher(full,kind,seed)
       for rep in WIDTHS:
        x=features(ms,rep)
        for cap in range(5):
         name=f'diagnostic-{seed}-{kind}-{rep}-c{cap}'
         def run_one():
            n=model(seed,rep,cap,'bounded_logit',device);n.fit_diagnostic(x[:192],y[:192],120)
            v=n.evaluate_batch(x[192:256]);t=n.evaluate_batch(x[256:]);return dict(seed=seed,teacher=kind,representation=rep,capacityIndex=cap,parameterCount=n.count(),validationMSE=float(np.mean((v-y[192:256])**2)),testMSE=float(np.mean((t-y[256:])**2)),trainingProtocol='fixed-lr-.025-120-epochs-no-selection')
         rows.append(cache(name,run_one))
        for simple in ['linear','quadratic','retrieval']:
         name=f'diagnostic-{seed}-{kind}-{rep}-{simple}'
         def run_simple():
            pred=baseline(x[:192],y[:192],x[192:],simple)
            return dict(seed=seed,teacher=kind,representation=rep,method=simple,validationMSE=float(np.mean((pred[:64]-y[192:256])**2)),testMSE=float(np.mean((pred[64:]-y[256:])**2)))
         rows.append(cache(name,run_simple))
    write(ROOT/'reports/diagnostic.json',dict(conditions=rows))

def run_performance():
    rows=[];calibration=[]
    for rep in WIDTHS:
     for cap in range(5):
      e=experience(training(5311,0,1)[0],rep);x=np.asarray([e['cue']]);cpu=model(5311,rep,cap,'bounded_logit','cpu');gpu=model(5311,rep,cap,'bounded_logit','cuda')
      initial_error=float(np.max(np.abs(cpu.evaluate_batch(x)-gpu.evaluate_batch(x))))
      for i in range(4):
        e=experience(training(5311,0,4)[i],rep);cpu.update(e);gpu.update(e)
      calibration.append(dict(representation=rep,capacityIndex=cap,initialMaxError=initial_error,fourUpdateMaxError=float(np.max(np.abs(cpu.evaluate_batch(x)-gpu.evaluate_batch(x))))))
      for device,n in [('cpu',cpu),('cuda',gpu)]:
        for _ in range(3):n.evaluate_batch(x)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        durations=[]
        for _ in range(20):
            synchronize(device);start=time.perf_counter();n.evaluate_batch(x);synchronize(device);durations.append(1000*(time.perf_counter()-start))
        update=[]
        for i in range(3):
            e=experience(training(5311,0,1)[0],rep);e['id']=f'benchmark-{i}';synchronize(device);start=time.perf_counter();n.update(e);synchronize(device);update.append(1000*(time.perf_counter()-start))
        mem=psutil.Process().memory_info();cuda=dict(allocatedBytes=torch.cuda.memory_allocated(),reservedBytes=torch.cuda.memory_reserved(),peakAllocatedBytes=torch.cuda.max_memory_allocated(),peakReservedBytes=torch.cuda.max_memory_reserved()) if device=='cuda' else None
        rows.append(dict(representation=rep,capacityIndex=cap,parameterCount=n.count(),device=device,weightBytes=sum(a.nbytes for a in n.arrays().values()),evaluateMedianMs=statistics.median(durations),evaluateP95Ms=float(np.percentile(durations,95)),updateMedianMs=statistics.median(update),rssBytes=mem.rss,privateBytes=getattr(mem,'private',None),cudaAllocator=cuda))
      # Save same GPU values and read on CPU: differs from separately retraining.
      p=ROOT/'models'/f'portability-{rep}-c{cap}';gpu.save(p);loaded=Personality.load(p)
      calibration[-1].update(sameWeightHash=loaded.parameter_hash()==gpu.parameter_hash(),sameWeightMaxError=float(np.max(np.abs(loaded.evaluate_batch(x)-gpu.evaluate_batch(x)))))
      del cpu,gpu,loaded;gc.collect();torch.cuda.empty_cache()
    write(ROOT/'reports/performance.json',dict(conditions=rows,calibration=calibration,processMemoryCaveat='Sequential process RSS/private include both devices and runtime; allocator is not whole-device VRAM.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['main','stress','diagnostic','performance']);p.add_argument('--device',default='cuda',choices=['cuda','cpu']);args=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);REG=frozen()
    runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device=args.device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,dtype='float64',threads=torch.get_num_threads(),platform=platform.platform(),preregistrationSha256=REG,startedAt=datetime.datetime.now(datetime.timezone.utc).isoformat())
    runtime_path=ROOT/'reports'/f'{args.mode}-runtime.json'
    if runtime_path.exists():
        old=json.loads(runtime_path.read_text(encoding='utf8'))
        if any(old[k]!=runtime[k] for k in ['torch','numpy','device','dtype','threads','preregistrationSha256']):raise ValueError('Cannot resume with a different runtime.')
    else:write(runtime_path,runtime)
    {'main':lambda:run_main(args.device),'stress':lambda:run_stress(args.device),'diagnostic':lambda:run_diagnostic(args.device),'performance':run_performance}[args.mode]()
