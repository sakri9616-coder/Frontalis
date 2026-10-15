"""Additional preregistered replication and long-run study; no API calls."""
import argparse,copy,hashlib,itertools,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnf.core import TemporalNPN,json_write
from npnf.dual_view import DualViewAgent
from npnf.data import GOOD,LOSS,acoustic,past_event,present_event,unrelated
from npnf.meaning import digest,encode
from scripts.freeze import verify as verify_primary
torch.set_num_threads(1)
SEEDS=[7901,7902,7903]
FREEZE=ROOT/'reference/replication-longrun-freeze.json'

def verify():
    primary=verify_primary();reg=json.loads(FREEZE.read_text(encoding='utf8'))
    for p,h in reg['sources'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Additional frozen source changed: '+p)
    return digest(reg),primary

def rms(a,b):return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def cue(n,m,space='relational'):return np.tanh(n.logits(encode(m,observed=False),space)[0])
def views(n,m):
    return dict(observed=n.evaluate(m)['longTerm'],dual50=DualViewAgent(n).evaluate(m)['longTerm'])

def cached(name,fn):
    h,primary=verify();p=ROOT/'reports/cache/replication-longrun'/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        out=json.loads(p.read_text(encoding='utf8'))
        if (out['sourceHash'],out['primaryHash'])!=(h,primary):raise ValueError('Cache provenance mismatch.')
        return out['result']
    start=time.perf_counter();out=fn();elapsed=time.perf_counter()-start
    json_write(p,dict(sourceHash=h,primaryHash=primary,elapsedSeconds=elapsed,result=out))
    print(name,round(elapsed,3),flush=True);return out

def replication():
    groups=[]
    for seed,cap,profile in itertools.product(SEEDS,range(5),range(4)):
        def condition():
            histories=[]
            for past in range(4):
                n=TemporalNPN(seed=seed,capacity=cap,profile=profile)
                for i in range(8):n.update(acoustic(i))
                for i in range(24):n.update(past_event(past,i,.08,seed))
                n.advance(8);branches=[]
                for current in range(4):
                    e=present_event(current,0,f'past-{past}-0' if current==2 else None)
                    before=views(n,e['interpretation']);b=n.clone()
                    for i in range(8):b.update(present_event(current,i,f'past-{past}-0' if current==2 else None))
                    branches.append(dict(current=current,before=before,after=views(b,e['interpretation'])))
                histories.append(dict(past=past,branches=branches,activeParameters=n.active_count()))
            metrics={}
            for phase,view in itertools.product(['before','after'],['observed','dual50']):
                diffs=[rms(a['branches'][c][phase][view],b['branches'][c][phase][view]) for c in range(4) for a,b in itertools.combinations(histories,2)]
                value=float(np.mean(diffs));metrics[phase+'-'+view]=dict(meanPastDifference=value,passed=value>.01)
            return dict(seed=seed,capacity=cap,profile=profile,histories=histories,metrics=metrics)
        groups.append(cached(f'rep-s{seed}-c{cap}-p{profile}',condition))
    return groups

def intervention(source,seed,cap,profile,mode,repeats):
    m=past_event(0)['interpretation'];u=unrelated();n=source
    baseline=TemporalNPN(seed=seed,capacity=cap,profile=profile,decay=mode,rate=.01)
    origin=cue(baseline,m);formed=cue(n,m);formation=rms(formed,origin)
    historical_hash=digest(n.episodes[0]);related_before=formed;outside=cue(n,u)
    audio=cue(n,acoustic()['interpretation'],'acoustic')
    idle=n.clone();idle.advance(1024)
    retention=rms(cue(idle,m),origin)/(formation+1e-12)
    decay_control=n.clone();decay_control.advance(256)
    b=n.clone();snapshots=[]
    for i in range(256):
        e=past_event(0,i+10000);e['text']='友人が善意で手伝ったが、結果は失敗となり損失が出た。'
        e['interpretation']['outcomeFacts8']=copy.deepcopy(LOSS);e['facts']=copy.deepcopy(e['interpretation'])
        row=b.update(e)
        if i in [0,7,63,255]:snapshots.append(dict(at=i+1,cue=cue(b,m).tolist(),targetLogit=row['targetLogit'],training=row['training']))
    final=cue(b,m);change=rms(final,related_before);spill=rms(cue(b,u),outside)
    learning_specific=rms(final,cue(decay_control,m))
    relearned_views=views(b,m);relearned_hash=b.parameter_hash()
    for i in range(16):b.update(past_event(0,i+20000))
    preserved=all(digest(x.episodes[0])==historical_hash for x in [b,idle,decay_control])
    if not preserved:raise ValueError('An earlier experience record changed.')
    restored=False;model=None
    if seed==SEEDS[0] and profile==0 and mode=='consolidation' and repeats==1024:
        model=f'models/replication-longrun/c{cap}-reacquired';prefix=ROOT/model
        b.save(prefix);loaded=TemporalNPN.load(prefix)
        restored=loaded.parameter_hash()==b.parameter_hash() and rms(cue(loaded,m),cue(b,m))<1e-12 and digest(loaded.episodes)==digest(b.episodes)
        if not restored:raise ValueError('Additional snapshot failed restoration.')
    return dict(repeats=repeats,formation=formation,retentionAfter1024=retention,relatedChange256=change,
        unrelatedChange256=spill,spillRatio=spill/(change+1e-12),learningSpecificChange256=learning_specific,
        plasticityPassed=change>.01,localityPassed=spill/(change+1e-12)<.5,learningSpecificPassed=learning_specific>.01,
        cueBefore=related_before.tolist(),cueAfter=final.tolist(),viewsBefore=views(n,m),viewsAfter=relearned_views,
        cueReacquired=cue(b,m).tolist(),reacquisitionChange=rms(cue(b,m),final),
        audioDrift256=rms(cue(decay_control,acoustic()['interpretation'],'acoustic'),audio),
        audioDriftRelearning=rms(cue(b,acoustic()['interpretation'],'acoustic'),audio),
        formedSaturatedFraction=float(np.mean(np.abs(formed)>.95)),afterSaturatedFraction=float(np.mean(np.abs(final)>.95)),
        weightAbsMax=float(n.weights.abs().max()),activeParameters=n.active_count(),snapshots=snapshots,
        oldRecordPreserved=preserved,numericallyFinite=True,savedModel=model,restored=restored,
        beforeHash=n.parameter_hash(),relearnedHash=relearned_hash,finalHash=b.parameter_hash())

def longrun():
    rows=[]
    for seed,cap,profile,mode in itertools.product(SEEDS,range(5),[0,2],['uniform','consolidation']):
        def condition():
            n=TemporalNPN(seed=seed,capacity=cap,profile=profile,decay=mode,rate=.01)
            for i in range(8):n.update(acoustic(i))
            points=[]
            for i in range(1024):
                n.update(past_event(0,i))
                if i+1 in [32,1024]:points.append(intervention(n,seed,cap,profile,mode,i+1))
            return dict(seed=seed,capacity=cap,profile=profile,decay=mode,rate=.01,checkpoints=points)
        rows.append(cached(f'long-s{seed}-c{cap}-p{profile}-{mode}',condition))
    return rows

def summarize(rep,long):
    replication={k:dict(passed=sum(r['metrics'][k]['passed'] for r in rep),total=len(rep),mean=float(np.mean([r['metrics'][k]['meanPastDifference'] for r in rep]))) for k in ['before-observed','before-dual50','after-observed','after-dual50']}
    long_summary=[]
    for mode,repeats in itertools.product(['uniform','consolidation'],[32,1024]):
        points=[p for r in long if r['decay']==mode for p in r['checkpoints'] if p['repeats']==repeats]
        long_summary.append(dict(mode=mode,repeats=repeats,total=len(points),
            plasticityPassed=sum(p['plasticityPassed'] for p in points),localityPassed=sum(p['localityPassed'] for p in points),
            learningSpecificPassed=sum(p['learningSpecificPassed'] for p in points),
            meanRetention=float(np.mean([p['retentionAfter1024'] for p in points])),
            meanLearningSpecificChange=float(np.mean([p['learningSpecificChange256'] for p in points])),
            meanRelatedChange=float(np.mean([p['relatedChange256'] for p in points])),
            meanSpillRatio=float(np.mean([p['spillRatio'] for p in points])),
            meanFormedSaturation=float(np.mean([p['formedSaturatedFraction'] for p in points])),
            meanGradientAtLast=float(np.mean([p['snapshots'][-1]['training']['gradientNorm'] for p in points])),
            allRecordsPreserved=all(p['oldRecordPreserved'] for p in points)))
    return dict(replication=replication,longrun=long_summary,
        replicationCriterionPassed=replication['before-dual50']['passed']/len(rep)>=.8,
        longrunCriterion={m:next(r for r in long_summary if r['mode']==m and r['repeats']==1024)['learningSpecificPassed']/30>=.8 for m in ['uniform','consolidation']},
        savedModels=sum(bool(p['savedModel']) for r in long for p in r['checkpoints']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['replication','longrun','all']);args=p.parse_args()
    if args.part=='replication':replication()
    elif args.part=='longrun':longrun()
    else:
        rep=replication();long=longrun();h,primary=verify()
        out=dict(type='Additional preregistered research after primary results',sourceHash=h,primaryHash=primary,seeds=SEEDS,
                 replication=rep,longrun=long,summary=summarize(rep,long))
        json_write(ROOT/'reports/replication-longrun.json',out);print(json.dumps(out['summary'],indent=2),flush=True)
