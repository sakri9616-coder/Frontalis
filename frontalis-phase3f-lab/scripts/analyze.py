"""Derived analysis; never changes frozen experiments or their cached outputs."""
import collections,copy,hashlib,html,itertools,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from npnf.meaning import scored_fields,digest,validate
from npnf.core import json_write
from experiments.language_plan import extraction

def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def mean(a):return float(np.mean(a)) if len(a) else None
def rms(a,b):return float(np.sqrt(np.mean((np.asarray(a)-b)**2)))
def rank(a):
    a=np.asarray(a);s=np.linalg.svd(a-a.mean(0),compute_uv=False);p=s/(s.sum()+1e-15)
    return float(np.exp(-np.sum(p*np.log(p+1e-15))))
def write(p,x):
    (ROOT/p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def numerical():
    t=read('reports/timelines.json')['rows'];bycap=[];groups=[]
    for (seed,cap,profile),rows in itertools.groupby(sorted(t,key=lambda x:(x['seed'],x['capacity'],x['profile'])),key=lambda x:(x['seed'],x['capacity'],x['profile'])):
        rows=list(rows);long=[];combined=[];cue=[];matrix=[]
        for now in range(4):
            for a,b in itertools.combinations(rows,2):
                long.append(rms(a['branches'][now]['before']['longTerm'],b['branches'][now]['before']['longTerm']))
                combined.append(rms(a['branches'][now]['before']['combined'],b['branches'][now]['before']['combined']))
                cue.append(rms(a['branches'][now]['preCue'],b['branches'][now]['preCue']))
        for row in rows:
            matrix.extend(b['after']['longTerm'] for b in row['branches'])
        groups.append(dict(seed=seed,capacity=cap,profile=profile,pastDifferenceLong=mean(long),pastDifferenceCombined=mean(combined),pastDifferenceCue=mean(cue),effectiveRank=rank(matrix)))
    for cap in range(5):
        r=[r for r in t if r['capacity']==cap];g=[g for g in groups if g['capacity']==cap]
        co=[b['after']['coactivation'] for x in r for b in x['branches']]
        bycap.append(dict(capacity=cap,parameters=r[0]['parameterCount'],activeParametersMean=mean([x['activeCount'] for x in r]),
            pastDifferenceLong=mean([x['pastDifferenceLong'] for x in g]),pastDifferenceCombined=mean([x['pastDifferenceCombined'] for x in g]),
            pastCriterionPassed=sum(x['pastDifferenceLong']>.01 for x in g),pastCriterionTotal=len(g),effectiveRank=mean([x['effectiveRank'] for x in g]),
            coactivationRate=mean([x>.05 for x in co]),orderDifference=mean([b['orderDifference'] for x in r for b in x['branches']]),
            shortRatio16=mean([x['shortRatio16'] for x in r]),moodRatio16=mean([x['moodRatio16'] for x in r]),longTermRetention16=mean([x['longTermRetention16'] for x in r]),
            acousticDrift=mean([x['acousticDrift'] for x in r])))
    st=read('reports/stress.json')['rows'];stress=[]
    for learner in ['local','dense']:
      for repeats in [32,512]:
        rs=[r for r in st if r['learner']==learner and r['repeats']==repeats]
        stress.append(dict(learner=learner,repeats=repeats,total=len(rs),passed=sum(r['relearningChange']>.01 and r['spillRatio']<.5 for r in rs),
            relearningChange=mean([r['relearningChange'] for r in rs]),spillRatio=mean([r['spillRatio'] for r in rs]),acousticDrift=mean([r['acousticDrift'] for r in rs])))
    d=read('reports/decay.json')['rows'];decay=[]
    for mode in ['uniform','selective','consolidation']:
        rs=[v for r in d for v in r['test'] if v['mode']==mode]
        decay.append(dict(mode=mode,testCount=len(rs),retention=mean([r['retention'] for r in rs]),relearningChange=mean([r['relearningChange'] for r in rs]),spillRatio=mean([r['spillRatio'] for r in rs])))
    diag=read('reports/diagnostic.json')['rows'];diagnostic=[]
    for kind in ['npn','linear','quadratic','retrieval']:
      for cap in range(5) if kind=='npn' else [None]:
        rs=[v for r in diag for v in r['rows'] if v['kind']==kind and v.get('capacity')==cap]
        diagnostic.append(dict(kind=kind,capacity=cap,testMSE=mean([v['testMSE'] for v in rs]),activeCount=mean([v['activeCount'] for v in rs]) if kind=='npn' else None))
    m=read('reports/mechanisms.json')['rows'];mechanisms=[]
    for mode in ['idle','rumination','correction','extinction','regulation']:
        rs=[r for r in m if r['mode']==mode]
        mechanisms.append(dict(mode=mode,count=len(rs),change=mean([r['change'] for r in rs]),shortAUC=mean([sum(v['short'] for v in r['timeline']) for r in rs]),
            moodAUC=mean([sum(v['mood'] for v in r['timeline']) for r in rs]),contextDifference=mean([rms(r['after'],r['alternate']) for r in rs]),recordsPreserved=all(r['recordPreserved'] for r in rs)))
    c=read('reports/controls.json')['rows'];structure=read('reports/structure.json')['rows']
    return dict(capacities=bycap,historyGroups=groups,stress=stress,decay=decay,validationSelectedRates=collections.Counter(str(r['selected']['rate']) for r in d),
        validationEligible=sum(r['selectionEligible'] for r in d),diagnostic=diagnostic,mechanisms=mechanisms,
        criteria=dict(historyPassed=sum(x['pastDifferenceLong']>.01 for x in groups),historyTotal=len(groups),
            shortPassed=sum(x['shortRatio16']<.01 for x in t),moodPassed=sum(x['moodRatio16']<.5 for x in t),retentionPassed=sum(x['longTermRetention16']>.5 for x in t),temporalTotal=len(t),
            plasticityPassed=sum(r['relearningChange']>.01 and r['spillRatio']<.5 for r in st),plasticityTotal=len(st),
            offWeightsInvariant=all(r['weightsUnchanged'] for r in c if r['mode']=='learning-off'),affectOffZero=all(r['emotionNormBeforeIdle']==0 and r['emotionNormAfterIdle']==0 for r in c if r['mode']=='affect-off')),
        structure=[dict(view=view,total=len(rs),relearningChange=mean([r['relearningChange'] for r in rs]),spillRatio=mean([r['spillRatio'] for r in rs])) for view in ['full','facts8-neutral-context'] for rs in [[r for r in structure if r['view']==view]]])

def semantics():
    gold=read('data/natural.json');rows=[];details=[]
    for engine in ['haiku','deepseek','qwen','qwen2048']:
        ex=extraction(engine);categorical=[];present=[];mae=[];exact=0;negation=[];separation=[]
        for e in gold:
            pred=ex.get(e['id']);diffs=[]
            if pred and all(pred[k]==e[k] for k in ['facts','interpretation']):exact+=1
            for name in ['facts','interpretation']:
                g=scored_fields(e[name]);p=scored_fields(pred[name]) if pred else {}
                for field,value in g.items():
                    correct=field in p and p[field]==value;categorical.append(correct)
                    if value is not None:present.append(correct)
                    if not correct:diffs.append(dict(view=name,field=field,gold=value,predicted=p.get(field)))
                for v in ['cueFacts8','outcomeFacts8']:mae.extend(np.abs(np.asarray(pred[name][v])-e[name][v]).tolist() if pred else [1.]*8)
            if e['id'] in ['n2','n10']:negation.append(bool(pred and pred['facts']['events'][0]['negated'] is True))
            if e['id'] in ['n1','n9']:separation.append(bool(pred and pred['facts']!=pred['interpretation'] and pred['interpretation']['events'][0]['uncertain']))
            details.append(dict(engine=engine,id=e['id'],valid=pred is not None,differences=diffs))
        rows.append(dict(engine=engine,valid=len(ex),total=16,exactRecords=exact,categoricalAccuracy=mean(categorical),presentFieldAccuracy=mean(present),facts8MAE=mean(mae),negationCorrect=sum(negation),negationTotal=len(negation),beliefSeparated=sum(separation),beliefTotal=len(separation)))
    return dict(rows=rows,details=details,annotation='Single authored synthetic gold; not independently double-annotated human ground truth.')

def conversations():
    results=[];outputs=[]
    for engine in ['haiku','deepseek','qwen','qwen2048']:
      for mode in ['dialogue','pipeline']:
        plan=read('data/'+('dialogue-plan' if mode=='dialogue' else 'pipeline-'+engine)+'.json')['cases'];lookup={r['id']:r for r in plan};raw=read(f'reports/{mode}-{engine}.json')
        records=[r for b in raw['batches'] if b['data'] for r in b['data']['records']];valid=0;stance=0;selection=0
        for r in records:
            case=lookup[r['id']]
            if set(r)!={'id','reply','stance','selection'} or not isinstance(r['reply'],str) or not r['reply'].strip():continue
            valid+=1;stance+=r['stance']==case['toolResult']['scene']['stance'];selection+=r['selection']==case['toolResult']['selection']
            outputs.append(dict(engine=engine,mode=mode,case=case,response=r))
        results.append(dict(engine=engine,mode=mode,planned=len(plan),valid=valid,stanceMatches=stance,selectionMatches=selection,proseFidelity='Requires separate evaluation; JSON metadata is insufficient.'))
    return dict(rows=results,outputs=outputs)

def review(data,prefix=''):
    out=data['outputs'];pairs=[];key=[]
    by=collections.defaultdict(list)
    for r in out:by[(r['mode'],r['case']['id'])].append(r)
    for (mode,id),items in by.items():
      for x,y in itertools.combinations(items,2):
        a,b=(x,y) if int(digest([mode,id,x['engine'],y['engine']])[:8],16)%2 else (y,x)
        pid=digest([mode,id,a['engine'],b['engine']])[:16];c=a['case']
        pairs.append(dict(id=pid,text=c['text'],history=[dict(update=i+1,text=e['text'],kind=e['kind']) for i,e in enumerate(c['history'])],
            A=dict(reply=a['response']['reply'],signal=c['toolResult']),B=dict(reply=b['response']['reply'],signal=b['case']['toolResult'])))
        key.append(dict(id=pid,mode=mode,case=id,A=a['engine'],B=b['engine']))
    pairs.sort(key=lambda p:digest(p['id']));dataset=dict(pairs=pairs);h=digest(dataset);write('reports/'+prefix+'review-data.json',dict(datasetHash=h,**dataset));write('reports/'+prefix+'review-key.json',dict(datasetHash=h,key=key))
    template=(ROOT/'scripts/review-template.html').read_text(encoding='utf8');(ROOT/('reports/'+prefix+'review.html')).write_text(template.replace('__DATA__',json.dumps(dict(datasetHash=h,**dataset),ensure_ascii=False).replace('</',r'<\/')),encoding='utf8')
    return dict(pairs=len(pairs),datasetHash=h,formalHumanEvaluationCompleted=False)

if __name__=='__main__':
    n=numerical();s=semantics();c=conversations();write('reports/numerical-analysis.json',n);write('reports/semantic-analysis.json',s);write('reports/conversation-analysis.json',c)
    print(json.dumps(dict(criteria=n['criteria'],semantics=s['rows'],conversation=c['rows'],review=review(c)),ensure_ascii=False,indent=2))
