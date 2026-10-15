import collections,copy,itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from npnf.meaning import *
from npnf.transport import normalize_meaning
from npnf.core import TemporalNPN
from scripts.analyze import rms,mean,write,review

def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def score(engine,normalize=False):
    raw=read(f'reports/extract-{engine}.json');pred={};converted=0
    for b in raw['batches']:
      if b['data']:
       for r in b['data']['records']:
        try:
            if set(r)!={'id','facts','interpretation'}:raise ValueError('Unexpected keys.')
            q=copy.deepcopy(r);changed=False
            for k in ['facts','interpretation']:
                if normalize:q[k],c=normalize_meaning(q[k]);changed|=c
                else:validate(q[k])
            pred[r['id']]=q;converted+=changed
        except (ValueError,KeyError,TypeError):pass
    gold=read('data/natural.json');accuracy=[];present=[];errors=[];negation=[];belief=[]
    for g in gold:
        p=pred.get(g['id'])
        for name in ['facts','interpretation']:
            gf=scored_fields(g[name]);pf=scored_fields(p[name]) if p else {}
            accuracy.extend(k in pf and pf[k]==v for k,v in gf.items());present.extend(k in pf and pf[k]==v for k,v in gf.items() if v is not None)
            for k in ['cueFacts8','outcomeFacts8']:errors.extend(np.abs(np.asarray(p[name][k])-g[name][k]).tolist() if p else [1]*8)
        if g['id'] in ['n2','n10']:negation.append(bool(p and p['facts']['events'][0]['negated']))
        if g['id'] in ['n1','n9']:belief.append(bool(p and p['facts']!=p['interpretation'] and p['interpretation']['events'][0]['uncertain']))
    return dict(engine=engine,normalized=normalize,valid=len(pred),converted=converted,total=16,categoricalAccuracy=mean(accuracy),presentFieldAccuracy=mean(present),facts8MAE=mean(errors),negationCorrect=sum(negation),beliefSeparated=sum(belief))

def language(mode):
    out=[];stats=[]
    for engine in ['haiku','deepseek']:
        plan={c['id']:c for c in read(f'data/{mode}-{engine}.json')['cases']};raw=read(f'reports/{mode}-{engine}.json');rows=[]
        for b in raw['batches']:
            if b['data']:
                for r in b['data']['records']:
                    if set(r)=={'id','reply','stance','selection'} and isinstance(r['reply'],str) and r['reply'].strip():
                        c=plan[r['id']];rows.append(r);out.append(dict(engine=engine,mode=mode,case=c,response=r))
        stats.append(dict(mode=mode,engine=engine,planned=len(plan),valid=len(rows),stanceMatches=sum(r['stance']==plan[r['id']]['toolResult']['scene']['stance'] for r in rows),selectionMatches=sum(r['selection']==plan[r['id']]['toolResult']['selection'] for r in rows)))
    return out,stats

if __name__=='__main__':
    dual=read('reports/dual-view.json')['rows'];groups=[]
    for seed in [6311,6312,6313]:
      for cap in range(5):
        rows=[r for r in dual if r['seed']==seed and r['capacity']==cap];diff=[]
        for current in range(4):
            for a,b in itertools.combinations(rows,2):diff.append(rms(a['branches'][current]['before']['longTerm'],b['branches'][current]['before']['longTerm']))
        groups.append(dict(seed=seed,capacity=cap,pastDifferenceLong=mean(diff)))
    audit=read('reports/expression-audit.json');auditkey={r['id']:r for r in read('reports/audit-key.json')['key']};auditrows=[]
    for batch in audit['batches']:
      if batch['data']:
       for r in batch['data']['records']:auditrows.append(dict(r,engine=auditkey[r['id']]['engine']))
    auditstats=[]
    for engine in ['haiku','deepseek','qwen','qwen2048']:
        r=[r for r in auditrows if r['engine']==engine]
        auditstats.append(dict(engine=engine,count=len(r),fidelity=mean([x['fidelity'] for x in r]),naturalness=mean([x['naturalness'] for x in r]),facts=mean([x['facts'] for x in r]),added=sum(x['emotionAdded'] for x in r),reversed=sum(x['emotionReversed'] for x in r),omitted=sum(x['emotionOmitted'] for x in r)))
    out1,s1=language('future');out2,s2=language('dual')
    # Preserve the primary 44-pair material and create a distinct additional dataset.
    extras=review(dict(outputs=out1+out2),prefix='additional-')
    actual=read('reports/actual-tool-calls.json')['rows'];verified=[]
    for row in actual:
        if not row.get('toolResult'):continue
        n=TemporalNPN.load(ROOT/'models'/('dialogue-'+row['id']))
        verified.append(row['toolResult']['parameterHash']==n.parameter_hash() and row['toolResult']['update']==n.clock)
    typed=score('haiku-structured');fixed=[score(e,True) for e in ['haiku','deepseek','qwen','qwen2048']]
    result=dict(dual=dict(groups=groups,passed=sum(r['pastDifferenceLong']>.01 for r in groups),total=len(groups),meanPastDifference=mean([r['pastDifferenceLong'] for r in groups]),weightsMatchAll=all(r['weightsMatchObservedControl'] for r in dual)),
        strictSchema=typed,losslessFormatDiagnosis=fixed,language=s1+s2,audit=dict(type='Post hoc LLM audit; not human scores',rows=auditstats,flags=[r for r in auditrows if r['emotionAdded'] or r['emotionOmitted'] or r['emotionReversed']]),
        actualTools=dict(total=len(actual),called=sum(bool(r.get('toolCalled')) for r in actual),requestExact=sum(bool(r.get('requestExact')) for r in actual),snapshotSignalsVerified=sum(verified),responses=sum(bool(r.get('response')) for r in actual),stanceMatches=sum(bool(r.get('stanceMatch')) for r in actual),selectionMatches=sum(bool(r.get('selectionMatch')) for r in actual)),additionalHumanMaterial=extras)
    write('reports/extended-analysis.json',result);write('reports/additional-conversations.json',dict(outputs=out1+out2));print(json.dumps({k:v for k,v in result.items() if k not in ['audit']},ensure_ascii=False,indent=2))
