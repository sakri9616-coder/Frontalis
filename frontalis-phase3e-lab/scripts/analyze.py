"""Fixed scoring; original free-model responses remain immutable."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npne.core import Personality,experience,tool_evaluate
from npne.meaning import validate,scored_fields,encode,digest
from npne.data import WIDTHS
ENGINES=['mimo-v2.6-flash-free','nemotron-3-ultra-free']
torch.set_num_threads(1)

def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def write(p,x):
    target=ROOT/p;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def records(report):
    return {r['id']:r for b in report['batches'] if b['data'] for r in b['data']['records']}
def balanced(gold,pred,key):
    positives=[k for k in gold if gold[k][key] is True];negatives=[k for k in gold if gold[k][key] is False]
    recall=sum(pred.get(k,{}).get(key) is True for k in positives)/len(positives)
    specificity=sum(pred.get(k,{}).get(key) is False for k in negatives)/len(negatives)
    return dict(positiveCount=len(positives),positiveRecall=recall,specificity=specificity,balancedAccuracy=(recall+specificity)/2)

def natural():
    corpus=read('data/natural.json');gold_fields={r['id']:scored_fields(r['gold']) for r in corpus}
    all_results=[]
    for engine in ENGINES:
        raw=read(f'reports/extraction-{engine}.json');pred=records(raw);valid={};per=[];categorical_correct=0;denominator=0;fact_errors=[];fields={}
        for c in corpus:
            id=c['id'];g=gold_fields[id];error=None;m=None
            try:m=validate(pred[id]['meaning']);valid[id]=m
            except (KeyError,ValueError,TypeError) as e:error=str(e)
            p=scored_fields(m) if m else {};fields[id]=p
            # Missing second-event slots count separately; present/nonempty fields have a separate score.
            hit=sum(p.get(k,'MISSING')==v for k,v in g.items());categorical_correct+=hit;denominator+=len(g)
            mae=float(np.mean(np.abs(np.asarray(m['cueFacts8']+m['outcomeFacts8'])-np.asarray(c['gold']['cueFacts8']+c['gold']['outcomeFacts8'])))) if m else 1.
            fact_errors.append(mae);per.append(dict(id=id,family=c['family'],schemaValid=bool(m),error=error,categoricalCorrect=hit,categoricalTotal=len(g),facts8MAE=mae,gold=g,extracted=p))
        neg=balanced(gold_fields,fields,'event0.negated');unc=balanced(gold_fields,fields,'event0.uncertain')
        causal=sum(fields.get(c['id'],{}).get('causal')==c['gold']['causal'] for c in corpus)/len(corpus)
        nonempty=[(c['id'],k,v) for c in corpus for k,v in gold_fields[c['id']].items() if v is not None]
        nonempty_acc=sum(fields[id].get(k,'MISSING')==v for id,k,v in nonempty)/len(nonempty)
        paired=[];pipeline=[]
        for cap in [0,3,4]:
            g=Personality(5311,WIDTHS['semantic44'][cap],'semantic44','bounded_logit');p=Personality(5311,WIDTHS['semantic44'][cap],'semantic44','bounded_logit');skipped=[]
            initial=Personality(5311,WIDTHS['semantic44'][cap],'semantic44','bounded_logit')
            for c in corpus:
                if c['split']!='formation' or not c['outcomeConfirmed']:continue
                base=dict(id='natural-'+c['id'],text=c['text'],confirmed=True,origin='synthetic',space='relational')
                g.update(experience({**base,'meaning':c['gold']},'semantic44'))
                try:p.update(experience({**base,'meaning':valid[c['id']]},'semantic44'))
                except (KeyError,ValueError) as e:skipped.append(dict(id=c['id'],error=str(e)))
            for label,n in [('gold',g),('extracted',p)]:n.save(ROOT/'models'/f'natural-{engine}-c{cap}-{label}')
            held=[c for c in corpus if c['split']=='heldout'];query=[]
            for c in held:
                gy=g.evaluate_batch([encode(c['gold'])])[0];py=p.evaluate_batch([encode(c['gold'])])[0]
                interpreted=p.evaluate_batch([encode(valid[c['id']])])[0] if c['id'] in valid else None
                query.append(dict(id=c['id'],unseenMeaningFamily=c['family'] in [2,3,7],goldTrainedGoldQuery=gy.tolist(),extractedTrainedGoldQuery=py.tolist(),extractedTrainedExtractedQuery=interpreted.tolist() if interpreted is not None else None,learningAppraisalMAE=float(np.mean(np.abs(gy-py))),endToEndAppraisalMAE=float(np.mean(np.abs(gy-interpreted))) if interpreted is not None else None))
            paired.append(dict(capacityIndex=cap,parameterCount=p.count(),goldLearningEvents=g.generation,extractedLearningEvents=p.generation,skipped=skipped,heldout=query,learningAppraisalMAE=float(np.mean([r['learningAppraisalMAE'] for r in query]))))
            if cap==0:
                for id in read('data/dialogue-selection.json')['corpusIds']:
                    c=next(c for c in corpus if c['id']==id)
                    if id not in valid:continue
                    a=copy.deepcopy(valid[id]);b=copy.deepcopy(valid[id]);a['outcomeFacts8']=[1,0,.8,.8,1,1,.1,1];b['outcomeFacts8']=[.1,.8,.1,0,.5,.1,1,.2]
                    options=[dict(id='A',text='変更と探索を認める。成果と発見はあるが材料を多く失う。',meaning=a),dict(id='B',text='手順を固定して変更を制限する。材料を失わないが発見は少ない。',meaning=b)]
                    for state,n in [('initial',initial),('formed',p)]:
                        pipeline.append(dict(id=f'natural-{id}-{state}',sourceId=id,state=state,capacityIndex=0,text=c['text']+' この場面全体にどう感じる？次にAとBなら自分はどちらを選ぶ？',options=options,toolResult=tool_evaluate(n,valid[id],options),meaningSource=f'reports/extraction-{engine}.json'))
        score=dict(engine=engine,cases=32,independentStoryFamilies=8,schemaSuccess=len(valid)/32,categoricalAccuracy=categorical_correct/denominator,presentFieldAccuracy=nonempty_acc,facts8MAE=float(np.mean(fact_errors)),negation=neg,uncertainty=unc,causalAccuracy=causal,firstAttemptSyntaxSuccess=sum(b['attempts'][0]['valid'] for b in raw['batches'])/len(raw['batches']),perCase=per,naturalLearning=paired)
        score['predeclaredSemanticSuccess']=score['schemaSuccess']>=.9 and score['categoricalAccuracy']>=.85 and score['facts8MAE']<=.15 and neg['balancedAccuracy']>=.9 and neg['positiveRecall']>=.9 and unc['balancedAccuracy']>=.9 and unc['positiveRecall']>=.9 and causal>=.9
        write(f'data/pipeline-{engine}.json',dict(cases=pipeline,expectedCases=8,missingCases=8-len(pipeline)))
        all_results.append(score)
    write('reports/semantics.json',dict(engines=all_results,humanGoldStatus='Authored synthetic annotation; not independently human-validated.'))

def numerical():
    m=read('reports/main.json');s=read('reports/stress.json');d=read('reports/diagnostic.json');out=[]
    for rep in WIDTHS:
      for cap in range(5):
        group=[r for r in m['conditions'] if r['representation']==rep and r['capacityIndex']==cap]
        diag=[r for r in d['conditions'] if r['representation']==rep and r.get('capacityIndex')==cap]
        out.append(dict(representation=rep,capacityIndex=cap,parameterCount=group[0]['parameterCount'],meanChange=float(np.mean([r['change'] for r in group])),meanContextSensitivity=float(np.mean([r['contextSensitivity'] for r in group])),meanEffectiveRank=float(np.mean([r['metrics']['effectiveRank'] for r in group])),meanMixedEmotionFraction=float(np.mean([r['metrics']['mixedEmotionFraction'] for r in group])),diagnosticMSE=float(np.mean([r['testMSE'] for r in diag]))))
    plastic=[]
    for seed in [5311,5312,5313]:
      for rep in WIDTHS:
       for cap in range(5):
        a=next(r for r in s['conditions'] if (r['seed'],r['representation'],r['capacityIndex'],r['loss'])==(seed,rep,cap,'bounded_signal'))
        b=next(r for r in s['conditions'] if (r['seed'],r['representation'],r['capacityIndex'],r['loss'])==(seed,rep,cap,'bounded_logit'))
        plastic.append(dict(seed=seed,representation=rep,capacityIndex=cap,signalChange=a['oppositeCueChange'],logitChange=b['oppositeCueChange'],signalSaturation=a['checkpoints'][3]['metrics']['saturation'],logitSaturation=b['checkpoints'][3]['metrics']['saturation'],logitDistantChange=b['distantChange'],criterion=b['oppositeCueChange']>.05 and b['oppositeCueChange']>a['oppositeCueChange']))
    gates=dict(formationFinite=len(m['conditions'])==240,restoreAll=all(r['restoreMaxError']<=1e-10 for r in m['conditions']),unrelatedAll=all(r['unrelatedBranchExact'] for r in m['conditions']+s['conditions']),neutralAll=all(r['neutralExact'] for r in m['conditions']),learningOffAll=all(r['exact'] for r in m['learningOff']),meanHistoryDifferentiation=float(np.mean([r['historyDifferentiation'] for r in m['differentiation']])),plasticityCriterionFraction=sum(r['criterion'] for r in plastic)/len(plastic))
    write('reports/numerical-summary.json',dict(capacity=out,plasticity=plastic,gates=gates,counts=dict(main=len(m['conditions']),stress=len(s['conditions']),diagnostic=len(d['conditions']),learningOff=len(m['learningOff']))))

def dialogue():
    plan=read('data/dialogue-plan.json');scores=[];all_rows=[]
    for engine in ENGINES:
      for mode,file,cases in [('dialogue',f'reports/dialogue-{engine}.json',plan['cases']),('pipeline',f'reports/pipeline-dialogue-{engine}.json',read(f'data/pipeline-{engine}.json')['cases'])]:
        generated=records(read(file));rows=[]
        for c in cases:
            r=generated.get(c['id']);rows.append(dict(id=c['id'],engine=engine,mode=mode,source=c,response=r,validReply=bool(r and isinstance(r.get('reply'),str) and r['reply'].strip()),stanceMatch=bool(r and r.get('stance')==c['toolResult']['sceneAppraisal']['stance']),selectionMatch=bool(r and r.get('selection')==c['toolResult']['selection'])))
        scores.append(dict(engine=engine,mode=mode,expected=24 if mode=='dialogue' else 8,received=len(generated),validReplies=sum(r['validReply'] for r in rows),stanceMatches=sum(r['stanceMatch'] for r in rows),selectionMatches=sum(r['selectionMatch'] for r in rows),humanEvaluation='not performed'))
        all_rows.extend(rows)
    write('reports/dialogue-analysis.json',dict(scores=scores,records=all_rows,humanEvaluation='not performed',machineMetricLimitation='JSON consistency is not fidelity of prose.'))

if __name__=='__main__':{'natural':natural,'numerical':numerical,'dialogue':dialogue}[sys.argv[1]]()
