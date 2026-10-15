"""Gate-triggered fixed supplementary conversation; keep primary plan intact."""
import copy
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnd.core import Personality,digest
from tools.evaluate import decision
PATH=ROOT/'reports/million-live-plan.json'
def read(p):return json.loads(p.read_text(encoding='utf8'))
gate=read(ROOT/'reports/capacity-gate.json')
if gate['millionParameterGatePassed'] is not True:raise ValueError('Gate not passed.')

if sys.argv[1]=='freeze':
    main=read(ROOT/'reports/live-plan.json');cases=[]
    for person in (0,1):
        for pid in ('promise-change','trolley'):
            c=copy.deepcopy(next(c for c in main['cases'] if c['id']==f'h16-p{person}-{pid}'))
            c.pop('expectedToolResult');c['id']=f'million-p{person}-{pid}';c['config']['width']=31250
            c['statePrefix']=str(ROOT/'models/raw8/bounded_signal/nn-h31250'/f'seed4311-p{person}')
            c['arguments']['query_id']=f'case-{len(cases)%2}'
            cases.append(c)
    unique=list({c['probeId']:c for c in main['cases']}.values())
    structure=[{'id':f'case-{i}','user':c['user'],'arguments':dict(c['arguments'],query_id=f'case-{i}')} for i,c in enumerate(unique)]
    value={'formatVersion':1,'frozenBeforeReplies':True,'models':main['models'],'cases':cases,'structureCases':structure,
           'conditionalGateSha256':hashlib.sha256((ROOT/'reports/capacity-gate.json').read_bytes()).hexdigest(),
           'primaryReviewUnchanged':True,'plannedAdditionalReplies':8}
    with PATH.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({'frozen':4,'planHash':digest(value)}))
elif sys.argv[1]=='materialize':
    value=read(PATH)
    if any('expectedToolResult' in c for c in value['cases']):raise ValueError('Already materialized.')
    for c in value['cases']:
        model=Personality.load(c['statePrefix'])
        if model.history!=c['history'] or model.parameter_count()!=1000014:raise ValueError('State differs from fixed supplemental plan.')
        c['expectedToolResult']=decision(model,c['arguments'])
    (ROOT/'reports/million-live-plan-pre-materialization.json').write_bytes(PATH.read_bytes())
    PATH.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'materialized':4,'planHash':digest(value)}))
