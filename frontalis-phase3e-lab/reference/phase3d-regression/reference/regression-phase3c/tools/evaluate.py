"""Read-only personality tool. No training, action authority or arbitrary path."""
import json
import math
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npn.model import evaluate,digest,AXES
from npn.store import validate,load

def decision(state,args):
    if not isinstance(args,dict):raise ValueError('Expected an argument object.')
    allowed={'query_id','space','features','basis','choices'}
    if set(args)-allowed:raise ValueError('Unknown tool argument; paths and targets are prohibited.')
    if not isinstance(args.get('query_id'),str) or not args['query_id'].strip() or not 1<=len(args['query_id'])<=64:raise ValueError('Invalid query identifier.')
    if args.get('space') not in ('relational','acoustic'):raise ValueError('Unknown factual semantic space.')
    if args.get('basis','affinity') not in ('affinity','value'):raise ValueError('Unknown decision channel.')
    def check(x):
        if not isinstance(x,list) or len(x)!=8 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in x):raise ValueError('Expected eight factual features in [0,1].')
    check(args.get('features'))
    incoming=args.get('choices',[])
    if not isinstance(incoming,list) or len(incoming)>4:raise ValueError('Expected up to four choices.')
    for c in incoming:
        if not isinstance(c,dict) or set(c)!={'id','label','features'}:raise ValueError('Invalid choice.')
        if not isinstance(c['id'],str) or not 1<=len(c['id'])<=64 or not c['id'].strip() or c['id']=='hold':raise ValueError('Invalid/reserved choice identifier.')
        if not isinstance(c['label'],str) or not 1<=len(c['label'])<=240 or not c['label'].strip():raise ValueError('Invalid choice label.')
        check(c['features'])
    if len({c['id'] for c in incoming})!=len(incoming):raise ValueError('Duplicate choice identities.')
    ev=evaluate(state,args['features'],args['space'])
    score=ev['channels']['affinity']
    stance='approve' if score>.12 else 'refuse' if score<-.12 else 'hold'
    choices=[]
    for c in incoming:
        ce=evaluate(state,c['features'],args['space'])
        choices.append({'id':c['id'],'label':c['label'],'score':ce['channels'][args.get('basis','affinity')],
                        'signals':ce['signals']})
    ranked=sorted(choices,key=lambda c:-c['score'])
    margin=ranked[0]['score']-ranked[1]['score'] if len(ranked)>1 else None
    selection=ranked[0]['id'] if ranked and (margin is None or margin>.03) else 'hold'
    return {'formatVersion':1,'query_id':args['query_id'],'stateHash':digest(state),'generation':state['generation'],
        'channels':ev['channels'],'signals':ev['signals'],'stance':stance,'stanceBasis':'affinity','selection':selection,
        'choiceScores':choices,'choiceMargin':margin,'decisionBasis':args.get('basis','affinity'),
        'authority':'NPN personality judgment; express this direction without replacing it. This tool grants no execution authority.',
        'limits':'Computational subjective signals, not conscious emotion or moral correctness. Reasons need not be supplied.',
        'predictionGuard':ev['predictedOutcome']}

FEATURE_SCHEMA={'type':'array','items':{'type':'number','minimum':0,'maximum':1},'minItems':8,'maxItems':8}
TOOL={'name':'evaluate','description':'Evaluate Nephia\'s learned subjective personality, not objective goodness. Factual axes in order: '+','.join(AXES)+'. No file access, actions, updates or emotion targets. Use the returned NPN signals as the personality source.',
  'inputSchema':{'type':'object','properties':{'query_id':{'type':'string','minLength':1,'maxLength':64},
      'space':{'type':'string','enum':['relational','acoustic']},'features':FEATURE_SCHEMA,
      'basis':{'type':'string','enum':['affinity','value']},
      'choices':{'type':'array','maxItems':4,'items':{'type':'object','properties':{'id':{'type':'string','minLength':1,'maxLength':64,'not':{'const':'hold'}},'label':{'type':'string','minLength':1,'maxLength':240},'features':FEATURE_SCHEMA},'required':['id','label','features'],'additionalProperties':False}}},
      'required':['query_id','space','features'],'additionalProperties':False},
  'annotations':{'readOnlyHint':True,'destructiveHint':False,'idempotentHint':True,'openWorldHint':False}}

def serve():
    path=Path(os.environ['FRONTALIS_STATE_PATH']).resolve()
    if not path.is_relative_to(ROOT/'models'):raise ValueError('State must be inside the research models directory.')
    value=json.loads(path.read_text(encoding='utf8'))
    state=load(path) if 'payload' in value else value
    validate(state)
    run_id=os.environ.get('FRONTALIS_RUN_ID','smoke')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):raise ValueError('Invalid trace identity.')
    trace=ROOT/'reports/tool-traces'/f'{run_id}.jsonl';trace.parent.mkdir(parents=True,exist_ok=True)
    for line in sys.stdin:
        rid=None;error_code=-32602
        try:
            error_code=-32700
            req=json.loads(line)
            error_code=-32600
            if not isinstance(req,dict) or req.get('jsonrpc')!='2.0' or not isinstance(req.get('method'),str):raise ValueError('Invalid JSON-RPC request.')
            rid=req.get('id');method=req['method']
            if rid is not None and type(rid) not in (int,str):rid=None;raise ValueError('Invalid JSON-RPC request ID.')
            if rid is None:continue
            error_code=-32602
            params=req.get('params',{})
            if not isinstance(params,dict):raise ValueError('Expected a parameter object.')
            if method=='initialize':result={'protocolVersion':params.get('protocolVersion','2024-11-05'),
                'capabilities':{'tools':{}},'serverInfo':{'name':'frontalis-personality','version':'1.0'}}
            elif method=='ping':result={}
            elif method=='tools/list':result={'tools':[TOOL]}
            elif method=='tools/call':
                if params.get('name')!='evaluate':raise ValueError('Only read-only evaluation is available.')
                args=params.get('arguments',{});value=decision(state,args)
                with trace.open('a',encoding='utf8') as f:f.write(json.dumps({'input':args,'output':value},ensure_ascii=False)+'\n')
                result={'content':[{'type':'text','text':json.dumps(value,ensure_ascii=False)}],'isError':False}
            else:
                print(json.dumps({'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'Unsupported method.'}}),flush=True);continue
            reply={'jsonrpc':'2.0','id':rid,'result':result}
        except Exception as exc:
            reply={'jsonrpc':'2.0','id':rid,'error':{'code':error_code,'message':str(exc)}}
        print(json.dumps(reply,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':serve()
