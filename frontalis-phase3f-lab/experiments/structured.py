import copy,hashlib,json,os,sys,time,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.core import TemporalNPN,json_write
from npnf.meaning import *
from npnf.tools import dispatch
from experiments.language_plan import options

KEY=os.environ.get('ANTHROPIC_API_KEY')
def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
def enum(values):return dict(type='string',enum=values)
def array(items,description=''):return dict(type='array',items=items,description=description)
event_schema=obj(dict(action=enum(ACTIONS),actor=enum(ACTORS),target=enum(['self','other','object']),negated=dict(type='boolean'),uncertain=dict(type='boolean'),intent=enum(['benefit','harm','unknown'])))
meaning_schema=obj(dict(cueFacts8=array(dict(type='number'),'Exactly 8 values in [0,1]. Unknown is 0.5. Use the specified axis order.'),outcomeFacts8=array(dict(type='number'),'Exactly 8 values in [0,1]. Unknown is 0.5.'),events=array(event_schema,'One or two events in text order.'),consent=enum(['yes','no','unknown']),trust=enum(['high','low','unknown']),causal=enum(['causes','prevents','unrelated','unknown']),temporal=enum(['after','simultaneous','unknown']),delayed=dict(type='boolean')))

def api(name,body):
    p=ROOT/'reports/live/structured'/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True);h=digest(body)
    if p.exists():
        raw=json.loads(p.read_text(encoding='utf8'))
        if raw['requestHash']!=h:raise ValueError('Structured cache request mismatch.')
        return raw
    response=None;error=None;status=None;start=time.perf_counter()
    try:
        req=urllib.request.Request('https://api.anthropic.com/v1/messages',data=json.dumps(body,ensure_ascii=False).encode(),headers={'content-type':'application/json','anthropic-version':'2023-06-01','x-api-key':KEY})
        with urllib.request.urlopen(req,timeout=180) as r:status=r.status;response=json.load(r)
    except urllib.error.HTTPError as e:
        status=e.code
        try:response=json.loads(e.read())
        except Exception:pass
        error='HTTP '+str(status)
    except Exception as e:error=type(e).__name__
    raw=dict(request=body,requestHash=h,response=response,error=error,status=status,elapsedSeconds=time.perf_counter()-start)
    text=json.dumps(raw,ensure_ascii=False,allow_nan=False)
    if KEY and KEY in text:raise ValueError('Secret serialization rejected.')
    json_write(p,raw);print(name,status,error,flush=True);return raw
def output(raw):
    if raw['error']:raise ValueError(raw['error'])
    return json.loads('\n'.join(x['text'] for x in raw['response']['content'] if x['type']=='text'))
def base(messages):return dict(model='claude-haiku-5-5',max_tokens=8192,messages=messages)

def extract():
    records=json.loads((ROOT/'data/natural.json').read_text(encoding='utf8'));contract=(ROOT/'data/extraction-contract.txt').read_text(encoding='utf8')
    schema=obj(dict(records=array(obj(dict(id=dict(type='string'),facts=meaning_schema,interpretation=meaning_schema)))))
    batches=[]
    for i in range(0,len(records),2):
        cases=[dict(id=r['id'],text=r['text']) for r in records[i:i+2]]
        request=base([dict(role='user',content=contract+'\n'+json.dumps(cases,ensure_ascii=False))]);request['output_config']=dict(format=dict(type='json_schema',schema=schema))
        raw=api('extract-'+str(i//2),request);error=raw['error'];data=None
        try:
            data=output(raw)
            if len(data['records'])!=2 or {r['id'] for r in data['records']}!={r['id'] for r in cases}:raise ValueError('Invalid extraction IDs/count.')
            for r in data['records']:validate(r['facts']);validate(r['interpretation'])
        except Exception as e:error=str(e);data=None
        batches.append(dict(data=data,error=error))
    json_write(ROOT/'reports/extract-haiku-structured.json',dict(engine='haiku-structured',model='claude-haiku-5-5',batches=batches))

def tools():
    plan=json.loads((ROOT/'data/dialogue-plan.json').read_text(encoding='utf8'))['cases'];cases=[c for c in plan if c['id'].endswith('-initial') or c['id'].endswith('-corrected')];rows=[]
    for i in range(0,len(cases),2):
        batch=cases[i:i+2];requests=[dict(caseId=c['id'],request=dict(modelId='dialogue-'+c['id'],meaning=c['meaning'],options=[dict(id=o['id'],meaning=o['meaning']) for o in options()])) for c in batch]
        req_schema=obj(dict(caseId=enum([c['id'] for c in batch]),request=obj(dict(modelId=enum([r['request']['modelId'] for r in requests]),meaning=meaning_schema,options=array(obj(dict(id=dict(type='string'),meaning=meaning_schema)))))))
        tool=dict(name='evaluate_npn',description='Read-only NPN appraisal. Copy the supplied model ID, factual meaning, and options exactly. No learning is available.',input_schema=req_schema,strict=True)
        messages=[dict(role='user',content='次の2件について各1回evaluate_npnを呼ぶ。要求はコピーし、感情や人格説明を追加しない。ツールの結果を得るまで発言は作らない。\n'+json.dumps(requests,ensure_ascii=False))]
        body=base(messages);body['tools']=[tool];raw=api('tool-request-'+str(i//2),body)
        if raw['error']:
            rows.extend(dict(id=c['id'],error=raw['error'],toolCalled=False) for c in batch);continue
        blocks=[b for b in raw['response']['content'] if b['type']=='tool_use'];tool_results=[];seen=set();group=[]
        for block in blocks:
            data=block['input'];cid=data.get('caseId');expected=next((r['request'] for r in requests if r['caseId']==cid),None);error=None;result=None
            try:
                if block['name']!='evaluate_npn' or expected is None or cid in seen:raise ValueError('Unexpected or duplicate tool call.')
                n=TemporalNPN.load(ROOT/'models'/data['request']['modelId']);before=n.parameter_hash();t=n.clock
                result=dispatch(ROOT,data['request']);readonly=n.parameter_hash()==before and n.clock==t
            except Exception as e:error=str(e);readonly=False
            seen.add(cid);group.append(dict(id=cid,toolCalled=True,requestExact=expected is not None and data.get('request')==expected,readOnly=readonly,error=error,toolResult=result))
            tool_results.append(dict(type='tool_result',tool_use_id=block['id'],content=json.dumps(result if not error else dict(error=error),ensure_ascii=False),is_error=bool(error)))
        for c in batch:
            if c['id'] not in seen:group.append(dict(id=c['id'],toolCalled=False,error='Missing tool call.'))
        messages.append(dict(role='assistant',content=raw['response']['content']));messages.append(dict(role='user',content=tool_results+[dict(type='text',text='各結果のscene.stanceとselectionを保持し、sceneとselectedを混同せず、場面へ日本語2〜4文で自然に返答する。感情の正値だけを表し、過去の事実を捏造しない。recordsのidはcaseId。理由は必須でない。')]))
        response_schema=obj(dict(records=array(obj(dict(id=dict(type='string'),reply=dict(type='string'),stance=enum(['approve','refuse','hold']),selection=dict(type='string'))))))
        final=base(messages);final.update(tools=[tool],output_config=dict(format=dict(type='json_schema',schema=response_schema)))
        answer=api('tool-response-'+str(i//2),final)
        try:
            data=output(answer)
            for r in group:
                reply=next((v for v in data['records'] if v['id']==r['id']),None);r['response']=reply
                if reply and r.get('toolResult'):
                    r['stanceMatch']=reply['stance']==r['toolResult']['scene']['stance'];r['selectionMatch']=reply['selection']==r['toolResult']['selection']
        except Exception as e:
            for r in group:r['responseError']=str(e)
        rows.extend(group)
    json_write(ROOT/'reports/actual-tool-calls.json',dict(type='Additional real tool-use test with gold interpretation',rows=rows))
if __name__=='__main__':
    if not KEY:raise ValueError('ANTHROPIC_API_KEY unavailable.')
    if sys.argv[1]=='extract':extract()
    elif sys.argv[1]=='tools':tools()
    else:raise ValueError('Use extract or tools.')
