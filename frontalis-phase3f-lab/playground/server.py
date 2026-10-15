"""Loopback-only temporal workbench; research checkpoints remain read-only."""
import argparse,copy,hashlib,io,json,os,re,secrets,subprocess,sys,threading,urllib.error,urllib.request,uuid,zipfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnf.core import TemporalNPN,tool_evaluate,validate_episode,json_write
from npnf.dual_view import DualViewAgent
from npnf.transport import EXTRACTION_SCHEMA,normalize_meaning
from npnf.data import *
from npnf.meaning import validate,digest
torch.set_num_threads(1)
STATES=ROOT/'playground/states';STATES.mkdir(parents=True,exist_ok=True)
TOKEN=secrets.token_hex(24);LOCK=threading.RLock();SESSION=None;NPN=None;TRACE=[];DRAFT=None
PORT=8905

def llm(engine,prompt,schema=None):
    if engine not in ['haiku','qwen','deepseek']:raise ValueError('Unknown expression engine.')
    if engine=='deepseek':
        env={k:v for k,v in os.environ.items() if not re.search('API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET',k,re.I)}
        r=subprocess.run([r'C:\Users\sakur\AppData\Local\Programs\@opencodedesktop\resources\opencode-cli.exe','run','--standalone','--format','json','--agent','plan','-m','opencode/deepseek-v4.1-flash-free'],input=prompt,text=True,encoding='utf8',capture_output=True,timeout=180,env=env,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if r.returncode:raise ValueError('OpenCode returned a nonzero exit status.')
        lines=[json.loads(x) for x in r.stdout.splitlines() if x.strip()]
        if any(x['type']=='tool_use' for x in lines):raise ValueError('Unexpected tool execution by expression model.')
        if any(x.get('part',{}).get('cost',0)!=0 for x in lines if x['type']=='step_finish'):raise ValueError('OpenCode returned nonzero cost.')
        return '\n'.join(x['part']['text'] for x in lines if x['type']=='text')
    key=os.environ.get('ANTHROPIC_API_KEY')
    if engine=='haiku' and not key:raise ValueError('ANTHROPIC_API_KEY is unavailable in this server process.')
    body=dict(model='claude-haiku-5-5' if engine=='haiku' else 'qwen3-4b-instruct-2507',max_tokens=4096,messages=[dict(role='user',content=prompt)])
    if engine=='haiku' and schema is not None:body['output_config']=dict(format=dict(type='json_schema',schema=schema))
    if engine=='qwen':body['temperature']=0
    headers={'Content-Type':'application/json'}
    if engine=='haiku':headers.update({'x-api-key':key,'anthropic-version':'2023-06-01'})
    req=urllib.request.Request('https://api.anthropic.com/v1/messages' if engine=='haiku' else 'http://127.0.0.1:1234/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=180) as response:data=json.load(response)
    except urllib.error.HTTPError as e:raise ValueError('LLM HTTP '+str(e.code)) from None
    return '\n'.join(x['text'] for x in data['content'] if x['type']=='text') if engine=='haiku' else data['choices'][0]['message']['content']

def parse(text):return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()))
def current_meaning():return present_event(0)['interpretation']
def result(n=None):
    n=n or NPN
    opts=[dict(id='repair',meaning=present_event(0)['interpretation']),dict(id='distance',meaning=unrelated())]
    return tool_evaluate(n,current_meaning(),opts)
def point(n):
    r=result(n)['scene'];return dict(update=n.clock,longTerm=r['longTerm'],shortTerm=r['shortTerm'],mood=r['mood'],combined=r['combined'])
def generation_path(sid,generation):return STATES/sid/f'g{generation:05d}'
def commit(candidate,sid,trace,revision):
    global NPN,SESSION,TRACE
    while Path(str(generation_path(sid,revision))+'.json').exists() or Path(str(generation_path(sid,revision))+'.npz').exists():revision+=1
    candidate.save(generation_path(sid,revision))
    evaluation_view='dual50' if isinstance(candidate,DualViewAgent) else 'observed'
    meta=dict(session=sid,revision=revision,trace=trace,evaluationView=evaluation_view)
    json_write(STATES/sid/f'view-{revision:05d}.json',meta)
    temp=STATES/('current-'+uuid.uuid4().hex+'.tmp');temp.write_text(json.dumps(dict(session=sid,revision=revision,evaluationView=evaluation_view)),encoding='utf8');os.replace(temp,STATES/'current.json')
    NPN=candidate;SESSION=dict(session=sid,revision=revision,evaluationView=evaluation_view);TRACE=trace
def init(capacity=0,profile=0,decay='consolidation',rate=.01,evaluation_view='observed'):
    global DRAFT
    if evaluation_view not in ['observed','dual50']:raise ValueError('Unknown evaluation architecture.')
    n=TemporalNPN(capacity=capacity,profile=profile,decay=decay,rate=rate)
    if evaluation_view=='dual50':n=DualViewAgent(n)
    sid=uuid.uuid4().hex;commit(n,sid,[point(n)],0);DRAFT=None
def restore():
    global NPN,SESSION,TRACE
    p=STATES/'current.json'
    if not p.exists():init();return
    m=json.loads(p.read_text(encoding='utf8'))
    if not re.fullmatch('[0-9a-f]{32}',m['session']) or type(m['revision']) is not int:raise ValueError('Invalid saved session pointer.')
    loader=DualViewAgent if m.get('evaluationView','observed')=='dual50' else TemporalNPN
    NPN=loader.load(generation_path(m['session'],m['revision']));SESSION=m
    TRACE=json.loads((STATES/m['session']/f"view-{m['revision']:05d}.json").read_text(encoding='utf8'))['trace']
def state():
    return dict(**SESSION,update=NPN.clock,parameters=NPN.count(),activeParameters=NPN.active_count(),config=NPN.config,
        toolResult=result(),trace=TRACE,episodes=NPN.episodes,token=TOKEN,hasHaiku=bool(os.environ.get('ANTHROPIC_API_KEY')))
def check_version(body):
    if body.get('session')!=SESSION['session'] or body.get('revision')!=SESSION['revision']:raise ValueError('The session changed. Reload before applying the experience.')
def mutate(body):
    global DRAFT
    check_version(body);n=NPN.clone();trace=copy.deepcopy(TRACE);action=body['action']
    if action=='elapsed':
        count=body['count']
        if type(count) is not int or count not in [1,8,16,32,64]:raise ValueError('Select a supported update interval.')
        for _ in range(count):n.advance();trace.append(point(n))
    elif action=='recall':n.reactivate(body['sourceId'],'ui-recall-'+str(n.clock+1));trace.append(point(n))
    elif action=='preset':
        if type(body['index']) is not int or not 0<=body['index']<=3:raise ValueError('Invalid scenario index.')
        if body['group']=='past':e=past_event(int(body['index']),n.clock)
        elif body['group']=='current':e=present_event(int(body['index']),n.clock,body.get('sourceId') if int(body['index'])==2 else None)
        else:raise ValueError('Unknown preset group.')
        e['id']='ui-'+str(n.clock+1);n.update(e);trace.append(point(n))
    elif action=='learn':
        if body.get('confirmed') is not True:raise ValueError('Confirm facts and interpretation before learning.')
        if not DRAFT or body.get('draftId')!=DRAFT['id']:raise ValueError('Unknown or stale extraction draft.')
        e=episode('ui-'+str(n.clock+1),text=DRAFT['text'],confidence=float(body['confidence']))
        e['facts']=validate(body['facts']);e['interpretation']=validate(body['interpretation'])
        if body.get('corrects'):
            e['kind']='correction';e['links']=[dict(id=body['corrects'],relation='corrects')]
        n.update(e);trace.append(point(n))
    elif action=='regulate':n.advance();n.short*=.5;n.mood*=.5;trace.append(point(n))
    else:raise ValueError('Unknown mutation.')
    commit(n,SESSION['session'],trace,SESSION['revision']+1)
    if action=='learn':DRAFT=None
    return state()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,format,*args):pass
    def send(self,status,data,kind='application/json; charset=utf-8'):
        raw=json.dumps(data,ensure_ascii=False,allow_nan=False).encode() if 'json' in kind else data
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
    def allowed(self):
        return self.headers.get('Host') in [f'127.0.0.1:{PORT}',f'localhost:{PORT}']
    def do_GET(self):
        if not self.allowed():self.send(403,dict(error='Loopback host required.'));return
        with LOCK:
            if self.path=='/':self.send(200,(ROOT/'playground/index.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/api/state':self.send(200,state())
            elif self.path=='/api/export':
                p=generation_path(SESSION['session'],SESSION['revision']);buffer=io.BytesIO();manifest={}
                with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
                    for suffix in ['.json','.npz']+(['.view.json'] if isinstance(NPN,DualViewAgent) else []):
                        b=Path(str(p)+suffix).read_bytes();z.writestr('personality'+suffix,b);manifest['personality'+suffix]=hashlib.sha256(b).hexdigest()
                    z.writestr('manifest.json',json.dumps(dict(files=manifest,parameterHash=NPN.parameter_hash(),update=NPN.clock,session=SESSION),ensure_ascii=False))
                self.send(200,buffer.getvalue(),'application/zip')
            elif self.path in ['/figure','/results.png']:self.send(200,(ROOT/'reports/results.png').read_bytes(),'image/png')
            elif self.path in ['/review','/review-additional','/report']:
                f=ROOT/'reports'/({'/review':'review.html','/review-additional':'additional-review.html','/report':'dashboard.html'}[self.path])
                if f.exists():self.send(200,f.read_bytes(),'text/html; charset=utf-8')
                else:self.send(404,dict(error='The analysis artifact has not been generated.'))
            else:self.send(404,dict(error='Unknown route.'))
    def do_POST(self):
        global DRAFT
        if not self.allowed() or self.headers.get('Origin') not in [None,f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}'] or self.headers.get('X-Frontalis-Token')!=TOKEN:
            self.send(403,dict(error='Invalid local request token.'));return
        if self.path!='/api/action':self.send(404,dict(error='Unknown route.'));return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=65536:raise ValueError('Request size outside the supported range.')
            body=json.loads(self.rfile.read(length))
            with LOCK:
                action=body['action']
                if action=='init':init(int(body['capacity']),int(body['profile']),body['decay'],float(body['rate']),body.get('evaluationView','observed'));out=state()
                elif action=='interpret':
                    text=body['text']
                    if not isinstance(text,str) or not 1<=len(text)<=4000:raise ValueError('Enter up to 4,000 characters of synthetic experience.')
                    prompt=CONTRACT+'\n'+json.dumps([dict(id='draft',text=text)],ensure_ascii=False)
                    raw=llm(body['engine'],prompt,EXTRACTION_SCHEMA);data=parse(raw)
                    if len(data['records'])!=1 or data['records'][0]['id']!='draft':raise ValueError('Invalid extraction record ID.')
                    r=data['records'][0];facts,normalized_facts=normalize_meaning(r['facts']);interpretation,normalized_interpretation=normalize_meaning(r['interpretation']);DRAFT=dict(id=uuid.uuid4().hex,text=text,engine=body['engine'])
                    out=dict(draftId=DRAFT['id'],facts=facts,interpretation=interpretation,transportNormalized=normalized_facts or normalized_interpretation)
                elif action=='express':
                    check_version(body);tool=result();prompt='指定のNPN信号に忠実に、架空場面へ日本語2〜4文で返答。場面全体sceneと選択対象selected、短期/気分/長期を混同しない。感情は正値だけ。喜び/怒り/悲しみを0や負値から足さない。記憶を推測で追加せず、数値や内部IDを本文へ出さない。人物名や人格説明は与えられていない。JSONのみ {"reply":"自然な返答","stance":"sceneのstance","selection":"selection"}。\n'+json.dumps(dict(text=NOW[0],options=['修復に取り組む','距離を置いて探索する'],toolResult=tool),ensure_ascii=False)
                    raw=llm(body['engine'],prompt);answer=parse(raw)
                    if not isinstance(answer.get('reply'),str) or answer.get('stance')!=tool['scene']['stance'] or answer.get('selection')!=tool['selection']:raise ValueError('Expression metadata did not preserve NPN judgment.')
                    out=dict(answer=answer,toolResult=tool,engine=body['engine']);json_write(STATES/SESSION['session']/('utterance-'+uuid.uuid4().hex+'.json'),out)
                else:out=mutate(body)
                self.send(200,out)
        except (ValueError,KeyError,TypeError,IndexError,json.JSONDecodeError) as e:self.send(400,dict(error=str(e)))
        except Exception as e:self.send(500,dict(error='Operation failed: '+type(e).__name__))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8905);args=p.parse_args();PORT=args.port
    restore();print(f'Phase 3-F workbench http://127.0.0.1:{PORT}',flush=True);ThreadingHTTPServer(('127.0.0.1',PORT),Handler).serve_forever()
