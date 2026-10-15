"""Local research workbench. Immutable generations; no body/memory integration."""
import argparse,copy,hashlib,io,json,os,re,secrets,sys,threading,time,urllib.error,urllib.request,uuid,zipfile
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from npne.core import Personality,experience,tool_evaluate
from npne.meaning import EXTRACTION_CONTRACT,validate,digest
from npne.data import WIDTHS
from npne.learning import prepare_experience
torch.set_num_threads(1)
STORE=ROOT/'playground/states';STORE.mkdir(parents=True,exist_ok=True)
TOKEN=secrets.token_hex(24);SESSIONS={};LOCK=threading.RLock()
MODEL='claude-haiku-5-5'

def json_write(path,x,exclusive=True):
    with path.open('x' if exclusive else 'w',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,indent=2)
def folder(id):
    if not isinstance(id,str) or not re.fullmatch('[a-f0-9]{32}',id):raise ValueError('不正な研究セッションIDです。')
    return STORE/id
def pointer(id,prefix):
    d=folder(id);tmp=d/('pointer-'+uuid.uuid4().hex+'.tmp');json_write(tmp,dict(prefix=prefix));os.replace(tmp,d/'current.json')
def current(id):
    d=folder(id);p=json.loads((d/'current.json').read_text(encoding='utf8'))['prefix']
    if not re.fullmatch(r'g[0-9]+-[a-f0-9]{32}',p):raise ValueError('保存された世代IDが不正です。')
    return d,p
def session(id):
    if id not in SESSIONS:
        d,p=current(id);origin=json.loads((d/'origin.json').read_text(encoding='utf8'));SESSIONS[id]=dict(model=Personality.load(d/p),pending={},prefix=p,outcomeView=origin.get('outcomeView','full'))
    return SESSIONS[id]
def state(id):
    s=session(id);n=s['model'];d=folder(id)
    versions=[dict(prefix=p.stem,generation=json.loads(p.read_text(encoding='utf8'))['metadata']['generation']) for p in d.glob('g*.json')]
    versions.sort(key=lambda v:(v['generation'],v['prefix']))
    return dict(sessionId=id,generation=n.generation,parameterHash=n.parameter_hash(),parameterCount=n.count(),representation=n.config['representation'],capacityIndex=WIDTHS[n.config['representation']].index(n.config['width']),prefix=s['prefix'],versions=versions,outcomeView=s.get('outcomeView','full'))
def commit(id,n):
    d=folder(id);prefix=f'g{n.generation}-{uuid.uuid4().hex}';n.save(d/prefix);pointer(id,prefix);SESSIONS[id]['model']=n;SESSIONS[id]['prefix']=prefix
def haiku(prompt):
    key=os.environ.get('ANTHROPIC_API_KEY')
    if not key:raise ValueError('ANTHROPIC_API_KEYが実行環境にありません。')
    data=json.dumps(dict(model=MODEL,max_tokens=8192,messages=[dict(role='user',content=prompt)])).encode('utf8')
    req=urllib.request.Request('https://api.anthropic.com/v1/messages',data=data,headers={'content-type':'application/json','anthropic-version':'2023-06-01','x-api-key':key},method='POST')
    try:
        with urllib.request.urlopen(req,timeout=180) as r:raw=json.loads(r.read().decode('utf8'))
    except urllib.error.HTTPError as e:raise ValueError(f'Haiku HTTP {e.code}。キーは表示しません。') from None
    except (urllib.error.URLError,TimeoutError):raise ValueError('Haikuへの接続に失敗しました。') from None
    text='\n'.join(c['text'] for c in raw.get('content',[]) if c.get('type')=='text')
    return text,dict(model=raw.get('model'),usage=raw.get('usage'),responseId=raw.get('id'))
def interpretation(text):
    raw,receipt=haiku(EXTRACTION_CONTRACT+'\n'+json.dumps([dict(id='interactive',text=text)],ensure_ascii=False))
    cleaned=re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip())
    try:r=json.loads(cleaned);m=validate(r['records'][0]['meaning'])
    except (ValueError,KeyError,IndexError,TypeError):raise ValueError('意味抽出のschemaが不正です。学習は行っていません。') from None
    return m,receipt

def dispatch(route,b):
    if route=='/api/new':
        rep=b.get('representation');cap=b.get('capacityIndex');formed=b.get('formed',False);view=b.get('outcomeView','full')
        if rep not in WIDTHS or type(cap) is not int or not 0<=cap<5 or type(formed) is not bool:raise ValueError('容量・入力方式が不正です。')
        if view not in ['full','facts8-neutral-context']:raise ValueError('学習条件が不正です。')
        n=Personality.load(ROOT/'models'/f'main-5311-{rep}-c{cap}-bounded_logit-h0') if formed else Personality(5311,WIDTHS[rep][cap],rep,'bounded_logit')
        id=uuid.uuid4().hex;folder(id).mkdir();SESSIONS[id]=dict(model=n,pending={},prefix=None,outcomeView=view);commit(id,n);json_write(folder(id)/'origin.json',dict(seed=5311,representation=rep,capacityIndex=cap,fromFormed=formed,outcomeView=view,exploratory=True));return state(id)
    id=b.get('sessionId');s=session(id);n=s['model']
    if route=='/api/resume':return state(id)
    if route=='/api/interpret':
        text=b.get('text')
        if not isinstance(text,str) or not text.strip() or len(text)>4000:raise ValueError('経験を1〜4000文字で入力してください。')
        m,receipt=interpretation(text);pending=uuid.uuid4().hex;s['pending'][pending]=dict(text=text,meaning=m,receipt=receipt,generation=n.generation,used=False)
        return dict(pendingId=pending,meaning=m,appraisal=tool_evaluate(n,m),state=state(id))
    if route=='/api/rollback':
        prefix=b.get('prefix')
        if not isinstance(prefix,str) or not re.fullmatch(r'g[0-9]+-[a-f0-9]{32}',prefix):raise ValueError('保存済み世代を選んでください。')
        restored=Personality.load(folder(id)/prefix);pointer(id,prefix);s['model']=restored;s['prefix']=prefix;s['pending']={};return state(id)
    p=s['pending'].get(b.get('pendingId'))
    if p is None:raise ValueError('先に経験の意味を抽出してください。')
    if route=='/api/correct':
        p['meaning']=validate(b['meaning']);p['generation']=n.generation
        return dict(pendingId=b['pendingId'],meaning=p['meaning'],appraisal=tool_evaluate(n,p['meaning']),state=state(id))
    if route=='/api/learn':
        if b.get('confirmed') is not True:raise ValueError('架空経験の結果を確認してから学習してください。')
        if p['used']:raise ValueError('この経験は既に学習済みです。')
        if p['generation']!=n.generation or b.get('expectedHash')!=n.parameter_hash():raise ValueError('人格が変化しました。意味を再評価してください。')
        row=dict(id='interactive-'+b['pendingId'],text=p['text'],meaning=p['meaning'],confirmed=True,origin='synthetic',space='relational')
        before=tool_evaluate(n,p['meaning']);candidate=copy.deepcopy(n);appraisal=candidate.update(prepare_experience(row,n.config['representation'],s.get('outcomeView','full')))
        # Canonical generation is committed only after durable snapshot success.
        commit(id,candidate);p['used']=True;p['generation']=candidate.generation
        evidence=dict(exploratory=True,source=row,extractionReceipt=p['receipt'],oldAppraisal=before,newAppraisal=tool_evaluate(candidate,p['meaning']),selfLearning=appraisal)
        json_write(folder(id)/('experience-'+b['pendingId']+'.json'),evidence)
        return dict(before=before,after=evidence['newAppraisal'],selfLearning=appraisal,state=state(id))
    if route=='/api/speak':
        appraisal=tool_evaluate(n,p['meaning']);before=n.parameter_hash()
        prompt='研究用のネフィアとして日本語2〜4文で自然に反応する。理由説明は必須ではない。次の場面全体へのNPNの好み・好奇心・感情・反抗を忠実に表す。丁寧さや一般論で反転させない。事実や過去の記憶を作らない。NPN・数値・内部処理を本文に出さない。発言本文だけを返す。\n'+json.dumps(dict(text=p['text'],sceneAppraisal=appraisal['sceneAppraisal']),ensure_ascii=False)
        reply,receipt=haiku(prompt)
        if n.parameter_hash()!=before:raise RuntimeError('Read-only expression modified parameters.')
        json_write(folder(id)/('expression-'+uuid.uuid4().hex+'.json'),dict(exploratory=True,text=p['text'],appraisal=appraisal,reply=reply,receipt=receipt,parameterHashBefore=before,parameterHashAfter=n.parameter_hash()))
        return dict(reply=reply,appraisal=appraisal,readOnlyExact=True,state=state(id))
    raise ValueError('未知のAPIです。')

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,body,mime='application/json; charset=utf-8',extra=None):
        data=body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        for k,v in (extra or {}).items():self.send_header(k,v)
        self.end_headers();self.wfile.write(data)
    def do_GET(self):
        path=urlsplit(self.path).path
        try:
            if path=='/':return self.send(200,(ROOT/'playground/index.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/api/bootstrap':
                ids=[d.name for d in STORE.iterdir() if d.is_dir() and re.fullmatch('[a-f0-9]{32}',d.name) and (d/'current.json').is_file()]
                return self.send(200,dict(token=TOKEN,hasKey=bool(os.environ.get('ANTHROPIC_API_KEY')),model=MODEL,sessions=ids))
            if path=='/api/export':
                id=parse_qs(urlsplit(self.path).query).get('sessionId',[''])[0];d,p=current(id);buf=io.BytesIO()
                with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                    for file in [d/(p+'.json'),d/(p+'.npz'),d/'origin.json',*d.glob('experience-*.json'),*d.glob('expression-*.json')]:z.write(file,file.name)
                return self.send(200,buf.getvalue(),'application/zip',{'Content-Disposition':'attachment; filename="frontalis-personality.zip"'})
            return self.send(404,dict(error='Not found'))
        except (ValueError,KeyError,FileNotFoundError):return self.send(400,dict(error='セッションが見つかりません。'))
    def do_POST(self):
        if self.headers.get('X-Frontalis-Token')!=TOKEN:return self.send(403,dict(error='セッショントークンが不正です。'))
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://{self.headers.get("Host")}':return self.send(403,dict(error='Origin mismatch'))
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=65536:raise ValueError('要求サイズが不正です。')
            body=json.loads(self.rfile.read(size))
            if not isinstance(body,dict):raise ValueError('JSON object required.')
            with LOCK:result=dispatch(urlsplit(self.path).path,body)
            self.send(200,result)
        except (ValueError,KeyError,FileNotFoundError,TypeError) as e:self.send(400,dict(error=str(e)))
        except Exception:self.send(500,dict(error='保存または処理に失敗しました。既存の世代ファイルは保持しています。'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8903);a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);print(f'http://127.0.0.1:{a.port}',flush=True);server.serve_forever()
