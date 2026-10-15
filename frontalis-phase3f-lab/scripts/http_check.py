"""Live HTTP verification, including snapshot export and rejected mutation requests."""
import io,json,sys,tempfile,urllib.request,urllib.error,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.core import TemporalNPN,json_write
from npnf.dual_view import DualViewAgent

base='http://127.0.0.1:8905'
def get(path):
    with urllib.request.urlopen(base+path,timeout=20) as r:return r.read()
def post(body,token):
    req=urllib.request.Request(base+'/api/action',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Frontalis-Token':token,'Origin':base})
    try:
        with urllib.request.urlopen(req,timeout=20) as r:return r.status,json.load(r)
    except urllib.error.HTTPError as e:return e.code,json.load(e)

if __name__=='__main__':
    s=json.loads(get('/api/state'));token=s.pop('token');checks=[]
    status,_=post(dict(action='elapsed',count=1,session=s['session'],revision=s['revision']),'invalid');checks.append(dict(name='csrf-rejected',passed=status==403))
    status,_=post(dict(action='elapsed',count=1,session=s['session'],revision=-1),token);checks.append(dict(name='stale-rejected',passed=status==400))
    r=json.loads(get('/api/state'));checks.append(dict(name='read-and-errors-do-not-advance',passed=r['update']==s['update'] and r['toolResult']['parameterHash']==s['toolResult']['parameterHash']))
    suffix='-'+sys.argv[1] if len(sys.argv)>1 else ''
    blob=get('/api/export');(ROOT/('reports/workbench-personality'+suffix+'.zip')).write_bytes(blob)
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            expected={'personality.json','personality.npz','manifest.json'}
            if r.get('evaluationView')=='dual50':expected.add('personality.view.json')
            if set(z.namelist())!=expected:raise ValueError('Unexpected ZIP files.')
            z.extractall(d)
        loader=DualViewAgent if r.get('evaluationView')=='dual50' else TemporalNPN
        n=loader.load(Path(d)/'personality');checks.append(dict(name='live-export-restores-exactly',passed=n.parameter_hash()==r['toolResult']['parameterHash'] and n.clock==r['update']))
    checks.append(dict(name='html-served',passed=b'PROJECT FRONTALIS' in get('/')))
    if not all(x['passed'] for x in checks):raise ValueError('HTTP verification failed.')
    json_write(ROOT/('reports/http-verification'+suffix+'.json'),dict(checks=checks,state={k:v for k,v in r.items() if k!='token'},browserRenderingVerified=False))
    print('HTTP checks passed; screenshot rendering is a separate check.')
