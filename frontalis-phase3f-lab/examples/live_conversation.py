"""An actual six-turn stateful conversation through the running isolated workbench."""
import json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.core import json_write
BASE='http://127.0.0.1:8905'
def get():
    with urllib.request.urlopen(BASE+'/api/state',timeout=20) as r:return json.load(r)
def call(body,state):
    req=urllib.request.Request(BASE+'/api/action',data=json.dumps(dict(body,session=state['session'],revision=state['revision'])).encode(),headers={'Content-Type':'application/json','Origin':BASE,'X-Frontalis-Token':state['token']})
    with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
if __name__=='__main__':
    dual='--dual' in sys.argv;suffix='-dual' if dual else ''
    s=get();s=call(dict(action='init',capacity=3,profile=2,decay='consolidation',rate=.01,evaluationView='dual50' if dual else 'observed'),s);rows=[]
    stages=['初期','未確認の裏切りという解釈を8回経験','16更新経過','最初の記録を反すう','責任がなかったという訂正','謝罪と修復を8回経験']
    for stage in range(6):
        if stage==1:
            for _ in range(8):s=call(dict(action='preset',group='past',index=2),s)
        elif stage==2:s=call(dict(action='elapsed',count=16),s)
        elif stage==3:s=call(dict(action='recall',sourceId='ui-1'),s)
        elif stage==4:s=call(dict(action='preset',group='current',index=2,sourceId='ui-1'),s)
        elif stage==5:
            for _ in range(8):s=call(dict(action='preset',group='current',index=0),s)
        reply=call(dict(action='express',engine='haiku'),s)
        rows.append(dict(stage=stages[stage],update=s['update'],answer=reply['answer'],signal=reply['toolResult'],parameterHash=s['toolResult']['parameterHash']))
        print(s['update'],stages[stage],reply['answer']['reply'],flush=True)
    json_write(ROOT/('reports/workbench-conversation'+suffix+'.json'),dict(session=s['session'],stages=rows,formalHumanEvaluation=False))
    with urllib.request.urlopen(BASE+'/api/export',timeout=30) as r:(ROOT/('reports/workbench-live'+suffix+'-personality.zip')).write_bytes(r.read())
