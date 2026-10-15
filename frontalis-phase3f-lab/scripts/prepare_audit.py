import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze import conversations
from npnf.meaning import digest
from npnf.core import json_write
out=conversations()['outputs'];cases=[];key=[]
for o in out:
    id=digest([o['engine'],o['mode'],o['case']['id']])[:20];c=o['case']
    cases.append(dict(id=id,text=c['text'],meaning=c['meaning'],options=c['options'],language=c['language'],toolResult=c['toolResult'],reply=o['response']['reply']))
    key.append(dict(id=id,engine=o['engine'],mode=o['mode'],case=c['id']))
cases.sort(key=lambda c:digest(c['id']));json_write(ROOT/'reports/audit-plan.json',dict(cases=cases));json_write(ROOT/'reports/audit-key.json',dict(key=key))
print('Prepared',len(cases),'anonymous prose checks.')
