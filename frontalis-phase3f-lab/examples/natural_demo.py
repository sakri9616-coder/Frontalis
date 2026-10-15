import json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples.live_conversation import get,call,BASE
from npnf.core import json_write
if __name__=='__main__':
    s=get();s=call(dict(action='init',capacity=2,profile=0,decay='consolidation',rate=.01,evaluationView='dual50'),s)
    text='友人が私を助けようとしたが、結果として損失が出て、成果はなかった。'
    d=call(dict(action='interpret',engine='haiku',text=text),s);after_interpret=get()
    if after_interpret['update']!=s['update'] or after_interpret['toolResult']['parameterHash']!=s['toolResult']['parameterHash']:raise ValueError('Interpretation unexpectedly learned.')
    before=s['toolResult']['parameterHash'];s=call(dict(action='learn',draftId=d['draftId'],confirmed=True,confidence=.8,facts=d['facts'],interpretation=d['interpretation'],corrects=None),s)
    if s['update']!=1 or s['toolResult']['parameterHash']==before:raise ValueError('Confirmed synthetic experience did not learn.')
    json_write(ROOT/'reports/natural-workbench-check.json',dict(text=text,draft=d,update=s['update'],learned=True,interpretationReadOnly=True,formalHumanVerification=False))
    print('Typed natural interpretation -> editable facts/belief -> explicit synthetic confirmation -> learned state verified.')
