import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnf.core import *
from npnf.data import *
from npnf.meaning import meaning,event
from experiments.language_plan import pack
if __name__=='__main__':
    cases=[];metrics=[]
    options=[dict(id='friend',text='友人に協力を頼む。結果はまだ不明。',meaning=meaning([event('help',actor='close_other')],cue=CUE,outcome=CUE)),
             dict(id='new_person',text='知らない人に協力を頼む。結果はまだ不明。',meaning=meaning([event('help',actor='stranger')],cue=CUE,outcome=CUE))]
    for cap in range(5):
      for state in ['initial','cooperation','misunderstanding','corrected']:
        n=TemporalNPN.load(ROOT/'models'/f'dialogue-c{cap}-{state}');m=present_event(3)['interpretation'];tool=tool_evaluate(n,m,options)
        cases.append(dict(id=f'future-c{cap}-{state}',text='次の作業で協力を頼む相手を選ぶ。どちらの結果もまだ分からない。',meaning=m,
            options=[dict(id=o['id'],text=o['text']) for o in options],language='ja',toolResult=tool,history=[r['episode'] for r in n.episodes]))
        metrics.append(dict(capacity=cap,state=state,selection=tool['selection'],candidateValues=[o['appraisal']['combined'][2] for o in tool['candidates']]))
    for engine in ['haiku','deepseek']:json_write(ROOT/f'data/future-{engine}.json',dict(cases=cases))
    json_write(ROOT/'reports/future-choice.json',dict(type='Post hoc exploratory; not the primary dialogue comparison',metrics=metrics))
