"""Materialize fixed dialogue selection, independent of observed replies."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npne.meaning import EXTRACTION_CONTRACT

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='contract':
        with (ROOT/'data/extraction-contract.txt').open('x',encoding='utf8') as f:f.write(EXTRACTION_CONTRACT)
        # Frozen selection before learning results and before reply generation.
        selection=dict(corpusIds=['s00','s04','s16','s24'],capacityIndices=[0,3,4],seriesByScene=[0,1,2,3],states=['initial','formed'],seed=5311,representation='semantic44',loss='bounded_logit')
        write(ROOT/'data/dialogue-selection.json',selection)
    elif mode=='materialize':
        from npne.core import Personality,tool_evaluate
        from npne.data import WIDTHS
        data={r['id']:r for r in json.loads((ROOT/'data/natural.json').read_text(encoding='utf8'))};s=json.loads((ROOT/'data/dialogue-selection.json').read_text(encoding='utf8'));cases=[]
        for cap in s['capacityIndices']:
          for index,id in enumerate(s['corpusIds']):
            r=data[id];m=r['gold'];a=copy.deepcopy(m);b=copy.deepcopy(m)
            # Declared thought-experiment outcomes: two explicit factual trade-offs.
            a['outcomeFacts8']=[1,0,.8,.8,1,1,.1,1];b['outcomeFacts8']=[.1,.8,.1,0,.5,.1,1,.2]
            options=[dict(id='A',text='変更と探索を認める。成果と発見はあるが材料を多く失う。',meaning=a),dict(id='B',text='手順を固定して変更を制限する。材料を失わないが発見は少ない。',meaning=b)]
            for state in s['states']:
                prefix=f'models/main-5311-semantic44-c{cap}-bounded_logit-h{s["seriesByScene"][index]}'
                n=Personality(5311,WIDTHS['semantic44'][cap],'semantic44','bounded_logit') if state=='initial' else Personality.load(ROOT/prefix)
                cases.append(dict(id=f'c{cap}-s{index}-{state}',capacityIndex=cap,state=state,sourceId=id,text=r['text']+' この場面全体にどう感じる？次にAとBなら自分はどちらを選ぶ？',options=options,modelPrefix=None if state=='initial' else prefix,toolResult=tool_evaluate(n,m,options)))
        write(ROOT/'data/dialogue-plan.json',dict(selection=s,cases=cases))
