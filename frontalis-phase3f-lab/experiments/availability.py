import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnf.core import *
from npnf.data import *
from experiments.language_plan import extraction
torch.set_num_threads(1)
if __name__=='__main__':
    gold=json.loads((ROOT/'data/natural.json').read_text(encoding='utf8'));rows=[]
    for engine in ['haiku','deepseek','qwen','qwen2048']:
      ex=extraction(engine)
      for cap in range(5):
       for profile in [0,2]:
        for reverse in [False,True]:
            a=TemporalNPN(capacity=cap,profile=profile);b=TemporalNPN.load(ROOT/'models'/f'natural-{engine}-c{cap}-p{profile}-r{int(reverse)}')
            for e in gold[::-1] if reverse else gold:
                if e['id'] in ex:a.update(e)
                else:a.advance()
            errors=[]
            for i in range(4):
                m=present_event(i)['interpretation'];errors.append(float(np.mean(np.abs(np.asarray(a.evaluate(m)['longTerm'])-b.evaluate(m)['longTerm']))))
            rows.append(dict(engine=engine,capacity=cap,profile=profile,reverse=reverse,matchedExamples=len(ex),meaningOnlyMAE=float(np.mean(errors)) if ex else None))
    json_write(ROOT/'reports/availability-matched.json',dict(rows=rows))
