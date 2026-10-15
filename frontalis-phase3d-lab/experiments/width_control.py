"""Auxiliary width-normalization control, not a personality success test."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnd.core import Personality
from npnd.data import training,held_out
torch.set_num_threads(1)
rows=[];query=held_out(phase='validation');events=training(count=8)
def outputs(m):return np.asarray([m.evaluate(e['cue'],e['space'])['raw'] for e in query])
for width in (16,64,312,3125):
    for scaled in (True,False):
        m=Personality(width=width,scaled=scaled);initial=outputs(m);item={'width':width,'parameters':m.parameter_count(),'scaled':scaled,'checkpoints':{}}
        for i,e in enumerate(events,1):
            m.update(e)
            if i in (1,8):
                out=outputs(m);item['checkpoints'][str(i)]={'meanInitialChange':float(np.abs(out-initial).mean()),'saturationFraction':float((np.abs(out)>.98).mean()),'finite':bool(np.isfinite(out).all())}
        rows.append(item)
value={'postMainAuxiliary':True,'purpose':'規格化なしで幅の増加が最初の関数更新を大きくするかの感度試験。主比較の条件・選択を変えない。','conditions':rows}
with (ROOT/'reports/width-control.json').open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
print(json.dumps(rows))
