import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npnf.core import *
from npnf.data import *
torch.set_num_threads(1)
if __name__=='__main__':
    folder=ROOT/'examples/demo-output';folder.mkdir(exist_ok=True);rows=[]
    for cap in range(5):
        n=TemporalNPN(capacity=cap)
        for i in range(4):n.update(past_event(2,i))
        n.advance(16)
        for i in range(4):n.update(present_event(2,i,'past-2-0'))
        n.save(folder/f'c{cap}');m=TemporalNPN.load(folder/f'c{cap}')
        if m.parameter_hash()!=n.parameter_hash():raise ValueError('Restoration mismatch.')
        rows.append(dict(capacity=cap,parameters=n.count(),activeParameters=n.active_count(),updates=n.clock,episodes=len(n.episodes),restored=True,evaluation=n.evaluate(present_event(0)['interpretation'])))
    json_write(folder/'summary.json',dict(rows=rows));print('Five capacities learned, elapsed, corrected, saved, and restored without API calls.')
