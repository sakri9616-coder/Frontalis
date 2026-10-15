import hashlib,json,os,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import psutil
import torch
from npnf.core import *
from npnf.data import *
from scripts.freeze import verify
torch.set_num_threads(1)

if __name__=='__main__':
    verify();rows=[];pairs=[]
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
      for cap in range(5):
        n=TemporalNPN(capacity=cap,device=device)
        for i in range(24):n.update(past_event(i%4,i,.08,6311))
        x=np.stack([encode(present_event(i)['interpretation']) for i in range(4)])
        for _ in range(5):n.logits(x)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for _ in range(40):n.logits(x)
        if device=='cuda':torch.cuda.synchronize()
        eval_ms=(time.perf_counter()-start)*1000/40/4
        start=time.perf_counter()
        for i in range(8):n.update(past_event(i%4,i+1000,.08,6311))
        if device=='cuda':torch.cuda.synchronize()
        update_ms=(time.perf_counter()-start)*1000/8
        mem=psutil.Process().memory_full_info()
        rows.append(dict(device=device,capacity=cap,parameters=n.count(),activeCount=n.active_count(),evalMsPerItemInBatch4=eval_ms,updateMs=update_ms,
            workingSetBytes=mem.rss,privateBytes=getattr(mem,'private',None),cudaAllocated=torch.cuda.memory_allocated() if device=='cuda' else 0,
            cudaReserved=torch.cuda.memory_reserved() if device=='cuda' else 0,cudaPeakAllocated=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
        print(device,cap,round(eval_ms,3),round(update_ms,3),flush=True)
    if torch.cuda.is_available():
      for cap in range(5):
        cpu=TemporalNPN(capacity=cap)
        for i in range(8):cpu.update(past_event(i%4,i,.08,6311))
        p=ROOT/'models'/f'device-c{cap}';cpu.save(p);gpu=TemporalNPN.load(p,'cuda')
        x=np.stack([encode(present_event(i)['interpretation']) for i in range(4)])
        same=rms=float(np.max(np.abs(cpu.logits(x)-gpu.logits(x))))
        for i in range(4):cpu.update(past_event(i%4,i+100));gpu.update(past_event(i%4,i+100))
        pairs.append(dict(capacity=cap,sameWeightMaxError=same,afterFourUpdatesMaxError=float(np.max(np.abs(cpu.logits(x)-gpu.logits(x))))))
    json_write(ROOT/'reports/performance.json',dict(sourceHash=verify(),rows=rows,devicePairs=pairs,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
