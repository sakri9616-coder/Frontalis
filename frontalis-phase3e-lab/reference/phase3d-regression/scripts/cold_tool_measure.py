"""Supplementary process-start measurements, separate from warm API timing."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path(r'C:\Users\sakur\AppData\Local\hermes\jef\venv\Scripts\python.exe')
rows=[]
for width in (16,31250):
    prefix=ROOT/f'models/raw8/bounded_signal/nn-h{width}/seed4311-p0'
    payload={'items':[{'modelPrefix':str(prefix),'calls':[{'name':'frontalis_evaluate','arguments':{'query_id':'cold-test','space':'relational','features':[.3,.65,.8,.45,.7,.2,.25,.8],'basis':'affinity'}}]}]}
    samples=[];hashes=[]
    for _ in range(3):
        start=time.perf_counter()
        result=subprocess.run([str(PYTHON),'-u',str(ROOT/'tools/evaluate.py')],input=json.dumps(payload),text=True,encoding='utf8',capture_output=True,cwd=ROOT,
            env={**os.environ,'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'},timeout=120)
        seconds=time.perf_counter()-start
        if result.returncode:raise RuntimeError(result.stderr)
        tool=json.loads(result.stdout);hashes.append(tool['toolResult']['results'][0]['parameterHash']);samples.append(seconds)
    rows.append({'width':width,'processSeconds':samples,'medianSeconds':statistics.median(samples),'sameReadoutParameterHash':len(set(hashes))==1})
value={'postResultPerformanceCheck':True,'scope':'process start + Torch import + state load + one tool appraisal; filesystem cache not reset; ordinary concurrent desktop/diagnostic load',
       'python':str(PYTHON),'conditions':rows}
with (ROOT/'reports/cold-tool-performance.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
