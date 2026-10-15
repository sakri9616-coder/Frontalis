"""Execution-only scheduling of frozen, independent capacity conditions.

No numerical routine, parameter, seed, update order within a condition, or
cache format is changed. Each worker owns exactly one capacity's cache/model
paths. The original frozen program aggregates and re-verifies the full set.
"""
import argparse,concurrent.futures,itertools,subprocess,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def worker(capacity):
    import experiments.replication_longrun as e
    original_product=itertools.product
    def capacity_product(*args):
        if len(args)!=4 or list(args[0])!=e.SEEDS or list(args[1])!=list(range(5)) or list(args[2])!=[0,2]:
            raise ValueError('Unexpected frozen experiment loop; scheduling aborted.')
        return (v for v in original_product(*args) if v[1]==capacity)
    e.itertools=types.SimpleNamespace(product=capacity_product)
    e.verify();rows=e.longrun()
    if len(rows)!=12 or any(r['capacity']!=capacity for r in rows):raise ValueError('Worker condition count mismatch.')

def launch(capacity):
    with (ROOT/f'reports/replication-worker-c{capacity}.txt').open('w',encoding='utf8') as f:
        r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--capacity',str(capacity)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Capacity worker failed: '+str(capacity))
    print('Capacity complete:',capacity,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capacity',type=int,choices=range(5));a=p.parse_args()
    if a.capacity is not None:worker(a.capacity)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(launch,range(5)))
        subprocess.run([sys.executable,'-B',str(ROOT/'experiments/replication_longrun.py'),'all'],cwd=ROOT,check=True)
