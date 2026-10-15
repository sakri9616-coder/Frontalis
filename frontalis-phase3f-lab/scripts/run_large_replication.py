"""Execution-only scheduling of the large capacity by disjoint seed paths."""
import argparse,concurrent.futures,itertools,subprocess,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def worker(seed):
    import experiments.replication_longrun as e
    original_product=itertools.product
    def selected_product(*args):
        if len(args)!=4 or list(args[0])!=e.SEEDS or list(args[1])!=list(range(5)) or list(args[2])!=[0,2]:
            raise ValueError('Unexpected frozen loop; scheduling aborted.')
        return (v for v in original_product(*args) if v[0]==seed and v[1]==4)
    e.itertools=types.SimpleNamespace(product=selected_product)
    e.verify();rows=e.longrun()
    if len(rows)!=4 or any(r['capacity']!=4 or r['seed']!=seed for r in rows):raise ValueError('Worker condition count mismatch.')

def launch(seed):
    with (ROOT/f'reports/replication-large-s{seed}.txt').open('w',encoding='utf8') as f:
        r=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--seed',str(seed)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Large capacity worker failed: '+str(seed))
    print('Large-capacity seed complete:',seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[7901,7902,7903]);a=p.parse_args()
    if a.seed is not None:worker(a.seed)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(launch,[7901,7902,7903]))
        subprocess.run([sys.executable,'-B',str(ROOT/'experiments/replication_longrun.py'),'all'],cwd=ROOT,check=True)
