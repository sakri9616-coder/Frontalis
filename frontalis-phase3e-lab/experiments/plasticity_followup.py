"""Post-hoc, independently frozen outcome-context ablation; no external targets."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import run
from npne.data import SEEDS

paths=['docs/01b-plasticity-followup.md','experiments/plasticity_followup.py']
seal={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
freeze=ROOT/'reference/plasticity-followup-freeze.json'
if sys.argv[1]=='freeze':
    run.write(freeze,dict(posthoc=True,sources=seal,mainResultSha256=hashlib.sha256((ROOT/'reports/numerical-summary.json').read_bytes()).hexdigest(),resultsInspected=False))
    print('Additional six conditions frozen.');raise SystemExit
old=json.loads(freeze.read_text(encoding='utf8'))
if old['sources']!=seal:raise ValueError('Additional frozen source changed.')
run.torch.set_num_threads(1);run.torch.use_deterministic_algorithms(True);run.frozen()
rows=[];main=json.loads((ROOT/'reports/stress.json').read_text(encoding='utf8'))['conditions']
for seed in SEEDS:
 for cap in [0,3]:
    start=time.perf_counter();rep='semantic44';n=run.model(seed,rep,cap,'bounded_logit','cuda');other=n.branch_hash('acoustic');anchor=run.stress_event(seed,0)['meaning'];cue=run.encode(anchor,rep,False)
    rng=run.np.random.default_rng(seed+765);related=[]
    for _ in range(32):
        m=run.copy.deepcopy(anchor);m['cueFacts8']=run.np.clip(run.np.asarray(anchor['cueFacts8'])+rng.normal(0,.04,8),0,1).tolist();m['outcomeFacts8']=m['cueFacts8'];related.append(m)
    x=run.features(related,rep);far=run.features(run.random_meanings(seed+786,32),rep);check=[]
    for i in range(1097):
        if i in [0,64,256,1000,1012,1048,1096]:
            y=n.evaluate_batch(x);f=n.evaluate_batch(far);q=n.evaluate_batch([cue])[0];check.append(dict(step=i,cue=q.tolist(),related=y.tolist(),distant=f.tolist(),metrics=run.stats(y),parameterHash=n.parameter_hash(),lastUpdate=n.history[-1]['training'] if n.history else None))
            if i==1000:strong=y.copy();strongfar=f.copy();strongcue=q.copy()
        if i<1096:
            e=run.experience(run.stress_event(seed,i,i>=1000),rep);e['observed'][8:]=[.5]*36
            # Provenance distinguishes an altered appraisal view from original meaning.
            e['sourceHash']=run.digest(dict(originalSourceHash=e['sourceHash'],view='facts8-neutral-context-v1'))
            n.update(e)
    prefix=f'models/followup-{seed}-semantic44-c{cap}-bounded_logit';n.save(ROOT/prefix)
    original=next(r for r in main if (r['seed'],r['representation'],r['capacityIndex'],r['loss'])==(seed,rep,cap,'bounded_logit'))
    change=float(run.np.mean(run.np.abs(q-strongcue)));row=dict(seed=seed,capacityIndex=cap,parameterCount=n.count(),cueChange=change,originalCueChange=original['oppositeCueChange'],relatedChange=float(run.np.mean(run.np.abs(y-strong))),distantChange=float(run.np.mean(run.np.abs(f-strongfar))),unrelatedExact=other==n.branch_hash('acoustic'),finite=True,criterion=change>.05 and change>original['oppositeCueChange'],modelPrefix=prefix,checkpoints=check,elapsedSeconds=time.perf_counter()-start)
    run.write(ROOT/'reports'/f'followup-{seed}-c{cap}.json',row);rows.append(row);print(json.dumps({k:row[k] for k in ['seed','capacityIndex','cueChange','originalCueChange','criterion','elapsedSeconds']}),flush=True)
run.write(ROOT/'reports/plasticity-followup.json',dict(posthoc=True,freezeSha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),conditions=rows,criterionFraction=sum(r['criterion'] for r in rows)/6,success=sum(r['criterion'] for r in rows)>=5 and all(r['unrelatedExact'] for r in rows)))
