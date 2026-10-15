"""Final evidence checks. This script never calls an external LLM."""
import hashlib,io,json,os,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT.parent/'frontalis-phase3e-lab'))
import numpy as np
import torch
from npnf.core import TemporalNPN
from npnf.dual_view import DualViewAgent
from npnf.data import present_event
from npne.core import Personality
from scripts.freeze import verify
torch.set_num_threads(1)

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    primary=verify();extra={}
    for p in sorted((ROOT/'reference').glob('*-freeze.json')):
        reg=read(p)
        for name,h in reg.get('sources',reg).items():
            if sha(ROOT/name)!=h:raise ValueError('Additional frozen source changed: '+name)
        extra[p.name]=sha(p)
    prior=read(ROOT/'reference/phase3e-artifact-sha256.json');old=[]
    for item in prior['files']:old.append(dict(path=str(ROOT.parent/'frontalis-phase3e-lab'/item['path']),bytes=item['bytes'],sha256=item['sha256']))
    older=read(ROOT.parent/'frontalis-phase3e-lab/reference/prior-sha256.json');old+=older
    changed=[]
    for item in old:
        p=Path(item['path'])
        if not p.exists() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:changed.append(str(p))
    if changed:raise ValueError('Protected prior artifacts differ: '+str(changed[:5]))
    source=read(ROOT/'reference/asist-source-sha256.json')
    for p,h in source.items():
        if sha(ROOT.parent/'asist'/p)!=h:raise ValueError('Read-only ASIST source changed: '+p)
    models=[]
    for p in sorted((ROOT/'models').glob('*.npz')):
        prefix=p.with_suffix('')
        if prefix.name.startswith('structure-'):n=Personality.load(prefix);output=list(n.evaluate([.5]*44)['channels'].values())
        elif prefix.name.startswith('dual-'):n=DualViewAgent.load(prefix);output=n.evaluate(present_event(0)['interpretation'])['longTerm']
        else:n=TemporalNPN.load(prefix);output=n.evaluate(present_event(0)['interpretation'])['longTerm']
        n.stable()
        if len(output)!=7 or not np.isfinite(output).all():raise ValueError('Model readout failed.')
        models.append(dict(id=prefix.name,parameterHash=n.parameter_hash(),parameters=n.count()))
    if len(models)!=265:raise ValueError('Expected 265 immutable research model states.')
    additional_models=[]
    for p in sorted((ROOT/'models/replication-longrun').glob('*.npz')):
        n=TemporalNPN.load(p.with_suffix(''));n.stable()
        additional_models.append(dict(id='replication-longrun/'+p.stem,parameterHash=n.parameter_hash(),parameters=n.count(),update=n.clock))
    if len(additional_models)!=5:raise ValueError('Expected five additional long-run snapshots.')
    replication=read(ROOT/'reports/replication-longrun.json')
    from experiments.replication_longrun import verify as verify_replication
    if replication['sourceHash']!=verify_replication()[0] or len(replication['replication'])!=60 or len(replication['longrun'])!=60:raise ValueError('Additional research provenance/condition count mismatch.')
    points=[p for r in replication['longrun'] for p in r['checkpoints']]
    if len(points)!=120 or not all(p['oldRecordPreserved'] and p['numericallyFinite'] for p in points):raise ValueError('Additional long-run integrity check failed.')
    expected_additional={p['savedModel'].removeprefix('models/'):p['finalHash'] for p in points if p['savedModel']}
    if {m['id']:m['parameterHash'] for m in additional_models}!=expected_additional:raise ValueError('Additional snapshots do not match recorded final states.')
    exports=[]
    for p in sorted((ROOT/'reports').glob('workbench-*.zip')):
        with tempfile.TemporaryDirectory() as d:
            with zipfile.ZipFile(p) as z:
                names=set(z.namelist())
                if not names<={'personality.json','personality.npz','personality.view.json','manifest.json'} or not {'personality.json','personality.npz','manifest.json'}<=names:raise ValueError('Unexpected ZIP contents.')
                manifest=json.loads(z.read('manifest.json'))
                for name,h in manifest['files'].items():
                    if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('ZIP manifest mismatch.')
                z.extractall(d)
            loader=DualViewAgent if 'personality.view.json' in names else TemporalNPN
            n=loader.load(Path(d)/'personality')
            if n.parameter_hash()!=manifest['parameterHash'] or n.clock!=manifest['update']:raise ValueError('Export restoration mismatch.')
            exports.append(dict(file=p.name,update=n.clock,parameterHash=n.parameter_hash()))
    for file,count in [('tests-complete.txt',33),('phase3e-regression.txt',34),('phase3d-regression.txt',94)]:
        text=(ROOT/'reports'/file).read_text(encoding='utf8')
        if f'Ran {count} tests' not in text or '\nOK' not in text or 'FAILED' in text:raise ValueError('Tests not confirmed: '+file)
    semantic=read(ROOT/'reports/extended-analysis.json');tool=semantic['actualTools']
    if tool['total']!=10 or tool['requestExact']!=10 or tool['snapshotSignalsVerified']!=10:raise ValueError('Actual NPN tool execution evidence incomplete.')
    for p in (ROOT/'reports/live/structured').glob('tool-response-*.json'):
        raw=read(p)
        if any(b['type']=='tool_use' for b in raw['response']['content']):raise ValueError('Unexecuted extra tool request.')
    for p in (ROOT/'reports/live/deepseek').glob('*.json'):
        raw=read(p);r=raw['response']
        if r and (r.get('cost',0)!=0 or r.get('tools',0)!=0):raise ValueError('Unexpected free CLI cost or tools.')
    for key in ['performance.json','performance-repeat.json']:
        r=read(ROOT/'reports'/key)
        if len(r['rows'])!=10 or any(v['sameWeightMaxError']>1e-12 or v['afterFourUpdatesMaxError']>1e-12 for v in r['devicePairs']):raise ValueError('Device calibration mismatch.')
    key=os.environ.get('ANTHROPIC_API_KEY');leaks=[]
    for p in ROOT.rglob('*'):
        if p.is_file() and 'states' not in p.relative_to(ROOT).parts and p.suffix in ['.json','.md','.txt','.html','.mjs','.py','.ps1','.csv']:
            if key and key.encode() in p.read_bytes():leaks.append(p.relative_to(ROOT).as_posix())
    if leaks:raise ValueError('Credential serialization detected; paths withheld.')
    result=dict(scope='Implementation and research experiments complete; scientific successes and failures remain separate.',primaryFreezeHash=primary,additionalFreezes=extra,
        primary=dict(pastSeries=240,forwardBranches=960,reverseBranches=960,decayValidation=135,decayTest=45,stress=60,mechanisms=75,controls=30,diagnosticMethods=24),
        additional=dict(structureConditions=30,dualPastSeries=60,dualForwardBranches=240,actualLLMToolCalls=10,replicationGroups=60,replicationPastSeries=240,replicationCurrentBranches=960,longrunSeries=60,longrunCheckpoints=120),researchModels=len(models),models=models,additionalResearchModels=additional_models,exports=exports,
        tests=dict(new=33,phase3e=34,phase3d=94),oldProtectedFiles=len(old),phase3eProtectedFiles=len(prior['files']),olderProtectedFiles=len(older),oldChanged=0,asistReadOnlySources=len(source),asistChanged=0,
        credentialMatches=0,formalHumanEvaluationCompleted=False,humanPairsPrimary=44,humanPairsAdditional=40,reproduction='Fresh-copy five-capacity offline learning/restoration and separate performance repeat; not full independent numeric replication.')
    (ROOT/'reports/final-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['models','additionalFreezes']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
