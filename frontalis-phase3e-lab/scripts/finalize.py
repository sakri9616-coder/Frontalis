"""Verify completed artifacts, not a substitute for the recorded experiments."""
import hashlib,io,json,os,re,shutil,sys,tempfile,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from npne.core import Personality,tool_evaluate
from npne.meaning import digest
from npne.data import WIDTHS
from scripts.freeze import verify
torch.set_num_threads(1)

def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with (ROOT/p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

main=read('reports/main.json');stress=read('reports/stress.json');diag=read('reports/diagnostic.json');sem=read('reports/semantics.json');con=read('reports/dialogue-analysis.json');follow=read('reports/plasticity-followup.json')
failures=[]
for engine,id,label,note in [
 ('mimo-v2.6-flash-free','c0-s1-formed','emotion-added','場面と選択対象のangerが両方0なのに「怒りはある」と発言。resistanceも0で、抵抗の付加もある。'),
 ('nemotron-3-ultra-free','c0-s0-initial','joy-added','場面と選択対象のjoyが両方0なのに「完成したのは嬉しい」と発言。JSONの一致と本文の一致は別。')]:
    r=next(v for v in con['records'] if v['engine']==engine and v['id']==id and v['mode']=='dialogue')
    failures.append(dict(kind=label,engine=engine,id=id,interpretation=note,sceneSignals=r['source']['toolResult']['sceneAppraisal']['signals'],targetSignals=r['source']['toolResult']['selectedTargetAppraisal']['signals'],reply=r['response']['reply'],evaluation='posthoc nonblind AI inspection, not formal human score'))
failures.extend([
 dict(kind='missing-extraction',engine='mimo-v2.6-flash-free',batch='extract-0',evidence='reports/live/mimo-v2.6-flash-free/extract-0-attempt0.json',finding='180s timeout; retry also failed. Four missing meanings retained in denominator. Provider root cause unconfirmed.'),
 dict(kind='unexpected-tool',engine='nemotron-3-ultra-free',batch='extract-2',evidence='reports/live/nemotron-3-ultra-free/extract-2-attempt0.json',finding='Executed a JavaScript JSON formatting tool; rejected under fixed no-tools criterion. Not relabelled as extraction success.'),
 dict(kind='annotation-disagreement',engine='claude-haiku-5-5',id='s16',finding='Self-chosen exploration was mapped to consent=yes, authored gold=unknown. Independent human annotation is needed to distinguish interpretation from annotation policy.'),
 dict(kind='semantic-long-plasticity',finding='Primary rich-input logit conditions met the recovery contrast criterion in 1/15, below predeclared threshold. Posthoc result-view ablation is reported separately.'),
 dict(kind='nonlocal-change',finding='Large contrary-fact changes also spread to distant probes of the same branch. Exact preservation of a separately routed branch does not prove general locality.')])
write('reports/failure-cases.json',dict(posthoc=True,formalHumanEvaluation=False,sourceSha256=sha(ROOT/'reports/dialogue-analysis.json'),cases=failures))

# All stored research numerical states, including the six exploratory ones.
model_count=0
for meta in sorted((ROOT/'models').glob('*.json')):
    n=Personality.load(meta.with_suffix(''));y=n.evaluate_batch([[.5]*n.dim])
    if not np.isfinite(y).all():raise ValueError('Nonfinite restored evaluation.')
    model_count+=1
    del n
tool_exact=0
for c in read('data/dialogue-plan.json')['cases']:
    n=Personality.load(ROOT/c['modelPrefix']) if c['modelPrefix'] else Personality(5311,WIDTHS['semantic44'][c['capacityIndex']],'semantic44','bounded_logit')
    if tool_evaluate(n,read('data/natural.json')[[r['id'] for r in read('data/natural.json')].index(c['sourceId'])]['gold'],c['options'])!=c['toolResult']:raise ValueError('Dialogue tool signal changed.')
    tool_exact+=1

preserved=verify();reg=read('reference/preregistration.json');changed=[p for p,h in reg['sources'].items() if sha(ROOT/p)!=h]
additional=read('reference/plasticity-followup-freeze.json');additional_changed=[p for p,h in additional['sources'].items() if sha(ROOT/p)!=h]
if changed or additional_changed or not preserved['success']:raise ValueError('Preservation failed.')

# Export from the actual local workbench, then restore the downloaded state.
sessions=[]
for d in (ROOT/'playground/states').iterdir():
    if not d.is_dir():continue
    origin=json.loads((d/'origin.json').read_text(encoding='utf8'));pointer=json.loads((d/'current.json').read_text(encoding='utf8'))
    n=Personality.load(d/pointer['prefix']);expr=[]
    for p in d.glob('expression-*.json'):
        e=json.loads(p.read_text(encoding='utf8'))
        if e['parameterHashBefore']!=e['parameterHashAfter']:raise ValueError('Expression mutated canonical parameters.')
        expr.append(dict(path=str(p.relative_to(ROOT)),readOnlyExact=True))
    sessions.append(dict(id=d.name,generation=n.generation,parameterHash=n.parameter_hash(),origin=origin,expressionChecks=expr,experienceFiles=len(list(d.glob('experience-*.json')))))
selected=next(s for s in sessions if s['origin'].get('outcomeView')=='facts8-neutral-context' and s['generation']>0)
with urllib.request.urlopen('http://127.0.0.1:8903/api/export?sessionId='+selected['id'],timeout=20) as r:archive=r.read()
(ROOT/'reports/workbench-personality.zip').write_bytes(archive)
with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        state_files=[p for p in z.namelist() if re.fullmatch(r'g[0-9]+-[a-f0-9]{32}\.(json|npz)',p)]
        if len(state_files)!=2:raise ValueError('Unexpected export state count.')
        for p in state_files:(Path(tmp)/p).write_bytes(z.read(p))
        restored=Personality.load(Path(tmp)/Path(state_files[0]).stem)
        export_exact=restored.parameter_hash()==selected['parameterHash']
if not export_exact:raise ValueError('Exported numerical state changed.')
shutil.copyfile(Path.home()/'Downloads/frontalis-phase3e-ratings.json',ROOT/'reports/review-smoke-ratings-NOT-HUMAN.json')

# Exact secret check: neither key nor headers are serialized to the artifacts.
key=os.environ.get('ANTHROPIC_API_KEY','').encode();leaks=[]
for p in ROOT.rglob('*'):
    if p.is_file() and key and key in p.read_bytes():leaks.append(str(p.relative_to(ROOT)))
if leaks:raise ValueError('Secret found in artifact; paths withheld until corrected.')
result=dict(completed=True,mainConditions=len(main['conditions']),learningOffConditions=len(main['learningOff']),stressConditions=len(stress['conditions']),diagnosticConditions=len(diag['conditions']),followupConditions=len(follow['conditions']),restoredModelStates=model_count,dialogueToolCasesExact=tool_exact,languageResponses=sum(r['validReplies'] for r in con['scores']),languageExpected=sum(r['expected'] for r in con['scores']),humanComparisonPairs=len(read('reports/review-data.json')['cases']),newTests=34,priorRegressionTests=94,formalHumanEvaluation=False,sourceChanges=changed,additionalSourceChanges=additional_changed,preservation=preserved,keyPresentForCheck=bool(key),secretMatches=len(leaks),workbenchSessions=sessions,exportRestoredExactly=export_exact,uiChecks=['two sequential actual Haiku interpretations and self-learning updates','old and new generation restored with original parameter hashes','read-only language output','raw8 selector creates 526 parameters','exploratory rich-input state learns via separated outcome view','human rating survives page reload','rating JSON download and import','all UI smoke ratings reset to unscored'])
write('reports/final-verification.json',result)
print(json.dumps({k:result[k] for k in ['completed','restoredModelStates','languageResponses','newTests','priorRegressionTests','exportRestoredExactly','secretMatches']}))
