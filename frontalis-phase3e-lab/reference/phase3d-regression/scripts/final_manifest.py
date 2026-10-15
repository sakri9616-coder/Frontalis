"""Validate completed local evidence and hash a portable research inventory."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'reports'/name).read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def tests(name):
    t=(ROOT/'reports'/name).read_text(encoding='utf-8-sig')
    count=re.search(r'Ran (\d+) tests',t)
    assert count and t.strip().endswith('OK'),name
    return int(count.group(1))

primary=read('numeric-summary.json'); million=read('million-online-summary.json');gpu=read('million-gpu-summary.json')
preservation=read('preservation.json');offline=read('final-offline-checks.json');browser=read('browser-checks.json')
main=read('live-analysis.json');extra=read('million-live-analysis.json');refusal=read('refusal-live-analysis.json')
refusal_verification=read('refusal-verification.json')
assert primary['mainCompleted']==228 and primary['mainFailed']==0
assert primary['stressCompleted']==24 and primary['stressFailed']==0
assert primary['diagnosticCompleted']==336 and primary['diagnosticFailed']==0
assert million['mainComplete']==8 and million['stressComplete']==2
assert gpu['gpuComplete']==48 and gpu['gpuFailed']==0
assert preservation['success'] and all(not x['changed'] and not x['missing'] for x in preservation['checks'])
assert all(x['exitCode']==0 and x['newExternalCalls']==0 for x in offline['checks'])
new_python=tests('final-python-tests.txt');old_python=tests('prior-regression-tests-retry1.txt')
node_text=(ROOT/'reports/final-ui-tests.txt').read_text(encoding='utf-8-sig')
node_count=int(re.search(r'(\d+) tests passed',node_text).group(1))
assert new_python>=94 and old_python==83 and node_count==32
assert len(main['records'])==52 and len(extra['records'])==8 and len(refusal['records'])==8
assert refusal_verification['planUnchanged'] and not refusal_verification['sourceEqualityIssues']
assert all(x['allUnchanged'] for x in refusal_verification['unchangedChecks'].values())
assert all(x['exitCode']==0 for x in refusal_verification['offlineVerifyOnly'].values())
assert browser['gradeExportImportRestoredTestValues'] and browser['restoredAfterReload']
verification={
 'completedAtUTC':datetime.now(timezone.utc).isoformat(),
 'newPythonRegressionTests':new_python,'newBrowserLogicTests':node_count,
 'oldPhase3CRegressionTestsInIsolatedCopy':old_python,
 'totalPassedTests':new_python+node_count+old_python,
 'oldRegressionFirstAttemptFailedBecauseWrongWorkingDirectory':True,
 'oldRegressionCorrectedRetryPassed':True,
 'primaryNumeric':{'formation':228,'longStress':24,'frozenDiagnostic':336,'failures':0},
 'conditionalMillionCPU':{'formation':8,'longStress':2,'diagnosticComplete':million['cpuDiagnosticCompleted'],
   'diagnosticPlannedBeforeGPUAmendment':24,'finalTestCompleted':False},
 'millionGPUPairedDiagnostic':{'complete':48,'failures':0,'validationBeforeFinalTest':True},
 'actualJapaneseReplies':{'primary':52,'conditionalMillion':8,'postHocFormedRefusal':8,'total':68},
 'primaryReviewPairs':20,'formalHumanRatingsIncluded':False,
 'browserChecks':'reports/browser-checks.json','offlineCacheChecks':'reports/final-offline-checks.json',
 'refusalOfflineCacheChecks':'reports/refusal-verification.json',
 'preservedOldFiles':sum(x['files'] for x in preservation['checks']),
 'changedOldFiles':0,'missingOldFiles':0,
 'reproductionDemo':{'newWorkspace':'../frontalis-phase3d-repro-smoke-2','states':4,'saveRestoreRollbackExact':True},
 'limitations':['AI qualitative audits are not human ratings.','Capacity effects are separate from input basis and fitting loss.',
 'Million online formation/stress uses one seed.','GPU protocol amendment preserves unfinished CPU evidence.',
 'Post-hoc refusal cases were selected after internal predictions, not representative prevalence estimates.',
 'Nemotron refusal annotation copy failed twice; body uses verified shared annotation, not a successful structure extraction claim.',
 'Warm API excludes new Python process startup.','No main ASIST integration or real-memory ingestion was performed.']}
(ROOT/'reports/final-verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf8')
files=[]
for area in ('npnd','tools','scripts','experiments','tests','docs','data','models','reference','reports','examples'):
    for p in (ROOT/area).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or p.name in ('artifact-sha256.json','final-manifest-run.txt'):continue
        if any(x in p.parts for x in ('node_modules','.venv','pip-cache')):continue
        files.append(p)
files.extend(ROOT/name for name in ('README.md','REPORT.md','CLAUDE_HANDOFF.md','.gitignore'))
inventory=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files))]
(ROOT/'reports/artifact-sha256.json').write_text(json.dumps({'createdAtUTC':verification['completedAtUTC'],
    'scope':'Research source, factual data, weights and evidence; excludes interpreter, dependency cache, sandbox and this command output log.',
    'files':inventory},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'verification':verification,'hashedFiles':len(inventory)},ensure_ascii=False))
