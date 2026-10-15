"""Freeze a post hoc refusal exploration without changing trained states."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npnd.core import Personality, digest
from npnd.data import probes
from tools.evaluate import decision

PLAN = ROOT / 'reports/refusal-live-plan.json'
FREEZE = ROOT / 'reports/refusal-live-freeze.json'
if PLAN.exists() or FREEZE.exists():
    raise FileExistsError('Refusal plan already frozen; no overwrite.')
if list((ROOT / 'reports/live').glob('*/refusal-*-input.json')):
    raise ValueError('Refusal generation has already started.')

source = {p['id']: p for p in probes()}
models = json.loads((ROOT / 'reports/live-plan.json').read_text(encoding='utf8'))['models']
cases = []
history = None
source_hashes = {}
for width, expected in ((16, (.363775, .380354)), (3125, (-.245915, -.126147))):
    prefix = ROOT / f'models/raw8/bounded_signal/nn-h{width}/seed4311-p3'
    model = Personality.load(prefix)
    if model.generation != 96 or len(model.history) != 96 or model.dtype_name != 'float64':
        raise ValueError('Expected the unchanged CPU float64 generation96 state.')
    if history is None:
        history = model.history
    elif model.history != history:
        raise ValueError('Widths must share exactly the same confirmed experiences.')
    for suffix in ('.json', '.npz'):
        fp = Path(str(prefix) + suffix)
        source_hashes[str(fp.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(fp.read_bytes()).hexdigest()
    before = model.parameter_hash()
    for index, probe_id in enumerate(('own-change', 'voluntary-project')):
        probe = source[probe_id]
        arguments = {'query_id': f'case-{index}', 'space': probe['space'], 'features': probe['features'], 'basis': 'affinity'}
        result = decision(model, arguments)
        if abs(result['signals']['preference'] - expected[index]) > 2e-6:
            raise ValueError('State differs from the post hoc selection evidence.')
        if result['stance'] != ('approve' if width == 16 else 'refuse') or result['selection'] != 'hold':
            raise ValueError('Unexpected fixed case direction.')
        cases.append({'id': f'refusal-h{width}-p3-{probe_id}',
                      'config': {'width': width, 'representation': 'raw8', 'loss': 'bounded_signal', 'kind': 'nn'},
                      'seed': 4311, 'person': 3, 'probeId': probe_id, 'user': probe['user'],
                      'statePrefix': str(prefix), 'arguments': arguments, 'generation': 96,
                      'historyCount': len(model.history), 'historySha256': digest(model.history),
                      'expectedToolResult': result})
    if model.parameter_hash() != before:
        raise ValueError('Read-only planning mutated the state.')

structure = [{'id': f'case-{index}', 'user': source[pid]['user'],
              'arguments': {'query_id': f'case-{index}', 'space': source[pid]['space'],
                            'features': source[pid]['features'], 'basis': 'affinity'}}
             for index, pid in enumerate(('own-change', 'voluntary-project'))]
plan = {'formatVersion': 1, 'frozenBeforeReplies': True, 'postHocExploration': True,
        'selectionScope': 'Formed refuse probes chosen after numerical/primary conversation inspection; not primary success or capacity superiority evidence.',
        'models': models, 'plannedAdditionalReplies': 8, 'cases': cases, 'structureCases': structure,
        'primaryReviewUnchanged': True, 'noNewLearning': True}
protected = ['reports/live-plan.json', 'reports/million-live-plan.json', 'reports/live-analysis.json',
             'reports/million-live-analysis.json', 'reports/conversations.md', 'reports/million-conversations.md',
             'reports/language-audit.json', 'reports/million-language-audit.json', 'reports/review.html',
             'reports/review-data.json', 'reports/review-key.json', 'reports/review-selection.json']
protected += [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'reports/live').glob('*/*') if p.is_file()]
protected_hashes = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in protected if (ROOT / rel).exists()}
with PLAN.open('x', encoding='utf8') as handle:
    json.dump(plan, handle, ensure_ascii=False, indent=2)
protocol = ROOT / 'docs/01d-formed-refusal-exploration.md'
receipt = {'formatVersion': 1, 'frozenAt': datetime.now(timezone.utc).isoformat(),
           'beforeAnyRefusalLLMCall': True, 'postHocSelection': True,
           'planFileSha256': hashlib.sha256(PLAN.read_bytes()).hexdigest(), 'planCanonicalSha256': digest(plan),
           'protocolSha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
           'modelSourceHashes': source_hashes, 'protectedSourceHashesBefore': protected_hashes}
with FREEZE.open('x', encoding='utf8') as handle:
    json.dump(receipt, handle, ensure_ascii=False, indent=2)
print(json.dumps({'casesFrozen': len(cases), 'planFileSha256': receipt['planFileSha256'],
                  'directions': [c['expectedToolResult']['stance'] for c in cases]}))
