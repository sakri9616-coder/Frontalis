"""Read-only Phase 3-D personality evaluation, without action authority."""
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from npnd.core import Personality
import torch
torch.set_num_threads(1)

def decision(model, args):
    if not isinstance(args, dict) or set(args) - {'query_id','space','features','basis','choices'}:
        raise ValueError('Only factual evaluation arguments allowed.')
    if not isinstance(args.get('query_id'), str) or not 1 <= len(args['query_id']) <= 64 or not args['query_id'].strip():
        raise ValueError('Invalid query identity.')
    basis = args.get('basis', 'affinity')
    if basis not in ('affinity', 'value'):
        raise ValueError('Unknown decision basis.')
    ev = model.evaluate(args.get('features'), args.get('space'))
    incoming = args.get('choices', [])
    if not isinstance(incoming, list) or len(incoming) > 4:
        raise ValueError('At most four choices.')
    choices = []
    for row in incoming:
        if not isinstance(row, dict) or set(row) != {'id','label','features'}:
            raise ValueError('Invalid choice.')
        if not isinstance(row['id'], str) or not 1 <= len(row['id']) <= 64 or not row['id'].strip() or row['id'] == 'hold':
            raise ValueError('Invalid/reserved choice identity.')
        if not isinstance(row['label'], str) or not 1 <= len(row['label']) <= 240 or not row['label'].strip():
            raise ValueError('Invalid choice label.')
        ce = model.evaluate(row['features'], args['space'])
        choices.append({'id':row['id'], 'label':row['label'], 'score':ce['channels'][basis], 'signals':ce['signals']})
    if len(set(c['id'] for c in choices)) != len(choices):
        raise ValueError('Duplicate choice identities.')
    ranked = sorted(choices, key=lambda c: -c['score'])
    margin = ranked[0]['score'] - ranked[1]['score'] if len(ranked) > 1 else None
    selection = ranked[0]['id'] if ranked and (margin is None or margin > .03) else 'hold'
    affinity = ev['channels']['affinity']
    return {'formatVersion':1, 'query_id':args['query_id'], 'parameterHash':ev['parameterHash'],
        'generation':ev['generation'], 'channels':ev['channels'], 'signals':ev['signals'],
        'stance':'approve' if affinity > .12 else 'refuse' if affinity < -.12 else 'hold',
        'stanceBasis':'affinity', 'selection':selection, 'choiceScores':choices, 'choiceMargin':margin,
        'decisionBasis':basis, 'predictionGuard':ev['predictedOutcome'],
        'authority':'NPN personality judgment; express this direction without replacing it. This tool grants no execution authority.',
        'limits':'Computational subjective signals, not conscious emotion or moral correctness. Reasons need not be supplied.'}

def dispatch(packet):
    traces = []
    for item in packet['items']:
        prefix = Path(item['modelPrefix']).resolve()
        if not prefix.is_relative_to(ROOT/'models'):
            raise ValueError('Research models only.')
        model = Personality.load(prefix)
        calls = item['calls']
        before = model.parameter_hash()
        result = []
        for call in calls:
            if set(call) != {'name','arguments'} or call['name'] != 'frontalis_evaluate':
                raise ValueError('Read-only evaluation only.')
            result.append(decision(model, call['arguments']))
        if model.parameter_hash() != before:
            raise ValueError('Read-only tool mutated personality.')
        traces.append({'modelPrefix':str(prefix), 'calls':calls, 'toolResult':{'results':result}})
    return {'traces':traces, 'toolResult':{'results':[r for t in traces for r in t['toolResult']['results']]}}

if __name__ == '__main__':
    print(json.dumps(dispatch(json.load(sys.stdin)), ensure_ascii=False, allow_nan=False))
