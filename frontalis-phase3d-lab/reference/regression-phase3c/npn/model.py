"""Language-independent self-reinforced valuation, separate from prediction.

No emotional target or desirable moral direction enters ordinary update().
Initial random readout, amplification, emotion gates and policy mappings are
designer assumptions; this prototype does not discover them from nothing.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import numpy as np

SCHEMA = 'frontalis-value-loop-v1'
AXES = ['autonomy', 'constraint', 'change', 'loss', 'achievement',
        'discovery', 'regularity', 'others']
CHANNELS = ['affinity', 'curiosity', 'value', 'aesthetic', 'anger', 'joy', 'sadness']
SPACES = ['relational', 'acoustic']
KINDS = ['nn', 'linear', 'quadratic', 'memory']
GUARD = {'kind': 'constant', 'outcomes': [.5] * 8, 'version': 1}


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf8')
    return hashlib.sha256(encoded).hexdigest()


def _vector(value, width=8, low=0., high=1.):
    array = np.asarray(value, dtype=float)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError('Invalid finite feature/target vector.')
    if np.any((array < low) | (array > high)):
        raise ValueError('Vector outside declared range.')
    return array


def _space(space):
    if space not in SPACES:
        raise ValueError('Unknown semantic space.')
    return space


def centered(features):
    return 2. * _vector(features) - 1.


def _basis(features, kind):
    array = np.asarray(features)
    if array.ndim == 1:
        array = array[None, :]
    if kind == 'quadratic':
        products = [array[:, i] * array[:, j] for i in range(8) for j in range(i, 8)]
        return np.column_stack([array] + products + [np.ones(len(array))])
    return np.column_stack([array, np.ones(len(array))])


def init(seed, kind='nn', gain=1.25, learning=True):
    if kind not in KINDS:
        raise ValueError('Unknown model family.')
    if type(seed) is not int or type(learning) is not bool:
        raise ValueError('Invalid initialization.')
    if not math.isfinite(gain) or not 0 < gain <= 4:
        raise ValueError('Invalid self-amplification gain.')
    base_rng = np.random.default_rng(seed)
    common = base_rng.normal(0., .16, (8, 7))
    hidden_rng = np.random.default_rng(seed + 100003)
    hidden = hidden_rng.normal(0., .28, (8, 16))
    branches = {}
    for space in SPACES:
        if kind == 'nn':
            branch = {'w1': hidden.tolist(), 'b1': [0.] * 16,
                      'w2': np.zeros((16, 7)).tolist(), 'b2': [0.] * 7}
        elif kind in ('linear', 'quadratic'):
            branch = {'coefficients': np.zeros((9 if kind == 'linear' else 45, 7)).tolist()}
        else:
            branch = {'memory': []}
        branches[space] = branch
    return {'formatVersion': 1, 'schema': SCHEMA, 'owner': 'frontalis',
            'seed': seed, 'kind': kind, 'gain': float(gain), 'learning': learning,
            'learningRate': .025, 'epochs': 8, 'replayLimit': 64,
            'gradientClip': 5., 'emergencyNorm': 50., 'memoryBandwidth': .65,
            'commonBase': common.tolist(), 'branches': branches,
            'fixedPredictionGuard': copy.deepcopy(GUARD),
            'replay': {space: [] for space in SPACES}, 'history': [],
            'appraisals': [], 'diagnostics': [], 'generation': 0,
            'ledgerHash': digest([])}


def parameter_digest(state):
    return digest({'commonBase': state['commonBase'], 'branches': state['branches']})


def predict(state, cue, space):
    _vector(cue); _space(space)
    if state['fixedPredictionGuard'] != GUARD:
        raise ValueError('Fixed prediction guard changed.')
    return copy.deepcopy(GUARD['outcomes'])


def _forward(state, features, space):
    f = np.asarray(features, dtype=float)
    if f.ndim == 1:
        f = f[None, :]
    base = f @ np.asarray(state['commonBase'])
    branch = state['branches'][space]
    kind = state['kind']
    if kind == 'nn':
        hidden = np.tanh(f @ np.asarray(branch['w1']) + np.asarray(branch['b1']))
        residual = hidden @ np.asarray(branch['w2']) + np.asarray(branch['b2'])
    elif kind in ('linear', 'quadratic'):
        hidden = None
        residual = _basis(f, kind) @ np.asarray(branch['coefficients'])
    else:
        hidden = None
        records = branch['memory']
        if not records:
            residual = np.zeros_like(base)
        else:
            remembered = np.asarray([row['centered'] for row in records])
            stored = np.asarray([row['residual'] for row in records])
            distances = ((f[:, None, :] - remembered[None, :, :]) ** 2).sum(axis=2)
            logits = -distances / (2. * state['memoryBandwidth'] ** 2)
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            residual = weights @ stored
    return np.tanh(base + residual), hidden


def evaluate(state, features, space):
    _space(space)
    f = centered(features)
    raw = _forward(state, f, space)[0][0]
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite appraisal.')
    channels = dict(zip(CHANNELS, raw.tolist()))
    affinity = channels['affinity']
    signals = {'preference': affinity, 'dislike': max(0., -affinity),
               'initiative': max(0., affinity), 'resistance': max(0., -affinity),
               'curiosity': max(0., channels['curiosity']),
               'curiositySigned': channels['curiosity'],
               'value': channels['value'], 'aesthetic': channels['aesthetic'],
               'anger': max(0., channels['anger']), 'joy': max(0., channels['joy']),
               'sadness': max(0., channels['sadness'])}
    return {'raw': raw.tolist(), 'channels': channels, 'signals': signals,
            'space': space, 'features': _vector(features).tolist(),
            'predictedOutcome': predict(state, features, space),
            'generation': state['generation'], 'parameterHash': parameter_digest(state),
            'signalLimit': 'Emotion names and positive gates are fixed conventions; '
                           'initiative/resistance derive from affinity. Reasons are optional.'}


def accepted(event):
    if not isinstance(event, dict):
        raise ValueError('Experience must be a record.')
    forbidden = {'reward', 'praise', 'userRating', 'valence', 'emotion',
                 'personalityTarget', 'teacherCoefficients', 'target', 'seriesId'}
    if forbidden.intersection(event):
        raise ValueError('External subjective/identity target is not accepted.')
    if not isinstance(event.get('id'), str) or not event['id'].strip():
        raise ValueError('Experience needs a source ID.')
    if not isinstance(event.get('text'), str):
        raise ValueError('Experience needs retained source text.')
    if event.get('confirmed') is not True or event.get('origin') != 'synthetic':
        raise ValueError('Only confirmed research observations are accepted.')
    _space(event.get('space'))
    _vector(event.get('cue'))
    _vector(event.get('observed'))
    return True


def _parameter_arrays(state):
    arrays = [np.asarray(state['commonBase'], dtype=float)]
    for branch in state['branches'].values():
        for key in ('w1', 'b1', 'w2', 'b2', 'coefficients'):
            if key in branch:
                arrays.append(np.asarray(branch[key], dtype=float))
        for row in branch.get('memory', []):
            arrays.append(np.asarray(row['residual'], dtype=float))
    return arrays


def parameter_norm(state):
    arrays = _parameter_arrays(state)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError('Non-finite learned state.')
    return float(np.sqrt(sum(float((array ** 2).sum()) for array in arrays)))


def _assert_stable(state):
    if state['fixedPredictionGuard'] != GUARD:
        raise ValueError('Fixed prediction guard changed.')
    # Memory rows are a dataset, so the emergency limit applies to the largest
    # individual retained vector rather than a norm growing with row count.
    if state['kind'] == 'memory':
        arrays = _parameter_arrays(state)
        norm = max(float(np.linalg.norm(array)) for array in arrays)
    else:
        norm = parameter_norm(state)
    if not np.isfinite(norm) or norm > state['emergencyNorm']:
        raise ValueError('Emergency norm exceeded; original state retained.')


def _amplified(raw, gain):
    if gain == 1.:
        return np.asarray(raw).copy()
    return np.tanh(gain * np.arctanh(np.clip(raw, -.999999, .999999)))


def _train_branch(state, space, labels):
    if not labels:
        return []
    f = np.asarray([centered(row['features']) for row in labels])
    targets = np.asarray([_vector(row['target'], 7, -1., 1.) for row in labels])
    branch = state['branches'][space]
    kind = state['kind']
    if kind == 'memory':
        branch['memory'] = [
            {'centered': point.tolist(),
             'residual': (np.arctanh(np.clip(target, -.999999, .999999)) -
                          point @ np.asarray(state['commonBase'])).tolist(),
             'eventId': row.get('eventId'), 'sourceStateHash': row.get('sourceStateHash')}
            for point, target, row in zip(f, targets, labels)]
        return [float(np.mean((_forward(state, f, space)[0] - targets) ** 2))]
    names = ['w1', 'b1', 'w2', 'b2'] if kind == 'nn' else ['coefficients']
    parameters = {name: np.asarray(branch[name], dtype=float) for name in names}
    # Fresh Adam per experience: no stale optimizer momentum on unrelated input.
    moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    velocity = {name: np.zeros_like(value) for name, value in parameters.items()}
    losses = []
    base = f @ np.asarray(state['commonBase'])
    for epoch in range(1, state['epochs'] + 1):
        if kind == 'nn':
            h = np.tanh(f @ parameters['w1'] + parameters['b1'])
            raw = np.tanh(base + h @ parameters['w2'] + parameters['b2'])
        else:
            basis = _basis(f, kind)
            raw = np.tanh(base + basis @ parameters['coefficients'])
        losses.append(float(np.mean((raw - targets) ** 2)))
        derivative = 2. * (raw - targets) * (1. - raw ** 2) / targets.size
        if kind == 'nn':
            hidden_derivative = (derivative @ parameters['w2'].T) * (1. - h ** 2)
            gradients = {'w1': f.T @ hidden_derivative,
                         'b1': hidden_derivative.sum(axis=0),
                         'w2': h.T @ derivative, 'b2': derivative.sum(axis=0)}
        else:
            gradients = {'coefficients': basis.T @ derivative}
        norm = np.sqrt(sum(float((gradient ** 2).sum()) for gradient in gradients.values()))
        if not np.isfinite(norm):
            raise ValueError('Non-finite gradient; original state retained.')
        scale = min(1., state['gradientClip'] / (norm + 1e-12))
        for name in names:
            gradient = gradients[name] * scale
            moment[name] = .9 * moment[name] + .1 * gradient
            velocity[name] = .999 * velocity[name] + .001 * gradient ** 2
            corrected_moment = moment[name] / (1. - .9 ** epoch)
            corrected_velocity = velocity[name] / (1. - .999 ** epoch)
            parameters[name] -= state['learningRate'] * corrected_moment / (np.sqrt(corrected_velocity) + 1e-8)
    for name, value in parameters.items():
        branch[name] = value.tolist()
    return losses


def train_on_labels(state, labels, provenance='diagnostic'):
    """Matched frozen-label diagnostic only; ordinary update creates own targets."""
    if not isinstance(provenance, str) or not provenance:
        raise ValueError('Diagnostic provenance required.')
    _assert_stable(state)
    result = copy.deepcopy(state)
    for row in labels:
        _space(row['space']); _vector(row['features']); _vector(row['target'], 7, -1., 1.)
    for space in SPACES:
        subset = [row for row in labels if row['space'] == space]
        if subset:
            _train_branch(result, space, subset[-state['replayLimit']:])
    result['diagnostics'].append({'kind': provenance, 'sourceStateHash': digest(state),
                                  'labelsHash': digest(labels), 'count': len(labels)})
    _assert_stable(result)
    return result


def update(state, event):
    """Experience -> prior NPN appraisal -> self-label -> parameter update."""
    accepted(event)
    if any(row['id'] == event['id'] for row in state['history']):
        raise ValueError('Duplicate source experience.')
    _assert_stable(state)
    result = copy.deepcopy(state)
    space = event['space']
    cue = evaluate(state, event['cue'], space)
    effect = evaluate(state, event['observed'], space)
    target = _amplified(np.asarray(effect['raw']), state['gain']).tolist()
    no_information = bool(np.all(centered(event['cue']) == 0.) and
                          np.all(centered(event['observed']) == 0.))
    source_hash = digest(state)
    labels = [{'features': copy.deepcopy(event[key]), 'target': target,
               'space': space, 'eventId': event['id'], 'role': key,
               'sourceStateHash': source_hash} for key in ('cue', 'observed')]
    losses = []
    if state['learning'] and not no_information:
        result['replay'][space] = (result['replay'][space] + labels)[-state['replayLimit']:]
        losses = _train_branch(result, space, result['replay'][space])
    appraisal = {'eventId': event['id'], 'sourceStateHash': source_hash,
                 'parameterHashBefore': parameter_digest(state),
                 'cueBefore': cue['raw'], 'effectBefore': effect['raw'],
                 'target': target, 'labelRule': 'old-effect-atanh-amplification',
                 'gain': state['gain'], 'noInformation': no_information,
                 'learningEnabled': state['learning'], 'trainingLoss': losses,
                 'parameterHashAfter': parameter_digest(result)}
    result['history'].append(copy.deepcopy(event))
    result['appraisals'].append(appraisal)
    result['generation'] += 1
    result['ledgerHash'] = digest([state['ledgerHash'], event, appraisal])
    _assert_stable(result)
    return result


def replay(events, seed, kind='nn', gain=1.25, learning=True):
    state = init(seed, kind=kind, gain=gain, learning=learning)
    for event in events:
        state = update(state, event)
    return state

