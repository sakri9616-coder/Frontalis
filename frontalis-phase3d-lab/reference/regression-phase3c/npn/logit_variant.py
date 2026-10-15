"""Post-result loss-space diagnostic; never replaces the primary core.

Old-state subjective targets are unchanged. Only fitting in preactivation
space and a finite ±6 target-logit precision cap differ from primary update.
"""
import copy
import json
from pathlib import Path
import numpy as np
from npn import model
from npn.store import validate as validate_primary


def init(seed, kind='nn', gain=1.25, learning=True):
    state = model.init(seed, kind=kind, gain=gain, learning=learning)
    state['lossSpace'] = 'logit-posthoc'
    state['logitCap'] = 6.
    return state


def evaluate(state, features, space):
    return model.evaluate(state, features, space)


def _logit_target(target, cap):
    return np.clip(np.arctanh(np.clip(target, -1 + 1e-12, 1 - 1e-12)), -cap, cap)


def preactivation(state, features, space):
    """Direct logits: avoid reconstructing atanh(tanh(z)) after saturation."""
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
        residual = model._basis(f, kind) @ np.asarray(branch['coefficients'])
    else:
        records = branch['memory']
        if not records:
            residual = np.zeros_like(base)
        else:
            remembered = np.asarray([row['centered'] for row in records])
            stored = np.asarray([row['residual'] for row in records])
            distances = ((f[:, None, :] - remembered[None, :, :]) ** 2).sum(axis=2)
            weights = -distances / (2. * state['memoryBandwidth'] ** 2)
            weights = np.exp(weights - weights.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            residual = weights @ stored
    return base + residual


def gradient_probe(state, event):
    model.accepted(event)
    old_cue = np.asarray(model.evaluate(state, event['cue'], event['space'])['raw'])
    old_effect = np.asarray(model.evaluate(state, event['observed'], event['space'])['raw'])
    target = model._amplified(old_effect, state['gain'])
    target_logit = _logit_target(target, 6.)
    bounded_target = np.tanh(target_logit)
    cue_logit = preactivation(state, model.centered(event['cue']), event['space'])[0]
    # Primitive derivative with respect to cue logit, single 7ch observation.
    # Hidden derivatives, replay dilution and optimizer dynamics are separate.
    signal_gradient = 2. * (old_cue - bounded_target) * (1 - old_cue ** 2) / 7.
    logit_gradient = 2. * (cue_logit - target_logit) / 7.
    output_derivative = 1. - old_cue ** 2
    return {'oldCueRaw': old_cue.tolist(), 'oldEffectRaw': old_effect.tolist(),
            'oldCueLogit': cue_logit.tolist(), 'frozenRawTarget': target.tolist(),
            'boundedTargetLogit': target_logit.tolist(),
            'boundedTargetRaw': bounded_target.tolist(),
            'cueEffectMeanAbsolute': float(np.abs(old_cue - old_effect).mean()),
            'cueTargetMeanAbsolute': float(np.abs(old_cue - target).mean()),
            'cueBoundedTargetMeanAbsolute': float(np.abs(old_cue - bounded_target).mean()),
            'outputTanhDerivativeMean': float(output_derivative.mean()),
            'outputTanhDerivativeMinimum': float(output_derivative.min()),
            'signalOutputGradientNorm': float(np.linalg.norm(signal_gradient)),
            'logitOutputGradientNorm': float(np.linalg.norm(logit_gradient)),
            'targetCapFraction': float((np.abs(target_logit) >= 6.).mean()),
            'scope': 'Both derivatives use the same ±6-bounded target. '
                     'Memory has no gradient fitting; its derivative is diagnostic only.'}


def _train_branch(state, space, labels):
    if not labels:
        return []
    features = np.asarray([model.centered(row['features']) for row in labels])
    targets = np.asarray([row['targetLogit'] for row in labels])
    branch = state['branches'][space]
    kind = state['kind']
    if kind == 'memory':
        branch['memory'] = [
            {'centered': point.tolist(),
             'residual': (target - point @ np.asarray(state['commonBase'])).tolist(),
             'eventId': row.get('eventId'), 'sourceStateHash': row.get('sourceStateHash')}
            for point, target, row in zip(features, targets, labels)]
        return [float(np.mean((preactivation(state, features, space) - targets) ** 2))]
    names = ['w1', 'b1', 'w2', 'b2'] if kind == 'nn' else ['coefficients']
    parameters = {name: np.asarray(branch[name], dtype=float) for name in names}
    moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    velocity = {name: np.zeros_like(value) for name, value in parameters.items()}
    losses = []
    base = features @ np.asarray(state['commonBase'])
    for epoch in range(1, state['epochs'] + 1):
        if kind == 'nn':
            hidden = np.tanh(features @ parameters['w1'] + parameters['b1'])
            logits = base + hidden @ parameters['w2'] + parameters['b2']
        else:
            basis = model._basis(features, kind)
            logits = base + basis @ parameters['coefficients']
        losses.append(float(np.mean((logits - targets) ** 2)))
        derivative = 2. * (logits - targets) / targets.size
        if kind == 'nn':
            hidden_derivative = (derivative @ parameters['w2'].T) * (1. - hidden ** 2)
            gradients = {'w1': features.T @ hidden_derivative,
                         'b1': hidden_derivative.sum(axis=0),
                         'w2': hidden.T @ derivative, 'b2': derivative.sum(axis=0)}
        else:
            gradients = {'coefficients': basis.T @ derivative}
        norm = np.sqrt(sum(float((value ** 2).sum()) for value in gradients.values()))
        if not np.isfinite(norm):
            raise ValueError('Non-finite variant gradient; original state retained.')
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


def update(state, event):
    model.accepted(event)
    if state.get('lossSpace') != 'logit-posthoc' or state.get('logitCap') != 6.:
        raise ValueError('Independent variant state required.')
    if any(row['id'] == event['id'] for row in state['history']):
        raise ValueError('Duplicate source experience.')
    model._assert_stable(state)
    result = copy.deepcopy(state)
    space = event['space']
    cue = evaluate(state, event['cue'], space)
    effect = evaluate(state, event['observed'], space)
    target = model._amplified(np.asarray(effect['raw']), state['gain']).tolist()
    target_logit = _logit_target(target, state['logitCap']).tolist()
    no_information = bool(np.all(model.centered(event['cue']) == 0.) and
                          np.all(model.centered(event['observed']) == 0.))
    source_hash = model.digest(state)
    labels = [{'features': copy.deepcopy(event[key]), 'target': target,
               'targetLogit': target_logit, 'space': space, 'eventId': event['id'],
               'role': key, 'sourceStateHash': source_hash} for key in ('cue', 'observed')]
    losses = []
    if state['learning'] and not no_information:
        result['replay'][space] = (result['replay'][space] + labels)[-state['replayLimit']:]
        losses = _train_branch(result, space, result['replay'][space])
    appraisal = {'eventId': event['id'], 'sourceStateHash': source_hash,
                 'parameterHashBefore': model.parameter_digest(state),
                 'cueBefore': cue['raw'], 'effectBefore': effect['raw'],
                 'target': target, 'targetLogit': target_logit,
                 'labelRule': 'old-effect-atanh-amplification', 'lossSpace': 'logit-posthoc',
                 'logitCap': state['logitCap'], 'gain': state['gain'],
                 'noInformation': no_information, 'learningEnabled': state['learning'],
                 'trainingLoss': losses, 'parameterHashAfter': model.parameter_digest(result)}
    result['history'].append(copy.deepcopy(event))
    result['appraisals'].append(appraisal)
    result['generation'] += 1
    result['ledgerHash'] = model.digest([state['ledgerHash'], event, appraisal])
    model._assert_stable(result)
    return result


def validate(state):
    validate_primary(state)
    if state.get('lossSpace') != 'logit-posthoc' or state.get('logitCap') != 6.:
        raise ValueError('Invalid independent loss-space variant.')
    for space in model.SPACES:
        for row in state['replay'][space]:
            if not np.array_equal(row['targetLogit'], _logit_target(np.asarray(row['target']), 6.)):
                raise ValueError('Frozen logit target mismatch.')
    for row in state['appraisals']:
        if not np.array_equal(row['targetLogit'], _logit_target(np.asarray(row['target']), 6.)):
            raise ValueError('Appraisal logit target mismatch.')
    return True


def save(path, state):
    validate(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as handle:
        json.dump({'sha256': model.digest(state), 'payload': state}, handle,
                  ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    if value.get('sha256') != model.digest(value['payload']):
        raise ValueError('Variant snapshot integrity mismatch.')
    validate(value['payload'])
    return value['payload']

