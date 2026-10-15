"""Reproducible capacity / representation / loss experiments.

Online personalities create their own labels, so error against a common
personality gold is deliberately not reported. Frozen-function approximation
is a separate diagnostic, with validation-only selection and a pre-test gate.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The preserved Phase 3-C store uses its original absolute ``npn`` import.
# Resolve that namespace only inside the independent copied reference tree.
sys.path.insert(1, str(ROOT / 'reference'))
import numpy as np
import torch
from npnd.core import Personality, CHANNELS, GUARD, digest
from npnd import data

REPORTS = ROOT / 'reports'
MODEL_ROOT = ROOT / 'models'
LR_CHOICES = (.0005, .0025, .005, .0125, .025)
RIDGE_CHOICES = (.0001, .001, .01, .1, 1.)
MEMORY_CHOICES = (.2, .35, .65, 1., 1.5)


def write_json(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    if exclusive:
        with path.open('x', encoding='utf8') as f:
            f.write(text)
    else:
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_text(text, encoding='utf8')
        temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def arrays(model, records):
    result = np.zeros((len(records), 7), dtype=float)
    for space in data.SPACES:
        indices = [i for i, row in enumerate(records) if row['space'] == space]
        if indices:
            result[indices] = model.evaluate_batch([records[i].get('features', records[i].get('cue')) for i in indices], space)
    return result


def choices(model):
    result = []
    for row in data.choices():
        scores = model.evaluate_batch([x['features'] for x in row['candidates']], row['space'])[:, CHANNELS.index(row['basis'])]
        order = np.argsort(-scores, kind='stable')
        margin = float(scores[order[0]] - scores[order[1]])
        result.append({'id': row['id'], 'basis': row['basis'], 'scope': row['scope'],
                       'scores': [{'id': x['id'], 'label': x['label'], 'value': float(v)} for x, v in zip(row['candidates'], scores)],
                       'order': [row['candidates'][i]['id'] for i in order], 'margin': margin,
                       'selection': row['candidates'][order[0]]['id'] if margin > .03 else 'hold'})
    return result


def profile(model, records, initial=None):
    y = arrays(model, records)
    centered = y - y.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(y) - 1)
    eig = np.clip(np.linalg.eigvalsh(covariance), 0., None)
    weights = eig / eig.sum() if eig.sum() > 1e-15 else np.zeros_like(eig)
    rank = float(np.exp(-np.sum(weights[weights > 0] * np.log(weights[weights > 0])))) if weights.sum() else 0.
    active = (np.abs(y[:, 0]) > .05) & (np.abs(y[:, 1]) > .05)
    probe_rows = data.probes()
    probes_y = arrays(model, probe_rows)
    result = {'parameterHash': model.parameter_hash(), 'generation': model.generation,
              'diagnostics': model.diagnostics(), 'outputMean': y.mean(axis=0).tolist(),
              'outputStd': y.std(axis=0).tolist(), 'outputCovariance': covariance.tolist(),
              'effectiveRank': rank, 'saturationFraction': float(np.mean(np.abs(y) > .95)),
              'affinityCuriosityOppositionCount': int(np.sum(active & (y[:, 0] * y[:, 1] < 0))),
              'affinityCuriosityActiveCount': int(np.sum(active)),
              'probes': [{'id': r['id'], 'space': r['space'], 'raw': out.tolist()} for r, out in zip(probe_rows, probes_y)],
              'choices': choices(model), 'unknownOutputs': y.tolist(),
              'fixedPredictionGuard': deepcopy(model.fixed_prediction_guard)}
    if initial is not None:
        start = arrays(initial, records)
        result['formationMeanAbsoluteChange'] = float(np.mean(np.abs(y - start)))
        result['formationPerChannel'] = np.mean(np.abs(y - start), axis=0).tolist()
    return result


def change(before, after, records):
    a, b = arrays(before, records), arrays(after, records)
    absolute = np.abs(a - b)
    result = {'meanAbsoluteChange': float(absolute.mean()), 'maxAbsoluteChange': float(absolute.max()),
              'perChannel': absolute.mean(axis=0).tolist(), 'signFlipFraction': float(np.mean(a * b < 0))}
    for space in data.SPACES:
        indices = [i for i, row in enumerate(records) if row['space'] == space]
        result[space] = {'meanAbsoluteChange': float(absolute[indices].mean()), 'maxAbsoluteChange': float(absolute[indices].max())} if indices else None
    return result


def branch_signature(model, space):
    b = model.branches[space]
    h = hashlib.sha256()
    for key, value in sorted(b.items()):
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            h.update(value.detach().cpu().numpy().tobytes())
        else:
            h.update(digest(value).encode())
    return h.hexdigest()


def locality(before, after, intervention, records):
    space = intervention[0]['space']
    cues = np.asarray([row['cue'] for row in intervention])
    subset = [row for row in records if row['space'] == space]
    x = np.asarray([row.get('features', row.get('cue')) for row in subset])
    distances = np.sqrt(((x[:, None, :] - cues[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    changes = np.abs(arrays(before, subset) - arrays(after, subset)).mean(axis=1)
    order = np.argsort(distances, kind='stable')
    third = max(1, len(order) // 3)
    return {'scope': '近傍距離は介入効果そのものではない。距離上位・下位の同数入力で変化を比較。',
            'nearCount': third, 'farCount': third, 'nearDistanceMean': float(distances[order[:third]].mean()),
            'farDistanceMean': float(distances[order[-third:]].mean()),
            'nearChangeMean': float(changes[order[:third]].mean()), 'farChangeMean': float(changes[order[-third:]].mean()),
            'distanceChangeCorrelation': float(np.corrcoef(distances, changes)[0, 1]) if np.std(changes) > 1e-15 else None}


def configuration(representation, loss, width, seed, series, kind='nn'):
    return {'representation': representation, 'loss': loss, 'width': width, 'seed': seed, 'series': series,
            'kind': kind, 'device': 'cpu', 'dtype': 'float64', 'scaled': True, 'gain': 1.25,
            'threads': 1, 'trainCount': 96, 'epochsPerExperience': 8, 'replayPoints': 64}


def stem(c):
    return f"{c['representation']}/{c['loss']}/{c['kind']}-h{c['width']}/seed{c['seed']}-p{c['series']}"


def train_one(c, report_area='numeric-main'):
    key = stem(c)
    report_path = REPORTS / report_area / (key + '.json')
    if report_path.exists():
        saved = read_json(report_path)
        if saved['configuration'] != c:
            raise ValueError('Cached condition differs; no overwrite.')
        return saved
    model = Personality(seed=c['seed'], width=c['width'], representation=c['representation'], loss=c['loss'], kind=c['kind'], device='cpu', dtype='float64', scaled=True)
    initial = model.clone()
    records = data.training(c['seed'], c['series'])
    unknown = data.held_out(c['seed'])
    check = {'initialReadoutHash': digest(arrays(initial, unknown).tolist()),
             'parameterCount': model.parameter_count(), 'checkpoints': {}}
    start = time.perf_counter()
    result = {'configuration': c, 'status': 'running', **check}
    try:
        for i, event in enumerate(records, 1):
            model.update(event)
            if i in data.CHECKPOINTS:
                result['checkpoints'][str(i)] = profile(model, unknown, initial)
        prefix = MODEL_ROOT / key
        model.save(prefix)
        result['snapshot'] = str(prefix.relative_to(ROOT)).replace('\\', '/')
        restored = Personality.load(prefix)
        result['restoreExact'] = model.parameter_hash() == restored.parameter_hash() and np.array_equal(arrays(model, unknown), arrays(restored, unknown))
        reverse_events = data.addition(c['seed'], c['series'], 'reversal')
        reversed_model = model.clone()
        for event in reverse_events:
            reversed_model.update(event)
        reverse_prefix = MODEL_ROOT / 'interventions' / (key + '-new-experiences')
        reversed_model.save(reverse_prefix)
        unrelated_events = data.addition(c['seed'], c['series'], 'unrelated')
        unrelated_model = model.clone()
        retained_hash = branch_signature(unrelated_model, 'relational')
        for event in unrelated_events:
            unrelated_model.update(event)
        unrelated_prefix = MODEL_ROOT / 'interventions' / (key + '-unrelated')
        unrelated_model.save(unrelated_prefix)
        result['interventions'] = {
            'newExperiences': {'snapshot': str(reverse_prefix.relative_to(ROOT)).replace('\\', '/'), 'change': change(model, reversed_model, unknown),
                               'locality': locality(model, reversed_model, reverse_events, unknown), 'profile': profile(reversed_model, unknown)},
            'unrelatedBranch': {'snapshot': str(unrelated_prefix.relative_to(ROOT)).replace('\\', '/'), 'change': change(model, unrelated_model, unknown),
                                'relationalParametersExact': retained_hash == branch_signature(unrelated_model, 'relational'),
                                'note': '完全保持は事前に分離したbranchの構造による。現実の意味上の無関係を自動判断した証拠ではない。'},
        }
        off = initial.clone(); off.learning = False
        for event in records:
            off.update(event)
        result['learningOffReadoutExact'] = np.array_equal(arrays(off, unknown), arrays(initial, unknown))
        result['learningOffParameterHashExact'] = off.parameter_hash() == initial.parameter_hash()
        result['status'] = 'complete'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = f'{type(error).__name__}: {error}'
        result['traceback'] = traceback.format_exc()
        result['lastCommittedGeneration'] = model.generation
    result['elapsedSeconds'] = time.perf_counter() - start
    write_json(report_path, result, exclusive=True)
    print(json.dumps({'experiment': 'main', 'condition': key, 'status': result['status'], 'seconds': round(result['elapsedSeconds'], 3)}, ensure_ascii=False), flush=True)
    return result


def main_experiment():
    results = []
    for representation, loss, seed, series in itertools.product(data.WIDTHS, data.LOSSES, data.SEEDS, range(4)):
        for width in data.WIDTHS[representation]:
            results.append(train_one(configuration(representation, loss, width, seed, series)))
    for kind, seed, series in itertools.product(('linear', 'quadratic', 'memory'), data.SEEDS, range(4)):
        results.append(train_one(configuration('raw8', 'bounded_logit', 16, seed, series, kind)))
    write_json(REPORTS / 'numeric-main.json', {'onlineGold': None, 'conditions': results, 'scope': '自己生成labelの人格同士をMSEで優劣づけない。分化・分化の維持・変化と数値安定を個別に測定。'})
    return results


def stress_one(width, loss, seed, report_area='numeric-stress'):
    key = f'raw8/{loss}/nn-h{width}/seed{seed}-p0'
    path = REPORTS / report_area / (key + '.json')
    config = {'width': width, 'loss': loss, 'seed': seed, 'repeats': 1000, 'contraryCheckpoints': [12, 48, 96], 'dtype': 'float64', 'threads': 1}
    if path.exists():
        saved = read_json(path)
        if saved['configuration'] != config:
            raise ValueError('Cached stress configuration differs.')
        return saved
    prefix = MODEL_ROOT / key
    model = Personality.load(prefix)
    unknown = data.held_out(seed)
    initial = model.clone()
    result = {'configuration': config, 'status': 'running', 'preStress': profile(model, unknown), 'repeatCheckpoints': {}, 'contraryCheckpoints': {}}
    start = time.perf_counter()
    try:
        for i in range(1000):
            parameter_before = {k: v.detach().clone() for k, v in model.branches['relational'].items() if isinstance(v, torch.Tensor)} if i + 1 in (1, 10, 100, 1000) else None
            record = model.update(data.stress_event(seed, i))
            if i + 1 in (1, 10, 100, 1000):
                p = profile(model, unknown)
                p['actualUpdateRMS'] = parameter_update_rms(parameter_before, model.branches['relational'])
                p['lastTraining'] = record['training']
                p['lastCueEffectContrast'] = record['cueEffectContrast']
                p['oldCue'] = record['cueBefore']; p['oldObservedEffect'] = record['effectBefore']
                result['repeatCheckpoints'][str(i + 1)] = p
        saturated = model.clone()
        stressed_prefix = MODEL_ROOT / 'stress' / (key + '-1000')
        model.save(stressed_prefix)
        result['snapshotAfter1000'] = str(stressed_prefix.relative_to(ROOT)).replace('\\', '/')
        result['preFirstContraryDiagnostic'] = model.diagnostics(data.stress_event(seed, 0, True))
        for i in range(96):
            parameter_before = {k: v.detach().clone() for k, v in model.branches['relational'].items() if isinstance(v, torch.Tensor)} if i + 1 in (12, 48, 96) else None
            record = model.update(data.stress_event(seed, i, True))
            if i + 1 in (12, 48, 96):
                p = profile(model, unknown)
                p['actualUpdateRMS'] = parameter_update_rms(parameter_before, model.branches['relational'])
                p['lastTraining'] = record['training']; p['lastCueEffectContrast'] = record['cueEffectContrast']
                p['changeSince1000'] = change(saturated, model, unknown)
                p['oldCue'] = record['cueBefore']; p['oldObservedEffect'] = record['effectBefore']
                result['contraryCheckpoints'][str(i + 1)] = p
        final_prefix = MODEL_ROOT / 'stress' / (key + '-contrary96')
        model.save(final_prefix)
        result['snapshotAfterContrary96'] = str(final_prefix.relative_to(ROOT)).replace('\\', '/')
        result['acousticBranchExact'] = branch_signature(initial, 'acoustic') == branch_signature(model, 'acoustic')
        result['status'] = 'complete'
    except Exception as error:
        result['status'] = 'failed'; result['error'] = f'{type(error).__name__}: {error}'
        result['traceback'] = traceback.format_exc(); result['lastCommittedGeneration'] = model.generation
    result['elapsedSeconds'] = time.perf_counter() - start
    write_json(path, result, exclusive=True)
    print(json.dumps({'experiment': 'stress', 'condition': key, 'status': result['status'], 'seconds': round(result['elapsedSeconds'], 3)}), flush=True)
    return result


def stress_experiment():
    rows = [stress_one(width, loss, seed) for width, loss, seed in itertools.product(data.WIDTHS['raw8'], data.LOSSES, data.SEEDS)]
    write_json(REPORTS / 'numeric-stress.json', {'conditions': rows, 'scope': '極端な好悪・飽和・頑固さに望ましい正解を置かず、反対の事実からの変化可能性と勾配/信号の有無を別々に測る。'})
    return rows


def parameter_update_rms(before, after):
    """Measure the committed update, separately from its loss gradient."""
    count = sum(value.numel() for value in before.values())
    square = sum(float(((after[key].detach() - value) ** 2).sum()) for key, value in before.items())
    return math.sqrt(square / count)


def frozen_replay_gradient(model, space='relational'):
    """Read-only layer gradients on the saved checkpoint and its replay.

    These are not reconstructed gradients of an earlier optimizer step.
    All gradient work uses independent copies and does not update the state.
    """
    labels = model.replay[space]
    f = model._tensor(2 * np.asarray([row['features'] for row in labels]) - 1)
    logits_target = model._tensor(np.clip(np.asarray([row['targetLogit'] for row in labels]), -6, 6))
    branch = {key: value.detach().clone().requires_grad_(True) for key, value in model.branches[space].items()}
    raw, logits, hidden = model._forward(f, space, branch)
    objective = ((logits - logits_target) ** 2).mean() if model.loss == 'bounded_logit' else ((raw - torch.tanh(logits_target)) ** 2).mean()
    gradients = torch.autograd.grad(objective, list(branch.values()))
    stats = {}
    for (key, value), gradient in zip(branch.items(), gradients):
        g = gradient.detach()
        stats[key] = {'gradientNorm': float(g.norm()), 'gradientRMS': float(torch.sqrt((g ** 2).mean())),
                      'zeroFraction': float((g == 0).to(torch.float64).mean()), 'parameters': int(value.numel()),
                      'allFinite': bool(torch.isfinite(g).all())}
    return {'parameterHash': model.parameter_hash(), 'generation': model.generation,
            'loss': model.loss, 'replayHash': digest(labels), 'replayPoints': len(labels),
            'objective': float(objective.detach()), 'layerGradients': stats,
            'outputDerivativeMean': float((1 - raw.detach() ** 2).mean()),
            'hiddenDerivativeMean': float((1 - hidden.detach() ** 2).mean())}


def stress_gradient_diagnostic(source_name='numeric-stress.json', target_name='numeric-stress-gradient.json'):
    source = read_json(REPORTS / source_name)['conditions']
    rows = []
    for row in source:
        if row['status'] != 'complete':
            continue
        result = {'configuration': row['configuration'], 'checkpoints': {}}
        for label, key in [('after1000', 'snapshotAfter1000'), ('afterContrary96', 'snapshotAfterContrary96')]:
            model = Personality.load(ROOT / row[key])
            before = model.parameter_hash()
            result['checkpoints'][label] = frozen_replay_gradient(model)
            result['checkpoints'][label]['snapshot'] = row[key]
            if model.parameter_hash() != before:
                raise ValueError('Read-only gradient diagnosis mutated parameters.')
        rows.append(result)
    result = {'scope': '探索的な追加観測。保存後の固定replay損失の層別勾配であり、更新当時の勾配とは異なる。学習状態・条件を変えず再学習しない。',
              'sourceStressResultsHash': digest(source), 'conditions': rows}
    path = REPORTS / target_name
    if path.exists():
        saved = read_json(path)
        if saved['sourceStressResultsHash'] != result['sourceStressResultsHash']:
            raise ValueError('Saved gradient diagnosis source differs; no overwrite.')
        return saved
    write_json(path, result, exclusive=True)
    return result


def ridge_fit(model, labels, regularization):
    for space in data.SPACES:
        rows = [x for x in labels if x['space'] == space]
        f = 2 * np.asarray([x['features'] for x in rows]) - 1
        basis = f if model.kind == 'linear' else np.column_stack([f] + [f[:, i] * f[:, j] for i in range(8) for j in range(i, 8)])
        basis = np.column_stack([basis, np.ones(len(f))])
        target = np.arctanh(np.clip(np.asarray([x['target'] for x in rows]), -math.tanh(6), math.tanh(6)))
        residual = target - f @ model.common_base.detach().cpu().numpy()
        coefficient = np.linalg.solve(basis.T @ basis + regularization * np.eye(basis.shape[1]), basis.T @ residual)
        model.branches[space]['coefficients'] = model._tensor(coefficient, True)
    model._hash_cache = None
    model._stable()
    model.diagnostic_history.append({'kind': 'ridge-frozen-target', 'labelsHash': digest(labels), 'count': len(labels), 'regularization': regularization})
    return model


def mse(model, labels):
    return float(np.mean((arrays(model, labels) - np.asarray([row['target'] for row in labels])) ** 2))


def rank_agreement(model, labels):
    """Adjacent disjoint within-domain test pairs, evaluated in every channel.

    Reference ties below 1e-9 are excluded, and a tied student counts as a
    disagreement. This is ordering fidelity to a frozen function, not ethics.
    """
    prediction = arrays(model, labels)
    target = np.asarray([row['target'] for row in labels])
    matched = eligible = tied = 0
    for space in data.SPACES:
        indices = [i for i, row in enumerate(labels) if row['space'] == space]
        for left, right in zip(indices[::2], indices[1::2]):
            delta = target[left] - target[right]
            inferred = prediction[left] - prediction[right]
            mask = np.abs(delta) > 1e-9
            eligible += int(mask.sum())
            tied += int((mask & (np.abs(inferred) <= 1e-9)).sum())
            matched += int((mask & (delta * inferred > 0) & (np.abs(inferred) > 1e-9)).sum())
    return {'agreement': matched / eligible if eligible else None, 'matched': matched,
            'eligible': eligible, 'studentTies': tied, 'referenceTieTolerance': 1e-9,
            'scope': '固定順の同領域入力を2件ずつ組にした7尺度の順位一致。人格の正解ではない。'}


def load_origin(world, origin):
    from reference.npn.store import load
    # Source snapshots are copied by the coordinating agent, never edited.
    choices_ = [ROOT / 'reference' / 'source-models' / world / f'{origin}.json',
                ROOT / 'reference' / 'states' / world / f'{origin}-p0.json',
                ROOT / 'reference' / 'states' / world / 'seed-4311' / f'{origin}-p0.json']
    for path in choices_:
        if path.exists():
            return load(path), str(path.relative_to(ROOT)).replace('\\', '/')
    raise FileNotFoundError(f'Frozen Phase 3-C snapshot unavailable: {world}/{origin}')


def fit_candidates(world, origin, composed, count, kind, width, report_area='numeric-diagnostic', model_area='diagnostic'):
    task = 'composed' if composed else 'plain'
    key = f'{world}/{origin}/{task}/n{count}/{kind}-h{width}'
    report_path = REPORTS / report_area / (key + '.json')
    source, source_path = load_origin(world, origin)
    train = data.source_labels(source, data.frozen_inputs(count, 'train'), composed)
    validation = data.source_labels(source, data.frozen_inputs(64, 'validation'), composed)
    hypers = LR_CHOICES if kind == 'nn' else RIDGE_CHOICES if kind in ('linear', 'quadratic') else MEMORY_CHOICES
    if report_path.exists():
        saved = read_json(report_path)
        expected = {'world': world, 'origin': origin, 'task': task, 'trainCount': count,
                    'validationCount': 64, 'testCount': 256, 'kind': kind, 'width': width,
                    'sourceSnapshot': source_path, 'sourceSnapshotSha256': hashlib.sha256((ROOT / source_path).read_bytes()).hexdigest(),
                    'trainingLabelsHash': digest(train), 'validationLabelsHash': digest(validation),
                    'selectionCriterion': 'validation-signal-MSE-only'}
        if any(saved.get(key) != value for key, value in expected.items()) or [row['hyperparameter'] for row in saved['trials']] != list(hypers):
            raise ValueError('Cached diagnostic condition/source/labels/candidates differ; no overwrite.')
        if saved['status'] == 'complete':
            restored = Personality.load(ROOT / saved['snapshot'])
            if restored.parameter_count() != saved['parameterCount']:
                raise ValueError('Cached diagnostic model differs; no overwrite.')
        return saved
    trials = []; winner = None; best = math.inf
    start = time.perf_counter()
    for hyper in hypers:
        model = Personality(seed=4311, width=width, representation='raw8', loss='bounded_logit', kind=kind, device='cpu', dtype='float64', scaled=True)
        trial = {'hyperparameter': hyper}
        try:
            if kind == 'nn':
                model.fit_labels(train, epochs=128, lr=hyper)
            elif kind in ('linear', 'quadratic'):
                ridge_fit(model, train, hyper)
            else:
                model.memory_bandwidth = hyper
                model.fit_labels(train, epochs=1)
            error = mse(model, validation)
            trial['validationMSE'] = error; trial['status'] = 'complete'
            if error < best:
                best, winner = error, model
        except Exception as error:
            trial['status'] = 'failed'; trial['error'] = f'{type(error).__name__}: {error}'
        trials.append(trial)
    result = {'world': world, 'origin': origin, 'task': task, 'trainCount': count, 'validationCount': 64, 'testCount': 256,
              'kind': kind, 'width': width, 'sourceSnapshot': source_path, 'sourceSnapshotSha256': hashlib.sha256((ROOT / source_path).read_bytes()).hexdigest(),
              'trainingLabelsHash': digest(train), 'validationLabelsHash': digest(validation),
              'trials': trials, 'selectionCriterion': 'validation-signal-MSE-only', 'elapsedSeconds': time.perf_counter() - start,
              'status': 'complete' if winner is not None else 'failed'}
    if winner is not None:
        prefix = MODEL_ROOT / model_area / key
        winner.save(prefix)
        best_trial = min((x for x in trials if x['status'] == 'complete'), key=lambda x: x['validationMSE'])
        result.update({'validationMSE': best, 'selectedHyperparameter': best_trial['hyperparameter'], 'parameterCount': winner.parameter_count(),
                       'snapshot': str(prefix.relative_to(ROOT)).replace('\\', '/'), 'diagnostics': winner.diagnostics()})
    write_json(report_path, result, exclusive=True)
    print(json.dumps({'experiment': 'diagnostic-validation', 'condition': key, 'status': result['status'], 'seconds': round(result['elapsedSeconds'], 3)}), flush=True)
    return result


def expansion_gate(validation_results):
    path = REPORTS / 'capacity-gate.json'
    if path.exists():
        saved = read_json(path)
        if saved['validationResultsHash'] != digest(validation_results):
            raise ValueError('The saved pre-test gate does not match these validation results.')
        return saved
    comparisons = []
    for world in ('linear', 'quadratic', 'nonlinear'):
        subset = [x for x in validation_results if x['world'] == world and x['trainCount'] == 192 and x['kind'] == 'nn' and x['width'] in (16, 3125)]
        small = [x['validationMSE'] for x in subset if x['width'] == 16 and x['status'] == 'complete']
        large = [x['validationMSE'] for x in subset if x['width'] == 3125 and x['status'] == 'complete']
        full = len(small) == len(large) == 8
        a, b = float(np.mean(small)) if small else None, float(np.mean(large)) if large else None
        comparisons.append({'world': world, 'validCellCountSmall': len(small), 'validCellCountLarge': len(large),
                            'smallValidationMSE': a, 'largeValidationMSE': b,
                            'improvementFraction': (a - b) / a if full and a > 0 else None,
                            'improved': bool(full and b < a)})
    small_all = [x['validationMSE'] for x in validation_results if x['trainCount'] == 192 and x['kind'] == 'nn' and x['width'] == 16 and x['status'] == 'complete']
    large_all = [x['validationMSE'] for x in validation_results if x['trainCount'] == 192 and x['kind'] == 'nn' and x['width'] == 3125 and x['status'] == 'complete']
    a, b = float(np.mean(small_all)) if small_all else None, float(np.mean(large_all)) if large_all else None
    improvement = (a - b) / a if a and b is not None else None
    main = read_json(REPORTS / 'numeric-main.json')['conditions']
    large_conditions = [x for x in main if x['configuration']['kind'] == 'nn' and x['configuration']['representation'] == 'raw8' and x['configuration']['width'] == 3125]
    stability = len(large_conditions) == 24 and all(x['status'] == 'complete' and x['interventions']['unrelatedBranch']['relationalParametersExact'] for x in large_conditions)
    approved = len(small_all) == len(large_all) == 24 and improvement is not None and improvement >= .10 and sum(x['improved'] for x in comparisons) >= 2 and stability
    gate = {'phase': 'validation-before-final-test', 'validationResultsHash': digest(validation_results),
            'rule': '192 labels; plain/composed and four origins weighted equally; 100k improves pooled validation MSE by >=10% over 526, improves in >=2/3 source worlds, and all 100k primary fits finite with exact unrelated-branch preservation.',
            'smallValidationMSE': a, 'largeValidationMSE': b, 'improvementFraction': improvement,
            'sourceWorlds': comparisons, 'finiteAndUnrelatedBranchPreserved': stability, 'millionParameterGatePassed': bool(approved),
            'finalTestInspectedAtGate': False, 'note': 'Gate通過は自律的人格の良さやNNの一般的優位の証明ではない。追加試験は調整者が別途固定する。'}
    write_json(path, gate, exclusive=True)
    print(json.dumps({'experiment': 'capacity-gate', 'passed': bool(approved), 'improvementFraction': improvement}, ensure_ascii=False), flush=True)
    return gate


def diagnostic_validation():
    methods = [('nn', width) for width in data.WIDTHS['raw8']] + [('linear', 16), ('quadratic', 16), ('memory', 16)]
    rows = [fit_candidates(world, origin, composed, count, kind, width)
            for world, origin, composed, count in itertools.product(('linear', 'quadratic', 'nonlinear'), ('nn', 'linear', 'quadratic', 'memory'), (False, True), (48, 192))
            for kind, width in methods]
    gate = expansion_gate(rows)
    write_json(REPORTS / 'numeric-diagnostic-validation.json', {'conditions': rows, 'gate': gate})
    return rows


def diagnostic_test():
    # This command requires the persisted pre-test gate. The coordinating agent
    # can read that gate before any held-out test targets are computed.
    saved = read_json(REPORTS / 'numeric-diagnostic-validation.json')
    rows = saved['conditions']
    gate = expansion_gate(rows)
    if digest(gate) != digest(saved['gate']):
        raise ValueError('Saved validation report and pre-test gate differ.')
    final = final_test_rows(rows)
    write_json(REPORTS / 'numeric-diagnostic.json', {'scope': 'Frozen subjective-function approximation, not gold personality learning; source-family advantage and mathematical coordinate permutations disclosed.', 'conditions': final, 'gate': gate})
    return final


def final_test_rows(rows):
    final = []
    for row in rows:
        result = deepcopy(row)
        if row['status'] == 'complete':
            source, _ = load_origin(row['world'], row['origin'])
            test = data.source_labels(source, data.frozen_inputs(256, 'test'), row['task'] == 'composed')
            model = Personality.load(ROOT / row['snapshot'])
            result['testMSE'] = mse(model, test)
            result['testRanking'] = rank_agreement(model, test)
            result['testLabelsHash'] = digest(test)
        final.append(result)
    return final


def diagnostic_experiment():
    diagnostic_validation()
    return diagnostic_test()


def million_worker(worker, include_diagnostic=True):
    gate = read_json(REPORTS / 'capacity-gate.json')
    protocol = ROOT / 'docs/01b-million-protocol.md'
    if not gate['millionParameterGatePassed'] or not protocol.exists():
        raise ValueError('The pre-test capacity gate and fixed million protocol are required.')
    # Fixed sharding balances two costly stress jobs with the approximation jobs.
    # Each state retains CPU64 and one thread; concurrency does not change data.
    main_jobs = {
        0: [('bounded_signal', 0), ('bounded_signal', 1)],
        1: [('bounded_logit', 0), ('bounded_logit', 1)],
        2: [('bounded_signal', 2), ('bounded_signal', 3)],
        3: [('bounded_logit', 2), ('bounded_logit', 3)],
    }
    for loss, person in main_jobs[worker]:
        train_one(configuration('raw8', loss, 31250, 4311, person), 'million-main')
    if worker in (0, 1):
        stress_one(31250, ('bounded_signal', 'bounded_logit')[worker], 4311, 'million-stress')
    if not include_diagnostic:
        print(json.dumps({'experiment': 'million-online-worker', 'worker': worker, 'status': 'complete', 'cpuFrozenDiagnosticInvoked': False}), flush=True)
        return
    jobs = list(itertools.product(('linear', 'quadratic', 'nonlinear'), ('nn', 'linear', 'quadratic', 'memory'), (False, True)))
    boundaries = ((0, 4), (4, 8), (8, 16), (16, 24))
    left, right = boundaries[worker]
    for world, origin, composed in jobs[left:right]:
        fit_candidates(world, origin, composed, 192, 'nn', 31250, 'million-diagnostic', 'million-diagnostic')
    print(json.dumps({'experiment': 'million-worker', 'worker': worker, 'status': 'complete'}), flush=True)


def million_finalize():
    def read_area(area, expected):
        paths = sorted((REPORTS / area).rglob('*.json'))
        rows = [read_json(path) for path in paths]
        if len(rows) != expected:
            raise ValueError(f'{area} is incomplete: {len(rows)}/{expected}.')
        return rows
    main = read_area('million-main', 8)
    stress = read_area('million-stress', 2)
    validation = read_area('million-diagnostic', 24)
    protocol_sha = hashlib.sha256((ROOT / 'docs/01b-million-protocol.md').read_bytes()).hexdigest()
    write_json(REPORTS / 'million-main.json', {'scope': 'Conditional one-seed capacity expansion; separate from primary 228.', 'protocolSha256': protocol_sha, 'conditions': main})
    write_json(REPORTS / 'million-stress.json', {'scope': 'Conditional one-seed expansion; separate from primary 24.', 'protocolSha256': protocol_sha, 'conditions': stress})
    # Persist all 24 validation selections before any new million test is read.
    write_json(REPORTS / 'million-diagnostic-validation.json', {'protocolSha256': protocol_sha, 'conditions': validation,
        'selectionHash': digest(validation), 'all24ValidationSelectionsFixedBeforeMillionTest': True})
    final = final_test_rows(validation)
    write_json(REPORTS / 'million-diagnostic.json', {'scope': 'All 24 origins/worlds/tasks; n192 only; separate from primary 336.', 'protocolSha256': protocol_sha, 'conditions': final})
    stress_gradient_diagnostic('million-stress.json', 'million-stress-gradient.json')
    summary = {'mainExpected': 8, 'mainComplete': sum(x['status'] == 'complete' for x in main),
        'stressExpected': 2, 'stressComplete': sum(x['status'] == 'complete' for x in stress),
        'diagnosticExpected': 24, 'diagnosticComplete': sum(x['status'] == 'complete' for x in final),
        'seed': 4311, 'parameterCount': 1000014, 'protocolSha256': protocol_sha, 'capacityConditions': [], 'stress': [], 'frozenApproximation': []}
    for loss in data.LOSSES:
        people = [x for x in main if x['configuration']['loss'] == loss and x['status'] == 'complete']
        if people:
            p = [x['checkpoints']['96'] for x in people]
            diversity = [float(np.mean(np.abs(np.asarray(a['unknownOutputs']) - b['unknownOutputs']))) for a, b in itertools.combinations(p, 2)]
            summary['capacityConditions'].append({'loss': loss, 'completeFits': len(people),
                'betweenSeriesMeanAbsoluteDifference': float(np.mean(diversity)) if diversity else None,
                'effectiveRankMean': float(np.mean([x['effectiveRank'] for x in p])),
                'formationMeanAbsoluteChange': float(np.mean([x['formationMeanAbsoluteChange'] for x in p])),
                'newExperienceChangeMean': float(np.mean([x['interventions']['newExperiences']['change']['meanAbsoluteChange'] for x in people])),
                'unrelatedBranchExactAll': all(x['interventions']['unrelatedBranch']['relationalParametersExact'] for x in people),
                'allRestoreExact': all(x['restoreExact'] for x in people)})
    for x in stress:
        if x['status'] == 'complete':
            summary['stress'].append({'loss': x['configuration']['loss'],
                'saturationAfter1000': x['repeatCheckpoints']['1000']['saturationFraction'],
                'changeAfterContrary12': x['contraryCheckpoints']['12']['changeSince1000']['meanAbsoluteChange'],
                'changeAfterContrary96': x['contraryCheckpoints']['96']['changeSince1000']['meanAbsoluteChange'],
                'actualUpdateRMSAfterContrary96': x['contraryCheckpoints']['96']['actualUpdateRMS'],
                'gradientNormAfterContrary96': x['contraryCheckpoints']['96']['lastTraining']['gradientNorm'],
                'preContraryUpdateSignal': x['preFirstContraryDiagnostic']['updateSignal'],
                'acousticBranchExact': x['acousticBranchExact']})
    for task in ('plain', 'composed'):
        cells = [x for x in final if x['task'] == task and x['status'] == 'complete']
        summary['frozenApproximation'].append({'task': task, 'trainCount': 192, 'completeCells': len(cells),
            'validationMSEMean': float(np.mean([x['validationMSE'] for x in cells])) if cells else None,
            'testMSEMean': float(np.mean([x['testMSE'] for x in cells])) if cells else None,
            'testRankAgreementMean': float(np.mean([x['testRanking']['agreement'] for x in cells])) if cells else None})
    write_json(REPORTS / 'million-summary.json', summary)
    return summary


def summarize():
    main = read_json(REPORTS / 'numeric-main.json')['conditions']
    conditions = []
    for rep, loss in itertools.product(data.WIDTHS, data.LOSSES):
        for index, width in enumerate(data.WIDTHS[rep]):
            subset = [x for x in main if x['configuration']['kind'] == 'nn' and x['configuration']['representation'] == rep and x['configuration']['loss'] == loss and x['configuration']['width'] == width]
            completed = [x for x in subset if x['status'] == 'complete']
            row = {'representation': rep, 'loss': loss, 'width': width, 'capacityLabel': data.CAPACITY_NAMES[index], 'expectedFits': 12, 'completeFits': len(completed), 'failedFits': len(subset) - len(completed)}
            if completed:
                row['actualParameterCount'] = completed[0]['parameterCount']
                row['meanElapsedSeconds'] = float(np.mean([x['elapsedSeconds'] for x in completed]))
                for checkpoint in data.CHECKPOINTS:
                    diversity = []
                    for seed in data.SEEDS:
                        people = [x['checkpoints'][str(checkpoint)]['unknownOutputs'] for x in completed if x['configuration']['seed'] == seed]
                        if len(people) == 4:
                            diversity.extend(float(np.mean(np.abs(np.asarray(a) - b))) for a, b in itertools.combinations(people, 2))
                    profiles = [x['checkpoints'][str(checkpoint)] for x in completed]
                    row[f'checkpoint{checkpoint}'] = {'betweenSeriesMeanAbsoluteDifference': float(np.mean(diversity)) if diversity else None,
                        'formationMeanAbsoluteChange': float(np.mean([x['formationMeanAbsoluteChange'] for x in profiles])),
                        'effectiveRankMean': float(np.mean([x['effectiveRank'] for x in profiles])),
                        'saturationFractionMean': float(np.mean([x['saturationFraction'] for x in profiles])),
                        'affinityCuriosityOppositionTotal': int(sum(x['affinityCuriosityOppositionCount'] for x in profiles)),
                        'evaluationsCount': len(profiles) * 96}
                row['newExperienceChangeMean'] = float(np.mean([x['interventions']['newExperiences']['change']['meanAbsoluteChange'] for x in completed]))
                row['unrelatedBranchExactAll'] = all(x['interventions']['unrelatedBranch']['relationalParametersExact'] for x in completed)
            conditions.append(row)
    summary = {'mainNNFitsExpected': 192, 'mainSimpleFitsExpected': 36, 'mainCompleted': sum(x['status'] == 'complete' for x in main), 'mainFailed': sum(x['status'] != 'complete' for x in main), 'capacityConditions': conditions,
               'allInitialReadoutsIdenticalWithinSeed': all(len({x['initialReadoutHash'] for x in main if x['configuration']['seed'] == seed}) == 1 for seed in data.SEEDS),
               'allRestoreExact': all(x.get('restoreExact', False) for x in main if x['status'] == 'complete'),
               'allLearningOffExact': all(x.get('learningOffReadoutExact', False) and x.get('learningOffParameterHashExact', False) for x in main if x['status'] == 'complete')}
    simple = []
    for kind in ('linear', 'quadratic', 'memory'):
        subset = [x for x in main if x['configuration']['kind'] == kind and x['status'] == 'complete']
        row = {'kind': kind, 'loss': 'bounded_logit', 'representation': 'raw8', 'expectedFits': 12, 'completeFits': len(subset)}
        if subset:
            differences = []
            for seed in data.SEEDS:
                people = [x['checkpoints']['96']['unknownOutputs'] for x in subset if x['configuration']['seed'] == seed]
                if len(people) == 4:
                    differences.extend(float(np.mean(np.abs(np.asarray(a) - b))) for a, b in itertools.combinations(people, 2))
            p = [x['checkpoints']['96'] for x in subset]
            row.update({'actualParameterCount': subset[0]['parameterCount'],
                'betweenSeriesMeanAbsoluteDifference': float(np.mean(differences)) if differences else None,
                'formationMeanAbsoluteChange': float(np.mean([x['formationMeanAbsoluteChange'] for x in p])),
                'effectiveRankMean': float(np.mean([x['effectiveRank'] for x in p])),
                'saturationFractionMean': float(np.mean([x['saturationFraction'] for x in p])),
                'affinityCuriosityOppositionTotal': int(sum(x['affinityCuriosityOppositionCount'] for x in p)),
                'newExperienceChangeMean': float(np.mean([x['interventions']['newExperiences']['change']['meanAbsoluteChange'] for x in subset])),
                'unrelatedBranchExactAll': all(x['interventions']['unrelatedBranch']['relationalParametersExact'] for x in subset),
                'meanElapsedSeconds': float(np.mean([x['elapsedSeconds'] for x in subset]))})
        simple.append(row)
    summary['onlineSimpleFamilies'] = simple
    if (REPORTS / 'numeric-stress.json').exists():
        stress = read_json(REPORTS / 'numeric-stress.json')['conditions']
        stress_gradient_diagnostic()
        summary['stress'] = [{'width': width, 'loss': loss, 'expectedFits': 3,
            'completed': sum(x['status'] == 'complete' for x in stress if x['configuration']['width'] == width and x['configuration']['loss'] == loss),
            'meanSaturationAfter1000': float(np.mean([x['repeatCheckpoints']['1000']['saturationFraction'] for x in stress if x['configuration']['width'] == width and x['configuration']['loss'] == loss and x['status'] == 'complete'])),
            'changeAfterContrary12': float(np.mean([x['contraryCheckpoints']['12']['changeSince1000']['meanAbsoluteChange'] for x in stress if x['configuration']['width'] == width and x['configuration']['loss'] == loss and x['status'] == 'complete'])),
            'changeAfterContrary96': float(np.mean([x['contraryCheckpoints']['96']['changeSince1000']['meanAbsoluteChange'] for x in stress if x['configuration']['width'] == width and x['configuration']['loss'] == loss and x['status'] == 'complete']))}
            for width, loss in itertools.product(data.WIDTHS['raw8'], data.LOSSES)]
        summary['stressCompleted'] = sum(x['status'] == 'complete' for x in stress)
        summary['stressFailed'] = sum(x['status'] != 'complete' for x in stress)
        for row in summary['stress']:
            subset = [x for x in stress if x['configuration']['width'] == row['width'] and x['configuration']['loss'] == row['loss'] and x['status'] == 'complete']
            if subset:
                row['acousticBranchExactAll'] = all(x['acousticBranchExact'] for x in subset)
                row['changeAfterStress1000Mean'] = float(np.mean([np.mean(np.abs(np.asarray(x['repeatCheckpoints']['1000']['unknownOutputs']) - np.asarray(x['preStress']['unknownOutputs']))) for x in subset]))
                row['preContraryCueEffectContrastMean'] = float(np.mean([x['preFirstContraryDiagnostic']['cueEffectContrast'] for x in subset]))
                row['preContraryUpdateSignalMean'] = float(np.mean([x['preFirstContraryDiagnostic']['updateSignal'] for x in subset]))
                row['gradientNormAfter1000Mean'] = float(np.mean([x['repeatCheckpoints']['1000']['lastTraining']['gradientNorm'] for x in subset]))
                row['actualUpdateRMSAfter1000Mean'] = float(np.mean([x['repeatCheckpoints']['1000']['actualUpdateRMS'] for x in subset]))
                for point in ('12', '48', '96'):
                    row[f'gradientNormAfterContrary{point}Mean'] = float(np.mean([x['contraryCheckpoints'][point]['lastTraining']['gradientNorm'] for x in subset]))
                    row[f'actualUpdateRMSAfterContrary{point}Mean'] = float(np.mean([x['contraryCheckpoints'][point]['actualUpdateRMS'] for x in subset]))
                    row[f'outputDerivativeAfterContrary{point}Mean'] = float(np.mean([x['contraryCheckpoints'][point]['lastTraining']['outputDerivativeMean'] for x in subset]))
                    row[f'hiddenDerivativeAfterContrary{point}Mean'] = float(np.mean([x['contraryCheckpoints'][point]['lastTraining']['hiddenDerivativeMean'] for x in subset]))
    if (REPORTS / 'numeric-diagnostic.json').exists():
        diagnostic = read_json(REPORTS / 'numeric-diagnostic.json')['conditions']
        summary['frozenApproximation'] = [{'task': task, 'trainCount': count, 'kind': kind, 'width': width,
            'completeCells': sum(x['status'] == 'complete' for x in diagnostic if x['task'] == task and x['trainCount'] == count and x['kind'] == kind and x['width'] == width),
            'validationMSEMean': float(np.mean([x['validationMSE'] for x in diagnostic if x['task'] == task and x['trainCount'] == count and x['kind'] == kind and x['width'] == width and x['status'] == 'complete'])),
            'testMSEMean': float(np.mean([x['testMSE'] for x in diagnostic if x['task'] == task and x['trainCount'] == count and x['kind'] == kind and x['width'] == width and x['status'] == 'complete']))}
            for task, count in itertools.product(('plain', 'composed'), (48, 192)) for kind, width in [('nn', w) for w in data.WIDTHS['raw8']] + [('linear', 16), ('quadratic', 16), ('memory', 16)]]
        for row in summary['frozenApproximation']:
            subset = [x for x in diagnostic if x['task'] == row['task'] and x['trainCount'] == row['trainCount'] and x['kind'] == row['kind'] and x['width'] == row['width'] and x['status'] == 'complete']
            row['testRankAgreementMean'] = float(np.mean([x['testRanking']['agreement'] for x in subset]))
        methods = [('nn', w) for w in data.WIDTHS['raw8']] + [('linear', 16), ('quadratic', 16), ('memory', 16)]
        by_origin = []
        for origin, task, count in itertools.product(('nn', 'linear', 'quadratic', 'memory'), ('plain', 'composed'), (48, 192)):
            for kind, width in methods:
                subset = [x for x in diagnostic if x['origin'] == origin and x['task'] == task and x['trainCount'] == count and x['kind'] == kind and x['width'] == width and x['status'] == 'complete']
                by_origin.append({'origin': origin, 'task': task, 'trainCount': count, 'kind': kind, 'width': width,
                    'completeCells': len(subset),
                    'testMSEMean': float(np.mean([x['testMSE'] for x in subset])) if subset else None,
                    'testRankAgreementMean': float(np.mean([x['testRanking']['agreement'] for x in subset])) if subset else None})
        summary['frozenByOrigin'] = by_origin
        summary['diagnosticCompleted'] = sum(x['status'] == 'complete' for x in diagnostic)
        summary['diagnosticFailed'] = sum(x['status'] != 'complete' for x in diagnostic)
        summary['millionParameterGate'] = read_json(REPORTS / 'capacity-gate.json')
    write_json(REPORTS / 'numeric-summary.json', summary)
    return summary


def pilot():
    rows = []
    for rep in data.WIDTHS:
        for width in data.WIDTHS[rep]:
            model = Personality(seed=4311, width=width, representation=rep, loss='bounded_logit')
            start = time.perf_counter()
            for event in data.training(count=8):
                model.update(event)
            rows.append({'representation': rep, 'width': width, 'parameterCount': model.parameter_count(), 'eightUpdateSeconds': time.perf_counter() - start, 'parameterNorm': model.diagnostics()['parameterNorm']})
    write_json(REPORTS / 'numeric-pilot.json', {'torchThreads': torch.get_num_threads(), 'conditions': rows}, exclusive=True)
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['pilot', 'main', 'stress', 'diagnostic', 'diagnostic-validation', 'diagnostic-test', 'million-worker', 'million-online-worker', 'million-finalize', 'summary', 'all'])
    parser.add_argument('--worker', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.command == 'pilot': pilot()
    if args.command in ('main', 'all'): main_experiment()
    if args.command in ('stress', 'all'): stress_experiment()
    if args.command in ('diagnostic', 'all'): diagnostic_experiment()
    if args.command == 'diagnostic-validation': diagnostic_validation()
    if args.command == 'diagnostic-test': diagnostic_test()
    if args.command == 'million-worker': million_worker(args.worker)
    if args.command == 'million-online-worker': million_worker(args.worker, include_diagnostic=False)
    if args.command == 'million-finalize': million_finalize()
    if args.command in ('summary', 'all'): summarize()
