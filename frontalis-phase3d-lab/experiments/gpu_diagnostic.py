"""Amended GPU64 frozen-function capacity diagnostic, isolated from CPU results.

Teacher labels are frozen with the original CPU runtime. Validation selections
for all 48 cells are committed before final test outputs are evaluated.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import numeric as n
import numpy as np
import torch

LABEL_PATH = ROOT / 'data/million-gpu-frozen-labels.json'
AREA = 'million-gpu-diagnostic'
WIDTHS = (3125, 31250)
JOBS = tuple(itertools.product(('linear', 'quadratic', 'nonlinear'), ('nn', 'linear', 'quadratic', 'memory'), ('plain', 'composed')))


def protocol_hash():
    return hashlib.sha256((ROOT / 'docs/01c-gpu-compute-amendment.md').read_bytes()).hexdigest()


def key(world, origin, task):
    return f'{world}/{origin}/{task}'


def commit_json(path, value):
    """An unchanged rerun is harmless; changed evidence never replaces originals."""
    if path.exists():
        if n.digest(n.read_json(path)) != n.digest(value):
            raise ValueError(f'Committed artifact differs: {path.name}; no overwrite.')
    else:
        n.write_json(path, value, exclusive=True)


def freeze_labels():
    """Only this command invokes the old teacher implementation."""
    if '+cpu' not in torch.__version__:
        raise RuntimeError('Freeze teacher labels with original CPU Torch runtime.')
    cells = {}
    for world, origin, task in JOBS:
        source, source_path = n.load_origin(world, origin)
        blocks = {phase: n.data.source_labels(source, n.data.frozen_inputs(count, phase), task == 'composed')
                  for phase, count in (('train', 192), ('validation', 64), ('test', 256))}
        cells[key(world, origin, task)] = {
            'sourceSnapshot': source_path,
            'sourceSnapshotSha256': hashlib.sha256((ROOT / source_path).read_bytes()).hexdigest(),
            'blocks': blocks,
            'blockHashes': {phase: n.digest(labels) for phase, labels in blocks.items()}}
    result = {'schema': 'phase3d-gpu-label-freeze-v1', 'protocolSha256': protocol_hash(),
              'teacherRuntime': {'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'dtype': 'float64'},
              'all24WorldOriginTaskCells': True, 'cells': cells}
    if LABEL_PATH.exists():
        if n.digest(n.read_json(LABEL_PATH)) != n.digest(result):
            raise ValueError('Frozen label cache differs; no overwrite.')
    else:
        n.write_json(LABEL_PATH, result, exclusive=True)
    print(json.dumps({'stage': 'freeze', 'cells': len(cells), 'bytesSha256': hashlib.sha256(LABEL_PATH.read_bytes()).hexdigest()}), flush=True)
    return result


def frozen_bundle():
    bundle = n.read_json(LABEL_PATH)
    if bundle['protocolSha256'] != protocol_hash() or set(bundle['cells']) != {key(*job) for job in JOBS}:
        raise ValueError('Frozen labels do not cover fixed protocol cells.')
    for cell in bundle['cells'].values():
        for phase, count in (('train', 192), ('validation', 64), ('test', 256)):
            if len(cell['blocks'][phase]) != count or n.digest(cell['blocks'][phase]) != cell['blockHashes'][phase]:
                raise ValueError('Frozen label block was changed.')
        if hashlib.sha256((ROOT / cell['sourceSnapshot']).read_bytes()).hexdigest() != cell['sourceSnapshotSha256']:
            raise ValueError('Frozen teacher snapshot was changed.')
    return bundle


def validation_cell(bundle, world, origin, task, width):
    cell = bundle['cells'][key(world, origin, task)]
    condition_key = f'{world}/{origin}/{task}/n192/nn-h{width}'
    path = ROOT / 'reports' / AREA / f'{condition_key}.json'
    expected = {'world': world, 'origin': origin, 'task': task, 'trainCount': 192, 'validationCount': 64, 'testCount': 256,
                'kind': 'nn', 'width': width, 'sourceSnapshot': cell['sourceSnapshot'], 'sourceSnapshotSha256': cell['sourceSnapshotSha256'],
                'trainingLabelsHash': cell['blockHashes']['train'], 'validationLabelsHash': cell['blockHashes']['validation'],
                'selectionCriterion': 'validation-signal-MSE-only', 'device': 'cuda', 'dtype': 'float64',
                'labelFileSha256': hashlib.sha256(LABEL_PATH.read_bytes()).hexdigest(), 'protocolSha256': protocol_hash(),
                'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0)}}
    if path.exists():
        saved = n.read_json(path)
        if any(saved.get(field) != value for field, value in expected.items()) or [x['hyperparameter'] for x in saved['trials']] != list(n.LR_CHOICES):
            raise ValueError('GPU validation cache differs; no overwrite.')
        if saved['status'] != 'complete':
            raise ValueError('Existing failed GPU cell retained; investigate instead of silently retrying.')
        restored = n.Personality.load(ROOT / saved['snapshot'], device='cuda')
        if restored.parameter_hash() != saved['parameterHash'] or restored.parameter_count() != saved['parameterCount']:
            raise ValueError('GPU validation model differs from committed cache.')
        return saved
    torch.cuda.reset_peak_memory_stats()
    trials = []; winner = None; best = math.inf
    start = time.perf_counter()
    for rate in n.LR_CHOICES:
        trial = {'hyperparameter': rate}; model = None
        trial_start = time.perf_counter()
        try:
            model = n.Personality(seed=4311, width=width, representation='raw8', loss='bounded_logit', kind='nn', device='cuda', dtype='float64', scaled=True)
            model.fit_labels(cell['blocks']['train'], epochs=128, lr=rate)
            torch.cuda.synchronize()
            error = n.mse(model, cell['blocks']['validation'])
            trial.update({'status': 'complete', 'validationMSE': error, 'elapsedSeconds': time.perf_counter() - trial_start})
            if error < best:
                best, winner = error, model
        except Exception as error:
            trial.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'elapsedSeconds': time.perf_counter() - trial_start})
        trials.append(trial)
    result = {**expected, 'trials': trials, 'elapsedSeconds': time.perf_counter() - start,
              'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0)},
              'maxMemoryAllocatedBytes': torch.cuda.max_memory_allocated(), 'maxMemoryReservedBytes': torch.cuda.max_memory_reserved(),
              'status': 'complete' if winner is not None and all(t['status'] == 'complete' for t in trials) else 'failed'}
    if result['status'] == 'complete':
        prefix = ROOT / 'models' / AREA / condition_key
        winner.save(prefix)
        best_trial = min(trials, key=lambda x: x['validationMSE'])
        result.update({'validationMSE': best, 'selectedHyperparameter': best_trial['hyperparameter'],
                       'parameterCount': winner.parameter_count(), 'parameterHash': winner.parameter_hash(),
                       'snapshot': str(prefix.relative_to(ROOT)).replace('\\', '/'), 'diagnostics': winner.diagnostics()})
    n.write_json(path, result, exclusive=True)
    print(json.dumps({'stage': 'gpu-validation', 'condition': condition_key, 'status': result['status'], 'seconds': round(result['elapsedSeconds'], 3)}), flush=True)
    if result['status'] != 'complete':
        raise RuntimeError('GPU diagnostic candidate failed; committed report retained.')
    return result


def calibration(gpu_row, labels, test_labels=None):
    cpu_area = 'numeric-diagnostic' if gpu_row['width'] == 3125 else 'million-diagnostic'
    cpu_path = ROOT / 'reports' / cpu_area / f"{gpu_row['world']}/{gpu_row['origin']}/{gpu_row['task']}/n192/nn-h{gpu_row['width']}.json"
    if not cpu_path.exists():
        return None
    cpu = n.read_json(cpu_path)
    if cpu['status'] != 'complete' or cpu['trainingLabelsHash'] != gpu_row['trainingLabelsHash'] or cpu['validationLabelsHash'] != gpu_row['validationLabelsHash']:
        raise ValueError('CPU calibration cache failed or used different labels.')
    cpu_model = n.Personality.load(ROOT / cpu['snapshot'], device='cpu')
    gpu_model = n.Personality.load(ROOT / gpu_row['snapshot'], device='cuda')
    a, b = n.arrays(cpu_model, labels), n.arrays(gpu_model, labels)
    result = {'world': gpu_row['world'], 'origin': gpu_row['origin'], 'task': gpu_row['task'], 'width': gpu_row['width'],
            'cpuReport': str(cpu_path.relative_to(ROOT)).replace('\\', '/'), 'gpuSnapshot': gpu_row['snapshot'],
            'sameTrainingLabels': True, 'sameValidationLabels': True,
            'cpuSelectedHyperparameter': cpu['selectedHyperparameter'], 'gpuSelectedHyperparameter': gpu_row['selectedHyperparameter'],
            'selectionSame': cpu['selectedHyperparameter'] == gpu_row['selectedHyperparameter'],
            'cpuValidationMSE': cpu['validationMSE'], 'gpuValidationMSE': gpu_row['validationMSE'],
            'validationMSEAbsoluteDifference': abs(cpu['validationMSE'] - gpu_row['validationMSE']),
            'validationOutputMaxAbsoluteDifference': float(np.max(np.abs(a-b))),
            'validationOutputMeanAbsoluteDifference': float(np.mean(np.abs(a-b))),
            'trials': [{'hyperparameter': ca['hyperparameter'], 'cpuValidationMSE': ca['validationMSE'], 'gpuValidationMSE': ga['validationMSE'],
                        'absoluteDifference': abs(ca['validationMSE'] - ga['validationMSE'])} for ca, ga in zip(cpu['trials'], gpu_row['trials'])]}
    if test_labels is not None and gpu_row['width'] == 3125 and (ROOT / 'reports/numeric-diagnostic.json').exists():
        original = next(x for x in n.read_json(ROOT / 'reports/numeric-diagnostic.json')['conditions']
                        if x['world'] == gpu_row['world'] and x['origin'] == gpu_row['origin'] and x['task'] == gpu_row['task']
                        and x['trainCount'] == 192 and x['kind'] == 'nn' and x['width'] == 3125)
        a, b = n.arrays(cpu_model, test_labels), n.arrays(gpu_model, test_labels)
        gpu_mse = float(np.mean((b-np.asarray([row['target'] for row in test_labels])) ** 2))
        result.update({'originalCPUFinalTestMSE': original['testMSE'], 'gpuFinalTestMSE': gpu_mse,
                       'finalTestMSEAbsoluteDifference': abs(original['testMSE']-gpu_mse),
                       'finalTestOutputMaxAbsoluteDifference': float(np.max(np.abs(a-b))),
                       'finalTestCalibrationScope': 'Existing complete primary CPU100k model/old-runtime testMSE versus amended GPU100k; evaluated only after all48 GPU choices fixed.'})
    return result


def pilot():
    bundle = frozen_bundle()
    paths = sorted((ROOT / 'reports/million-diagnostic').rglob('*.json'))
    if not paths:
        raise ValueError('At least one completed CPU1M cell needed for GPU calibration pilot.')
    cpu = next(n.read_json(path) for path in paths if n.read_json(path)['status'] == 'complete')
    row = validation_cell(bundle, cpu['world'], cpu['origin'], cpu['task'], 31250)
    result = calibration(row, bundle['cells'][key(cpu['world'], cpu['origin'], cpu['task'])]['blocks']['validation'])
    result['gpuElapsedSeconds'] = row['elapsedSeconds']; result['cpuElapsedSeconds'] = cpu['elapsedSeconds']
    path = ROOT / 'reports/million-gpu-pilot-calibration.json'
    if not path.exists():
        n.write_json(path, result, exclusive=True)
    else:
        if n.digest(n.read_json(path)) != n.digest(result):
            raise ValueError('Pilot calibration differs; no overwrite.')
    print(json.dumps({'stage': 'pilot-calibration', 'result': result}), flush=True)
    return result


def validation():
    bundle = frozen_bundle()
    rows = [validation_cell(bundle, *job, width) for width, job in itertools.product(WIDTHS, JOBS)]
    result = {'scope': 'Supplemental GPU64 comparison; separate from primary CPU and incomplete CPU1M diagnostic.',
              'protocolSha256': protocol_hash(), 'conditions': rows, 'selectionHash': n.digest(rows),
              'all48ValidationSelectionsFixedBeforeTest': True, 'finalTestEvaluated': False}
    path = ROOT / 'reports/million-gpu-validation.json'
    if path.exists():
        if n.digest(n.read_json(path)) != n.digest(result):
            raise ValueError('Frozen GPU selections differ; no overwrite.')
    else:
        n.write_json(path, result, exclusive=True)
    return result


def finalize():
    bundle = frozen_bundle()
    saved = n.read_json(ROOT / 'reports/million-gpu-validation.json')
    rows = saved['conditions']
    required = {(width, *job) for width, job in itertools.product(WIDTHS, JOBS)}
    actual = {(x['width'], x['world'], x['origin'], x['task']) for x in rows if x['status'] == 'complete'}
    if len(rows) != 48 or actual != required or n.digest(rows) != saved['selectionHash'] or not saved['all48ValidationSelectionsFixedBeforeTest']:
        raise ValueError('All 48 validation selections must be fixed before final test.')
    final, comparisons = [], []
    for row in rows:
        cell = bundle['cells'][key(row['world'], row['origin'], row['task'])]
        model = n.Personality.load(ROOT / row['snapshot'], device='cuda')
        if model.parameter_hash() != row['parameterHash'] or model.parameter_count() != row['parameterCount']:
            raise ValueError('Final test snapshot differs from fixed validation selection.')
        result = deepcopy(row)
        result.update({'testMSE': n.mse(model, cell['blocks']['test']), 'testRanking': n.rank_agreement(model, cell['blocks']['test']),
                       'testLabelsHash': cell['blockHashes']['test']})
        final.append(result)
        comparison = calibration(row, cell['blocks']['validation'], cell['blocks']['test'])
        if comparison:
            comparisons.append(comparison)
    commit_json(ROOT / 'reports/million-gpu-diagnostic.json', {'protocolSha256': protocol_hash(), 'selectionHash': saved['selectionHash'], 'conditions': final})
    commit_json(ROOT / 'reports/million-gpu-device-calibration.json', {'scope': 'CPU1M completed subset and all24 CPU100k validation calibration, no CPU completion claim.', 'conditions': comparisons})
    summary = {'protocolSha256': protocol_hash(), 'gpuExpected': 48, 'gpuComplete': len(final), 'gpuFailed': 0,
               'cpuMillionDiagnosticCompleted': len(list((ROOT / 'reports/million-diagnostic').rglob('*.json'))),
               'cpuMillionDiagnosticExpected': 24, 'frozenApproximation': [], 'deviceCalibration': {
                   'comparisons': len(comparisons), 'selectedRateChanges': sum(not x['selectionSame'] for x in comparisons),
                   'validationOutputMaxAbsoluteDifference': max((x['validationOutputMaxAbsoluteDifference'] for x in comparisons), default=None),
                   'validationMSEAbsoluteDifferenceMax': max((x['validationMSEAbsoluteDifference'] for x in comparisons), default=None),
                   'hundredThousandFinalTestMSEAbsoluteDifferenceMax': max((x['finalTestMSEAbsoluteDifference'] for x in comparisons if 'finalTestMSEAbsoluteDifference' in x), default=None),
                   'hundredThousandFinalTestOutputMaxAbsoluteDifference': max((x['finalTestOutputMaxAbsoluteDifference'] for x in comparisons if 'finalTestOutputMaxAbsoluteDifference' in x), default=None)}}
    for width, task in itertools.product(WIDTHS, ('plain', 'composed')):
        cells = [x for x in final if x['width'] == width and x['task'] == task]
        summary['frozenApproximation'].append({'width': width, 'parameterCount': cells[0]['parameterCount'], 'task': task, 'trainCount': 192, 'completeCells': len(cells),
            'validationMSEMean': float(np.mean([x['validationMSE'] for x in cells])), 'testMSEMean': float(np.mean([x['testMSE'] for x in cells])),
            'testRankAgreementMean': float(np.mean([x['testRanking']['agreement'] for x in cells])), 'elapsedSecondsTotal': sum(x['elapsedSeconds'] for x in cells)})
    for task in ('plain', 'composed'):
        a, b = [next(x for x in summary['frozenApproximation'] if x['width'] == width and x['task'] == task)['testMSEMean'] for width in WIDTHS]
        summary.setdefault('millionVsHundredThousand', []).append({'task': task, 'testMSEImprovementFraction': (a-b)/a})
    commit_json(ROOT / 'reports/million-gpu-summary.json', summary)
    print(json.dumps({'stage': 'gpu-finalize', 'summary': summary}), flush=True)
    return summary


def cpu_online():
    """Publish complete CPU formation/stress without pretending 24 CPU fits ran."""
    def completed(area, count):
        rows = [n.read_json(path) for path in sorted((ROOT / 'reports' / area).rglob('*.json'))]
        if len(rows) != count or any(x['status'] != 'complete' for x in rows):
            raise ValueError(f'{area} is not fully successful: {len(rows)}/{count}.')
        return rows
    main = completed('million-main', 8); stress = completed('million-stress', 2)
    sha = hashlib.sha256((ROOT / 'docs/01b-million-protocol.md').read_bytes()).hexdigest()
    for name, value in [('million-main.json', {'scope': 'Conditional one-seed CPU formation; separate from primary228.', 'protocolSha256': sha, 'conditions': main}),
                        ('million-stress.json', {'scope': 'Conditional one-seed CPU stress; separate from primary24.', 'protocolSha256': sha, 'conditions': stress})]:
        path = ROOT / 'reports' / name
        if path.exists():
            if n.digest(n.read_json(path)) != n.digest(value):
                raise ValueError('Existing CPU online aggregate differs; no overwrite.')
        else:
            n.write_json(path, value, exclusive=True)
    n.stress_gradient_diagnostic('million-stress.json', 'million-stress-gradient.json')
    summary = {'mainExpected': 8, 'mainComplete': 8, 'stressExpected': 2, 'stressComplete': 2,
               'cpuDiagnosticExpected': 24, 'cpuDiagnosticCompleted': len(list((ROOT / 'reports/million-diagnostic').rglob('*.json'))),
               'cpuDiagnosticFinalTestCompleted': False, 'seed': 4311, 'parameterCount': 1000014,
               'protocolSha256': sha, 'capacityConditions': [], 'stress': []}
    for loss in n.data.LOSSES:
        people = [x for x in main if x['configuration']['loss'] == loss]
        profiles = [x['checkpoints']['96'] for x in people]
        diversity = [float(np.mean(np.abs(np.asarray(a['unknownOutputs'])-b['unknownOutputs']))) for a, b in itertools.combinations(profiles, 2)]
        summary['capacityConditions'].append({'loss': loss, 'completeFits': len(people),
            'betweenSeriesMeanAbsoluteDifference': float(np.mean(diversity)),
            'effectiveRankMean': float(np.mean([x['effectiveRank'] for x in profiles])),
            'formationMeanAbsoluteChange': float(np.mean([x['formationMeanAbsoluteChange'] for x in profiles])),
            'newExperienceChangeMean': float(np.mean([x['interventions']['newExperiences']['change']['meanAbsoluteChange'] for x in people])),
            'unrelatedBranchExactAll': all(x['interventions']['unrelatedBranch']['relationalParametersExact'] for x in people),
            'allRestoreExact': all(x['restoreExact'] for x in people)})
    for row in stress:
        summary['stress'].append({'loss': row['configuration']['loss'],
            'saturationAfter1000': row['repeatCheckpoints']['1000']['saturationFraction'],
            'changeAfterContrary12': row['contraryCheckpoints']['12']['changeSince1000']['meanAbsoluteChange'],
            'changeAfterContrary96': row['contraryCheckpoints']['96']['changeSince1000']['meanAbsoluteChange'],
            'actualUpdateRMSAfterContrary96': row['contraryCheckpoints']['96']['actualUpdateRMS'],
            'gradientNormAfterContrary96': row['contraryCheckpoints']['96']['lastTraining']['gradientNorm'],
            'preContraryUpdateSignal': row['preFirstContraryDiagnostic']['updateSignal'], 'acousticBranchExact': row['acousticBranchExact']})
    path = ROOT / 'reports/million-online-summary.json'
    if path.exists():
        if n.digest(n.read_json(path)) != n.digest(summary):
            raise ValueError('Existing CPU online summary differs; no overwrite.')
    else:
        n.write_json(path, summary, exclusive=True)
    print(json.dumps({'stage': 'cpu-online', 'summary': summary}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'pilot', 'validation', 'finalize', 'cpu-online'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'freeze':
        freeze_labels()
        return
    if args.command == 'cpu-online':
        cpu_online()
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently replace GPU protocol with CPU.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == 'pilot': pilot()
    if args.command == 'validation': validation()
    if args.command == 'finalize': finalize()


if __name__ == '__main__':
    main()
