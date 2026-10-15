"""Integrity-checked, portable snapshots with explicit append-only rollback."""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import numpy as np
from npn.model import (SCHEMA, SPACES, CHANNELS, KINDS, GUARD, accepted,
                       digest, evaluate, _assert_stable, _vector, _amplified)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate(state):
    if not isinstance(state, dict):
        raise ValueError('State must be a record.')
    try:
        if state.get('formatVersion') != 1 or state.get('schema') != SCHEMA or state.get('owner') != 'frontalis':
            raise ValueError('State schema/owner mismatch.')
        if state['kind'] not in KINDS or type(state['seed']) is not int or type(state['learning']) is not bool:
            raise ValueError('Invalid model family/initialization.')
        if not math.isfinite(state['gain']) or not 0 < state['gain'] <= 4:
            raise ValueError('Invalid amplification.')
        for key in ('learningRate', 'gradientClip', 'emergencyNorm', 'memoryBandwidth'):
            if not math.isfinite(state[key]) or state[key] <= 0:
                raise ValueError('Invalid positive update constant.')
        if type(state['epochs']) is not int or not 1 <= state['epochs'] <= 1000:
            raise ValueError('Invalid epoch count.')
        if type(state['replayLimit']) is not int or not 1 <= state['replayLimit'] <= 10000:
            raise ValueError('Invalid replay window.')
        if state['fixedPredictionGuard'] != GUARD or set(state['branches']) != set(SPACES) or set(state['replay']) != set(SPACES):
            raise ValueError('Invalid branch/predictor structure.')
        base = np.asarray(state['commonBase'], dtype=float)
        if base.shape != (8, 7) or not np.isfinite(base).all():
            raise ValueError('Invalid initial readout.')
        for space in SPACES:
            branch = state['branches'][space]
            if state['kind'] == 'nn':
                shapes = {'w1': (8, 16), 'b1': (16,), 'w2': (16, 7), 'b2': (7,)}
            elif state['kind'] in ('linear', 'quadratic'):
                shapes = {'coefficients': (9 if state['kind'] == 'linear' else 45, 7)}
            else:
                shapes = {}
                if len(branch['memory']) > state['replayLimit']:
                    raise ValueError('Memory window exceeded.')
                for row in branch['memory']:
                    _vector(row['centered'], 8, -1, 1)
                    value = np.asarray(row['residual'], dtype=float)
                    if value.shape != (7,) or not np.isfinite(value).all():
                        raise ValueError('Invalid learned memory residual.')
            for key, shape in shapes.items():
                value = np.asarray(branch[key], dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('Invalid trainable parameter shape/value.')
            if len(state['replay'][space]) > state['replayLimit']:
                raise ValueError('Replay window exceeded.')
            for row in state['replay'][space]:
                if row['space'] != space or not _hash(row['sourceStateHash']):
                    raise ValueError('Invalid frozen-label provenance.')
                _vector(row['features']); _vector(row['target'], 7, -1, 1)
        ids = []
        for event in state['history']:
            accepted(event); ids.append(event['id'])
        if type(state['generation']) is not int or len(ids) != len(set(ids)) or state['generation'] != len(ids) or len(state['appraisals']) != len(ids):
            raise ValueError('Invalid experience ancestry.')
        ledger = digest([])
        for event, appraisal in zip(state['history'], state['appraisals']):
            if appraisal['eventId'] != event['id'] or not _hash(appraisal['sourceStateHash']):
                raise ValueError('Invalid appraisal attribution.')
            for key in ('parameterHashBefore', 'parameterHashAfter'):
                if not _hash(appraisal[key]):
                    raise ValueError('Invalid parameter attribution.')
            _vector(appraisal['cueBefore'], 7, -1, 1)
            effect = _vector(appraisal['effectBefore'], 7, -1, 1)
            target = _vector(appraisal['target'], 7, -1, 1)
            if not np.array_equal(target, _amplified(effect, appraisal['gain'])):
                raise ValueError('Frozen self-label rule mismatch.')
            losses = np.asarray(appraisal['trainingLoss'], dtype=float)
            if not np.isfinite(losses).all() or np.any(losses < 0):
                raise ValueError('Invalid training loss.')
            ledger = digest([ledger, event, appraisal])
        if ledger != state['ledgerHash']:
            raise ValueError('Experience/appraisal provenance mismatch.')
        for diagnostic in state['diagnostics']:
            if not _hash(diagnostic['sourceStateHash']) or not _hash(diagnostic['labelsHash']) or not isinstance(diagnostic['kind'], str) or not diagnostic['kind']:
                raise ValueError('Invalid diagnostic attribution.')
        _assert_stable(state)
        for space in SPACES:
            if not np.isfinite(evaluate(state, [.5] * 8, space)['raw']).all():
                raise ValueError('Invalid inference state.')
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Incomplete or malformed state.') from error
    return True


def save(path, state):
    validate(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {'sha256': digest(state), 'payload': state}
    path.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def load(path):
    wrapper = json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(wrapper, dict) or not isinstance(wrapper.get('payload'), dict):
        raise ValueError('Invalid snapshot wrapper.')
    if wrapper.get('sha256') != digest(wrapper['payload']):
        raise ValueError('Snapshot integrity mismatch.')
    validate(wrapper['payload'])
    return wrapper['payload']


class Store:
    """Single-writer research store; expected_head is not a process lock."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = self.directory / 'generations.json'
        if not self.file.exists():
            self._write({'head': None, 'generations': []})

    def _write(self, value):
        pending = self.file.with_suffix('.pending')
        with pending.open('w', encoding='utf8') as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.flush(); os.fsync(output.fileno())
        pending.replace(self.file)

    def _read(self):
        return json.loads(self.file.read_text(encoding='utf8'))

    def commit(self, state, expected_head=None, kind='learn', rollback_target=None):
        validate(state)
        if expected_head is not None and type(expected_head) is not int:
            raise ValueError('Invalid expected generation.')
        manifest = self._read()
        if manifest['head'] != expected_head:
            raise ValueError('Stale generation.')
        if kind not in ('learn', 'rollback', 'initial', 'diagnostic'):
            raise ValueError('Unknown store operation.')
        sha = digest(state)
        snapshot = self.directory / (sha + '.json')
        if snapshot.exists():
            if load(snapshot) != state:
                raise ValueError('Immutable snapshot mismatch.')
        else:
            save(snapshot, state)
        generation = len(manifest['generations'])
        manifest['generations'].append({'generation': generation, 'parent': manifest['head'],
            'kind': kind, 'hash': sha, 'rollbackTarget': rollback_target})
        manifest['head'] = generation
        self._write(manifest)
        return generation

    def load(self, generation=None):
        manifest = self._read()
        selected = manifest['head'] if generation is None else generation
        if type(selected) is not int or not 0 <= selected < len(manifest['generations']):
            raise ValueError('Unknown generation.')
        sha = manifest['generations'][selected]['hash']
        if not _hash(sha):
            raise ValueError('Invalid snapshot hash.')
        return load(self.directory / (sha + '.json'))

    def rollback(self, target, expected_head):
        return self.commit(self.load(target), expected_head, 'rollback', target)

    def export(self):
        manifest = self._read()
        return {'manifest': manifest, 'snapshots': [self.load(i) for i in range(len(manifest['generations']))]}

    def restore(self, bundle):
        if self._read()['head'] is not None:
            raise ValueError('Restore requires an empty store.')
        rows = bundle['manifest']['generations']
        states = bundle['snapshots']
        if len(rows) != len(states) or bundle['manifest']['head'] != (len(rows) - 1 if rows else None):
            raise ValueError('Invalid export.')
        for i, (row, state) in enumerate(zip(rows, states)):
            validate(state)
            if row['generation'] != i or row['parent'] != (i - 1 if i else None) or row['hash'] != digest(state):
                raise ValueError('Invalid export ancestry.')
            if row['kind'] not in ('learn', 'rollback', 'initial', 'diagnostic'):
                raise ValueError('Invalid export operation.')
            if row['kind'] == 'rollback' and (type(row['rollbackTarget']) is not int or not 0 <= row['rollbackTarget'] < i):
                raise ValueError('Invalid rollback ancestry.')
        for i, (row, state) in enumerate(zip(rows, states)):
            self.commit(state, i - 1 if i else None, row['kind'], row['rollbackTarget'])

