"""Independent subjective appraisal; factual outcome prediction stays fixed.

The initial random appraisal, semantic axes, self-amplification and named
signal gates are research assumptions. No desired emotional answer is learned.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

AXES = ['autonomy', 'constraint', 'change', 'loss', 'achievement', 'discovery', 'regularity', 'others']
CHANNELS = ['affinity', 'curiosity', 'value', 'aesthetic', 'anger', 'joy', 'sadness']
SPACES = ['relational', 'acoustic']
GUARD = {'kind': 'constant', 'outcomes': [.5] * 8, 'version': 1}
SCHEMA = 'frontalis-capacity-v1'
KINDS = ('nn', 'linear', 'quadratic', 'memory')
LOSSES = ('legacy_signal', 'bounded_signal', 'bounded_logit')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()

def _vectors(value, width=8, lo=0., hi=1., batch=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.dtype.kind == 'b':
        raise ValueError('Numeric feature/target vectors required.')
    a = np.asarray(value, dtype=np.float64)
    expected = a.ndim == 2 and a.shape[1] == width if batch else a.shape == (width,)
    if not expected or not np.isfinite(a).all() or np.any(a < lo) or np.any(a > hi):
        raise ValueError('Invalid finite vector or declared range.')
    return a

def _space(space):
    if space not in SPACES:
        raise ValueError('Unknown semantic branch.')
    return space

def _expanded(f):
    return torch.cat([f] + [(f[:, i] * f[:, j])[:, None] for i in range(8) for j in range(i, 8)], dim=1)

def _amplified(raw, gain, loss):
    a = np.asarray(raw, dtype=np.float64)
    if loss == 'legacy_signal' and gain == 1.:
        return a.copy(), np.arctanh(np.clip(a, -.999999, .999999))
    logits = gain * np.arctanh(np.clip(a, -.999999, .999999))
    if loss != 'legacy_signal':
        logits = np.clip(logits, -6., 6.)
    return np.tanh(logits), logits

class Personality:
    def __init__(self, seed=4311, width=16, representation='raw8', loss='bounded_signal', kind='nn', device='cpu', dtype='float64', scaled=True, learning=True, gain=1.25):
        if type(seed) is not int or type(width) is not int or width < 1 or width > 100000:
            raise ValueError('Invalid seed or hidden width.')
        if representation not in ('raw8', 'quadratic44') or loss not in LOSSES or kind not in KINDS:
            raise ValueError('Invalid representation, loss or model family.')
        if type(scaled) is not bool or type(learning) is not bool or not math.isfinite(gain) or not 0 < gain <= 4:
            raise ValueError('Invalid amplification/learning configuration.')
        if dtype not in ('float64', 'float32'):
            raise ValueError('Use float64 or float32.')
        self.seed, self.width, self.representation, self.loss, self.kind = seed, width, representation, loss, kind
        self.device, self.dtype_name = str(device), dtype
        self.dtype = getattr(torch, dtype)
        self.scaled, self.learning, self.gain = scaled, learning, float(gain)
        self.learning_rate, self.epochs, self.replay_limit = .025, 8, 64
        self.gradient_clip, self.memory_bandwidth = 5., .65
        self.input_dim = 8 if representation == 'raw8' else 44
        self.readout_scale = 16. / width if scaled and kind == 'nn' else 1.
        self.common_base = self._tensor(np.random.default_rng(seed).normal(0., .16, (8, 7)))
        hidden = np.random.default_rng(seed + 100003).normal(0., .28 * math.sqrt(8. / self.input_dim), (self.input_dim, width))
        self.branches = {}
        for space in SPACES:
            if kind == 'nn':
                self.branches[space] = {'w1': self._tensor(hidden, True), 'b1': self._tensor(np.zeros(width), True), 'w2': self._tensor(np.zeros((width, 7)), True), 'b2': self._tensor(np.zeros(7), True)}
            elif kind in ('linear', 'quadratic'):
                self.branches[space] = {'coefficients': self._tensor(np.zeros((9 if kind == 'linear' else 45, 7)), True)}
            else:
                self.branches[space] = {'memory': []}
        self.emergency_norm = 50. * math.sqrt((self.parameter_count() or 526) / 526.)
        self.fixed_prediction_guard = copy.deepcopy(GUARD)
        self.replay = {s: [] for s in SPACES}
        self.history, self.appraisals, self.diagnostic_history = [], [], []
        self.generation, self.ledger_hash = 0, digest([])
        self._hash_cache = None

    def _tensor(self, value, grad=False):
        return torch.tensor(value, dtype=self.dtype, device=self.device, requires_grad=grad)

    def _representation(self, f):
        return f if self.representation == 'raw8' else _expanded(f)

    def _basis(self, f):
        features = _expanded(f) if self.kind == 'quadratic' else f
        return torch.cat([features, torch.ones((len(f), 1), dtype=self.dtype, device=self.device)], dim=1)

    def _forward(self, f, space, branch=None):
        branch = self.branches[space] if branch is None else branch
        base = f @ self.common_base
        hidden = None
        if self.kind == 'nn':
            hidden = torch.tanh(self._representation(f) @ branch['w1'] + branch['b1'])
            residual = self.readout_scale * (hidden @ branch['w2']) + branch['b2']
        elif self.kind in ('linear', 'quadratic'):
            residual = self._basis(f) @ branch['coefficients']
        elif branch['memory']:
            points = self._tensor([row['centered'] for row in branch['memory']])
            values = self._tensor([row['residual'] for row in branch['memory']])
            distances = ((f[:, None, :] - points[None, :, :]) ** 2).sum(dim=2)
            weights = torch.softmax(-distances / (2. * self.memory_bandwidth ** 2), dim=1)
            residual = weights @ values
        else:
            residual = torch.zeros_like(base)
        logits = base + residual
        return torch.tanh(logits), logits, hidden

    def parameter_count(self):
        return sum(int(v.numel()) for branch in self.branches.values() for v in branch.values() if isinstance(v, torch.Tensor))

    def _arrays(self):
        arrays = {'commonBase': self.common_base.detach().cpu().numpy().copy()}
        for space, branch in self.branches.items():
            for name, value in branch.items():
                if isinstance(value, torch.Tensor):
                    arrays[space + '__' + name] = value.detach().cpu().numpy().copy()
        return arrays

    def parameter_hash(self):
        if self._hash_cache is None:
            h = hashlib.sha256()
            h.update(digest({'schema': SCHEMA, 'kind': self.kind, 'representation': self.representation, 'readoutScale': self.readout_scale}).encode())
            for name, a in sorted(self._arrays().items()):
                # Normalize portability hash to the numerical values, not device.
                a = np.ascontiguousarray(a, dtype='<f8')
                h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
            if self.kind == 'memory':
                h.update(digest(self.branches).encode())
            self._hash_cache = h.hexdigest()
        return self._hash_cache

    def _stable(self, branches=None):
        if self.fixed_prediction_guard != GUARD:
            raise ValueError('Factual predictor guard modified.')
        branches = self.branches if branches is None else branches
        arrays = [self.common_base] + [v for b in branches.values() for v in b.values() if isinstance(v, torch.Tensor)]
        if any(not bool(torch.isfinite(a).all()) for a in arrays):
            raise ValueError('Non-finite learned parameters.')
        norm = math.sqrt(sum(float((a.detach() ** 2).sum()) for a in arrays))
        if self.kind == 'memory':
            values = [np.asarray(row['residual'], dtype=float) for b in branches.values() for row in b['memory']]
            if any(not np.isfinite(a).all() for a in values):
                raise ValueError('Non-finite memory residual.')
            norm = max([norm] + [float(np.linalg.norm(a)) for a in values])
        if not math.isfinite(norm) or norm > self.emergency_norm:
            raise ValueError('Technical emergency norm exceeded; no update committed.')
        return norm

    def evaluate_batch(self, features, space):
        _space(space); self._stable()
        x = _vectors(features, batch=True)
        if self.loss == 'legacy_signal' and self.device == 'cpu' and self.dtype_name == 'float64':
            result = self._numpy_forward(2. * x - 1., space)[0]
        else:
            with torch.no_grad():
                raw = self._forward(self._tensor(2. * x - 1.), space)[0]
            result = raw.cpu().numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError('Non-finite appraisal.')
        return result

    def evaluate(self, features, space):
        x = _vectors(features)
        raw = self.evaluate_batch(x[None, :], space)[0]
        channels = dict(zip(CHANNELS, raw.tolist()))
        affinity = channels['affinity']
        signals = {'preference': affinity, 'initiative': max(0., affinity), 'dislike': max(0., -affinity), 'resistance': max(0., -affinity), 'curiosity': max(0., channels['curiosity']), 'curiositySigned': channels['curiosity'], 'value': channels['value'], 'aesthetic': channels['aesthetic'], 'anger': max(0., channels['anger']), 'joy': max(0., channels['joy']), 'sadness': max(0., channels['sadness'])}
        return {'raw': raw.tolist(), 'channels': channels, 'signals': signals, 'space': space, 'features': x.tolist(), 'predictedOutcome': copy.deepcopy(GUARD['outcomes']), 'generation': self.generation, 'parameterHash': self.parameter_hash()}

    def _accept(self, event):
        forbidden = {'reward', 'praise', 'userRating', 'valence', 'emotion', 'personalityTarget', 'teacherCoefficients', 'target', 'seriesId'}
        if not isinstance(event, dict) or forbidden.intersection(event):
            raise ValueError('External subjective targets forbidden.')
        if not isinstance(event.get('id'), str) or not event['id'].strip() or not isinstance(event.get('text'), str):
            raise ValueError('Experience source ID/text required.')
        if event.get('confirmed') is not True or event.get('origin') != 'synthetic':
            raise ValueError('Only confirmed synthetic observations accepted.')
        _space(event.get('space')); _vectors(event.get('cue')); _vectors(event.get('observed'))
        if any(row['id'] == event['id'] for row in self.history):
            raise ValueError('Duplicate experience source ID.')
        # Validate retained provenance before any model mutation.
        digest(event)

    def _train_branch(self, space, labels, epochs=None, lr=None):
        if not labels:
            return self.branches[space], {'losses': [], 'gradientNorm': 0.}
        f = self._tensor(np.asarray([2. * _vectors(row['features']) - 1. for row in labels]))
        targets_np = np.asarray([_vectors(row['target'], 7, -1., 1.) for row in labels])
        targets = self._tensor(targets_np)
        target_logits_np = np.asarray([row.get('targetLogit', np.arctanh(np.clip(row['target'], -.999999, .999999))) for row in labels])
        if self.loss != 'legacy_signal':
            target_logits_np = np.clip(target_logits_np, -6., 6.)
            targets_np = np.tanh(target_logits_np)
            targets = self._tensor(targets_np)
        target_logits = self._tensor(target_logits_np)
        if self.loss == 'legacy_signal' and self.device == 'cpu' and self.dtype_name == 'float64' and self.kind != 'memory':
            return self._legacy_numpy_train(space, labels, epochs, lr)
        if self.kind == 'memory':
            centered = f.detach().cpu().numpy()
            base = centered @ self.common_base.detach().cpu().numpy()
            residual = np.arctanh(np.clip(targets_np, -.999999, .999999)) - base if self.loss == 'legacy_signal' else target_logits_np - base
            branch = {'memory': [{'centered': p.tolist(), 'residual': v.tolist(), 'eventId': row.get('eventId'), 'sourceStateHash': row.get('sourceStateHash')} for p, v, row in zip(centered, residual, labels)]}
            raw = self._forward(f, space, branch)[0]
            return branch, {'losses': [float(((raw - targets) ** 2).mean())], 'gradientNorm': 0., 'outputDerivativeMean': float((1. - raw ** 2).mean())}
        branch = {name: value.detach().clone().requires_grad_(True) for name, value in self.branches[space].items()}
        moment = {name: torch.zeros_like(value) for name, value in branch.items()}
        velocity = {name: torch.zeros_like(value) for name, value in branch.items()}
        epochs = self.epochs if epochs is None else epochs
        rate = self.learning_rate if lr is None else lr
        losses, norms, output_derivatives, hidden_derivatives = [], [], [], []
        for epoch in range(1, epochs + 1):
            raw, logits, hidden = self._forward(f, space, branch)
            objective = ((logits - target_logits) ** 2).mean() if self.loss == 'bounded_logit' else ((raw - targets) ** 2).mean()
            losses.append(float(objective.detach()))
            output_derivatives.append(float((1. - raw.detach() ** 2).mean()))
            hidden_derivatives.append(float((1. - hidden.detach() ** 2).mean()) if hidden is not None else 1.)
            gradients = torch.autograd.grad(objective, list(branch.values()))
            norm = math.sqrt(sum(float((g.detach() ** 2).sum()) for g in gradients))
            if not math.isfinite(norm) or not math.isfinite(losses[-1]):
                raise ValueError('Non-finite gradient/loss; no update committed.')
            norms.append(norm)
            scale = min(1., self.gradient_clip / (norm + 1e-12))
            with torch.no_grad():
                for (name, value), gradient in zip(branch.items(), gradients):
                    g = gradient * scale
                    moment[name].mul_(.9).add_(g, alpha=.1)
                    velocity[name].mul_(.999).addcmul_(g, g, value=.001)
                    m = moment[name] / (1. - .9 ** epoch)
                    v = velocity[name] / (1. - .999 ** epoch)
                    value.addcdiv_(m, torch.sqrt(v) + 1e-8, value=-rate)
        return branch, {'losses': losses, 'gradientNorm': norms[-1], 'gradientNormFirst': norms[0], 'outputDerivativeMean': output_derivatives[-1], 'hiddenDerivativeMean': hidden_derivatives[-1]}

    def _numpy_basis(self, f, quadratic=False, bias=True):
        parts = [f]
        if quadratic:
            parts += [(f[:, i] * f[:, j])[:, None] for i in range(8) for j in range(i, 8)]
        if bias:
            parts.append(np.ones((len(f), 1)))
        return np.column_stack(parts)

    def _numpy_forward(self, f, space, parameters=None):
        """Narrow legacy CPU64 bridge avoids self-label roundoff amplification."""
        branch = self.branches[space]
        p = ({name: value.detach().cpu().numpy() for name, value in branch.items() if isinstance(value, torch.Tensor)} if parameters is None else parameters)
        base = f @ self.common_base.detach().cpu().numpy()
        hidden = None
        if self.kind == 'nn':
            inputs = f if self.representation == 'raw8' else self._numpy_basis(f, True, False)
            hidden = np.tanh(inputs @ p['w1'] + p['b1'])
            residual = self.readout_scale * (hidden @ p['w2']) + p['b2']
        elif self.kind in ('linear', 'quadratic'):
            residual = self._numpy_basis(f, self.kind == 'quadratic') @ p['coefficients']
        elif branch['memory']:
            points = np.asarray([row['centered'] for row in branch['memory']])
            values = np.asarray([row['residual'] for row in branch['memory']])
            distances = ((f[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
            logits = -distances / (2. * self.memory_bandwidth ** 2)
            weights = np.exp(logits - logits.max(axis=1, keepdims=True)); weights /= weights.sum(axis=1, keepdims=True)
            residual = weights @ values
        else:
            residual = np.zeros_like(base)
        return np.tanh(base + residual), hidden

    def _legacy_numpy_train(self, space, labels, epochs=None, lr=None):
        f = np.asarray([2. * _vectors(row['features']) - 1. for row in labels])
        targets = np.asarray([row['target'] for row in labels])
        p = {name: np.asarray(value.detach().cpu().numpy().tolist(), dtype=float) for name, value in self.branches[space].items()}
        moment = {name: np.zeros_like(value) for name, value in p.items()}
        velocity = {name: np.zeros_like(value) for name, value in p.items()}
        losses, norms, output_derivatives, hidden_derivatives = [], [], [], []
        rate = self.learning_rate if lr is None else lr
        base = f @ np.asarray(self.common_base.detach().cpu().numpy().tolist())
        for epoch in range(1, (self.epochs if epochs is None else epochs) + 1):
            if self.kind == 'nn':
                inputs = f if self.representation == 'raw8' else self._numpy_basis(f, True, False)
                hidden = np.tanh(inputs @ p['w1'] + p['b1'])
                residual = hidden @ p['w2'] if self.readout_scale == 1. else self.readout_scale * (hidden @ p['w2'])
                raw = np.tanh(base + residual + p['b2'])
            else:
                hidden = None
                raw = np.tanh(base + self._numpy_basis(f, self.kind == 'quadratic') @ p['coefficients'])
            losses.append(float(np.mean((raw - targets) ** 2)))
            derivative = 2. * (raw - targets) * (1. - raw ** 2) / targets.size
            output_derivatives.append(float((1. - raw ** 2).mean()))
            hidden_derivatives.append(float((1. - hidden ** 2).mean()) if hidden is not None else 1.)
            if self.kind == 'nn':
                hd = (derivative @ p['w2'].T) * (1. - hidden ** 2)
                if self.readout_scale != 1.:
                    hd *= self.readout_scale
                inputs = f if self.representation == 'raw8' else self._numpy_basis(f, True, False)
                gradients = {'w1': inputs.T @ hd, 'b1': hd.sum(axis=0), 'w2': hidden.T @ derivative, 'b2': derivative.sum(axis=0)}
                if self.readout_scale != 1.:
                    gradients['w2'] *= self.readout_scale
            else:
                gradients = {'coefficients': self._numpy_basis(f, self.kind == 'quadratic').T @ derivative}
            norm = float(np.sqrt(sum(float((gradient ** 2).sum()) for gradient in gradients.values())))
            if not math.isfinite(norm):
                raise ValueError('Non-finite gradient; original parameters retained.')
            norms.append(norm)
            scale = min(1., self.gradient_clip / (norm + 1e-12))
            for name in p:
                gradient = gradients[name] * scale
                moment[name] = .9 * moment[name] + .1 * gradient
                velocity[name] = .999 * velocity[name] + .001 * gradient ** 2
                m = moment[name] / (1. - .9 ** epoch)
                v = velocity[name] / (1. - .999 ** epoch)
                p[name] -= rate * m / (np.sqrt(v) + 1e-8)
        return {name: self._tensor(value, True) for name, value in p.items()}, {'losses': losses, 'gradientNorm': norms[-1], 'gradientNormFirst': norms[0], 'outputDerivativeMean': output_derivatives[-1], 'hiddenDerivativeMean': hidden_derivatives[-1], 'backend': 'phase3c-numpy-compatibility'}

    def update(self, event):
        self._accept(event); self._stable()
        space = event['space']
        cue, effect = self.evaluate(event['cue'], space), self.evaluate(event['observed'], space)
        target, target_logits = _amplified(effect['raw'], self.gain, self.loss)
        no_info = bool(np.all(np.asarray(event['cue']) == .5) and np.all(np.asarray(event['observed']) == .5))
        before = self.parameter_hash()
        source_hash = digest({'parameters': before, 'ledger': self.ledger_hash, 'generation': self.generation, 'loss': self.loss, 'gain': self.gain, 'learning': self.learning, 'learningRate': self.learning_rate, 'epochs': self.epochs, 'replayLimit': self.replay_limit})
        labels = [{'features': copy.deepcopy(event[key]), 'target': target.tolist(), 'targetLogit': target_logits.tolist(), 'space': space, 'eventId': event['id'], 'role': key, 'sourceStateHash': source_hash} for key in ('cue', 'observed')]
        new_replay = self.replay[space]
        stats = {'losses': [], 'gradientNorm': 0.}
        new_branch = self.branches[space]
        if self.learning and not no_info:
            new_replay = (new_replay + labels)[-self.replay_limit:]
            new_branch, stats = self._train_branch(space, new_replay)
            candidates = dict(self.branches); candidates[space] = new_branch
            self._stable(candidates)
        self.branches[space], self.replay[space] = new_branch, new_replay
        self._hash_cache = None
        record = {'eventId': event['id'], 'sourceStateHash': source_hash, 'parameterHashBefore': before, 'parameterHashAfter': self.parameter_hash(), 'cueBefore': cue['raw'], 'effectBefore': effect['raw'], 'target': target.tolist(), 'targetLogit': target_logits.tolist(), 'noInformation': no_info, 'learningEnabled': self.learning, 'labelRule': 'old-effect-self-amplification', 'lossRule': self.loss, 'gain': self.gain, 'training': stats, 'cueEffectContrast': float(np.mean(np.abs(np.asarray(cue['raw']) - effect['raw'])))}
        self.history.append(copy.deepcopy(event)); self.appraisals.append(record); self.generation += 1
        self.ledger_hash = digest([self.ledger_hash, event, record])
        return copy.deepcopy(record)

    def fit_labels(self, labels, epochs=None, lr=None, provenance='frozen-target-diagnostic'):
        if not isinstance(labels, list) or not provenance or (epochs is not None and (type(epochs) is not int or epochs < 1)) or (lr is not None and (not math.isfinite(lr) or lr <= 0)):
            raise ValueError('Invalid diagnostic training request.')
        for row in labels:
            _space(row['space']); _vectors(row['features']); _vectors(row['target'], 7, -1., 1.)
            if 'targetLogit' in row:
                _vectors(row['targetLogit'], 7, -1e6, 1e6)
        self._stable()
        branches = dict(self.branches); stats = {}
        for space in SPACES:
            subset = [row for row in labels if row['space'] == space]
            if subset:
                branches[space], stats[space] = self._train_branch(space, subset, epochs, lr)
        self._stable(branches)
        self.branches = branches; self._hash_cache = None
        self.diagnostic_history.append({'kind': provenance, 'labelsHash': digest(labels), 'count': len(labels), 'epochs': epochs or self.epochs, 'stats': stats})
        return stats

    def diagnostics(self, event=None):
        norm = self._stable()
        layers = {s: {name: {'norm': float(v.detach().norm()), 'rms': float(torch.sqrt((v.detach() ** 2).mean())), 'count': int(v.numel())} for name, v in b.items() if isinstance(v, torch.Tensor)} for s, b in self.branches.items()}
        result = {'parameterCount': self.parameter_count(), 'fixedBaseParameters': 56, 'parameterNorm': norm, 'emergencyNorm': self.emergency_norm, 'layerStats': layers, 'readoutScale': self.readout_scale, 'device': self.device, 'dtype': self.dtype_name, 'generation': self.generation, 'memoryRows': {s: len(b.get('memory', [])) for s, b in self.branches.items()}}
        if event is not None:
            self._accept(event)
            cue = self.evaluate(event['cue'], event['space'])['raw']
            effect = self.evaluate(event['observed'], event['space'])['raw']
            target, logits = _amplified(effect, self.gain, self.loss)
            result.update({'cueBefore': cue, 'effectBefore': effect, 'target': target.tolist(), 'targetLogit': logits.tolist(), 'cueEffectContrast': float(np.mean(np.abs(np.asarray(cue) - effect))), 'outputDerivativeMean': float(np.mean(1. - np.asarray(cue) ** 2)), 'updateSignal': float(np.mean(np.abs(np.asarray(cue) - target)))})
        return result

    def snapshot(self):
        self._stable()
        configuration = {key: getattr(self, key) for key in ('seed', 'width', 'representation', 'loss', 'kind', 'device', 'dtype_name', 'scaled', 'learning', 'gain', 'learning_rate', 'epochs', 'replay_limit', 'gradient_clip', 'memory_bandwidth', 'emergency_norm')}
        return {'schema': SCHEMA, 'configuration': configuration, 'fixedPredictionGuard': copy.deepcopy(self.fixed_prediction_guard), 'arrays': {k: v.tolist() for k, v in self._arrays().items()}, 'memory': {s: copy.deepcopy(b.get('memory', [])) for s, b in self.branches.items()}, 'replay': copy.deepcopy(self.replay), 'history': copy.deepcopy(self.history), 'appraisals': copy.deepcopy(self.appraisals), 'diagnostics': copy.deepcopy(self.diagnostic_history), 'generation': self.generation, 'ledgerHash': self.ledger_hash, 'parameterHash': self.parameter_hash()}

    def clone(self):
        """Copy independent parameters and retained evidence, without file writes."""
        return copy.deepcopy(self)

    def save(self, prefix):
        prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
        array_path = Path(str(prefix) + '.npz'); meta_path = Path(str(prefix) + '.json')
        if array_path.exists() or meta_path.exists():
            raise FileExistsError('Snapshot generation already exists; no overwrite.')
        snapshot = self.snapshot(); snapshot.pop('arrays')
        with array_path.open('xb') as handle:
            np.savez_compressed(handle, **self._arrays())
        snapshot['arrayFileSha256'] = hashlib.sha256(array_path.read_bytes()).hexdigest()
        envelope = {'metadata': snapshot, 'metadataSha256': digest(snapshot)}
        try:
            with meta_path.open('x', encoding='utf8') as handle:
                json.dump(envelope, handle, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        except Exception:
            array_path.unlink(missing_ok=True)
            raise
        return str(prefix)

    @classmethod
    def load(cls, prefix, device='cpu', dtype=None):
        prefix = Path(prefix)
        envelope = json.loads(Path(str(prefix) + '.json').read_text(encoding='utf8'))
        metadata = envelope['metadata']
        if digest(metadata) != envelope['metadataSha256'] or metadata['schema'] != SCHEMA or metadata['fixedPredictionGuard'] != GUARD:
            raise ValueError('Invalid/tampered snapshot metadata.')
        array_path = Path(str(prefix) + '.npz')
        if hashlib.sha256(array_path.read_bytes()).hexdigest() != metadata['arrayFileSha256']:
            raise ValueError('Invalid/tampered parameter archive.')
        c = metadata['configuration']
        model = cls(seed=c['seed'], width=c['width'], representation=c['representation'], loss=c['loss'], kind=c['kind'], device=device, dtype=c['dtype_name'] if dtype is None else dtype, scaled=c['scaled'], learning=c['learning'], gain=c['gain'])
        for key in ('learning_rate', 'epochs', 'replay_limit', 'gradient_clip', 'memory_bandwidth', 'emergency_norm'):
            setattr(model, key, c[key])
        expected = model._arrays()
        with np.load(array_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ValueError('Wrong snapshot array keys.')
            for name in archive.files:
                array = archive[name]
                if array.shape != expected[name].shape or array.dtype.kind != 'f' or not np.isfinite(array).all():
                    raise ValueError('Invalid snapshot array shape/type/value.')
                if name == 'commonBase':
                    model.common_base = model._tensor(array)
                else:
                    space, key = name.split('__')
                    model.branches[space][key] = model._tensor(array, True)
        if model.kind == 'memory':
            for space in SPACES:
                model.branches[space]['memory'] = copy.deepcopy(metadata['memory'][space])
        model.replay, model.history, model.appraisals = copy.deepcopy(metadata['replay']), copy.deepcopy(metadata['history']), copy.deepcopy(metadata['appraisals'])
        model.diagnostic_history = copy.deepcopy(metadata['diagnostics'])
        model.generation, model.ledger_hash = metadata['generation'], metadata['ledgerHash']
        model._hash_cache = None
        model._stable()
        # Numerical cast may alter parameters; exact hash check applies without cast.
        if model.dtype_name == c['dtype_name'] and model.parameter_hash() != metadata['parameterHash']:
            raise ValueError('Snapshot parameter hash mismatch.')
        return model
