"""Adaptive radial-basis NPN with separate transient affect and immutable episodes.

All inference of long-term personality uses numerical parameters, never diary text.
The learning target is the pre-update NPN's appraisal of the interpreted outcome.
Prototype allocation and adaptation are local unsupervised parameter updates.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .meaning import digest, encode, validate, vector

WIDTHS = [5, 19, 98, 980, 9803]
SEEDS = [6311, 6312, 6313]
SPACES = ['relational', 'acoustic']
SCHEMA = 'frontalis-temporal-rbf-v1'
CHANNELS = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6']
# The first four abstract initial tendencies change numerical parameters only.
PROFILES = [
    [1., 1., 1., 1., 1., 1., 1., 1.],
    [1.6, .6, 1.5, .7, .8, 1.6, .5, .8],
    [.6, 1.5, .5, 1.6, 1.2, .7, 1.6, 1.4],
    [1., 1., 1., 1.5, .8, 1.4, .9, 1.7],
]

def json_write(path, obj):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def validate_episode(e):
    fields = {'id', 'kind', 'text', 'facts', 'interpretation', 'confidence', 'links', 'space', 'origin'}
    if not isinstance(e, dict) or set(e) != fields or e['origin'] != 'synthetic':
        raise ValueError('An exact synthetic episode schema is required.')
    if not isinstance(e['id'], str) or not e['id'] or len(e['id']) > 100:
        raise ValueError('Invalid episode ID.')
    if not isinstance(e['text'], str) or not 1 <= len(e['text']) <= 8000:
        raise ValueError('Invalid episode text.')
    if e['kind'] not in ['experience', 'correction', 'reactivation'] or e['space'] not in SPACES:
        raise ValueError('Unknown episode kind or domain.')
    for k in ['facts', 'interpretation']:
        validate(e[k])
    if type(e['confidence']) not in [float, int] or not math.isfinite(e['confidence']) or not 0 <= e['confidence'] <= 1:
        raise ValueError('Confidence must be finite in [0,1].')
    if not isinstance(e['links'], list) or len(e['links']) > 8:
        raise ValueError('Invalid temporal links.')
    for link in e['links']:
        if not isinstance(link, dict) or set(link) != {'id', 'relation'} or link['relation'] not in ['follows', 'corrects', 'recalls'] or not isinstance(link['id'], str):
            raise ValueError('Invalid episode reference.')
    if e['kind'] == 'correction' and not any(x['relation'] == 'corrects' for x in e['links']):
        raise ValueError('A correction must reference its earlier interpretation.')
    if e['kind'] == 'reactivation' and not any(x['relation'] == 'recalls' for x in e['links']):
        raise ValueError('A reactivation must reference an earlier episode.')
    return copy.deepcopy(e)

class TemporalNPN:
    def __init__(self, seed=6311, capacity=0, profile=0, decay='consolidation', rate=.01,
                 learner='local', device='cpu', learning=True, emotional=True):
        if type(seed) is not int or type(capacity) is not int or capacity not in range(5) or type(profile) is not int or profile not in range(4):
            raise ValueError('Invalid personality configuration.')
        if decay not in ['uniform', 'selective', 'consolidation', 'none'] or learner not in ['local', 'dense']:
            raise ValueError('Unknown update rule.')
        if type(rate) not in [float, int] or not math.isfinite(rate) or not 0 <= rate <= .25 or type(learning) is not bool or type(emotional) is not bool:
            raise ValueError('Invalid update configuration.')
        self.config = dict(seed=seed, capacity=capacity, profile=profile, decay=decay, rate=rate, learner=learner, learning=learning, emotional=emotional)
        self.device = str(device); self.width = WIDTHS[capacity]
        # Fixed initial factual evaluation, common across capacities. It is part of the parameter artifact.
        self.base = self.t(np.random.default_rng(seed).normal(0, .22, (8, 7)) * np.asarray(PROFILES[profile])[:, None])
        self.centers = self.t(np.zeros((2, self.width, 44)))
        self.weights = self.t(np.zeros((2, self.width, 7)))
        self.used = np.zeros((2, self.width), dtype=bool)
        self.consolidation = self.t(np.zeros((2, self.width)))
        self.short = np.zeros(7); self.mood = np.zeros(7)
        self.clock = 0; self.episodes = []; self.ticks = []; self._hash = None

    def t(self, x): return torch.as_tensor(np.asarray(x), dtype=torch.float64, device=self.device).clone()
    def count(self): return 2 * self.width * 51
    def active_count(self): return int(self.used.sum()) * 51
    def arrays(self):
        return dict(base=self.base.cpu().numpy().copy(), centers=self.centers.cpu().numpy().copy(), weights=self.weights.cpu().numpy().copy(), used=self.used.copy())
    def parameter_hash(self):
        if self._hash is None:
            h = hashlib.sha256(SCHEMA.encode())
            for k, v in sorted(self.arrays().items()):
                h.update(k.encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
            self._hash = h.hexdigest()
        return self._hash
    def branch_hash(self, space):
        s = SPACES.index(space)
        return digest([self.centers[s].cpu().tolist(), self.weights[s].cpu().tolist(), self.used[s].tolist()])
    def _features(self, a, s):
        idx = np.flatnonzero(self.used[s])
        if len(idx) == 0: return idx, self.t(np.zeros((len(a), 0)))
        c = self.centers[s, idx]
        # Factual and contextual distances each have a defined, capacity-independent scale.
        delta = a[:, None, :] - c[None, :, :]
        distance = .5 * (delta[:, :, :8] ** 2).mean(2) + .5 * (delta[:, :, 8:] ** 2).mean(2)
        h = torch.exp(-distance / (2 * .18 ** 2))
        return idx, h / (1 + h.sum(1, keepdim=True))
    def logits(self, x, space='relational'):
        if space not in SPACES: raise ValueError('Unknown domain.')
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 1: a = a[None, :]
        if a.ndim != 2 or a.shape[1] != 44: raise ValueError('Expected semantic44 input.')
        for row in a: vector(row, 44)
        a = self.t(2 * a - 1); s = SPACES.index(space)
        idx, h = self._features(a, s)
        z = a[:, :8] @ self.base
        if len(idx): z = z + h @ self.weights[s, idx]
        return z.cpu().numpy()
    def evaluate(self, meaning, space='relational'):
        z = self.logits(encode(meaning), space)[0]; long = np.tanh(z)
        combined = np.tanh(z + .3 * self.short + .2 * self.mood) if self.config['emotional'] else long.copy()
        return dict(longTerm=long.tolist(), shortTerm=self.short.tolist(), mood=self.mood.tolist(), combined=combined.tolist(),
                    stance='approve' if combined[0] > .08 else 'refuse' if combined[0] < -.08 else 'hold',
                    valence=float((combined[5] - combined[6]) / 2), activation=float(np.mean(np.abs(combined[4:]))),
                    coactivation=float(min(max(0, combined[5]), max(0, combined[6]))),
                    update=self.clock, parameterHash=self.parameter_hash())
    def _decay(self, s=None, activation=None):
        if not self.config['learning'] or self.config['decay'] == 'none' or self.config['rate'] == 0: return
        mode = self.config['decay']; rates = torch.ones_like(self.consolidation) * self.config['rate']
        if mode == 'consolidation': rates /= (1 + self.consolidation)
        if mode == 'selective' and s is not None and activation is not None:
            idx, a = activation
            rates[s, idx] *= (1 - .9 * torch.clamp(a * 2, 0, 1))
        self.weights *= (1 - rates[:, :, None]); self.consolidation *= .995; self._hash = None
    def _emotion(self, stimulus=None):
        if not self.config['emotional']: return
        self.short = .45 * self.short + (.55 * stimulus if stimulus is not None else 0)
        self.mood = .9 * self.mood + .1 * self.short
    def stable(self):
        for v in [self.base, self.centers, self.weights, self.consolidation]:
            if not bool(torch.isfinite(v).all()): raise ValueError('Nonfinite numerical state.')
        if float(torch.max(torch.abs(self.weights))) > 100 or not np.isfinite([self.short, self.mood]).all():
            raise ValueError('Numerical bound exceeded.')
    def advance(self, count=1):
        if type(count) is not int or not 1 <= count <= 4096: raise ValueError('Invalid update count.')
        for _ in range(count):
            self._decay(); self._emotion(); self.clock += 1
            self.ticks.append(dict(update=self.clock, kind='elapsed', shortNorm=float(np.linalg.norm(self.short)), moodNorm=float(np.linalg.norm(self.mood))))
        self.stable()
    def _learn(self, cue, target, confidence, s):
        a = self.t(2 * cue - 1); idx = np.flatnonzero(self.used[s])
        if len(idx):
            d = .5 * ((self.centers[s, idx, :8] - a[:8]) ** 2).mean(1) + .5 * ((self.centers[s, idx, 8:] - a[8:]) ** 2).mean(1)
            near = int(idx[int(torch.argmin(d))]); distance = float(d.min())
        else: near = 0; distance = math.inf
        free = np.flatnonzero(~self.used[s])
        if len(free) and distance > .015:
            near = int(free[0]); self.centers[s, near] = a; self.used[s, near] = True
        elif self.config['learner'] == 'local':
            self.centers[s, near] += .03 * confidence * (a - self.centers[s, near])
        idx, h = self._features(a[None, :], s); h = h[0]
        z = a[:8] @ self.base + h @ self.weights[s, idx]
        delta = self.t(target) - z
        credit = torch.ones_like(h) / (1 + len(h)) if self.config['learner'] == 'dense' else h
        step = .3 * confidence * credit[:, None] * delta[None, :] / (1e-6 + (credit ** 2).sum())
        self.weights[s, idx] = torch.clamp(self.weights[s, idx] + torch.clamp(step, -.5, .5), -32, 32)
        self.consolidation[s, idx] += h
        self._hash = None
        return dict(targetError=float(torch.sqrt((delta ** 2).mean())), gradientNorm=float(torch.linalg.norm(delta) * torch.linalg.norm(h)),
                    activeUnits=int(self.used[s].sum()), outputDerivative=float((1 - torch.tanh(z) ** 2).mean()),
                    clippedSteps=int((torch.abs(step) > .5).sum()), allocationDistance=None if not math.isfinite(distance) else distance)
    def update(self, e):
        e = validate_episode(e)
        if any(x['episode']['id'] == e['id'] for x in self.episodes): raise ValueError('Duplicate experience ID.')
        ids = {x['episode']['id'] for x in self.episodes}
        if any(link['id'] not in ids for link in e['links']): raise ValueError('Unknown earlier reference.')
        before = self.parameter_hash(); appraisal = self.evaluate(e['interpretation'], e['space'])
        cue = encode(e['interpretation'], observed=False)
        outcome = encode(e['interpretation']); outcome[8:] = .5
        # Outcome isolation is the explicitly labeled Phase 3-E follow-up hypothesis.
        old_outcome = self.logits(outcome, e['space'])[0]
        target = np.clip(1.1 * old_outcome, -4, 4)
        s = SPACES.index(e['space']); idx, h = self._features(self.t(2 * cue[None, :] - 1), s)
        self._decay(s, (idx, h[0])); self._emotion(np.asarray(appraisal['longTerm']))
        training = self._learn(cue, target, float(e['confidence']), s) if self.config['learning'] and e['confidence'] > 0 else {}
        self.clock += 1; self.stable()
        row = dict(episode=e, update=self.clock, before=before, after=self.parameter_hash(), appraisalBefore=appraisal,
                   outcomeAppraisalBefore=np.tanh(old_outcome).tolist(), targetLogit=target.tolist(), training=training)
        row['recordHash'] = digest(row); self.episodes.append(copy.deepcopy(row))
        self.ticks.append(dict(update=self.clock, kind=e['kind'], id=e['id'], shortNorm=float(np.linalg.norm(self.short)), moodNorm=float(np.linalg.norm(self.mood))))
        return copy.deepcopy(row)
    def reactivate(self, source_id, new_id):
        old = next((x['episode'] for x in self.episodes if x['episode']['id'] == source_id), None)
        if old is None: raise ValueError('Unknown reactivation source.')
        e = copy.deepcopy(old); e.update(id=new_id, kind='reactivation', confidence=min(.6, old['confidence']), links=[dict(id=source_id, relation='recalls')])
        return self.update(e)
    def clone(self): return copy.deepcopy(self)
    def save(self, prefix):
        p = Path(prefix); p.parent.mkdir(parents=True, exist_ok=True)
        a = Path(str(p) + '.npz'); j = Path(str(p) + '.json')
        if a.exists() or j.exists(): raise FileExistsError('Existing snapshots are immutable.')
        self.stable()
        with a.open('xb') as f: np.savez_compressed(f, **self.arrays(), consolidation=self.consolidation.cpu().numpy())
        meta = dict(schema=SCHEMA, config=self.config, update=self.clock, short=self.short.tolist(), mood=self.mood.tolist(),
                    episodes=self.episodes, ticks=self.ticks, parameterHash=self.parameter_hash(), arraySha256=hashlib.sha256(a.read_bytes()).hexdigest())
        try: json_write(j, dict(metadata=meta, metadataSha256=digest(meta)))
        except Exception: a.unlink(missing_ok=True); raise
    @classmethod
    def load(cls, prefix, device='cpu'):
        p = Path(prefix); m = json.loads(Path(str(p) + '.json').read_text(encoding='utf8')); meta = m['metadata']; a = Path(str(p) + '.npz')
        if meta['schema'] != SCHEMA or digest(meta) != m['metadataSha256'] or hashlib.sha256(a.read_bytes()).hexdigest() != meta['arraySha256']:
            raise ValueError('Snapshot integrity mismatch.')
        n = cls(**meta['config'], device=device)
        expected = dict(n.arrays(), consolidation=n.consolidation.cpu().numpy())
        with np.load(a, allow_pickle=False) as data:
            if set(data.files) != set(expected): raise ValueError('Snapshot array keys mismatch.')
            for k, v in data.items():
                if v.shape != expected[k].shape or v.dtype != expected[k].dtype or not np.isfinite(v).all(): raise ValueError('Invalid snapshot shape/type/value.')
                setattr(n, k, v.copy() if k == 'used' else n.t(v))
        n.clock = meta['update']; n.short = np.asarray(meta['short']); n.mood = np.asarray(meta['mood'])
        n.episodes = meta['episodes']; n.ticks = meta['ticks']
        for row in n.episodes:
            validate_episode(row['episode']); r = dict(row); r.pop('recordHash')
            if digest(r) != row['recordHash']: raise ValueError('Episode history integrity mismatch.')
        n.stable()
        if n.parameter_hash() != meta['parameterHash']: raise ValueError('Parameter hash mismatch.')
        return n

def tool_evaluate(n, meaning, options=None, space='relational'):
    scene = n.evaluate(meaning, space)
    candidates = [dict(id=o['id'], appraisal=n.evaluate(o['meaning'], space)) for o in options or []]
    chosen = max(candidates, key=lambda o:o['appraisal']['combined'][2]) if candidates else None
    return dict(scene=scene, candidates=candidates, selection=chosen['id'] if chosen else 'hold', selected=chosen['appraisal'] if chosen else None,
                channels=CHANNELS, channelGuide=['好意/接近', '好奇心', '価値判断', '美的関心', '怒り方向', '喜び方向', '悲しみ方向'],
                interpretation='各軸は独立した連続値。感情の強度は正の部分。負値は逆の感情を意味しない。数値のゼロを感情があると説明しない。',
                parameterHash=n.parameter_hash(), update=n.clock)
