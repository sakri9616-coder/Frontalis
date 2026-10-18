"""Sim0: analysis of the existing rule (prototype.py) -- toy sketch, not an experiment.
(a) 1-D reductions of felt-q delta rule: feature in cue+outcome (c=1) vs outcome-only (c=0).
(b) Long histories with the unmodified prototype: does v stay bounded? does the sign lock in?
(c) Content-permutation control: shuffle which outcome features occur, keep marginals -> are values changed?
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prototype as P

print('=== (a) 1-D reductions: v <- (1-d) v + a_eff * delta ===')
d = 0.002
def run1d(i, v0, c, a_eff, o=1.0, T=20000, d=d, beta_m=0.0):
    v = v0
    for t in range(T):
        felt = np.tanh((i + v) * o + beta_m); q = np.tanh(c * v)
        v = (1 - d) * v + a_eff * (felt - q) * (c + (1.0 if c == 0 else 0.0) * 0 + 0)  # placeholder, replaced below
    return v

def run1d_proto(i, v0, c, alpha=0.25, kappa=0.4, gam=0.5, o=1.0, T=20000, d=0.002, mood=0.0, confirm=None):
    """prototype-faithful: v += rate*delta*c  (situation) + gam*rate*delta*o (co-occurrence); rate=alpha*(1+-kappa)."""
    v = v0
    for t in range(T):
        felt = np.tanh((i + v) * o + mood); q = np.tanh(v * c); delta = felt - q
        confirms = (q != 0 and np.sign(delta) == np.sign(q))
        rate = alpha * ((1 + kappa) if confirms else (1 - kappa))
        v = v - d * v + rate * delta * c + gam * rate * delta * o
    return v

print('case c=1 (feature in cue and outcome), innate i:')
for i in (0.5, -0.5, 0.1, 0.0):
    print('  i=%+.1f  v* from v0=0: %+.3f   from v0=+3: %+.3f   from v0=-3: %+.3f' % (
        i, run1d_proto(i, 0, 1), run1d_proto(i, 3, 1), run1d_proto(i, -3, 1)))
print('case c=0 (outcome-only feature), innate i  (bistable: unstable point near -i):')
for i in (0.5, -0.5, 0.0):
    for v0 in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.6):
        pass
    print('  i=%+.1f  ' % i + '  '.join('v0=%+.1f->%+.1f' % (v0, run1d_proto(i, v0, 0, T=60000)) for v0 in (-0.7, -0.55, -0.45, -0.3, 0.0, 0.3, 0.7)))
print('outcome-only, o sign flips (o=-1): dv independent of sign(o)?')
for i in (0.5,):
    print('  i=%+.1f o=-1 ' % i + '  '.join('v0=%+.1f->%+.1f' % (v0, run1d_proto(i, v0, 0, o=-1.0, T=60000)) for v0 in (-0.7, -0.3, 0.0, 0.3, 0.7)))
print('no decay, outcome-only, i=0.5, v0=0: v after t steps:', [round(run1d_proto(0.5, 0, 0, T=t, d=0.0), 2) for t in (10, 100, 1000, 10000)])

print('\n=== (b) prototype, long histories (repeat each history block n times) ===')
def long_history(kind, reps):
    ev = []
    for r in range(reps):
        for k, (e, c, o, by) in enumerate(P.history(kind)):
            ev.append((f'{e}-r{r}', c, o, by))
    return ev
for kind in P.KINDS:
    for reps in (1, 5, 25, 100):
        p = P.Personality(kind)
        for e in long_history(kind, reps): p.experience(*e)
        top = np.argsort(-np.abs(p.v))[:3]
        print(f'  {kind:26s} reps={reps:3d} (events={len(long_history(kind,reps)):4d}) max|v|={np.abs(p.v).max():6.2f}  ' + ', '.join(f'{P.F[i]} {p.v[i]:+.2f}' for i in top))

print('\n=== (c) sign lock-in and content-permutation control (prototype) ===')
# Content-permutation control: for each event keep cue, but replace the outcome by the outcome of a random other event in the same history.
rng = np.random.default_rng(0)
def run_hist(events):
    p = P.Personality('x')
    for e in events: p.experience(*e)
    return p
for kind in P.KINDS:
    h = long_history(kind, 3)
    p0 = run_hist(h)
    diffs = []
    for s in range(20):
        perm = rng.permutation(len(h))
        h2 = [(h[i][0], h[i][1], h[perm[i]][2], None) for i in range(len(h))]
        diffs.append(run_hist(h2).v)
    D = np.array(diffs)
    print(f'  {kind:26s} ||v_true||={np.linalg.norm(p0.v):.2f}  mean ||v_perm||={np.mean(np.linalg.norm(D,axis=1)):.2f}  mean||v_true - v_perm||={np.mean(np.linalg.norm(D-p0.v,axis=1)):.2f}')

# Sign lock-in: first 5 events 'positive' vs 'negative' noise for an outcome-only feature, then 200 events of opposite evidence.
print('\nSign lock-in test (outcome-only feature "authorship", other features fixed): first n_early events with extra +/-0.6 of achievement-like push, then late evidence')
def lock(early_sign, late_sign, n_early=5, n_late=200):
    p = P.Personality('x', decay=0.002)
    # event: outcome has authorship=+1 only, plus an "ambient" feature push (achievement) whose sign differs early/late
    for t in range(n_early):
        p.experience(f'e{t}', P.vec(autonomy=1), P.vec(authorship=1, achievement=early_sign * 1.0), None)
    v_mid = p.v[P.IDX['authorship']]
    for t in range(n_late):
        p.experience(f'l{t}', P.vec(autonomy=1), P.vec(authorship=1, achievement=late_sign * 1.0), None)
    return v_mid, p.v[P.IDX['authorship']], p.v[P.IDX['achievement']]
for es, ls in ((+1, +1), (+1, -1), (-1, +1), (-1, -1)):
    vm, vf, va = lock(es, ls)
    print(f'  early achievement push {es:+d}, late {ls:+d}:  authorship after early {vm:+.2f} -> after late {vf:+.2f}   (achievement v {va:+.2f})')
