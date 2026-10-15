import sys, copy; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
pool = []
for k in P.KINDS: pool += list(P.history(k))
pool = [(f'p{i}', c_, o, b) for i, (_, c_, o, b) in enumerate(pool)] + [P.NEW] * 3
rng = np.random.default_rng(1); stream = [pool[i] for i in rng.integers(0, len(pool), size=128)]
print("open-loop: 1e-3 random perturbation of W (and c if commit) after 'credit-taken' history, 128 shared events; ||d state_final||/||d state_0|| and probe ratio (10 seeds)")
for nm, kw in (('core RB=.4', dict(commit=False, RB=0.4)), ('core RB=.4 q=0', dict(commit=False, RB=0.4, q=0.0)), ('+c lam_c=.1 RB=.4', dict(commit=True, lam_c=0.1, RB=0.4)), ('+c lam_c=1 RB=.4', dict(commit=True, lam_c=1.0, RB=0.4))):
    ra = []; pr = []
    for sd in range(10):
        r = np.random.default_rng(100 + sd); a = run_persona(lambda: V2(**kw), 'credit-taken'); b = copy.deepcopy(a)
        d = r.standard_normal(a.W.shape); d *= 1e-3 / np.linalg.norm(d); b.W = b.W + d; n0 = np.linalg.norm(d)
        if kw['commit']:
            dc = r.standard_normal(NF); dc *= 1e-3 / np.linalg.norm(dc); b.c = b.c + dc; n0 = np.sqrt(n0**2 + 1e-6)
        for ev in stream: a.experience(ev); b.experience(ev)
        ra.append(np.sqrt(np.linalg.norm(a.W - b.W)**2 + np.linalg.norm(a.c - b.c)**2) / n0); pr.append(np.linalg.norm(a.probe() - b.probe()) / n0)
    print(f"  {nm}: state ratio mean {np.mean(ra):.3f} max {np.max(ra):.3f}; probe ratio mean {np.mean(pr):.3f}")
