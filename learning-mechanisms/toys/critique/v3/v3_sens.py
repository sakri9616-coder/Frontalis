import sys, copy; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v3model import *
pool = []
for k in P.KINDS: pool += list(P.history(k))
pool = [(f'p{i}', c_, o, b) for i, (_, c_, o, b) in enumerate(pool)] + [P.NEW] * 3
rng0 = np.random.default_rng(1); stream = [pool[i] for i in rng0.integers(0, len(pool), size=128)]
print("open-loop 1e-3 perturbation of W after the K history, 128 shared events (fresh individual for each 'others' event); ratio ||dW_final||/||dW_0|| over the weights present at the start; unseen-collaborator value ratio. 10 seeds.")
print("| q rule | weight-space ratio (mean/max) | unseen-collaborator value ratio |\n|---|---|---|")
for nm, kw in (('fixed', dict(qmode='fixed')), ('literal (eta=.05)', dict(qmode='literal')), ('EMA (eta=.01)', dict(qmode='ema', eta_q=0.01))):
    ra = []; rv = []
    for sd in range(10):
        r = np.random.default_rng(100 + sd); a = run_persona('credit-taken', **kw); b = copy.deepcopy(a); n0 = a.hk.n
        d = r.standard_normal((n0, K)); d *= 1e-3 / np.linalg.norm(d); b.hk.W[:n0] += d
        for ev in stream: a.experience(ev); b.experience(ev)
        ra.append(np.linalg.norm(a.hk.W[:n0] - b.hk.W[:n0]) / 1e-3); rv.append(abs(a.inherited() - b.inherited()) / 1e-3)
    print(f"| {nm} | {np.mean(ra):.3f} / {np.max(ra):.3f} | {np.mean(rv):.3f} |")
