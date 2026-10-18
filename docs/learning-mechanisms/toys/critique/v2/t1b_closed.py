"""T1b: closed-loop (value-guided choice) sensitivity to a 1e-3 perturbation with common random numbers; commit ON/OFF; RB sweep (arbitration scale)."""
import sys, copy; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
np.set_printoptions(precision=3, suppress=True)
OPTS = {
 'collab':  ('collab', vec(connection=1, friendA=1, music=1), vec(connection=1, friendA=1, music=1, achievement=.8, authorship=-.8, novelty=.6), None),
 'solo':    ('solo', vec(autonomy=1, music=1), vec(autonomy=1, authorship=1, achievement=.9, music=1), None),
 'explore': ('explore', vec(autonomy=1), vec(autonomy=1, novelty=1, achievement=.3), None)}
names = list(OPTS)
def Qopt(c, ev):
    act = np.nonzero(zvec(ev[1], ev[2]))[0]; return float(np.mean([c.Q(x) for x in act]))
def run(c, uu, beta, T=150):
    seq = []
    for t in range(T):
        q = np.array([Qopt(c, OPTS[n]) for n in names]); p = np.exp(beta * (q - q.max())); p /= p.sum()
        i = int(np.searchsorted(np.cumsum(p), uu[t])); i = min(i, 2); seq.append(i)
        c.experience(OPTS[names[i]]); c.idle(2)
    return seq
for label, kw in (('v2 core', dict(commit=False)), ('v2 + c', dict(commit=True)), ('v2 core RB=0.1', dict(commit=False, RB=0.1)), ('v2 + c, RB=0.1', dict(commit=True, RB=0.1))):
    for beta in (10.0, 30.0):
        div = 0; ratios = []; nseeds = 40
        for sd in range(nseeds):
            rg = np.random.default_rng(sd); uu = rg.random(150)
            a = run_persona(lambda: V2(**kw), 'credit-taken'); b = copy.deepcopy(a)
            d = rg.standard_normal(a.W.shape); d *= 1e-3 / np.linalg.norm(d); b.W = b.W + d
            sa = run(a, uu, beta); sb = run(b, uu, beta)
            same = sa == sb; div += (not same)
            ratios.append((same, np.linalg.norm(a.W - b.W) / 1e-3, np.linalg.norm(a.probe() - b.probe()) / 1e-3))
        rs = [r[1] for r in ratios if r[0]]; ps = [r[2] for r in ratios if r[0]]; rd = [r[1] for r in ratios if not r[0]]
        print(f"{label:16s} beta={beta:4.0f}: runs whose choice sequence diverged {div}/{nseeds}; if not diverged ||dW||/1e-3 mean={np.mean(rs) if rs else float('nan'):.2f}, probe ratio {np.mean(ps) if ps else float('nan'):.2f}; if diverged ||dW||={np.mean(rd)*1e-3 if rd else float('nan'):.3f}")
