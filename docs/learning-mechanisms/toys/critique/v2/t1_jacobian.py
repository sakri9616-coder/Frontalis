"""T1: unrolled one-step Jacobian of the learned state (W, S, c, vEC) over the R/K/X histories + NEW (same measure as c10), and 1e-3 perturbation sensitivity."""
import sys, copy; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
np.set_printoptions(precision=3, suppress=True)
def jac(c0, ev, h=1e-6):
    x0 = c0.get_state(); n = len(x0)
    def step(x):
        c = copy.deepcopy(c0); c.set_state(x); c.experience(ev); return c.get_state()
    f0 = step(x0); J = np.zeros((n, n))
    for i in range(n):
        x = x0.copy(); x[i] += h; J[:, i] = (step(x) - f0) / h
    return J
def radius(J, idx): return max(abs(np.linalg.eigvals(J[np.ix_(idx, idx)])))
cfgs = {'v2 core (no c, no EC)': dict(commit=False, ec=False), 'v2 + c': dict(commit=True, ec=False), 'v2 + c + EC': dict(commit=True, ec=True)}
for nm, kw in cfgs.items():
    rows = {k: [] for k in ('full', 'no_c', 'W', 'S')}; over = 0; tot = 0; worst = None
    for kind in P.KINDS:
        c0 = run_persona(lambda: V2(**kw), kind); b = c0.blocks()
        allidx = np.arange(len(c0.get_state())); noc = np.setdiff1d(allidx, b['c'])
        for ev in P.history(kind) + [P.NEW]:
            J = jac(c0, ev)
            rf = radius(J, allidx); rn = radius(J, noc); rows['full'].append(rf); rows['no_c'].append(rn); rows['W'].append(radius(J, b['W'])); rows['S'].append(radius(J, b['S']))
            tot += 1; over += rn > 1.0001
            if worst is None or rn > worst[0]: worst = (rn, kind, ev[0])
    print(f"{nm}: events={tot}  max radius full={max(rows['full']):.4f}  excluding c-block={max(rows['no_c']):.6f}  W-block={max(rows['W']):.6f}  S-block={max(rows['S']):.6f}; events with radius(no c)>1.0001: {over}/{tot}")
# learning phase from scratch (credit-taken history)
c = V2(commit=True, ec=False); rs = []; rf = []
for ev in P.history('credit-taken'):
    J = jac(c, ev); b = c.blocks(); allidx = np.arange(J.shape[0]); noc = np.setdiff1d(allidx, b['c']); rs.append(radius(J, noc)); rf.append(radius(J, allidx)); c.experience(ev)
print("learning phase (credit-taken from scratch), radius excl. c:", np.round(rs[:12], 4), " full:", np.round(rf[:12], 3))
# open-loop 1e-3 perturbation: perturb W (random direction, norm 1e-3) after history, run the shared 128-event stream, final probe/W difference
pool = []
for k in P.KINDS: pool += list(P.history(k))
pool = [(f'p{i}', c_, o, b) for i, (_, c_, o, b) in enumerate(pool)] + [P.NEW] * 3
rng = np.random.default_rng(1); stream = [pool[i] for i in rng.integers(0, len(pool), size=128)]
for nm, kw in cfgs.items():
    ratios = []; pr = []
    for sd in range(10):
        r = np.random.default_rng(100 + sd)
        a = run_persona(lambda: V2(**kw), 'credit-taken'); b = copy.deepcopy(a)
        d = r.standard_normal(a.W.shape); d *= 1e-3 / np.linalg.norm(d); b.W = b.W + d
        if kw['commit']: dc = r.standard_normal(NF); b.c = b.c + 1e-3 * dc / np.linalg.norm(dc)
        n0 = np.sqrt(np.linalg.norm(d) ** 2 + (1e-3 ** 2 if kw['commit'] else 0))
        for ev in stream: a.experience(ev); b.experience(ev)
        dW = np.linalg.norm(a.W - b.W); dc_ = np.linalg.norm(a.c - b.c)
        ratios.append(np.sqrt(dW ** 2 + dc_ ** 2) / n0); pr.append(np.linalg.norm(a.probe() - b.probe()) / n0)
    print(f"open-loop 1e-3 perturbation, 128 shared events, {nm}: ||dstate_final||/||dstate_0|| = {np.mean(ratios):.3f} (max {np.max(ratios):.3f}); probe ratio {np.mean(pr):.3f}")
