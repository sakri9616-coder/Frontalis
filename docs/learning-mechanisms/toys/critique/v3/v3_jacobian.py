"""V3: one-step Jacobian spectral radius of the learned state (W, Sigma, q) per need (needs are decoupled), over R/K/X histories + NEW; and 1e-3 perturbation sensitivity."""
import sys, copy; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v3model import *
def compact(a, ev):
    """returns n, per-need (W,S,q), z, r for the event (features created on a copy)."""
    b = copy.deepcopy(a); vm = b.vm; p = vm.p; _, cue, out, _ = ev
    Dh = p['gM'] * (M @ out); r = vm.u(np.clip(vm.h + Dh, 0, p['hmax'])) - vm.u(vm.h)
    d = b.zdict(ev); n = b.hk.n; z = np.zeros(n)
    for nm, v in d.items(): z[b.hk.idx[nm]] = v
    return b, n, z, r
def radius_need(b, n, z, r, k, h=1e-6):
    hk = b.hk; iu = np.triu_indices(n); m = len(iu[0]); R, eta, mode = hk.p['R'], hk.p['eta_q'], hk.p['qmode']
    x0 = np.concatenate([hk.W[:n, k], hk.S[k, :n, :n][iu], hk.q[k, :n]])
    def unpack(x):
        W = x[:n]; s = np.zeros((n, n)); s[iu] = x[n:n + m]; s = s + s.T - np.diag(np.diag(s)); return W, s, x[n + m:]
    def step(x):
        W, S, q = unpack(x); W2, S2, q2, e, d = kal_step(W, S, q, z, r[k], R, eta, mode); return np.concatenate([W2, S2[iu], q2])
    f0 = step(x0); N = len(x0); J = np.zeros((N, N))
    for i in range(N):
        x = x0.copy(); x[i] += h; J[:, i] = (step(x) - f0) / h
    ev = np.linalg.eigvals(J); return max(abs(ev)), J
res = {}
for mode in ('fixed', 'literal', 'ema'):
    kw = dict(qmode=mode) if mode != 'ema' else dict(qmode='ema', eta_q=0.01)
    worst = (0, None); cnt = 0; tot = 0; mx = 0
    for kind in P.KINDS:
        a = V3Agent(**kw)
        for ev in P.history(kind) + [P.NEW]:
            b, n, z, r = compact(a, ev)
            for k in range(K):
                rad, J = radius_need(b, n, z, r, k); tot += 1; cnt += rad > 1.0001
                if rad > worst[0]: worst = (rad, (kind, ev[0], NEEDS_k := k))
            a.experience(ev)
    res[mode] = (worst, cnt, tot); print(f"q mode={mode}: max radius {worst[0]:.4f} at {worst[1]}; (event, need) pairs with radius>1.0001: {cnt}/{tot}")
