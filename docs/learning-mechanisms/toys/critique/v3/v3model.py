"""v3 value-net toy (spec: learning-mechanisms/specs/v3-spec.md): hierarchical MF (individual / class / general) with Kalman credit assignment and learned per-weight q.
Reuses utility, need effects M, need-state dynamics and the doc's R/K/X histories from learning-mechanisms/toys/critique/v2/v2model.py. Hand-set parameters in P3. Scratchpad only."""
import sys, numpy as np
sys.dont_write_bytecode = True
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'v2'))
import v2model as V2M
from v2model import P, F, IDX, vec, zvec, NF, K, M, PARAMS
KIND = {'ind': 0, 'cls': 1, 'gen': 2}
P3 = dict(var=dict(ind=0.25, cls=0.04, gen=0.01),            # prior variance by layer (hand-set)
          q0=dict(ind=1e-4, cls=2e-5, gen=5e-6),             # initial process noise by layer (q_ind > q_cls > q_gen)
          R=0.004, eta_q=0.05, qmode='literal',              # q learning: 'literal' (spec) / 'ema' (Sage-Husa, clipped at 0) / 'fixed'
          fmax=220)

def kal_step(W, S, q, z, r, R, eta, mode):
    """One Kalman regression step for ONE output. W:(F,), S:(F,F), q:(F,), z:(F,), r scalar. Returns W', S', q', innovation, predictive variance."""
    Sz = S @ z; den = float(z @ Sz) + R; Kg = Sz / den; e = float(r - z @ W)
    W2 = W + Kg * e
    S2 = S - np.outer(Kg, Sz); S2 = 0.5 * (S2 + S2.T)
    act = np.nonzero(z)[0]; q2 = q.copy()
    if mode == 'literal':   q2[act] = q[act] + eta * (max(0.0, e * e - den) - q[act])
    elif mode == 'ema':     q2[act] = np.maximum(0.0, q[act] + eta * (e * e - den - q[act]))
    S2[act, act] += q2[act]
    return W2, S2, q2, e, den

class HK:
    """Hierarchical Kalman regression for K independent outputs with dynamically added features."""
    def __init__(self, Kout=K, **kw):
        p = dict(P3); p.update(kw); self.p = p; self.K = Kout; fm = p['fmax']
        self.name = []; self.layer = []; self.idx = {}
        self.W = np.zeros((fm, Kout)); self.S = np.zeros((Kout, fm, fm)); self.q = np.zeros((Kout, fm))
    def feat(self, name, layer):
        if name in self.idx: return self.idx[name]
        i = len(self.name); self.name.append(name); self.layer.append(layer); self.idx[name] = i
        for k in range(self.K): self.S[k, i, i] = self.p['var'][layer]; self.q[k, i] = self.p['q0'][layer]
        return i
    @property
    def n(self): return len(self.name)
    def update(self, zdict, r):
        """zdict: {feature name: value}; r: (K,) teacher vector. Features must already exist."""
        n = self.n; z = np.zeros(n)
        for nm, v in zdict.items(): z[self.idx[nm]] = v
        err = np.zeros(self.K); den = np.zeros(self.K)
        for k in range(self.K):
            W2, S2, q2, e, d = kal_step(self.W[:n, k], self.S[k, :n, :n], self.q[k, :n], z, r[k], self.p['R'], self.p['eta_q'], self.p['qmode'])
            self.W[:n, k] = W2; self.S[k, :n, :n] = S2; self.q[k, :n] = q2; err[k] = e; den[k] = d
        return err
    def val(self, names):
        """sum over listed features of the weights, summed over needs."""
        return float(sum(self.W[self.idx[n]].sum() for n in names if n in self.idx))

CONCEPT_LAYER = {'connection': 'cls', 'authorship': 'cls', 'autonomy': 'cls', 'music': 'cls', 'others': 'cls', 'friendA': 'ind'}
class V3Agent:
    """Doc harness agent: R/K/X histories, utility/needs as in v2. 'others' events get a fresh anonymous individual (declared modelling choice)."""
    def __init__(self, **kw):
        self.hk = HK(**kw); self.vm = V2M.V2(commit=False); self.nanon = 0; self.log = []
    def zdict(self, ev, fresh=True):
        _, cue, out, _ = ev; z = zvec(cue, out); d = {}
        for x in np.nonzero(z)[0]:
            nm = F[x]; d[nm] = float(z[x]); self.hk.feat(nm, CONCEPT_LAYER[nm])
        social = ('others' in d) or ('friendA' in d)
        if social:
            if 'others' not in d: d['others'] = 1.0; self.hk.feat('others', 'cls')
            if 'friendA' not in d and fresh:
                nm = f'o{self.nanon}'; self.nanon += 1; d[nm] = 1.0; self.hk.feat(nm, 'ind')
            d['people'] = 1.0; self.hk.feat('people', 'gen')
        return d
    def experience(self, ev):
        vm = self.vm; p = vm.p; _, cue, out, _ = ev
        Dh = p['gM'] * (M @ out); r = vm.u(np.clip(vm.h + Dh, 0, p['hmax'])) - vm.u(vm.h)
        d = self.zdict(ev); err = self.hk.update(d, r)
        vm.h = np.clip(vm.h + Dh, 0, p['hmax']); vm.h = vm.h + p['lam'] * (p['h0'] - vm.h)
        return r, err
    def inherited(self): return self.hk.val(['others', 'people'])                 # value of an unseen collaborator
    def value_of(self, ind): return self.hk.val(['others', 'people', ind])
def run_persona(kind, **kw):
    a = V3Agent(**kw)
    for e in P.history(kind): a.experience(e)
    return a
