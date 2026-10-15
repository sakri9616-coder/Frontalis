"""v2 value-net toy (spec: learning-mechanisms/specs/v2-spec.md). Hand-set parameters, see PARAMS. Scratchpad only; NOT an experiment.
Concept features / histories come from the doc's prototype (learning-mechanisms/toys/value/prototype.py), reused unchanged."""
import sys, os, numpy as np, copy
sys.dont_write_bytecode = True
V = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'value'))
sys.path.insert(0, V)
import prototype as P
F, IDX, vec = P.F, P.IDX, P.vec
NF = len(F)
INN_MASK = np.zeros(NF); INN_MASK[[IDX[k] for k in ('achievement', 'loss', 'betrayal', 'novelty')]] = 1
NI = 1 - INN_MASK
PROBES = ['connection', 'authorship', 'autonomy', 'music', 'friendA', 'others']
PI = [IDX[k] for k in PROBES]
def zvec(cue, out):
    o = out * NI
    return np.where(o != 0, o, cue * NI)
NEEDS = ['relatedness', 'competence', 'autonomy', 'stimulation', 'safety']
K = 5
# fixed, persona-independent effect matrix M(o): outcome feature -> need change (designer's hand-set)
M = np.zeros((K, NF))
M[0, IDX['connection']] = 1; M[0, IDX['betrayal']] = -1
M[1, IDX['achievement']] = 1; M[1, IDX['loss']] = -1
M[2, IDX['autonomy']] = 1; M[2, IDX['authorship']] = 1
M[3, IDX['novelty']] = 1
M[4, IDX['betrayal']] = -.5; M[4, IDX['loss']] = -.5

# hand-set parameters (set before running; changes after seeing results are logged in v2-verify.md)
PARAMS = dict(gM=0.3, lam=0.10, h0=1.0, hmax=6.0, s=3.0, hstar_safe=1.0,
              R=0.004, q=1e-4, s0=0.25,                 # MF Kalman
              dB=0.05, RB=0.004, NM=40,                 # belief memory (window ~1/dB), belief obs-noise scale
              a_c=0.1, g=1.3, k_e=2.0, lam_c=1.0,       # commitment
              gamma_ec=0.5, a_ec=0.1, lam_ec=0.5)       # evaluative conditioning (one stage)

class V2:
    name = 'v2'
    def __init__(self, commit=True, ec=False, **kw):
        p = dict(PARAMS); p.update(kw); self.p = p; self.commit = commit; self.ec = ec
        self.h = np.full(K, p['h0']); self.W = np.zeros((NF, K)); self.S = np.array([np.eye(NF) * p['s0'] for _ in range(K)])
        self.c = np.zeros(NF); self.vEC = np.zeros(NF)
        self.Bw = np.zeros((NF, p['NM'])); self.BD = np.zeros((NF, p['NM'], K)); self.bptr = np.zeros(NF, int)
        self.last = {}
    # ---- fixed utility
    def u(self, h):                               # per-need utility, vector
        p = self.p; out = p['s'] * (1 - np.exp(-h / p['s']))
        out = np.array(out, float); out[..., 4] = -0.5 * np.maximum(p['hstar_safe'] - h[..., 4], 0) ** 2
        return out
    def U(self, h): return self.u(h).sum(-1)
    # ---- readouts
    def V_MF(self, x): return float(self.W[x].sum())
    def n_eff(self, x): return float(self.Bw[x].sum())
    def V_MB(self, x):
        w = self.Bw[x]
        if w.sum() <= 0: return 0.0
        d = self.U(self.h[None, :] + self.BD[x]) - self.U(self.h)
        return float((w * d).sum() / w.sum())
    def sig_mf(self, x): return float(np.mean([self.S[k][x, x] for k in range(K)]))
    def sig_mb(self, x): return self.p['RB'] / (self.n_eff(x) + 1.0)
    def omega(self, x):
        a, b = self.sig_mb(x), self.sig_mf(x); return a / (a + b)
    def U_val(self, x):
        w = self.omega(x); return w * self.V_MF(x) + (1 - w) * self.V_MB(x)
    def Q(self, x):
        p = self.p; return self.U_val(x) + (p['lam_c'] * self.c[x] if self.commit else 0.0) + (p['lam_ec'] * self.vEC[x] if self.ec else 0.0)
    def probe(self): return np.array([self.Q(i) for i in range(NF)])      # full concept vector (harness indexes it with PI)
    # ---- MF update (vector teacher r, concept vector z; per-need z allowed)
    def mf_update(self, z, r, zk=None):
        p = self.p; err = np.zeros(K); act = np.nonzero(z)[0]
        for k in range(K):
            zz = z if zk is None else zk[k]
            Sz = self.S[k] @ zz; den = zz @ Sz + p['R']; Kg = Sz / den
            e = r[k] - zz @ self.W[:, k]; self.W[:, k] += Kg * e
            self.S[k] = self.S[k] - np.outer(Kg, Sz); self.S[k] = 0.5 * (self.S[k] + self.S[k].T)
            for x in act: self.S[k][x, x] += p['q']
            err[k] = e
        return err
    def experience(self, ev, dh=None, r=None):
        p = self.p; _, cue, out, _ = ev; z = zvec(cue, out); act = np.nonzero(z)[0]
        Dh = p['gM'] * (M @ out) if dh is None else dh
        rk = (self.u(np.clip(self.h + Dh, 0, p['hmax'])) - self.u(self.h)) if r is None else r   # hedonic vector at event level
        if self.ec and len(act) > 1:
            Vd = {x: self.U_val(x) for x in act}                                              # direct (non-EC) values, before this event
        err = self.mf_update(z, rk)
        if self.commit:
            e_tot = err.sum()
            for x in act:
                self.c[x] += p['a_c'] * (-self.c[x] + p['g'] * np.tanh(self.c[x]) + p['k_e'] * z[x] * e_tot)
        if self.ec and len(act) > 1:
            for y in act:
                tgt = p['gamma_ec'] * np.mean([Vd[x] for x in act if x != y]); self.vEC[y] += p['a_ec'] * (tgt - self.vEC[y])
        for x in act:                                                                           # belief memory (event-level outcome, forgetting)
            self.Bw[x] *= (1 - p['dB']); i = self.bptr[x] % p['NM']; self.Bw[x, i] = 1.0; self.BD[x, i] = Dh; self.bptr[x] += 1
        self.h = np.clip(self.h + Dh, 0, p['hmax']); self.h = self.h + p['lam'] * (p['h0'] - self.h)
        return rk
    def idle(self, n):                                                                          # irrelevant updates: only the need state relaxes
        for _ in range(n): self.h = self.h + self.p['lam'] * (self.p['h0'] - self.h)
    # ---- state vector (learned, for Jacobians): W, S (upper tri), c, vEC
    def get_state(self):
        iu = np.triu_indices(NF)
        return np.concatenate([self.W.ravel(), np.concatenate([self.S[k][iu] for k in range(K)]), self.c, self.vEC])
    def set_state(self, x):
        iu = np.triu_indices(NF); n = len(iu[0]); a = NF * K
        self.W = x[:a].reshape(NF, K).copy()
        for k in range(K):
            s = np.zeros((NF, NF)); s[iu] = x[a + k * n: a + (k + 1) * n]; s = s + s.T - np.diag(np.diag(s)); self.S[k] = s
        b = a + K * n; self.c = x[b:b + NF].copy(); self.vEC = x[b + NF:b + 2 * NF].copy()
    def blocks(self):
        iu = np.triu_indices(NF); n = len(iu[0]); a = NF * K; b = a + K * n
        return dict(W=np.arange(0, a), S=np.arange(a, b), c=np.arange(b, b + NF), vEC=np.arange(b + NF, b + 2 * NF))

def run_persona(cls, kind):
    c = cls()
    for e in P.history(kind): c.experience(e)
    return c
