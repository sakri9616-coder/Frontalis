"""Sim1: toy implementations of candidate value-learning rules on the prototype's 3 histories and its NEW event.
Scratchpad only. NOT an experiment: parameters are hand-set to make mechanisms visible; no tuning toward a target pattern.
Candidates implemented here: A (existing, prototype.py), B (TD, linear signed features), C (homeostatic + set point),
E (model-based -> model-free consolidation), I (Kalman/Bayesian TD).  D, F, G, H are in sim2_custom.py.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prototype as P
F, IDX, vec, INNATE = P.F, P.IDX, P.vec, P.INNATE
NF = len(F)
INN_MASK = np.zeros(NF); INN_MASK[[IDX[k] for k in ('achievement', 'loss', 'betrayal', 'novelty')]] = 1
NI = 1 - INN_MASK           # non-innate-bearing features ("states/means")
PROBES = ['connection', 'authorship', 'autonomy', 'music', 'friendA', 'others']
PI = [IDX[k] for k in PROBES]

def r_innate(out): return float(INNATE @ out)
def zvec(cue, out):
    o = out * NI
    return np.where(o != 0, o, cue * NI)

# ---------------------------------------------------------------- A
class A_existing:
    name = 'A'
    def __init__(self): self.p = P.Personality('A')
    def experience(self, ev):
        return self.p.experience(*ev)
    def probe(self): return self.p.v.copy()

# ---------------------------------------------------------------- B: TD on signed features
class B_TD:
    name = 'B'
    def __init__(self, gamma=0.9, alpha=0.1):
        self.w = np.zeros(NF); self.g, self.a = gamma, alpha
    def experience(self, ev):
        _, cue, out, _ = ev
        phi0, phi1 = cue, out * NI
        r = r_innate(out)
        d0 = self.g * self.w @ phi1 - self.w @ phi0; self.w += self.a * d0 * phi0
        d1 = r - self.w @ phi1; self.w += self.a * d1 * phi1
        return dict(d0=d0, d1=d1, V1=float(self.w @ phi1))
    def probe(self): return self.w.copy()

# ---------------------------------------------------------------- I: Kalman TD
class I_KTD:
    name = 'I'
    def __init__(self, gamma=0.9, s0=1.0, obs=0.25, q=0.002):
        self.mu = np.zeros(NF); self.S = np.eye(NF) * s0; self.g, self.o, self.q = gamma, obs, q
    def _upd(self, phi, y):
        Sphi = self.S @ phi; k = Sphi / (phi @ Sphi + self.o)
        self.mu = self.mu + k * (y - self.mu @ phi); self.S = self.S - np.outer(k, Sphi)
    def experience(self, ev):
        _, cue, out, _ = ev
        phi0, phi1 = cue, out * NI
        self._upd(phi0, self.g * self.mu @ phi1)
        self._upd(phi1, r_innate(out))
        self.S = self.S + self.q * np.eye(NF)
    def probe(self): return self.mu.copy()
    def sd(self): return np.sqrt(np.diag(self.S))

# ---------------------------------------------------------------- C: homeostatic with set points
M = np.zeros((4, NF))   # needs: relatedness, competence, autonomy, stimulation
M[0, IDX['connection']] = 1; M[0, IDX['betrayal']] = -1
M[1, IDX['achievement']] = 1; M[1, IDX['loss']] = -1
M[2, IDX['autonomy']] = 1; M[2, IDX['authorship']] = 1
M[3, IDX['novelty']] = 1
NEEDS = ['relatedness', 'competence', 'autonomy', 'stimulation']
class C_Homeo:
    name = 'C'
    def __init__(self, g=0.06, lam=0.05, eta=0.02, rho=0.005, au=0.1, h0=1.0, adapt=True):
        self.h = np.full(4, h0); self.hs0 = np.full(4, h0); self.hs = self.hs0.copy()
        self.u = np.zeros((NF, 4)); self.g, self.lam, self.eta, self.rho, self.au, self.adapt = g, lam, eta, rho, au, adapt
    @staticmethod
    def D(h, hs): return 0.5 * float(np.sum((hs - h) ** 2))
    def experience(self, ev):
        _, cue, out, _ = ev
        dh = self.g * (M @ out)
        felt = self.D(self.h, self.hs) - self.D(self.h + dh, self.hs)       # drive reduction
        z = zvec(cue, out); err = dh - z @ self.u; self.u += self.au * np.outer(z, err)
        self.h = np.clip(self.h + dh, 0, 2)
        if self.adapt: self.hs = self.hs + self.eta * (self.h - self.hs) + self.rho * (self.hs0 - self.hs)
        self.h = self.h * (1 - self.lam)
        return dict(felt=felt, dh=dh)
    def value_of(self, x):          # value of meeting concept x at unit level in the current state
        return self.D(self.h, self.hs) - self.D(self.h + self.u[x], self.hs)
    def probe(self): return np.array([self.value_of(i) for i in range(NF)])

# ---------------------------------------------------------------- E: MB -> MF consolidation
class E_MBMF:
    name = 'E'
    def __init__(self, ab=0.2, nh=10.0, amin=0.02):
        self.beta = np.zeros(NF); self.vmf = np.zeros(NF); self.n = np.zeros(NF); self.ab, self.nh, self.amin = ab, nh, amin
    def experience(self, ev):
        _, cue, out, _ = ev
        z = zvec(cue, out); r = r_innate(out)
        self.beta += self.ab * (r - self.beta @ z) * z                       # belief about means-ends (fast)
        for k in np.nonzero(z)[0]:
            self.n[k] += 1; a = max(1.0 / (self.n[k] + 1), self.amin)
            self.vmf[k] += a * (self.beta[k] - self.vmf[k])                  # slow cache distilled from MB value
    def omega(self): return self.n / (self.n + self.nh)
    def probe(self):
        w = self.omega(); return w * self.vmf + (1 - w) * self.beta

CANDS = [A_existing, B_TD, C_Homeo, E_MBMF, I_KTD]

def run_persona(cls, kind):
    c = cls()
    for e in P.history(kind): c.experience(e)
    return c

def table(title, fn):
    print('\n' + title)
    print('%-14s' % '' + ''.join('%11s' % k[:10] for k in PROBES))
    for kind in P.KINDS: print('%-14s' % kind[:14] + ''.join('%11.3f' % x for x in fn(kind)))

if __name__ == '__main__':
    NEW = P.NEW
    print('==== Persona probe vectors after history, then after the NEW event (value of concept at unit level) ====')
    res = {}
    for cls in CANDS:
        print('\n##### Candidate', cls.name)
        before, after = {}, {}
        for kind in P.KINDS:
            c = run_persona(cls, kind); before[kind] = c.probe()[PI]
            extra = c.experience(NEW); after[kind] = c.probe()[PI]
            res[(cls.name, kind)] = (before[kind], after[kind], extra, c)
        print('%-26s' % '' + ''.join('%11s' % k[:10] for k in PROBES))
        for kind in P.KINDS:
            print('%-26s' % (kind + ' before') + ''.join('%11.3f' % x for x in before[kind]))
            print('%-26s' % (kind + ' after') + ''.join('%11.3f' % x for x in after[kind]))
            print('%-26s' % (kind + ' DELTA') + ''.join('%+11.3f' % x for x in (after[kind] - before[kind])))
        if cls is B_TD:
            for kind in P.KINDS:
                ex = res[('B', kind)][2]; print('   B', kind, 'd0=%.3f d1=%.3f V(s1)=%.3f (TD error of the NEW event second step = surprise)' % (ex['d0'], ex['d1'], ex['V1']))
        if cls is C_Homeo:
            for kind in P.KINDS:
                c = res[('C', kind)][3]; ex = res[('C', kind)][2]
                print('   C', kind, 'felt(drive reduction)=%+.3f' % ex['felt'], 'h=', np.round(c.h, 2), 'set point h*=', np.round(c.hs, 3))
        if cls is A_existing:
            for kind in P.KINDS:
                ex = res[('A', kind)][2]; print('   A', kind, 'felt %+.3f expected %+.3f delta %+.3f' % (ex['felt'], ex['expected'], ex['delta']))
        if cls is I_KTD:
            for kind in P.KINDS:
                c = res[('I', kind)][3]; print('   I', kind, 'sd(friendA) %.3f sd(authorship) %.3f' % (c.sd()[IDX['friendA']], c.sd()[IDX['authorship']]))
        if cls is E_MBMF:
            for kind in P.KINDS:
                c = res[('E', kind)][3]; i = IDX['authorship']
                print('   E', kind, 'authorship: belief(MB) %+.3f cache(MF) %+.3f omega %.2f n=%d' % (c.beta[i], c.vmf[i], c.omega()[i], c.n[i]))

    # --------- continuity / forgetting test: persona differences after n shared new events
    print('\n==== Retention of persona differences after n shared new events (mean pairwise L2 distance of probe vectors, relative to n=0) ====')
    pool = []
    for kind in P.KINDS: pool += [e for e in P.history(kind)]
    pool = [(f'pool{i}', c, o, b) for i, (_, c, o, b) in enumerate(pool)] + [NEW] * 3
    rng = np.random.default_rng(1)
    stream = [pool[i] for i in rng.integers(0, len(pool), size=128)]
    def dist(cs):
        ps = [c.probe()[PI] for c in cs]
        return float(np.mean([np.linalg.norm(ps[i] - ps[j]) for i in range(3) for j in range(i + 1, 3)]))
    marks = (0, 1, 4, 8, 16, 32, 64, 128)
    print('%-4s' % '' + ''.join('%8d' % m for m in marks))
    for cls in CANDS:
        cs = [run_persona(cls, k) for k in P.KINDS]; d0 = dist(cs); row = [1.0]; n = 0
        for m in marks[1:]:
            while n < m:
                for c in cs: c.experience(stream[n])
                n += 1
            row.append(dist(cs) / d0)
        print('%-4s' % cls.name + ''.join('%8.2f' % x for x in row), '  (d0=%.2f)' % d0)

    # --------- irrelevant-experience leak test: 200 events on an unrelated concept (here: reuse 'others' cue only w/ no outcome)
    print('\n==== Irrelevant-experience test: 200 null events (cue=music, outcome=music only, no innate-bearing features) ====')
    null = ('null', vec(music=1), vec(music=1), None)
    for cls in CANDS:
        for kind in ('credit-taken',):
            c = run_persona(cls, kind); p0 = c.probe()[PI].copy()
            for i in range(200): c.experience(null)
            p1 = c.probe()[PI]
            excl = [j for j, k in enumerate(PROBES) if k != 'music']
            print(f'  {cls.name} {kind}: ||probe(excluding music)|| {np.linalg.norm(p0[excl]):.3f} -> {np.linalg.norm(p1[excl]):.3f}  (ratio {np.linalg.norm(p1[excl])/max(np.linalg.norm(p0[excl]),1e-9):.2f});  music {p0[3]:+.3f} -> {p1[3]:+.3f}')

    # --------- reversal (plasticity) test: persona 'credit-taken' receives n events of the 'rescued-by-collaboration' history (opposite experience)
    print('\n==== Reversal test: credit-taken persona then receives n events drawn (cyclically) from the rescued history; fraction of the initial probe gap to the *rescued persona own end state* that is closed ====')
    marks = (0, 4, 8, 16, 32, 64, 128)
    print('%-4s' % '' + ''.join('%8d' % m for m in marks))
    for cls in CANDS:
        k_ = run_persona(cls, 'credit-taken'); r_ = run_persona(cls, 'rescued-by-collaboration')
        target = r_.probe()[PI]; d0 = np.linalg.norm(k_.probe()[PI] - target); hist = P.history('rescued-by-collaboration'); n = 0; row = []
        for m in marks:
            while n < m:
                k_.experience(hist[n % len(hist)]); n += 1
            row.append(1 - np.linalg.norm(k_.probe()[PI] - target) / d0)
        print('%-4s' % cls.name + ''.join('%8.2f' % x for x in row), '  (d0=%.2f)' % d0)
