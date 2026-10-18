"""V1: persistence (difference on an UNSEEN individual of the same class) vs plasticity (the new individual N's own value), hierarchical value vs baselines.
Utility scale: credit-taking collaborator r=-0.30, ordinary collaborator r=+0.20 (sd 0.06 = sqrt(R)). History: P1 = 8 events with 'taking' collaborators, P2 = 8 events with ordinary ones,
split over n_ind individuals. Then new individual N (true r=+0.20) for m events. 200 seeds."""
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v3model import *
SD = 0.06
def hist_events(mu, n_ind, per, rng): return [(f'h{i}', mu + SD * rng.standard_normal()) for i in range(n_ind) for _ in range(per)]
class Hier:
    name = 'v3 hierarchical'
    def __init__(self, **kw): self.hk = HK(Kout=1, **kw); self.hk.feat('gen', 'gen'); self.hk.feat('cls', 'cls')
    def obs(self, ind, r):
        self.hk.feat(ind, 'ind'); self.hk.update({'gen': 1.0, 'cls': 1.0, ind: 1.0}, np.array([r]))
    def unseen(self): return self.hk.val(['gen', 'cls'])
    def value(self, ind): return self.hk.val(['gen', 'cls', ind])
class Flat:
    """v2 core: one class weight (individuals not represented), Kalman with fixed q"""
    name = 'v2 core (no hierarchy)'
    def __init__(self, q=1e-4, **kw): self.hk = HK(Kout=1, qmode='fixed', **kw); self.hk.p['q0'] = dict(ind=q, cls=q, gen=q); self.hk.feat('cls', 'cls'); self.hk.S[0, 0, 0] = 0.25; self.hk.q[0, 0] = q
    def obs(self, ind, r): self.hk.update({'cls': 1.0}, np.array([r]))
    def unseen(self): return self.hk.val(['cls'])
    def value(self, ind): return self.hk.val(['cls'])
class FixedRate:
    name = 'fixed rate a=0.25'
    def __init__(self, a=0.25): self.w = 0.0; self.a = a
    def obs(self, ind, r): self.w += self.a * (r - self.w)
    def unseen(self): return self.w
    def value(self, ind): return self.w
class Count:
    name = '1/n count'
    def __init__(self): self.n = 0; self.w = 0.0
    def obs(self, ind, r): self.n += 1; self.w += (r - self.w) / self.n
    def unseen(self): return self.w
    def value(self, ind): return self.w
def trial(mk, n_ind, per, ms=(8, 32), seed=0):
    rng = np.random.default_rng(seed); out = {}
    ag = {}
    for nm, mu in (('P1', -0.30), ('P2', +0.20)):
        a = mk(); [a.obs(i, r) for i, r in hist_events(mu, n_ind, per, rng)]; ag[nm] = a
    d0 = ag['P1'].unseen() - ag['P2'].unseen(); before = {k: ag[k].unseen() for k in ag}
    done = 0
    for m in ms:
        while done < m:
            for k in ag: ag[k].obs('N', 0.20 + SD * rng.standard_normal())
            done += 1
        d = ag['P1'].unseen() - ag['P2'].unseen()
        pers = d / d0
        plas = np.mean([(ag[k].value('N') - before[k]) / (0.20 - before[k]) for k in ('P1',)])      # fraction of the gap closed for P1 (the persona with the wrong prior)
        out[m] = (pers, plas)
    return out
def table(mk, n_ind=8, per=1, N=200):
    res = [trial(mk, n_ind, per, seed=s) for s in range(N)]
    return {m: (np.mean([r[m][0] for r in res]), np.mean([r[m][1] for r in res])) for m in (8, 32)}
print("V1a persistence P (difference on an unseen individual, after N-experience / before) and plasticity L (gap closed for N's own value, P1), history = 8 individuals x 1 event")
print("| model | P m=8 | L m=8 | P m=32 | L m=32 | both >=0.7 (m=8,32) |\n|---|---|---|---|---|---|")
models = [('v3 hier, q fixed (q0 by layer)', lambda: Hier(qmode='fixed')), ('v3 hier, q learned (spec literal)', lambda: Hier(qmode='literal')), ('v3 hier, q learned (EMA, eta=.01)', lambda: Hier(qmode='ema', eta_q=0.01)),
          ('v2 core (no hierarchy)', lambda: Flat()), ('fixed rate 0.25', FixedRate), ('1/n count', Count)]
for nm, mk in models:
    t = table(mk); ok = all(v >= 0.7 for m in t for v in t[m])
    print(f"| {nm} | {t[8][0]:.2f} | {t[8][1]:.2f} | {t[32][0]:.2f} | {t[32][1]:.2f} | {'PASS' if ok else 'FAIL'} |")
print("\nV1b retention (persistence) vs the number of individuals that produced the history (8 events in total), v3 hier q fixed / literal / EMA, and 1/n count; m=8 and m=32")
print("| n individuals x events | v3 fixed m=8 | m=32 | v3 literal m=8 | m=32 | v3 EMA m=8 | m=32 | 1/n m=8 |\n|---|---|---|---|---|---|---|---|")
for n_ind, per in ((1, 8), (2, 4), (4, 2), (8, 1)):
    row = []
    for mk in (lambda: Hier(qmode='fixed'), lambda: Hier(qmode='literal'), lambda: Hier(qmode='ema', eta_q=0.01)):
        t = table(mk, n_ind, per, N=100); row += [t[8][0], t[32][0]]
    t = table(Count, n_ind, per, N=20)
    print(f"| {n_ind}x{per} | " + " | ".join(f"{v:.2f}" for v in row) + f" | {t[8][0]:.2f} |")
