"""V1 extras: (a) class-level reversal by distinct counter-evidence individuals, (b) prior-variance sensitivity (coefficients), (c) LLM class errors."""
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v3model import *
exec(open(os.path.join(HERE, 'v1_persist.py')).read().split("print(\"V1a")[0])      # reuse Hier, Flat, hist_events, SD, trial, table
print("V1c reversal at the CLASS level: P1 (history: 8 collaborators who took credit) meets k NEW distinct collaborators (true r=+0.20); fraction of the class gap closed on an unseen collaborator")
print("| k distinct individuals x events each | v3 fixed q | v3 q literal | v2 core |\n|---|---|---|---|")
def reversal(mk, k, per, seed):
    rng = np.random.default_rng(seed); a = mk(); [a.obs(i, r) for i, r in hist_events(-0.30, 8, 1, rng)]; w0 = a.unseen()
    for j in range(k):
        for _ in range(per): a.obs(f'n{j}', 0.20 + SD * rng.standard_normal())
    return (a.unseen() - w0) / (0.20 - w0)
for k, per in ((1, 8), (2, 4), (4, 2), (8, 1), (16, 1), (32, 1), (8, 8), (32, 8)):
    row = [np.mean([reversal(mk, k, per, s) for s in range(60)]) for mk in (lambda: Hier(qmode='fixed'), lambda: Hier(qmode='literal'), Flat)]
    print(f"| {k}x{per} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
print("\nV1d coefficient sensitivity (prior variances; q fixed), history 8x1, m=8: persistence P / plasticity L")
print("| var_ind \\ var_cls | 0.01 | 0.04 | 0.16 |\n|---|---|---|---|")
for vi in (0.06, 0.25, 1.0):
    row = []
    for vc in (0.01, 0.04, 0.16):
        t = table(lambda: Hier(qmode='fixed', var=dict(ind=vi, cls=vc, gen=0.01)), 8, 1, N=60); row.append(f"{t[8][0]:.2f} / {t[8][1]:.2f}")
    print(f"| {vi} | " + " | ".join(row) + " |")
print("\nV1e LLM class errors. True classes: 'taker' individuals (r=-0.30) and 'friend' individuals (r=+0.20). Persona history: 8 takers + 8 friends, each 1 event; the LLM assigns the wrong class with prob p.")
print("Unseen new TAKER (true -0.30) and unseen new FRIEND (+0.20) are assigned their (possibly wrong) class; inherited prior vs truth; 200 seeds. Also over-split (every individual its own class) and over-merge (one class for all).")
class H2:
    def __init__(self, **kw): self.hk = HK(Kout=1, **kw); self.hk.feat('gen', 'gen')
    def obs(self, ind, cls, r):
        self.hk.feat(cls, 'cls'); self.hk.feat(ind, 'ind'); self.hk.update({'gen': 1.0, cls: 1.0, ind: 1.0}, np.array([r]))
    def prior(self, cls): return self.hk.val(['gen', cls]) if cls in self.hk.idx else self.hk.val(['gen'])
def llm(p, seed, mode='noise'):
    rng = np.random.default_rng(seed); h = H2(qmode='fixed'); ev = [(f't{i}', 'taker', -0.30) for i in range(8)] + [(f'f{i}', 'friend', 0.20) for i in range(8)]; rng.shuffle(ev)
    for ind, c, r in ev:
        if mode == 'noise': cl = c if rng.random() > p else ('friend' if c == 'taker' else 'taker')
        elif mode == 'split': cl = f'c_{ind}'
        else: cl = 'all'
        h.obs(ind, cl, r + SD * rng.standard_normal())
    if mode == 'noise': return h.prior('taker'), h.prior('friend')
    if mode == 'split': return h.prior('c_new1'), h.prior('c_new2')
    return h.prior('all'), h.prior('all')
print("| LLM class assignment | prior for a new TAKER (true -0.30) | prior for a new FRIEND (true +0.20) | separation |\n|---|---|---|---|")
for lab, p, mode in (('correct', 0.0, 'noise'), ('10% wrong', 0.10, 'noise'), ('25% wrong', 0.25, 'noise'), ('50% wrong (random)', 0.5, 'noise'), ('over-split (class per individual)', 0, 'split'), ('over-merge (one class)', 0, 'merge')):
    r = np.array([llm(p, s, mode) for s in range(200)]); print(f"| {lab} | {r[:,0].mean():+.3f} | {r[:,1].mean():+.3f} | {r[:,1].mean()-r[:,0].mean():+.3f} |")
