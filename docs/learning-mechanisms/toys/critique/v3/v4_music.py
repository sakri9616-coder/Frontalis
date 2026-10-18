"""V4: music starvation under v3. Concept probes (MF + EC) and situation U_val (MF situation sum + situation-level MB)."""
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v3model import *
SD = 0.06
class Sit:
    """situation-level belief memory (event utility), forgetting dB per observation of that situation"""
    def __init__(self, dB=0.05, RB=0.004, NM=40): self.d = {}; self.dB = dB; self.RB = RB; self.NM = NM
    def obs(self, key, r):
        w, x = self.d.setdefault(key, ([], [])); w[:] = [a * (1 - self.dB) for a in w]; w.append(1.0); x.append(r)
        if len(w) > self.NM: w.pop(0); x.pop(0)
    def mb(self, key):
        w, x = self.d[key]; w = np.array(w); x = np.array(x); m = (w * x).sum() / w.sum(); v = (w * (x - m) ** 2).sum() / w.sum()
        return m, (v + self.RB) / (w.sum() + 1)
def world(music_layer, p_alone, pre, T, seed, noise, ec=False, qmode='fixed'):
    rng = np.random.default_rng(seed); hk = HK(Kout=1, qmode=qmode)
    hk.feat('people', 'gen'); hk.feat('collab', 'cls'); hk.feat('F', 'ind'); hk.feat('music', music_layer)
    sit = Sit(); vec_ec = 0.0
    def ev(together, r):
        nonlocal vec_ec
        z = {'people': 1.0, 'collab': 1.0, 'F': 1.0}
        if together: z['music'] = 1.0
        if ec and together:                                                  # one-stage EC from the direct value of F (frozen snapshot)
            vF = hk.val(['people', 'collab', 'F']); vec_ec += 0.1 * (0.5 * vF - vec_ec)
        hk.update(z, np.array([r + noise * rng.standard_normal()])); sit.obs('F+music' if together else 'F', r + noise * rng.standard_normal())
    for t in range(pre): ev(False, 0.5)
    for t in range(T): ev(False, 0.5) if rng.random() < p_alone else ev(True, 0.8)
    return hk, sit, vec_ec
def report(music_layer, p_alone, pre, noise=0.0, ec=False, seeds=1):
    out = []
    for s in range(seeds):
        hk, sit, vec_ec = world(music_layer, p_alone, pre, 400, s, noise, ec)
        F = hk.val(['people', 'collab', 'F']); Mu = hk.val(['music'])
        zt = np.zeros(hk.n); [zt.__setitem__(hk.idx[n], 1.0) for n in ('people', 'collab', 'F', 'music')]; za = np.zeros(hk.n); [za.__setitem__(hk.idx[n], 1.0) for n in ('people', 'collab', 'F')]
        S = hk.S[0, :hk.n, :hk.n]; res = []
        for key, z in (('F+music', zt), ('F', za)):
            mf = float(z @ hk.W[:hk.n, 0]); sf = float(z @ S @ z)
            if key not in sit.d: res += [mf, float('nan'), float('nan'), mf]; continue
            mb, sb = sit.mb(key); om = sb / (sb + sf); res += [mf, mb, om, om * mf + (1 - om) * mb]
        out.append([F, Mu, vec_ec] + res)
    o = np.mean(out, 0); return o
print("| setting | friend probe | music probe (MF) | music probe + 0.5*EC | situation F+music: MF / MB / omega / U_val | situation F alone: U_val |\n|---|---|---|---|---|---|")
for lab, ml, pa, pre, nz, ec in (('music = class feature; friend established; friend alone 30%', 'cls', 0.3, 150, 0.0, False),
                                  ('music = class feature; friend established; always together', 'cls', 0.0, 150, 0.0, False),
                                  ('music = class feature; both new; always together', 'cls', 0.0, 0, 0.0, False),
                                  ('music = individual song; both new; always together', 'ind', 0.0, 0, 0.0, False),
                                  ('music = individual song; friend established; always together', 'ind', 0.0, 150, 0.0, False),
                                  ('class music; friend established; alone 30%; noise sd .06; EC on', 'cls', 0.3, 150, SD, True)):
    o = report(ml, pa, pre, nz, ec, seeds=1 if nz == 0 else 40)
    print(f"| {lab} | {o[0]:.3f} | {o[1]:.3f} | {o[1]+0.5*o[2]:.3f} | {o[3]:.3f} / {o[4]:.3f} / {o[5]:.2f} / {o[6]:.3f} | {o[10]:.3f} |")
print("(true marginal values: friend 0.5, music 0.3; situation F+music 0.8, F alone 0.5)")
