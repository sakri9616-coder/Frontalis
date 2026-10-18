"""T4: NV3 (value remaining after the reinforcer is removed) and G1' elasticities (coefficient vs evidence)."""
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
MU = IDX['music']
train = ('train', vec(music=1), vec(music=1, achievement=.9), None)
nullev = ('rem', vec(music=1), vec(music=1), None)
def readout(c, kind):                    # state-corrected: h reset to baseline, so the number is the learned value not the satiation state
    h = c.h.copy(); c.h = np.full(K, PARAMS['h0']); v = {'Q': c.Q(MU), 'MF': c.V_MF(MU), 'U': c.U_val(MU), 'c': c.c[MU]}[kind]; c.h = h; return v
def nv3(n=100, ms=(20, 100), kind='Q', **kw):
    c = V2(**kw)
    for t in range(n): c.experience(train); c.idle(4)
    v0 = readout(c, kind); out = []; m_done = 0
    for m in ms:
        while m_done < m: c.experience(nullev); c.idle(4); m_done += 1
        out.append(readout(c, kind) / v0 if abs(v0) > 1e-9 else float('nan'))
    return v0, out
print("T4a NV3 retention Q(music)/Q_before after m removal events (m=20,100); training n=100 reinforcements (music + achievement .9)")
print("| config | Q before | retention m=20 | m=100 |\n|---|---|---|---|")
cfgs = {'v2 core (RB=.004)': dict(commit=False), 'v2 core RB=.4': dict(commit=False, RB=0.4), 'v2 core RB=.4, q=0': dict(commit=False, RB=0.4, q=0.0),
        'v2 + c (lam_c=1, g=1.3)': dict(commit=True), 'v2 + c (lam_c=.1, g=1.3)': dict(commit=True, lam_c=0.1), 'v2 + c (lam_c=.1) RB=.4': dict(commit=True, lam_c=0.1, RB=0.4)}
for nm, kw in cfgs.items():
    v0, r = nv3(**kw); print(f"| {nm} | {v0:.3f} | {r[0]:.2f} | {r[1]:.2f} |")
print("\nT4b G1' elasticities on retention at m=20 (|E_coef| <= 0.25|E_evidence| required). E = ln(ret(2x)/ret(x/2))/ln 4. Evidence = training trials n (50 vs 200) or reinforcer magnitude")
def ret(m=20, kind='Q', **kw):
    return max(nv3(ms=(m,), kind=kind, **kw)[1][0], 1e-3)
for nm, base in (('v2 core RB=.4', dict(commit=False, RB=0.4)), ('v2 + c (lam_c=.1) RB=.4', dict(commit=True, lam_c=0.1, RB=0.4))):
    print(f"-- {nm}")
    ev = np.log(ret(n=200, **base) / ret(n=50, **base)) / np.log(4)
    print(f"   E_evidence (n 50->200): ret {ret(n=50,**base):.3f} -> {ret(n=200,**base):.3f}, E={ev:+.3f}")
    keys = {'R': PARAMS['R'], 'q': PARAMS['q'], 'dB': PARAMS['dB']}
    if base['commit']: keys.update({'a_c': PARAMS['a_c'], 'g-1': None})
    for kname, v in keys.items():
        if kname == 'g-1':
            lo = ret(**{**base, 'g': 1 + (PARAMS['g'] - 1) / 2}); hi = ret(**{**base, 'g': 1 + (PARAMS['g'] - 1) * 2}); lab = 'g-1'
        else:
            lo = ret(**{**base, kname: v / 2}); hi = ret(**{**base, kname: v * 2}); lab = kname
        E = np.log(hi / lo) / np.log(4)
        print(f"   coef {lab:4s}: ret(x/2)={lo:.3f} ret(2x)={hi:.3f}  E_coef={E:+.3f}  |E_coef|/|E_evid|={abs(E)/max(abs(ev),1e-9):.2f}  G1' {'PASS' if abs(E) <= 0.25*abs(ev) else 'FAIL'}")
print("\nT4c q=0 (static world) retention vs evidence n, Kalman gain 1/(n+..): m=20 / m=100")
for n in (25, 50, 100, 200, 400):
    v0, r = nv3(n=n, commit=False, RB=0.4, q=0.0); print(f"   n={n}: Q_before={v0:.3f} retention m=20: {r[0]:.2f}  m=100: {r[1]:.2f}   (n/(n+m) = {n/(n+20):.2f}, {n/(n+100):.2f})")
