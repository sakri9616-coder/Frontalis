"""T5: commitment reversal threshold vs the closed form |u_c| = sqrt(g(g-1)) - acosh(sqrt g); effect of an endogenous choice-conflict term; time to reverse."""
import numpy as np
def fixed_upper(g, u=0.0):
    c = 2.0
    for _ in range(20000): c += 0.2 * (-c + g * np.tanh(c) + u)
    return c
def reverses(g, u, ac=0.1, N=60000):
    c = fixed_upper(g)
    for t in range(N):
        c += ac * (-c + g * np.tanh(c) + u)
        if c < 0: return True, t
    return False, N
def threshold(g, shift=0.0, **kw):
    lo, hi = 0.0, 3.0
    for _ in range(40):
        mid = (lo + hi) / 2; ok, _ = reverses(g, -(mid) + shift, **kw)    # counter-evidence of size mid, offset by an endogenous push `shift`
        if ok: hi = mid
        else: lo = mid
    return hi
print("T5a closed form vs numeric (exogenous input only, a_c=0.1)")
print("| g | closed form u_c | numeric u_c | steps to reverse at 1.1 u_c / 1.5 u_c / 2 u_c |\n|---|---|---|---|")
for g in (1.1, 1.2, 1.3, 1.5):
    uc = np.sqrt(g * (g - 1)) - np.arccosh(np.sqrt(g)); num = threshold(g)
    st = [reverses(g, -f * uc)[1] for f in (1.1, 1.5, 2.0)]
    print(f"| {g} | {uc:.4f} | {num:.4f} | {st} |")
print("\nT5b endogenous choice-conflict push (k_ch * conflict, conflict in [0,1], pushes toward the chosen option's basin): effective threshold on EXTERNAL counter-evidence")
print("| g | k_ch*conflict | threshold on external evidence | ratio to closed form |\n|---|---|---|---|")
for g in (1.3,):
    uc = np.sqrt(g * (g - 1)) - np.arccosh(np.sqrt(g))
    for kc in (0.0, 0.05, 0.1, 0.2):
        th = threshold(g, shift=kc); print(f"| {g} | {kc} | {th:.4f} | {th/uc:.2f} |")
print("\nT5c what e_x level (MF innovation, k_e=2) is needed: u_c(g=1.3)/k_e =", round((np.sqrt(1.3*.3)-np.arccosh(np.sqrt(1.3)))/2, 4), "innovation-sum per update; typical |innovation sum| in R/K/X histories (v2 core):")
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
vals = []
for kind in P.KINDS:
    c = V2(commit=False)
    for ev in P.history(kind):
        z = zvec(ev[1], ev[2]); Dh = c.p['gM'] * (M @ ev[2]); r = c.u(np.clip(c.h + Dh, 0, c.p['hmax'])) - c.u(c.h)
        err = c.mf_update(z, r); vals.append(abs(err.sum())); c.h = np.clip(c.h + Dh, 0, c.p['hmax']); c.h = c.h + c.p['lam'] * (c.p['h0'] - c.h)
vals = np.array(vals); print(f"   |sum innovation| median {np.median(vals):.3f}, 90% {np.quantile(vals,.9):.3f}, max {vals.max():.3f}; k_e*|e| median {2*np.median(vals):.3f} vs u_c=0.101")
