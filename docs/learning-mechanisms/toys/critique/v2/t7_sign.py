"""T7: sign of the value of friend A after the §6 NEW event, same Delta h for all personas; only need state h (from their histories) and attribution differ."""
import sys, copy; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
np.set_printoptions(precision=3, suppress=True)
A = IDX['friendA']
p = PARAMS; s = p['s']
Dh = p['gM'] * (M @ P.NEW[2])                      # identical for every persona (fixed M)
print("Delta h (all personas) [rel, comp, aut, stim, safe] =", Dh)
print("per-need hedonic r_k = s e^{-h_k/s} (1-e^{-D_k/s}) for growth needs (sign = sign(D_k) always); only magnitude depends on h")
ATT = {'A takes everything (a=1)': np.array([1, 1, 1, 1, 1.]),
       'section-6 attribution': {'R': np.array([1, 1, 0, 0, 0.]), 'K': np.array([1, 0, 1, 0, 0.]), 'X': np.array([1, 0, 0, 1, 0.])}}
names = {'rescued-by-collaboration': 'R', 'credit-taken': 'K', 'explorer': 'X'}
print("| persona | h at end of history | r_k for NEW | V_A (a=1) | V_A (section-6 attribution) | V_MF(A) after NEW (Kalman, no attribution) |\n|---|---|---|---|---|---|")
res = {}
for kind in P.KINDS:
    c = run_persona(lambda: V2(commit=False), kind); h = c.h.copy(); r = c.u(np.clip(h + Dh, 0, p['hmax'])) - c.u(h); nm = names[kind]
    vA1 = r.sum(); vA6 = (ATT['section-6 attribution'][nm] * r).sum()
    c.experience(P.NEW); res[nm] = (h, r)
    print(f"| {nm} | {np.round(h,2)} | {np.round(r,3)} | {vA1:+.3f} | {vA6:+.3f} | {c.V_MF(A):+.3f} |")
print("\nAnalytic condition (small D): V_A = sum_k a_k m_k D_k with m_k = e^{-h_k/s}. K's V_A<0 needs a_aut m_aut |D_aut| > a_rel m_rel D_rel (+ other attributed gains).")
D = Dh; print(f"   D_rel={D[0]:.2f}, D_aut={D[2]:.2f} -> need m_aut/m_rel > {D[0]/abs(D[2]):.2f}*(a_rel/a_aut), i.e. h_rel - h_aut > s*ln(...) = {s*np.log(D[0]/abs(D[2])):.2f} (small-D); concave loss amplification lowers it slightly")
for nm,(h,r) in res.items(): print(f"   {nm}: h_rel-h_aut = {h[0]-h[2]:+.2f}")
# Exact thresholds for K attribution (rel gain, aut loss to A): solve for h_rel - h_aut such that V_A=0, others baseline
def vA(h_rel, h_aut, Drel=D[0], Daut=D[2]):
    return s*np.exp(-h_rel/s)*(1-np.exp(-Drel/s)) + s*np.exp(-h_aut/s)*(1-np.exp(-Daut/s))
def bis(f, lo, hi):
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return (lo + hi) / 2
th = bis(lambda x: vA(1.0 + x, 1.0), -3, 6); print(f"   exact (K attribution: rel gain + aut loss to A; h_aut=1): V_A<0 iff h_rel - h_aut >= {th:.2f}")
th2 = bis(lambda x: vA(1.0, 1.0 - x), 0, 6) if False else None
print("   equivalently with h_rel=1: V_A<0 iff h_aut <= ", round(bis(lambda ha: vA(1.0, ha), -2, 6),2), " (autonomy depleted below this level)")
print("\nWhat produces the section-6 sign pattern under v2 (R:+, K:-, X:+) with a = section-6 attribution:")
for nm,(h,r) in res.items():
    a = ATT['section-6 attribution'][nm]; print(f"   {nm}: V_A = {(a*r).sum():+.3f}  (parts: {np.round(a*r,3)})")
# attribution-only variation: same h (baseline) for everyone
print("\nSame h = baseline for everyone; only attribution differs:")
r0 = c.u(np.full(K, 1.0) + Dh) - c.u(np.full(K, 1.0))
for nm,a in (('R',ATT['section-6 attribution']['R']),('K',ATT['section-6 attribution']['K']),('X',ATT['section-6 attribution']['X'])): print(f"   {nm}: V_A={(a*r0).sum():+.3f} parts {np.round(a*r0,3)}")
print("   -> with baseline h, the K negative requires a_aut=1 AND a_rel,a_comp,a_stim such that the aut loss outweighs the attributed gains; K attributing rel (+0.3 gain) to A too gives:", f"{(np.array([1,0,1,0,0.])*r0).sum():+.3f}")

print("\nK with history order reversed (solo-success first, then credit-stolen: autonomy ends DEPLETED) vs the prototype order:")
for nm, order in (('prototype order (stolen x8, then solo-success x12)', False), ('reversed order', True)):
    hist = P.history('credit-taken'); stolen = [e for e in hist if e[0].startswith('credit-stolen')]; solo = [e for e in hist if e[0].startswith('solo-success')]
    seq = (solo + stolen) if order else (stolen + solo)
    c = V2(commit=False)
    for e in seq: c.experience(e)
    h = c.h.copy(); r = c.u(np.clip(h + Dh, 0, p['hmax'])) - c.u(h); a = ATT['section-6 attribution']['K']
    print(f"   {nm}: h={np.round(h,2)} r={np.round(r,3)} V_A(a=1)={r.sum():+.3f} V_A(section-6 attribution)={(a*r).sum():+.3f}")
