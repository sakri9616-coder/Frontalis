"""Sim3: (1) A's fixed point vs innate magnitude and decay; (2) LP niche dynamics (concave vs sigmoidal learning curves);
(3) operational-definition demo: how much of each candidate's acquired value is reproduced by an 'innate x co-occurrence' reference;
(4) belief-swap decomposition for E.  Scratchpad only."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prototype as P
from sim1_candidates import *
from sim0_existing import run1d_proto  # noqa (prints sim0 output again; harmless)
print('\n\n======== sim3 ========')
print('(1) A, c=1 fixed point v*(i, d):  (rows: innate i; cols: decay d)')
ds = (0.0005, 0.002, 0.01, 0.05)
print('        ' + ''.join(f'd={d:<8}' for d in ds))
for i in (0.05, 0.1, 0.2, 0.5, 1.0):
    print(f'  i={i:<4} ' + ''.join(f'{run1d_proto(i, 0, 1, d=d, T=60000):<10.3f}' for d in ds))

print('\n(2) LP niche dynamics: two identical domains, softmax(beta*iota) choice, iota=EMA of progress')
def niche(curve, seed, T=600, beta=12.0, eta=0.1, k=3.0, n0=15.0, lam=0.04):
    rg = np.random.default_rng(seed); n = np.zeros(2); iota = np.array([0.05, 0.05])
    def err(x):
        if curve == 'exp': return 1 - (1 - np.exp(-lam * x))
        return 1 - (x ** k / (x ** k + n0 ** k))
    for t in range(T):
        p = np.exp(beta * iota); p /= p.sum(); d = rg.choice(2, p=p)
        lp = err(n[d]) - err(n[d] + 1) if curve == 'exp' else err(n[d]) - err(n[d] + 1)
        n[d] += 1; iota[d] += eta * (lp * 10 - iota[d])
        iota[1 - d] *= 1.0
    return n / n.sum()
for curve in ('exp', 'sigmoid'):
    shares = np.array([niche(curve, s) for s in range(12)])
    print(f'  {curve:8s}: share of time in domain 1 over 12 seeds: ' + ' '.join(f'{x:.2f}' for x in shares[:, 0]))

print('\n(3) Reducibility to "innate x co-occurrence" reference (pairwise least squares of event innate reward on each feature alone)')
def ec_reference(kind):
    ref = np.zeros(NF)
    ev = P.history(kind); Z = np.array([zvec(c, o) for (_, c, o, _) in ev]); r = np.array([r_innate(o) for (_, _, o, _) in ev])
    for k in range(NF):
        z = Z[:, k]; ref[k] = (z @ r) / (z @ z) if (z @ z) > 0 else 0.0
    return ref
def td_reference(kind):   # converged TD(gamma=.9) = reference 2 (innate + temporal stats): use B itself at many sweeps
    c = B_TD()
    for _ in range(200):
        for e in P.history(kind): c.experience(e)
    return c.probe()
def r2(U, R):
    U = np.concatenate(U); R = np.concatenate(R); s = (U @ R) / (R @ R); res = U - s * R
    return 1 - (res @ res) / (U @ U), s
refs_ec = [ec_reference(k)[PI] for k in P.KINDS]
refs_td = [td_reference(k)[PI] for k in P.KINDS]
for cls in (A_existing, B_TD, E_MBMF, I_KTD, C_Homeo):
    U = [run_persona(cls, k).probe()[PI] for k in P.KINDS]
    a, sa = r2(U, refs_ec); b, sb = r2(U, refs_td)
    print(f'  {cls.name}: R^2 vs EC-reference {a:.2f} (scale {sa:+.2f});  R^2 vs TD-reference {b:.2f} (scale {sb:+.2f})')

print('\n(4) Belief-swap decomposition for E (value in the cache vs belief): persona K vs R, probe = authorship, connection')
K_ = run_persona(E_MBMF, 'credit-taken'); R_ = run_persona(E_MBMF, 'rescued-by-collaboration')
def U(cache_from, belief_from, i):
    w = cache_from.omega()[i]; return w * cache_from.vmf[i] + (1 - w) * belief_from.beta[i]
for nm in ('authorship', 'connection', 'music'):
    i = IDX[nm]
    UKK, UKR, URK, URR = U(K_, K_, i), U(K_, R_, i), U(R_, K_, i), U(R_, R_, i)
    # K minus R, decomposed into: belief swap effect (hold cache) and cache swap effect (hold belief), averaged over the two orderings
    belief_eff = 0.5 * ((UKK - UKR) + (URK - URR)); cache_eff = 0.5 * ((UKK - URK) + (UKR - URR))
    print(f'  {nm:11s} U(K)={UKK:+.3f} U(R)={URR:+.3f} diff {UKK-URR:+.3f} = belief part {belief_eff:+.3f} + cache(value) part {cache_eff:+.3f} (interaction {UKK-URR-belief_eff-cache_eff:+.3f})')
