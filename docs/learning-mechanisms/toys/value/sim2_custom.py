"""Sim2: 1-D analytic checks and small 3-persona demos for candidates B..I (toy sketch; not an experiment).
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prototype as P
from sim1_candidates import *   # classes, F, IDX, NI, zvec, r_innate

np.set_printoptions(precision=3, suppress=True)
print('############ B: TD -- order dependence and contraction')
def td_chain(order, gamma=0.9, alpha=0.2, n=400, r=1.0):
    V = {'A': 0., 'B': 0.}
    for _ in range(n):
        s = order  # sequence of states then terminal reward
        for i, x in enumerate(s):
            nxt = V[s[i + 1]] if i + 1 < len(s) else r
            V[x] += alpha * ((0 if i + 1 < len(s) else 0) + gamma * (nxt if i + 1 < len(s) else 0) + (r if i + 1 == len(s) else 0) - V[x])
    return V
print('  A->B->reward :', {k: round(v, 3) for k, v in td_chain(['A', 'B']).items()}, '   B->A->reward :', {k: round(v, 3) for k, v in td_chain(['B', 'A']).items()}, '(same co-occurrence counts, different order)')
print('  1-D bootstrapped fixed point v*=r/(1-gamma*p): gamma=.9, p=1 ->', 1 / (1 - .9), '; gamma=1.0 diverges (no fixed point)')

print('\n############ C: set-point dynamics, 1 need (deficit e=h*-h)')
def setpoint(eta, rho, lam=0.3, k=0.5, T=4000, h0s=1.0):
    h, hs = 1.0, h0s
    for t in range(T):
        e = hs - h
        h = h - lam + k * max(e, 0)             # consumption lam per step, regulation gain k on deficit
        hs = hs + eta * (h - hs) + rho * (h0s - hs)
    return h, hs
for eta, rho in ((0.02, 0.0), (0.02, 0.005), (0.02, 0.02), (0.0, 0.0)):
    h, hs = setpoint(eta, rho)
    print(f'  eta={eta} rho(anchor)={rho}: after 4000 steps of constant consumption: h={h:.3f} set point={hs:.3f} (innate 1.0)')
print('  same, but consumption halves for the first 2000 steps then doubles (history-dependence of set point):')
def setpoint2(eta, rho, lam1, lam2, T1=2000, T2=2000, k=0.5):
    h, hs, h0s = 1.0, 1.0, 1.0
    out = []
    for t in range(T1 + T2):
        lam = lam1 if t < T1 else lam2
        e = hs - h; h = h - lam + k * max(e, 0); hs = hs + eta * (h - hs) + rho * (h0s - hs)
        if t in (T1 - 1, T1 + T2 - 1): out.append(round(hs, 3))
    return out
print('   lam 0.1 then 0.3 -> set point at end of phase 1, 2:', setpoint2(0.02, 0.005, 0.1, 0.3), '; lam 0.3 then 0.1 ->', setpoint2(0.02, 0.005, 0.3, 0.1))

print('\n############ C: 3-persona NEW event, felt = drive reduction and set-point change')
for kind in P.KINDS:
    c = run_persona(C_Homeo, kind); hs_before = c.hs.copy(); h_before = c.h.copy()
    ex = c.experience(P.NEW)
    print(f'  {kind:26s} deficit before (h*-h)={np.round(hs_before-h_before,2)}  NEW dh={np.round(ex["dh"],3)}  felt={ex["felt"]:+.4f}  set point change={np.round(c.hs-hs_before,4)}')

print('\n############ D: learning-progress interest (3 domains), persona states from histories; then NEW x6')
DOM = {'M': lambda c, o: (c[IDX['music']] != 0) or (o[IDX['music']] != 0),
       'Cn': lambda c, o: any(c[IDX[k]] != 0 or o[IDX[k]] != 0 for k in ('others', 'connection', 'friendA')),
       'Ex': lambda c, o: o[IDX['novelty']] != 0 or c[IDX['autonomy']] != 0}
class D_LP:
    def __init__(self, W=4, eta=0.3, lr=0.15, beta=0.3):
        self.m = {d: np.zeros(NF) for d in DOM}; self.E = {d: None for d in DOM}; self.Eh = {d: [] for d in DOM}
        self.iota = {d: 0.0 for d in DOM}; self.W, self.eta, self.lr, self.beta = W, eta, lr, beta; self.LP = {d: 0.0 for d in DOM}
    def experience(self, ev):
        _, cue, out, _ = ev
        for d, f in DOM.items():
            if not f(cue, out): continue
            e = float(np.sum((out - self.m[d]) ** 2)); self.m[d] += self.lr * (out - self.m[d])
            self.E[d] = e if self.E[d] is None else (1 - self.beta) * self.E[d] + self.beta * e
            self.Eh[d].append(self.E[d])
            if len(self.Eh[d]) > self.W:
                self.LP[d] = self.Eh[d][-1 - self.W] - self.Eh[d][-1]        # progress = error drop over the last W domain events
                self.iota[d] += self.eta * (self.LP[d] - self.iota[d])
    def state(self): return {d: (round(self.E[d] or 0, 2), round(self.LP[d], 2), round(self.iota[d], 3)) for d in DOM}
for kind in P.KINDS:
    c = D_LP()
    for e in P.history(kind): c.experience(e)
    s0 = c.state(); out = [f'  {kind:26s} (E,LP,interest) after history {s0}']
    for n in (1, 3, 6):
        c2 = D_LP(); [c2.experience(e) for e in P.history(kind)]
        for _ in range(n): c2.experience(P.NEW)
        out.append(f'      after NEW x{n}: ' + str(c2.state()))
    print('\n'.join(out))

print('\n############ E: devaluation persistence (cache vs belief)')
def devalue(n_train, m_after, nh=10.0, amin=0.02, beta_after=0.0):
    v = 0.0; n = 0; beta = 0.5
    for _ in range(n_train):
        n += 1; v += max(1 / (n + 1), amin) * (beta - v)
    om = n / (n + nh)
    res = []
    for m in m_after:
        vv, nn = v, n
        for _ in range(m):
            nn += 1; vv += max(1 / (nn + 1), amin) * (beta_after - vv)
        om2 = nn / (nn + nh)
        res.append((m, round(om2 * vv + (1 - om2) * beta_after, 3), round(vv, 3)))
    return round(om, 2), round(v, 3), res
for ntr in (3, 10, 30, 100):
    print(f'  trained {ntr:3d} times (MB value 0.5): omega, v_MF, then after the reinforcer is removed (belief->0): (m extinction trials, felt value, v_MF) =', devalue(ntr, (1, 5, 20, 80)))

print('\n############ F: choice-induced preference change')
rng = np.random.default_rng(0)
def naive_selfperception(v0, eta=0.05, beta=3.0, T=3000, d=0.0):
    v = v0;
    for t in range(T):
        a = 1.0 if rng.random() < 1 / (1 + np.exp(-beta * v)) else 0.0
        v += eta * (a - 0.5) * 2 - d * v    # chosen -> v up, not chosen -> v down (uncalibrated: no expectation subtracted)
    return v
print('  naive self-perception, v0=0 (symmetric), 10 runs final v (bistable polarisation):', np.round([naive_selfperception(0.0) for _ in range(10)], 2))
def calibrated(v0, eta=0.05, beta=3.0, T=3000):
    v = v0
    for t in range(T):
        p = 1 / (1 + np.exp(-beta * v)); a = 1.0 if rng.random() < p else 0.0
        v += eta * (a - p) * 2
    return v
fin = np.array([calibrated(0.0) for _ in range(200)])
print('  calibrated (subtract own prediction), v0=0: mean final v %.3f, sd %.3f (martingale: no drift, only random walk)' % (fin.mean(), fin.std()))
def cost_anchored(v0, c, eta=0.2, T=60):
    v = v0
    for t in range(T): v += eta * max(c - v, 0)
    return v
print('  cost-anchored (effort justification) v <- v + eta*max(c - v, 0), c=0.5: from v0=-0.54 ->', round(cost_anchored(-0.54, 0.5), 3), ', from 0.755 ->', round(cost_anchored(0.755, 0.5), 3), ', from 0.52 ->', round(cost_anchored(0.52, .5), 3))
print('  3-persona NEW (forced choice: co-create, cost c=.5, eta=.3); prior value = tanh(A-value of connection / autonomy):')
for kind in P.KINDS:
    p = P.Personality(kind)
    for e in P.history(kind): p.experience(*e)
    vcc, vsl = np.tanh(p.v[IDX['connection']]), np.tanh(p.v[IDX['autonomy']])
    dcc = 0.3 * max(0.5 - vcc, 0); dsl = -0.3 * max(vsl, 0) * (1.0 if vcc > -1 else 0)
    print(f'   {kind:26s} v_cocreate {vcc:+.2f} -> {vcc+dcc:+.2f} ({dcc:+.3f});  v_solo {vsl:+.2f} -> {vsl+dsl:+.2f} ({dsl:+.3f})')

print('\n############ G: trust-weighted social value')
def trust_bif(s, T=20000, eta=0.02, w0=0.0, agree=True):
    """two sources u=+1, u=-1, equally frequent; trust tau(u,w)=exp(-(u-w)^2/(2 s^2)) (agreement-based)."""
    w = w0; rng = np.random.default_rng(3)
    for t in range(T):
        u = 1.0 if rng.random() < .5 else -1.0
        tau = np.exp(-(u - w) ** 2 / (2 * s * s)) if agree else 1.0
        w += eta * tau * (u - w)
    return w
for s in (0.5, 0.9, 1.0, 1.2, 2.0):
    print(f'  agreement-trust width s={s}: final w from w0=+0.01: {trust_bif(s, w0=0.01):+.2f}; from w0=-0.01: {trust_bif(s, w0=-0.01):+.2f}')
print('  fixed-trust control (outcome-anchored): final w from +-0.01:', round(trust_bif(1, w0=0.01, agree=False), 2), round(trust_bif(1, w0=-0.01, agree=False), 2), '(sources average out)')
print('  3-persona: trust from Beta(1+benevolent,1+betrayed) over "others", then testimony u=-0.5 on authorship from NEW friend A; eta=.4; prior w=tanh(A value of authorship)')
benev = {'rescued-by-collaboration': (16, 0), 'credit-taken': (0, 8), 'explorer': (4, 0)}
for kind in P.KINDS:
    p = P.Personality(kind)
    for e in P.history(kind): p.experience(*e)
    w = np.tanh(p.v[IDX['authorship']]); b, bt = benev[kind]; tau = (1 + b) / (2 + b + bt); s = 2 * tau - 1
    u = -0.5; d_signed = 0.4 * s * (u - w); d_clamp = 0.4 * max(s, 0) * (u - w)
    print(f'   {kind:26s} trust tau={tau:.2f} signed={s:+.2f}; w_auth {w:+.2f} -> signed {w+d_signed:+.2f} ({d_signed:+.3f}) | clamped(no reactance) {w+d_clamp:+.2f} ({d_clamp:+.3f})')

print('\n############ H: relevance (attention) weights over appraisal dimensions')
def attn_sym_break(rho, T=6000, a=0.05, noise=0.0, seed=0):
    """two perfectly redundant features x1=x2=1 predicting r=1. weights w1,w2; attention softmax(rho*|w|); dw_k = a*omega_k*err"""
    rg = np.random.default_rng(seed); w = np.array([0.0, 0.0]) + rg.normal(0, 1e-3, 2)
    for t in range(T):
        om = np.exp(rho * np.abs(w)); om /= om.sum()
        err = 1.0 - w.sum()
        w = w + a * 2 * om * err
    return w
for rho in (0, 2, 5, 10, 20):
    ws = [attn_sym_break(rho, seed=s) for s in range(4)]
    print(f'  rho={rho:2d}: final (w1,w2) for 4 seeds: ' + ' '.join(f'({w[0]:.2f},{w[1]:.2f})' for w in ws), ' sum=1 always (anchored); split depends on seed')
dims = ['connection', 'authorship', 'autonomy']
valence = {'connection': 1.0, 'authorship': 1.0, 'autonomy': 1.0}     # designer-fixed valence prior (limitation)
for kind in P.KINDS:
    X, y = [], []
    for e in P.history(kind):
        _, cue, out, _ = e; z = zvec(cue, out); X.append([z[IDX[k]] for k in dims]); y.append(r_innate(out))
    X, y = np.array(X), np.array(y); beta = np.linalg.solve(X.T @ X + 0.5 * np.eye(3), X.T @ y)
    Pk = np.abs(beta) * X.std(0); om = np.exp(4 * Pk / (Pk.max() + 1e-9)); om /= om.sum()
    out = P.NEW[2]; a_new = sum(om[i] * valence[k] * out[IDX[k]] for i, k in enumerate(dims))
    print(f'   {kind:26s} beta {np.round(beta,2)} relevance omega {np.round(om,2)}  appraisal of NEW (sum omega*valence*level) = {a_new:+.3f}')

print('\n############ I: Kalman steady-state gain (volatility keeps plasticity) and hierarchical prior for a NEW entity')
def kgain(q, r=0.25, T=3000):
    S = 1.0
    for _ in range(T):
        Sp = S + q; K = Sp / (Sp + r); S = (1 - K) * Sp
    return K
for q in (0.0, 0.001, 0.01, 0.1):
    print(f'  process noise q={q}: steady Kalman gain {kgain(q):.3f}')
print('  hierarchical prior for friendA: prior mean = mean value of known "others"-class entities; observation y=+0.3; prior var tau0^2=0.3, obs var .25')
for kind, prior in (('rescued-by-collaboration', +0.8), ('credit-taken', -0.6), ('explorer', +0.4)):
    t0, r = 0.3, 0.25; K = t0 / (t0 + r); post = prior + K * (0.3 - prior)
    print(f'   {kind:26s} prior {prior:+.2f} -> posterior {post:+.2f} (update {post-prior:+.2f}, posterior sd {np.sqrt((1-K)*t0):.2f})')
