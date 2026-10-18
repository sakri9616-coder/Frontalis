import numpy as np
def run(curve, seed, ND=3, T=120, beta=25.0, eta=0.15, k=3.0, n0=12.0, lam=0.05, forget=0.0):
    rg = np.random.default_rng(seed); n = np.zeros(ND); iota = np.full(ND, 0.05)
    def S(x):
        return 1-np.exp(-lam*x) if curve=='exp' else x**k/(x**k+n0**k)
    for t in range(T):
        p = np.exp(beta*iota); p/=p.sum(); d = rg.choice(ND, p=p)
        lp = (S(n[d]+1)-S(n[d])) * 5.0
        n[d]+=1; iota[d] += eta*(lp-iota[d])
        iota = iota*(1-forget)
    return n/ n.sum()
for T in (40, 120, 600):
  for curve in ('exp','sigmoid'):
    sh = np.array([run(curve, s, T=T) for s in range(10)])
    print(f'T={T:4d} {curve:8s} max share of the most-visited domain over 10 seeds: ' + ' '.join(f'{x:.2f}' for x in sh.max(1)) + f'   (3 domains; uniform=0.33)  mean {sh.max(1).mean():.2f}')
