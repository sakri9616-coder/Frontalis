"""C4: replay re-attribution with online-EM counts; prior-head distillation targets. Scratchpad only."""
import numpy as np
def post_mean(a,b): return a/(a+b)
def replay(N, remove, kap_e=2.0, kap_g=20.0, th=0.5):
    # two scope units for one outcome type; one observed event y=1
    ae,be=kap_e*th,kap_e*(1-th); ag,bg=kap_g*th,kap_g*(1-th)
    pi=np.array([1.0,1.0])               # Dirichlet counts pi_e, pi_g
    y=1; contrib=None
    def resp():
        lik=np.array([post_mean(ae,be), post_mean(ag,bg)]) ; r=(pi/pi.sum())*lik; return r/r.sum()
    r=resp(); ae+=r[0]*y; be+=r[0]*(1-y); ag+=r[1]*y; bg+=r[1]*(1-y); pi+=r; contrib=r.copy()
    out=[(0,post_mean(ae,be),pi[0]/pi.sum(),ae+be)]
    for n in range(1,N+1):
        if remove:
            ae-=contrib[0]*y; be-=contrib[0]*(1-y); ag-=contrib[1]*y; bg-=contrib[1]*(1-y); pi-=contrib
        r=resp(); ae+=r[0]*y; be+=r[0]*(1-y); ag+=r[1]*y; bg+=r[1]*(1-y); pi+=r; contrib=r.copy()
        if n in (1,5,20,100): out.append((n,post_mean(ae,be),pi[0]/pi.sum(),ae+be))
    return out
print("(1) ONE observed event (y=1), replayed N times with re-attribution by the CURRENT belief")
print("    columns: N, E[theta_person], P(scope=person), evidence mass in person unit (prior kappa=2)")
for remove in (False,True):
    print("   remove old contribution before re-adding =",remove)
    for row in replay(100,remove): print("     N=%3d  E[theta_e]=%.3f  pi_e=%.3f  mass_e=%.1f"%row)

# (2) prior-head targets
rng=np.random.default_rng(0)
def sim_head(target, kap_true=3.0, th=0.3, n=5, U=400, iters=12, kap0=8.0):
    thetas=rng.beta(kap_true*th,kap_true*(1-th),size=U); k=rng.binomial(n,thetas); kap=kap0; hist=[kap]
    for it in range(iters):
        a0,b0=kap*th,kap*(1-th)
        if target=='posterior_mean_moment':      # head trained to the shrunk posterior means, kappa from their dispersion
            pm=(a0+k)/(kap+n); v=pm.var(); kap=max(th*(1-th)/v-1,0.1)
        elif target=='posterior_params':         # head regressed on posterior Beta params
            kap=kap+n
        elif target=='marginal_likelihood':      # empirical Bayes: maximise beta-binomial marginal likelihood on grid
            from math import lgamma
            grid=np.exp(np.linspace(np.log(0.3),np.log(200),120)); best=None
            lb=lambda a,b: lgamma(a)+lgamma(b)-lgamma(a+b)
            ll=[sum(lb(g*th+kk,g*(1-th)+n-kk)-lb(g*th,g*(1-th)) for kk in k) for g in grid]
            kap=grid[int(np.argmax(ll))]
        hist.append(kap)
    return hist
print("(2) concentration kappa (true kappa=3: 'people differ') learned for the prior head, kappa0=8, 12 replay rounds")
for tgt in ('marginal_likelihood','posterior_mean_moment','posterior_params'):
    print("   %-24s"%tgt, np.round(sim_head(tgt),1))
