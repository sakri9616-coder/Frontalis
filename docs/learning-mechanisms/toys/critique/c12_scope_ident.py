"""C12: scope attribution (person e vs scene c vs general) when e and c always co-occur (single user, single setting): identifiable?"""
import numpy as np
def run(seed, T=300, p=0.7, prior_e=(1.5,4.5), prior_c=(1.5,4.5), xi=(1.0,1.0), kappa_diff=0.0):
    rng=np.random.default_rng(seed)
    a=np.array([prior_e[0],prior_c[0]],float); b=np.array([prior_e[1],prior_c[1]],float); pi=np.array(xi,float)
    for t in range(T):
        y=rng.random()<p
        th=a/(a+b); lik=np.where(y,th,1-th); r=(pi/pi.sum())*lik; r/=r.sum()
        a+=r*y; b+=r*(1-y); pi+=r
    return pi[0]/pi.sum(), a/(a+b)
print("always-together e,c; y~Bern(0.7); online EM; final P(scope=person) over 40 seeds")
for nm,kw in (("identical priors",{}),("prior means differ by 0.01 (0.26 vs 0.25)",dict(prior_e=(1.56,4.44))),("person prior more flexible (kappa 2 vs 20)",dict(prior_e=(0.5,1.5),prior_c=(5,15)))):
    ps=np.array([run(s,**kw)[0] for s in range(40)]); print(f"  {nm:46s} pi_e: mean {ps.mean():.2f} sd {ps.std():.2f} min {ps.min():.2f} max {ps.max():.2f}")
print("predictive P(y) is identical in all cases (data cannot tell the split):", [round(float(np.mean([ (lambda pi,th:(pi*th).sum()/pi.sum())(np.array([run(s)[0],1-run(s)[0]]), run(s)[1]) for s in range(3)])),3)])
# hyper-parameter identifiability with N partners: marginal-likelihood curvature of kappa_e
from math import lgamma
def lb(a,b): return lgamma(a)+lgamma(b)-lgamma(a+b)
def ll(kap, ks, ns, th=0.3): return sum(lb(kap*th+k,kap*(1-th)+n-k)-lb(kap*th,kap*(1-th)) for k,n in zip(ks,ns))
rng=np.random.default_rng(0)
print("kappa_e marginal likelihood: 95%-ish support width on a log grid, true kappa=3, n=8 trials per partner")
grid=np.exp(np.linspace(np.log(0.3),np.log(300),200))
for N in (1,2,4,8,32):
    wd=[]
    for rep in range(30):
        th=rng.beta(3*0.3,3*0.7,size=N); ks=rng.binomial(8,th); l=np.array([ll(g,ks,[8]*N) for g in grid]); w=np.exp(l-l.max()); w/=w.sum()
        cdf=np.cumsum(w); lo=grid[np.searchsorted(cdf,0.025)]; hi=grid[min(np.searchsorted(cdf,0.975),len(grid)-1)]; wd.append(np.log10(hi/lo))
    print(f"   N partners={N:2d}: median width of 95% interval = {np.median(wd):.2f} decades")
