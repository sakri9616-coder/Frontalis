import numpy as np
def sim(hs_init, eta=0.02, rho_h=0.005, lam=0.05, a=0.3, p_opp=0.5, beta=8.0, T=3000, seed=0, h0s=1.0, record=(50,100,200,500,1000,2000)):
    rng=np.random.default_rng(seed); h=hs_init; hs=hs_init; out={}
    D=lambda h,hs:0.5*(hs-h)**2
    # common random numbers across initial conditions
    for t in range(1,T+1):
        r=0.0
        if rng.random()<p_opp:
            val=D(h,hs)-D(h+a,hs); p=1/(1+np.exp(-beta*val))
            if rng.random()<p: h=min(h+a,2.0); r=1.0
        hs=hs+eta*r*(h-hs)+rho_h*(h0s-hs); h=h*(1-lam)
        if t in record: out[t]=hs
    return out
import statistics
a=[sim(0.3,seed=s) for s in range(20)]; b=[sim(1.8,seed=s) for s in range(20)]
print("closed-loop (value-guided approach) memory of the initial set point: mean |h*(0.3 start) - h*(1.8 start)| with common random numbers")
for t in (50,100,200,500,1000,2000):
    d=np.mean([abs(x[t]-y[t]) for x,y in zip(a,b)]); print(f"   t={t:5d}: {d:.3f} (initial gap 1.5)")
# 200 irrelevant updates then relevant events: gated vs ungated
lam=0.05
for gated in (True,False):
    hs=1.0; h=1.0
    for t in range(200):
        h*=(1-lam)
        hs=hs+0.005*(1.0-hs)+(0 if gated else 0.02*(h-hs))
    print(f"   after 200 irrelevant updates, gated={gated}: h*={hs:.3f}")
