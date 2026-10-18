"""C3: trust tau_j=(1+b)/(2+b+h) where b,h count 'satisfied my needs / predicted right' vs 'harmed / wrong'. Scratchpad only."""
import numpy as np
def run(seed, T=400, beta=6.0, use_hedonic=True, flat_fact_acc=0.5, hon_fact_acc=0.8):
    rng=np.random.default_rng(seed)
    b=np.zeros(2); h=np.zeros(2)         # 0=honest, 1=flatterer
    w=0.0                                # adopted social value of topic x whose TRUE need-effect is -0.5
    u=np.array([-1.0, +1.0])             # what each source says about x (honest says bad, flatterer says good)
    n=np.zeros(2)
    for t in range(T):
        tau=(1+b)/(2+b+h)
        p=np.exp(beta*tau)/np.exp(beta*tau).sum()
        j=rng.choice(2,p=p); n[j]+=1
        # need effect of the interaction itself (relatedness): praise +, criticism -
        if j==1: dh_rel=+0.3
        else:    dh_rel=+0.3 if rng.random()<0.5 else -0.3
        fact_ok = rng.random() < (flat_fact_acc if j==1 else hon_fact_acc)   # factual prediction right?
        if use_hedonic:
            ok = (dh_rel>0) + fact_ok          # b counts both
            ko = (dh_rel<0) + (not fact_ok)
        else:
            ok = fact_ok*1.0; ko = (not fact_ok)*1.0     # only external predictive accuracy
        b[j]+=ok; h[j]+=ko
        tj=(1+b[j])/(2+b[j]+h[j])
        w += 0.1*tj*(u[j]-w)
    return (1+b)/(2+b+h), n/T, w
for mode in (True, False):
    taus=[];fr=[];ws=[]
    for s in range(100):
        t,f,w=run(s,use_hedonic=mode); taus.append(t); fr.append(f); ws.append(w)
    taus=np.array(taus); fr=np.array(fr); ws=np.array(ws)
    print(f"b_j includes own need satisfaction (hedonic) = {mode}: tau_honest={taus[:,0].mean():.2f} tau_flatterer={taus[:,1].mean():.2f}; share of interactions with flatterer={fr[:,1].mean():.2f}; adopted w_soc(x) mean={ws.mean():+.2f} (true need effect of x is -0.5; honest says -1, flatterer +1); frac runs w>0 = {np.mean(ws>0):.2f}")
