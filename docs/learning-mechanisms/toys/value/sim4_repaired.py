"""Sim4: A' = existing rule repaired: exogenous (innate-reward) teacher, concern-weighted credit allocation (sum-conserving).  Scratchpad only."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prototype as P
from sim1_candidates import *

class Aprime:
    name="A'"
    def __init__(self, alpha=0.3, eps=0.05, rho=3.0):
        self.v=np.zeros(NF); self.alpha, self.eps, self.rho = alpha, eps, rho
    def experience(self, ev):
        _, cue, out, _ = ev
        z = zvec(cue, out); act = np.nonzero(z)[0]
        r = r_innate(out); pred = float(self.v @ z); delta = r - pred            # exogenous target, prediction contains the same parameters (Lemma: P' >= T' = 0)
        conc = np.abs(self.v[act]*z[act]) + self.eps                               # concern = current contribution magnitude (personal)
        s = conc**self.rho / np.sum(conc**self.rho)                                # allocation shares, sum to 1
        self.v[act] += self.alpha * len(act) * s * delta * z[act] / (z[act]@z[act] + 1e-9)
        return dict(delta=delta, shares=dict(zip([F[i] for i in act], np.round(s,2))))
    def probe(self): return self.v.copy()

if __name__=='__main__':
    # candidates comparison
    for cls in (Aprime, B_TD):
        D=[]
        for k in P.KINDS:
            c=run_persona(cls,k); b=c.probe()[PI].copy(); ex=c.experience(P.NEW); D.append(c.probe()[PI]-b)
            print(cls.name, k, 'before', np.round(b,3), 'delta', np.round(D[-1],3), (ex if cls is Aprime else ''))
        D=np.array(D); cos=[D[i]@D[j]/(np.linalg.norm(D[i])*np.linalg.norm(D[j])+1e-12) for i in range(3) for j in range(i+1,3)]
        print(' ', cls.name, 'mean pairwise cosine %.2f' % np.mean(cos), 'friendA signs', np.sign(D[:,4]).astype(int))
    # retention / reversal
    pool=[]
    for k in P.KINDS: pool += list(P.history(k))
    pool=[(f'p{i}',c,o,b) for i,(_,c,o,b) in enumerate(pool)]+[P.NEW]*3
    rng=np.random.default_rng(1); stream=[pool[i] for i in rng.integers(0,len(pool),size=128)]
    cs=[run_persona(Aprime,k) for k in P.KINDS]
    def dist(cs):
        ps=[c.probe()[PI] for c in cs]; return np.mean([np.linalg.norm(ps[i]-ps[j]) for i in range(3) for j in range(i+1,3)])
    d0=dist(cs); n=0; row=[]
    for m in (1,4,8,16,32,64,128):
        while n<m:
            for c in cs: c.experience(stream[n])
            n+=1
        row.append(dist(cs)/d0)
    print("A' retention n=1,4,8,16,32,64,128:", np.round(row,2), 'd0=%.2f'%d0)
    k_=run_persona(Aprime,'credit-taken'); r_=run_persona(Aprime,'rescued-by-collaboration'); tgt=r_.probe()[PI]; g0=np.linalg.norm(k_.probe()[PI]-tgt); hist=P.history('rescued-by-collaboration'); n=0; row=[]
    for m in (4,8,16,32,64,128):
        while n<m: k_.experience(hist[n%len(hist)]); n+=1
        row.append(1-np.linalg.norm(k_.probe()[PI]-tgt)/g0)
    print("A' reversal closed fraction n=4..128:", np.round(row,2))
    # sum conservation / boundedness: long run
    c=Aprime()
    for r in range(100):
        for e in P.history('credit-taken'): c.experience(e)
    print("A' after 2000 events max|v| %.2f" % np.abs(c.v).max())
    # content permutation
    rg=np.random.default_rng(0); h=[e for r in range(3) for e in P.history('credit-taken')]
    c0=Aprime(); [c0.experience(e) for e in h]
    dd=[]
    for s in range(20):
        perm=rg.permutation(len(h)); c1=Aprime(); [c1.experience((h[i][0],h[i][1],h[perm[i]][2],None)) for i in range(len(h))]; dd.append(np.linalg.norm(c1.v-c0.v))
    print("A' content-permutation ||v_true||=%.2f mean||v_true-v_perm||=%.2f" % (np.linalg.norm(c0.v), np.mean(dd)))
