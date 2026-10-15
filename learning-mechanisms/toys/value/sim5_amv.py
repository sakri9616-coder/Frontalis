"""Sim5: integrated toy of the recommended AMV at levels L1 (B/A'+E) and L2 (+C needs/set points). Scratchpad only; not an experiment."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prototype as P
from sim1_candidates import *

class AMV_L1:
    """belief beta learns r_innate with concern-weighted allocation (A'); cache v distills beta (E); U = omega v + (1-omega) beta."""
    name='AMV-L1'
    def __init__(self, alpha=0.3, eps=0.05, rho=3.0, nh=10.0, amin=0.02):
        self.beta=np.zeros(NF); self.v=np.zeros(NF); self.n=np.zeros(NF)
        self.alpha, self.eps, self.rho, self.nh, self.amin = alpha, eps, rho, nh, amin
    def omega(self): return self.n/(self.n+self.nh)
    def U(self):
        w=self.omega(); return w*self.v+(1-w)*self.beta
    def experience(self, ev):
        _,cue,out,_=ev; z=zvec(cue,out); act=np.nonzero(z)[0]
        delta=r_innate(out)-self.beta@z
        U=self.U(); conc=np.abs(U[act]*z[act])+self.eps; s=conc**self.rho/np.sum(conc**self.rho)
        self.beta[act]+=self.alpha*len(act)*s*delta*z[act]/(z[act]@z[act]+1e-9)
        for k in act:
            self.n[k]+=1; a=max(1/(self.n[k]+1),self.amin); self.v[k]+=a*(self.beta[k]-self.v[k])
    def probe(self): return self.U()

class AMV_L2:
    """needs-vector teacher with concern-weighted allocation; V_MB(x|h)=D(h)-D(h+u_x); cache distills V_MB; set points adapt."""
    name='AMV-L2'
    def __init__(self, g=0.06, lam=0.05, eta=0.02, rho_h=0.005, alpha=0.3, eps=0.02, rho=3.0, nh=10.0, amin=0.02):
        self.h=np.ones(4); self.hs0=np.ones(4); self.hs=self.hs0.copy(); self.u=np.zeros((NF,4)); self.v=np.zeros(NF); self.n=np.zeros(NF)
        self.g,self.lam,self.eta,self.rho_h,self.alpha,self.eps,self.rho,self.nh,self.amin=g,lam,eta,rho_h,alpha,eps,rho,nh,amin
    def D(self,h): return 0.5*float(np.sum((self.hs-h)**2))
    def VMB(self,x): return self.D(self.h)-self.D(self.h+self.u[x])
    def omega(self): return self.n/(self.n+self.nh)
    def U(self):
        w=self.omega(); mb=np.array([self.VMB(i) for i in range(NF)]); return w*self.v+(1-w)*mb
    def experience(self, ev):
        _,cue,out,_=ev; z=zvec(cue,out); act=np.nonzero(z)[0]; dh=self.g*(M@out)
        err=dh-z@self.u                                           # vector-valued teacher (needs)
        U=self.U(); conc=np.abs(U[act]*z[act])+self.eps*np.abs(U).max()+1e-9; s=conc**self.rho/np.sum(conc**self.rho)
        for k,sk in zip(act,s): self.u[k]+=self.alpha*len(act)*sk*z[k]*err/(z[act]@z[act]+1e-9)
        self.h=np.clip(self.h+dh,0,2); self.hs=self.hs+self.eta*(self.h-self.hs)+self.rho_h*(self.hs0-self.hs); self.h=self.h*(1-self.lam)
        for k in act:
            self.n[k]+=1; a=max(1/(self.n[k]+1),self.amin); self.v[k]+=a*(self.VMB(k)-self.v[k])
    def probe(self): return self.U()

def metrics(cls):
    out={}
    D=[]
    for k in P.KINDS:
        c=run_persona(cls,k); b=c.probe()[PI].copy(); c.experience(P.NEW); D.append(c.probe()[PI]-b)
    D=np.array(D); cos=[D[i]@D[j]/(np.linalg.norm(D[i])*np.linalg.norm(D[j])+1e-12) for i in range(3) for j in range(i+1,3)]
    out['cos']=np.mean(cos); out['friendA']=D[:,4]; out['authorship']=D[:,1]
    pool=[]
    for k in P.KINDS: pool+=list(P.history(k))
    pool=[(f'p{i}',c,o,b) for i,(_,c,o,b) in enumerate(pool)]+[P.NEW]*3
    rng=np.random.default_rng(1); stream=[pool[i] for i in rng.integers(0,len(pool),size=128)]
    cs=[run_persona(cls,k) for k in P.KINDS]
    def dist(cs):
        ps=[c.probe()[PI] for c in cs]; return np.mean([np.linalg.norm(ps[i]-ps[j]) for i in range(3) for j in range(i+1,3)])
    d0=dist(cs); n=0; row=[]
    for m in (8,32,128):
        while n<m:
            for c in cs: c.experience(stream[n])
            n+=1
        row.append(dist(cs)/d0)
    out['retention']=row
    k_=run_persona(cls,'credit-taken'); r_=run_persona(cls,'rescued-by-collaboration'); tgt=r_.probe()[PI]; g0=np.linalg.norm(k_.probe()[PI]-tgt); hist=P.history('rescued-by-collaboration'); n=0; rr=[]
    for m in (32,128):
        while n<m: k_.experience(hist[n%len(hist)]); n+=1
        rr.append(1-np.linalg.norm(k_.probe()[PI]-tgt)/g0)
    out['reversal']=rr
    c=run_persona(cls,'credit-taken'); p0=c.probe()[PI].copy(); null=('null',vec(music=1),vec(music=1),None)
    for i in range(200): c.experience(null)
    p1=c.probe()[PI]; ex=[j for j,k in enumerate(PROBES) if k!='music']
    out['null_ratio']=np.linalg.norm(p1[ex])/max(np.linalg.norm(p0[ex]),1e-9)
    return out
for cls in (AMV_L1, AMV_L2):
    o=metrics(cls)
    print(cls.name,'cos %.2f'%o['cos'],'friendA delta',np.round(o['friendA'],4),'authorship delta',np.round(o['authorship'],4),'retention 8/32/128',np.round(o['retention'],2),'reversal 32/128',np.round(o['reversal'],2),'null ratio %.2f'%o['null_ratio'])
# belief-swap for L1: share of K-R difference in cache vs belief
K_=run_persona(AMV_L1,'credit-taken'); R_=run_persona(AMV_L1,'rescued-by-collaboration')
def Uc(cv,bl,i):
    w=cv.omega()[i]; return w*cv.v[i]+(1-w)*bl.beta[i]
for nm in ('authorship','connection'):
    i=IDX[nm]; UKK,UKR,URK,URR=Uc(K_,K_,i),Uc(K_,R_,i),Uc(R_,K_,i),Uc(R_,R_,i)
    b=.5*((UKK-UKR)+(URK-URR)); c=.5*((UKK-URK)+(UKR-URR))
    print('L1 belief-swap',nm,'diff %+.3f = belief %+.3f + cache %+.3f'%(UKK-URR,b,c))
