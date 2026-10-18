"""C8: simpler alternative = exogenous RW teacher with UNIFORM split (rho=0) + one explicit bounded bistable commitment per concept.
Compare on the doc's own toy harness (cos, retention, reversal, null ratio) with AMV-L1/L2 (sim5)."""
import sys, io, contextlib, os, numpy as np
sys.dont_write_bytecode=True
V=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'value'))
sys.path.insert(0,V); os.chdir(V)
with contextlib.redirect_stdout(io.StringIO()):
    import sim5_amv as S
import prototype as P
from sim1_candidates import *
np.set_printoptions(precision=3, suppress=True)

class RW:      # A' with rho=0: exogenous teacher, uniform split of credit
    name="RW(rho=0)"
    def __init__(self, alpha=0.3): self.e=np.zeros(NF); self.alpha=alpha
    def experience(self, ev):
        _,cue,out,_=ev; z=zvec(cue,out); act=np.nonzero(z)[0]
        err=r_innate(out)-self.e@z; self.e[act]+=self.alpha*err*z[act]/(z[act]@z[act]+1e-9)
    def probe(self): return self.e.copy()

class RWC(RW):  # + explicit bounded bistable commitment driven ONLY by the exogenous estimate
    name="RW+commit"
    def __init__(self, alpha=0.3, g=1.5, k=1.0, ac=0.2, w=1.0):
        super().__init__(alpha); self.c=np.zeros(NF); self.g,self.k,self.ac,self.w=g,k,ac,w
    def experience(self, ev):
        super().experience(ev); _,cue,out,_=ev; act=np.nonzero(zvec(cue,out))[0]
        for x in act: self.c[x]+=self.ac*(-self.c[x]+self.g*np.tanh(self.c[x])+self.k*self.e[x])
    def probe(self): return self.e+self.w*self.c

def metrics(cls):
    out={}
    D=[]
    for k in P.KINDS:
        c=run_persona(cls,k); b=c.probe()[PI].copy(); c.experience(P.NEW); D.append(c.probe()[PI]-b)
    D=np.array(D); cos=[D[i]@D[j]/(np.linalg.norm(D[i])*np.linalg.norm(D[j])+1e-12) for i in range(3) for j in range(i+1,3)]
    out['cos']=np.mean(cos)
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
    out['retention']=np.round(row,2)
    k_=run_persona(cls,'credit-taken'); r_=run_persona(cls,'rescued-by-collaboration'); tgt=r_.probe()[PI]; g0=np.linalg.norm(k_.probe()[PI]-tgt); hist=P.history('rescued-by-collaboration'); n=0; rr=[]
    for m in (32,128):
        while n<m: k_.experience(hist[n%len(hist)]); n+=1
        rr.append(1-np.linalg.norm(k_.probe()[PI]-tgt)/g0)
    out['reversal']=np.round(rr,2)
    c=run_persona(cls,'credit-taken'); p0=c.probe()[PI].copy(); null=('null',vec(music=1),vec(music=1),None)
    for i in range(200): c.experience(null)
    p1=c.probe()[PI]; ex=[j for j,k in enumerate(PROBES) if k!='music']
    out['null_ratio']=round(float(np.linalg.norm(p1[ex])/max(np.linalg.norm(p0[ex]),1e-9)),2)
    return out
rows=[S.AMV_L1,S.AMV_L2,RW,RWC]
for cls in rows:
    o=metrics(cls); print(f"{cls.name:10s} cos={o['cos']:.2f} retention8/32/128={o['retention']} reversal32/128={o['reversal']} null_ratio={o['null_ratio']}")
print("-- sweep of the commitment gain g (explicit, closed-form reversal threshold u_c = sqrt(g(g-1)) - acosh(sqrt g))")
for g in (1.2,1.5,2.0,2.5):
    uc=np.sqrt(g*(g-1))-np.arccosh(np.sqrt(g)); 
    class X(RWC):
        def __init__(self): super().__init__(g=g)
    X.name=f"commit g={g}"
    o=metrics(X); print(f"   g={g}: u_c={uc:.3f} cos={o['cos']:.2f} retention={o['retention']} reversal={o['reversal']} null={o['null_ratio']}")
# G1 test for both: scale learning-rate (ac) by 2 and 1/2; compare sign / rank of probe vector for credit-taken persona
for nm,mk in (("commit ac x0.5",lambda:RWC(ac=0.1)),("commit ac x1",lambda:RWC()),("commit ac x2",lambda:RWC(ac=0.4))):
    c=run_persona(lambda:mk(),'credit-taken'); print(f"   G1 {nm}: probe={np.round(c.probe()[PI],2)}")
