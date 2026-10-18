"""C10: loop gain of the doc's own L1 toy: spectral radius of d(state_{t+1})/d(state_t) over the history events (finite differences).
G4 (one-step static graph) says 0; the unrolled Jacobian is what decides self-amplification."""
import sys, io, contextlib, os, copy, numpy as np
sys.dont_write_bytecode=True
V=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'value'))
sys.path.insert(0,V); os.chdir(V)
with contextlib.redirect_stdout(io.StringIO()):
    import sim5_amv as S
import prototype as P
from sim1_candidates import run_persona, NF, zvec
np.set_printoptions(precision=3, suppress=True)
def state(c): return np.concatenate([c.beta, c.v])
def setstate(c, x): c.beta=x[:NF].copy(); c.v=x[NF:].copy()
def step(c0, x, ev):
    c=copy.deepcopy(c0); setstate(c,x); c.experience(ev); return state(c)
def jac(c0, ev, h=1e-6):
    x0=state(c0); f0=step(c0,x0,ev); J=np.zeros((len(x0),len(x0)))
    for i in range(len(x0)):
        x=x0.copy(); x[i]+=h; J[:,i]=(step(c0,x,ev)-f0)/h
    return J
for kind in P.KINDS:
    c=run_persona(S.AMV_L1,kind); hist=P.history(kind)+[P.NEW]
    srs=[]
    for ev in hist:
        J=jac(c,ev); srs.append(max(abs(np.linalg.eigvals(J))))
    print(f"L1 {kind:26s} spectral radius over {len(hist)} events: max={max(srs):.3f} mean={np.mean(srs):.3f}  (#events with radius>1.0001: {sum(s>1.0001 for s in srs)})")
# same on freshly started persona (learning phase) 
c=S.AMV_L1(); srs=[]
for ev in P.history('credit-taken'):
    J=jac(c,ev); srs.append(max(abs(np.linalg.eigvals(J)))); c.experience(ev)
print("L1 learning-phase (credit-taken history, from scratch): spectral radius per event:", np.round(srs,3))
print("--- same with allocation exponent rho=0 (uniform split) and rho=1")
for rho in (0.0,1.0):
    class X(S.AMV_L1):
        def __init__(self): super().__init__(rho=rho)
    mx=[]; cnt=0; tot=0
    for kind in P.KINDS:
        c=run_persona(X,kind)
        for ev in P.history(kind)+[P.NEW]:
            r=max(abs(np.linalg.eigvals(jac(c,ev)))); mx.append(r); cnt+= r>1.0001; tot+=1
    print(f"  rho={rho}: max radius {max(mx):.3f}, events with radius>1.0001: {cnt}/{tot}")
