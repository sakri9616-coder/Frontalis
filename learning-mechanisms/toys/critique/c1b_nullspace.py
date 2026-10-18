import numpy as np
np.set_printoptions(precision=3, suppress=True)
def run_red(rho, u0, noise=0.0, mu=1.0, T=1500, alpha=0.1, eps=0.05, seed=0):
    rng=np.random.default_rng(seed); u=np.array(u0,float)
    for t in range(T):
        r=mu+noise*rng.standard_normal(); err=r-u.sum()
        c=np.abs(u)+eps; s=c**rho/np.sum(c**rho); u=u+alpha*s*err
    return u
print("(a') exact redundancy, learn from scratch: final (u1,u2) vs rho and the tiny initial asymmetry (u1_0,u2_0)")
for rho in (0.0,0.5,1.0,2.0,3.0):
    rows=[]
    for u0 in ((0.01,0.0),(0.0,0.01),(0.1,0.0),(0.3,0.0)):
        u=run_red(rho,u0); rows.append(u)
    print(f"  rho={rho}: ", " ".join(f"{u0}->({u[0]:.2f},{u[1]:.2f})" for u0,u in zip(((0.01,0),(0,0.01),(0.1,0),(0.3,0)),rows)))
print("   -> sum fixed to 1 by the teacher, but the split is set by rho and the initial seed asymmetry (G1 violation); rho=0 -> split equals initial split")
print("(a'') same with noisy teacher sd=0.3: share of u1 over 30 seeds, start (0.01,0)")
for rho in (0.0,3.0):
    sh=[ (lambda u:u[0]/u.sum())(run_red(rho,(0.01,0.0),noise=0.3,seed=s)) for s in range(30)]
    print(f"  rho={rho}: mean share {np.mean(sh):.2f}, sd {np.std(sh):.2f}")
print("(a''') start symmetric (0,0)+ eps-level noise in the first events (rephrasing / tag noise): winner flips?")
for rho in (0.0,3.0):
    sh=[]
    for s in range(30):
        rng=np.random.default_rng(s); u0=(1e-3*rng.standard_normal(),1e-3*rng.standard_normal()); sh.append(run_red(rho,u0)[0]/1.0)
    print(f"  rho={rho}: share of u1, 30 random 1e-3 initial perturbations: {np.round(np.sort(sh),2)}")

def run_ov(rho, y1, A0=0.05, B0=0.0, T=8000, alpha=0.05, eps=0.02, p=(0.5,0.25,0.25), y2=0.3, seed=0):
    rng=np.random.default_rng(seed); u=np.array([A0,B0],float)
    ev=[((0,1),y1),((0,),y2),((1,),y2)]
    for t in range(T):
        k=rng.choice(3,p=p); act,y=ev[k]; act=list(act)
        err=y-u[act].sum(); c=np.abs(u[act])+eps; s=c**rho/np.sum(c**rho); u[act]+=alpha*s*err
    return u
print("(d') partially overlapping non-realizable design, p=(.5,.25,.25): events {A,B}->y1, {A}->0.3, {B}->0.3")
for y1 in (1.0,2.0,4.0):
    for rho in (0.0,3.0):
        a=run_ov(rho,y1,0.05,0.0); b=run_ov(rho,y1,0.0,0.05)
        print(f"  y1={y1} rho={rho}: init(A rich)->({a[0]:.2f},{a[1]:.2f})  init(B rich)->({b[0]:.2f},{b[1]:.2f})")
print("(d'') rarer disambiguation p=(.9,.05,.05)")
for y1 in (2.0,4.0):
    for rho in (0.0,3.0):
        a=run_ov(rho,y1,0.05,0.0,p=(0.9,0.05,0.05),T=20000); b=run_ov(rho,y1,0.0,0.05,p=(0.9,0.05,0.05),T=20000)
        print(f"  y1={y1} rho={rho}: init(A rich)->({a[0]:.2f},{a[1]:.2f})  init(B rich)->({b[0]:.2f},{b[1]:.2f})")
print("(f) one single-feature event {B} erases the null-space individuality of a perfectly redundant pair?")
for rho in (0.0,3.0):
    u=run_red(rho,(0.01,0.0))   # A took everything
    print(f"  rho={rho}: after redundant phase u=({u[0]:.2f},{u[1]:.2f})")
    for n in range(1,6):
        # event {B} alone with target 0.4 (B's marginal effect)
        err=0.4-u[1]; u[1]+=0.3*err   # s=1 for single-active event
        # then redundant events again resume
        if n in (1,3,5): print(f"    after {n} single-feature {{B}} events (alpha=.3): u=({u[0]:.2f},{u[1]:.2f}); sum={u.sum():.2f}")
