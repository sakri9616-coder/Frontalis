"""C13: minimal fix test: precision-weighted (Kalman) credit allocation instead of concern-weighted s ∝ (|U|+eps)^rho."""
import numpy as np
def kalman(events, u0, var0, r_obs=0.05, q=0.0):
    u=np.array(u0,float); S=np.diag(var0).astype(float)
    for act,y in events:
        z=np.zeros(len(u)); z[act]=1.0
        Sz=S@z; k=Sz/(z@Sz+r_obs); u=u+k*(y-u@z); S=S-np.outer(k,Sz)+q*np.eye(len(u))
    return u, np.diag(S)
rng=np.random.default_rng(0)
print("music scenario (friend established u=0.5 var 0.02; music new var 1.0). together->0.8, alone->0.5")
for p_alone in (0.0,0.3):
    ev=[(( [0] if rng.random()<p_alone else [0,1]), (0.5 if False else 0)) for _ in range(0)]
    evs=[]
    for t in range(300):
        if rng.random()<p_alone: evs.append(([0],0.5))
        else: evs.append(([0,1],0.8))
    u,v=kalman(evs,[0.5,0.0],[0.02,1.0]); print(f"  p_alone={p_alone}: friend={u[0]:.2f} music={u[1]:.2f}  (true 0.5 / 0.3 if identifiable; if never alone: sum=0.8 fixed, split by prior variance)")
print("exact redundancy, learn from scratch, prior variance ratio 1.01:1 (the 'tiny asymmetry'): split of a total of 1.0")
for ratio in (1.0,1.01,1.5,3.0):
    evs=[([0,1],1.0)]*200
    u,_=kalman(evs,[0,0],[ratio,1.0]); print(f"  prior var ratio {ratio}: u=({u[0]:.2f},{u[1]:.2f})  share of unit1 = {u[0]/u.sum():.2f}")
print("noise-level perturbation of the prior variance (1e-3): spread of share over 30 draws:")
sh=[]
for s in range(30):
    r=np.random.default_rng(s); v1=1.0+1e-3*r.standard_normal(); u,_=kalman([([0,1],1.0)]*200,[0,0],[v1,1.0]); sh.append(u[0]/u.sum())
print("  share range:",round(min(sh),3),round(max(sh),3), "(concern-weighted rho=3 gave 0.31-0.65 for 1e-3 perturbations)")
