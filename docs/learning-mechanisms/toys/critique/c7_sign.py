"""C7: sign of 'value of A' in §6 under the doc's own rule: V = D(h;h*) - D(h+dh;h*), D=0.5*sum (h*-h)^2, dh = M(o) identical for all personas."""
import numpy as np
np.set_printoptions(precision=3, suppress=True)
needs=['rel','comp','aut','stim']
# SAME observed Delta h for all personas (M(o) is fixed and persona-independent): collab success, new technique, own ideas rejected
dh=np.array([+0.2,+0.2,-0.2,+0.2])
def V(g, dh):          # per-dimension drive reduction; g = h*-h (deficit if >0)
    return g*dh-0.5*dh**2
print("per-dimension value g*dh - dh^2/2 for dh=(rel+.2, comp+.2, aut-.2, stim+.2)")
for name,g in {"R (rel satisfied as in §6 table: g_rel=0)":[0,0.1,-0.1,0.0],
               "R' (rel deprived: g_rel=0.3)":[0.3,0.1,-0.1,0.0],
               "K (aut set point high, deficit 0.5)":[0.1,0.1,0.5,0.0],
               "K' (aut set point high but regulated: g_aut=0)":[0.1,0.1,0.0,0.0],
               "X (stim set point high, deficit .6)":[0.1,0.1,0.0,0.6]}.items():
    v=V(np.array(g,float),dh); print(f"  {name:48s} per-dim {v}, total {v.sum():+.3f}")
print("  -> for g_rel=0 (satisfied) any +dh_rel is a LOSS (-dh^2/2): R's '+' needs g_rel > dh_rel/2 = 0.1, i.e. 'deprived', contradicting '関係性は満たされている'")
print("  -> K' (set point high but at set point): same tiny loss as anyone: a high set point only matters through the deficit g=h*-h, which homeostatic control keeps ~lam*h*/k")
print("  -> since dh is identical across personas, §6 row 'Δh' (R: rel+,comp+ / K: rel+,aut- / X: stim+,rel+) is not what the frame computes; differences must come from g and from attribution r")
# attribution by part: r_A,part
r_R=np.array([1,1,0,0.]); r_K=np.array([1,0,1,0.]); r_X=np.array([1,0,0,1.])
for nm,g,r in (("R'",[0.3,0.1,-0.1,0.0],r_R),("K",[0.1,0.1,0.5,0.0],r_K),("X",[0.1,0.1,0.0,0.6],r_X)):
    v=V(np.array(g,float),dh); print(f"  V_A for {nm} with part-wise attribution to A r={r}: {float((r*v).sum()):+.3f}")
print("  with deficit-only D (max(h*-h,0)): satisfied persona gets 0, never positive; liking requires a prior deficit.")
