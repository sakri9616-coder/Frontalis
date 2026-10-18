"""C5: does G1 (sign and rank invariant to 2x / 1/2 of learning rate or decay) detect the A-rule pathology? (c=1 case, doc §1)"""
import numpy as np
def vstar(i, d, a1=0.35, T=400000):
    v=0.0
    for t in range(T):
        v += a1*(np.tanh(i+v)-np.tanh(v)) - d*v
    return v
I=(0.05,0.2,0.5,1.0)
print("fixed point v*(i) of the (self-referential) rule for personas with innate value i, and for decay d, 2d, d/2")
res={}
for d in (0.001,0.002,0.004):
    res[d]=[vstar(i,d) for i in I]; print(f"  d={d}: v* for i={I}: {np.round(res[d],2)}")
ranks={d:tuple(np.argsort(np.argsort(res[d]))) for d in res}
print("  ranks across personas:",ranks, "-> rank and sign identical under d/2, d, 2d: G1 as written in doc (sign & rank) PASSES")
print("  but magnitude moves:", [round(res[0.004][k]/res[0.001][k],2) for k in range(4)] , "(ratio v*(4d)/v*(d)... i.e. size is not experience-determined)")
# NV3: cache devaluation retention is a coefficient: exp(-a_min m)
for amin in (0.01,0.02,0.04):
    print(f"  NV3 cache retention after m=20 / 100 / 300 post-removal updates, a_min={amin}: {[round((1-amin)**m,2) for m in (20,100,300)]}")
