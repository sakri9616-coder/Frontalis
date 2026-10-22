# mean-field fixed points of g on one cue dim: g = c + lam*tanh(g)  (c = E[o|z] slope in late world)
import numpy as np
for lam in (0.5,1.0,1.5,2.0):
    for c in (0.0,-0.2,-0.5,-1.0):
        gs=np.linspace(-6,6,240001); f=c+lam*np.tanh(gs)-gs
        roots=gs[np.where(np.diff(np.sign(f))!=0)[0]]
        # stability: d/dg (c+lam tanh g - g) <0
        st=[round(r,2) for r in roots if lam*(1-np.tanh(r)**2)-1<0]
        print(f"lam={lam} c={c}: stable g*={st}")
# flip threshold for lam=1.5: largest |c| with 2 stable points
for lam in (1.5,2.0):
    gstar=np.arccosh(np.sqrt(lam)); cth=gstar-lam*np.tanh(gstar)
    print(f"lam={lam}: bistable iff |c| < {abs(cth):.3f}")
