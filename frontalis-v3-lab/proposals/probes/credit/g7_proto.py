# Prototype: perceptual learning gated by credit. Besides reconstruction, each feature unit j also learns to make
# its own credited share predict the innate outcome: the error (alpha_j*o - c_j) is sent back into unit j's input
# weights, scaled by alpha_j (only units that currently receive credit refine). "You learn finer distinctions where you care."
import numpy as np
exec(open("g6_proto.py", encoding="utf-8").read().split("for beta in (0.0, 4.0):")[0])
LAMBDA = 0.02
def credit_refine(ag, x, o, eta=0.02):
    h = ag.feat(x); p = (h / ag.s > 0.5).astype(float)
    if p.sum() == 0: return
    z = np.where(p == 1, ag.beta * ag.c, -np.inf); a = np.exp(z - z[p == 1].max()); a /= a.sum()
    err = a * o - ag.c                                   # per-unit credit error
    ag.c += eta * p * err
    # refine: unit j's activation should rise when its credited outcome exceeds its value, fall otherwise
    g = LAMBDA * a * err * (h > 0)
    ag.W += np.outer(g, x); ag.b += g
Ag.credit = lambda self, x, o, eta=0.02: credit_refine(self, x, o, eta)
for beta in (0.0, 4.0):
    R = []
    for s in range(10):
        pa, Wd, ca = life2(beta, s, 0, True); pb, _, cb = life2(beta, s, 2, True)
        R.append([ca[0], cb[2], perc_discrim(pa, Wd, 0) - perc_discrim(pa, Wd, 2), perc_discrim(pb, Wd, 0) - perc_discrim(pb, Wd, 2),
                  val_discrim(pa, Wd, 0) - val_discrim(pa, Wd, 2), val_discrim(pb, Wd, 0) - val_discrim(pb, Wd, 2)])
    m = np.array(R)
    print(f"beta={beta}: Pa share opt0 {m[:,0].mean():.2f} Pb share opt2 {m[:,1].mean():.2f} | perc(0)-perc(2) Pa {m[:,2].mean():+.2f} Pb {m[:,3].mean():+.2f} (Pa>Pb {np.mean(m[:,2]>m[:,3]):.1f}) | "
          f"val(0)-val(2) Pa {m[:,4].mean():+.2f} Pb {m[:,5].mean():+.2f} (Pa>Pb {np.mean(m[:,4]>m[:,5]):.1f})")
