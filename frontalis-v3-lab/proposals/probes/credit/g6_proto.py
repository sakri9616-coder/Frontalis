# Prototype: evaluation -> choice -> exposure -> perception -> evaluation (the philosophy's cycle).
# Options k = 0..K-1: aspect k present w.p. 0.9, others 0.3; variants random; innate o as in g5 (variant good/bad, DELTA).
# Agent: features (autoencoder, plain weighting, keeps learning on what it experiences), credit criterion c over perceived
# features, prediction model M[k] = running mean of perceived features after choosing option k. Choice: softmax(tau * c . M[k]).
# Early nudge: for the first N_NUDGE steps the partner picks the option (Pa: option 0, Pb: option 2). Afterwards the agent chooses.
import numpy as np
exec(open("g5_proto2.py", encoding="utf-8").read().split("def life(")[0])
TAU, N_NUDGE, N_FREE = 3.0, 400, 6000
def opt_freq(k): f = np.full(K, .3); f[k] = .9; return f
def life2(beta, seed, nudge, free_choice=True):
    Wd = world(seed); ag = Ag(beta, 0.0, seed); r = np.random.default_rng([seed, 1])
    for _ in range(2000): ag.ae(sample(Wd, r, np.full(K, .5))[2])
    ag.cal([sample(Wd, r, np.full(K, .5))[2] for _ in range(300)])
    M = np.full((K, H), 0.5); rc = np.random.default_rng([seed, 7]); re = np.random.default_rng([seed, 3]); counts = np.zeros(K)
    for t in range(N_NUDGE + N_FREE):
        if t < N_NUDGE: k = nudge
        elif free_choice:
            q = TAU * (M @ ag.c); p = np.exp(q - q.max()); p /= p.sum(); k = rc.choice(K, p=p)
        else: k = rc.integers(K)
        a, v, x, o = sample(Wd, re, opt_freq(k)); ag.credit(x, o); ag.ae(x); M[k] += 0.05 * (ag.pres(x) - M[k]); counts[k] += 1
        if t % 500 == 499: ag.cal([sample(Wd, r, np.full(K, .5))[2] for _ in range(300)])
    return ag, Wd, counts / counts.sum()
def perc_discrim(ag, Wd, k):
    A, B = Wd; return float(np.abs(ag.pres(A[k] + B[k]) - ag.pres(A[k] - B[k])).sum())
def val_discrim(ag, Wd, k):
    A, B = Wd; return ag.V(A[k] + B[k]) - ag.V(A[k] - B[k])
for beta in (0.0, 4.0):
    for free in (True, False):
        R = []
        for s in range(10):
            pa, Wd, ca = life2(beta, s, 0, free); pb, _, cb = life2(beta, s, 2, free)
            R.append([ca[0], ca[2], cb[0], cb[2], perc_discrim(pa, Wd, 0) - perc_discrim(pa, Wd, 2), perc_discrim(pb, Wd, 0) - perc_discrim(pb, Wd, 2),
                      val_discrim(pa, Wd, 0) - val_discrim(pa, Wd, 2), val_discrim(pb, Wd, 0) - val_discrim(pb, Wd, 2)])
        m = np.array(R); print(f"beta={beta} free={free}: share(opt0,opt2) Pa {m[:,0].mean():.2f},{m[:,1].mean():.2f} Pb {m[:,2].mean():.2f},{m[:,3].mean():.2f} | "
              f"perc(0)-perc(2) Pa {m[:,4].mean():+.2f} Pb {m[:,5].mean():+.2f} (Pa>Pb in {np.mean(m[:,4]>m[:,5]):.1f}) | val(0)-val(2) Pa {m[:,6].mean():+.2f} Pb {m[:,7].mean():+.2f}")
