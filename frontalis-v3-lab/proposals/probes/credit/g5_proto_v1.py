# Prototype: evaluation shapes perception. Limited feature capacity (H=6) for 4 aspects x 2 variants.
# Innate: aspect k present -> o_k = 1 + delta*(2v-1) (good / bad variant). Raw x = sum a_k (A_k + (2v_k-1) B_k) + noise.
# Feature learning: autoencoder whose per-sample reconstruction weight is w(x) = 1 + kappa * max(V(x), 0) (valued experience is rehearsed more).
# Criterion: credit rule over perceived features (as v11). Early: Pa meets aspect 0 often, Pb meets aspect 2 often.
import numpy as np
K, D, H, DELTA = 4, 16, 6, 0.5
def world(ws):
    r = np.random.default_rng([ws, 0]); return r.normal(0, 1, (K, D)), r.normal(0, 0.7, (K, D))
def sample(W, rng, freq):
    A, B = W; a = (rng.random(K) < freq).astype(float); v = (rng.random(K) < 0.5).astype(float)
    x = (a[:, None] * (A + (2 * v - 1)[:, None] * B)).sum(0) + rng.normal(0, 0.1, D)
    o = (a * (1 + DELTA * (2 * v - 1))).sum() + rng.normal(0, 0.1)
    return a, v, x, o
class Ag:
    def __init__(self, beta, kappa, fs):
        r = np.random.default_rng([fs, 5]); self.W = r.normal(0, .3, (H, D)); self.b = np.zeros(H); self.U = r.normal(0, .3, (D, H))
        self.c = np.zeros(H); self.beta, self.kappa = beta, kappa; self.s = np.ones(H)
    def feat(self, x): return np.maximum(0, self.W @ x + self.b)
    def pres(self, x): return (self.feat(x) / self.s > 0.5).astype(float)
    def V(self, x): return float(self.c @ self.pres(x))
    def ae(self, x, lr=0.01, l1=0.01):
        wgt = 1 + self.kappa * max(self.V(x), 0)
        h = self.feat(x); e = self.U @ h - x; gh = (self.U.T @ e + l1 * np.sign(h)) * (h > 0)
        self.U -= lr * wgt * np.outer(e, h); self.W -= lr * wgt * np.outer(gh, x); self.b -= lr * wgt * gh
    def cal(self, xs): self.s = np.maximum(np.percentile([self.feat(x) for x in xs], 90, axis=0), 1e-3)
    def credit(self, x, o, eta=0.02):
        p = self.pres(x)
        if p.sum() == 0: return
        z = np.where(p == 1, self.beta * self.c, -np.inf); a = np.exp(z - z[p == 1].max()); a /= a.sum()
        self.c += eta * p * (a * o - self.c)
def life(beta, kappa, seed, focus, n_neutral=2000, n_early=6000):
    Wd = world(seed); ag = Ag(beta, kappa, seed); r = np.random.default_rng([seed, 1])
    for _ in range(n_neutral): ag.ae(sample(Wd, r, np.full(K, .5))[2])
    ag.cal([sample(Wd, r, np.full(K, .5))[2] for _ in range(300)])
    freq = np.full(K, .3); freq[focus] = .9
    re = np.random.default_rng([seed, 3])
    for t in range(n_early):
        a, v, x, o = sample(Wd, re, freq); ag.credit(x, o); ag.ae(x)
        if t % 500 == 499: ag.cal([sample(Wd, r, np.full(K, .5))[2] for _ in range(300)])
    return ag, Wd
def discrim(ag, Wd, k):
    """How differently the agent VALUES the good and bad variant of aspect k (alone)."""
    A, B = Wd; xg = A[k] + B[k]; xb = A[k] - B[k]
    return ag.V(xg) - ag.V(xb), float(np.abs(ag.feat(xg) / ag.s - ag.feat(xb) / ag.s).sum())
for kappa in (0.0, 2.0):
    for beta in (0.0, 4.0):
        rows = []
        for s in range(10):
            pa, Wd = life(beta, kappa, s, 0); pb, _ = life(beta, kappa, s, 2)
            rows.append([*discrim(pa, Wd, 0), *discrim(pb, Wd, 0), *discrim(pa, Wd, 2), *discrim(pb, Wd, 2)])
        m = np.array(rows).mean(0).round(2)
        print(f"kappa={kappa} beta={beta}: aspect0 dV/featdiff Pa {m[0]},{m[1]} Pb {m[2]},{m[3]} | aspect2 Pa {m[4]},{m[5]} Pb {m[6]},{m[7]}")
