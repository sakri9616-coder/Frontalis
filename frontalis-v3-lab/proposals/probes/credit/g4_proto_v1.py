# Prototype (construction check) for G4: credit attribution over SELF-LEARNED features.
# Latent aspects a in {0,1}^K; the agent sees only raw x = a @ A + noise. Innate o = sum(a) (all aspects equally good).
# Features h = relu(W x + b) learned by a small autoencoder (reconstruct x) with an L1 sparsity penalty: unsupervised.
# Criterion c over features; credit alpha_j ∝ h_j * exp(beta c_j); c_j <- c_j + eta * h_j/max(h) * (alpha_j * o - c_j).
import numpy as np
K, D, H = 4, 16, 8
def make_world(rng): return rng.normal(0, 1, (K, D))
def sample(A, rng, lead=None, partner=None, q=0.5):
    a = (rng.random(K) < q).astype(float)
    if lead is not None:
        a[lead] = float(rng.random() < 0.8)
        if partner is not None: a[partner] = a[lead] if rng.random() < 0.8 else float(rng.random() < q)
    return a, a @ A + rng.normal(0, 0.1, D)
class Agent:
    def __init__(self, rng, beta, eta=0.02):
        self.W = rng.normal(0, 0.3, (H, D)); self.b = np.zeros(H); self.U = rng.normal(0, 0.3, (D, H))
        self.c = np.zeros(H); self.beta, self.eta = beta, eta
    def feat(self, x): return np.maximum(0, self.W @ x + self.b)
    def ae_step(self, x, lr=0.01, l1=0.01):
        h = self.feat(x); xr = self.U @ h; e = xr - x
        gU = np.outer(e, h); gh = self.U.T @ e + l1 * np.sign(h); gh *= (h > 0)
        self.U -= lr * gU; self.W -= lr * np.outer(gh, x); self.b -= lr * gh
    def credit(self, x, o):
        h = self.feat(x); m = h.max()
        if m <= 1e-6: return np.zeros(H)
        w = h / m; z = np.log(w + 1e-12) + self.beta * self.c; a = np.exp(z - z.max()); a /= a.sum()
        dc = self.eta * w * (a * o - self.c); self.c += dc; return dc
    def V(self, x): return self.c @ (self.feat(x) / max(self.feat(x).max(), 1e-6))
def life(beta, partner, seed, n_ae=3000, n_early=2000):
    rng = np.random.default_rng(seed); A = make_world(np.random.default_rng(999))   # same world for all
    ag = Agent(np.random.default_rng(seed + 1), beta)
    for _ in range(n_ae): ag.ae_step(sample(A, rng)[1])          # feature learning on neutral experience
    for _ in range(n_early):
        a, x = sample(A, rng, lead=1, partner=partner); ag.ae_step(x); ag.credit(x, a.sum() + rng.normal(0, .1))
    return ag, A
def same_new(beta, seed, n_new=100):
    out = []
    for partner in (2, 0):                     # Pa: aspect 2 comes with lead 1 ; Pb: aspect 0 comes with lead 1
        ag, A = life(beta, partner, seed); r = np.random.default_rng(seed + 77)
        xa = np.eye(K)[2] @ A; xb = np.eye(K)[0] @ A
        va0, vb0 = ag.V(xa), ag.V(xb)
        for _ in range(n_new):
            a = np.zeros(K); a[[0, 2]] = 1; x = a @ A + r.normal(0, .1, D); ag.credit(x, 2 + r.normal(0, .1))
        out.append(np.sign((ag.V(xb) - vb0) - (ag.V(xa) - va0)))   # >0: the new events taught "aspect 0" more
    return out
for beta in (0, 2, 4):
    res = [same_new(beta, s) for s in range(12)]
    print(f"beta={beta}: opposite-sign pairs {sum(a != b for a, b in res)}/12  (Pa,Pb signs: {[(int(a), int(b)) for a, b in res[:6]]})")
