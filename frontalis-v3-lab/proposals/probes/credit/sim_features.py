"""Credit attribution over self-learned features (DESIGN-features, v11 F-G3a).

World: hidden aspects a in {0,1}^K, raw x = a A + noise (A random per world seed); innate o = sum(a).
Features: ReLU autoencoder (H units, L1) trained on neutral experience only, then frozen.
Perception: feature j present iff h_j > 0.5 * (90th percentile of h_j on neutral experience).
Criterion: the credit rule of sim_credit.Agent over perceived features.
Streams: world [ws,0]; neutral/features [fs,1]; calibration [fs,2]; early [es,3]; new events [ws,4].
"""
import numpy as np

K, D, H = 4, 16, 8
N_AE, L1, LR_AE, N_CAL = 3000, 0.01, 0.01, 500
N_EARLY, ETA, N_NEW = 2000, 0.02, 100


def world(ws):
    return np.random.default_rng([ws, 0]).normal(0, 1, (K, D))


def sample(A, rng, lead=None, partner=None, q=0.5):
    a = (rng.random(K) < q).astype(float)
    if lead is not None:
        a[lead] = float(rng.random() < 0.8)
        if partner is not None:
            a[partner] = a[lead] if rng.random() < 0.8 else float(rng.random() < q)
    return a, a @ A + rng.normal(0, 0.1, D)


class FeatureAgent:
    def __init__(self, beta, fs):
        r = np.random.default_rng([fs, 5])
        self.W, self.b, self.U = r.normal(0, 0.3, (H, D)), np.zeros(H), r.normal(0, 0.3, (D, H))
        self.c, self.beta = np.zeros(H), beta

    def feat(self, x):
        return np.maximum(0, self.W @ x + self.b)

    def ae_step(self, x):
        h = self.feat(x); e = self.U @ h - x
        gh = (self.U.T @ e + L1 * np.sign(h)) * (h > 0)
        self.U -= LR_AE * np.outer(e, h); self.W -= LR_AE * np.outer(gh, x); self.b -= LR_AE * gh

    def calibrate(self, xs):
        self.s = np.maximum(np.percentile([self.feat(x) for x in xs], 90, axis=0), 1e-6)

    def pres(self, x):
        return (self.feat(x) / self.s > 0.5).astype(float)

    def credit(self, x, o):
        p = self.pres(x)
        if p.sum() == 0:
            return
        z = np.where(p == 1, self.beta * self.c, -np.inf); a = np.exp(z - z[p == 1].max()); a /= a.sum()
        self.c += ETA * p * (a * o - self.c)

    def V(self, x):
        return float(self.c @ self.pres(x))


def grow(beta, A, fs, es, partner, lead=1):
    ag = FeatureAgent(beta, fs)
    r1, r2, r3 = (np.random.default_rng([fs, 1]), np.random.default_rng([fs, 2]), np.random.default_rng([es, 3]))
    for _ in range(N_AE):
        ag.ae_step(sample(A, r1)[1])
    ag.calibrate([sample(A, r2)[1] for _ in range(N_CAL)])
    for _ in range(N_EARLY):
        a, x = sample(A, r3, lead=lead, partner=partner)
        ag.credit(x, a.sum() + r3.normal(0, 0.1))
    return ag


def new_events(ag, A, ws):
    """Same new events (aspects 0 and 2 together, o = 2); returns d = dV(aspect 0 alone) - dV(aspect 2 alone)."""
    x0, x2 = np.eye(K)[0] @ A, np.eye(K)[2] @ A          # noiseless probes (fixed by design)
    v0, v2 = ag.V(x0), ag.V(x2)
    r = np.random.default_rng([ws, 4])
    for _ in range(N_NEW):
        a = np.zeros(K); a[[0, 2]] = 1
        ag.credit(a @ A + r.normal(0, 0.1, D), 2 + r.normal(0, 0.1))
    return (ag.V(x0) - v0) - (ag.V(x2) - v2)


def f_g3a(beta, seed):
    """Pa (aspect 2 with the lead) vs Pb (aspect 0 with the lead): same world, same features, same early stream."""
    A = world(seed)
    da = new_events(grow(beta, A, seed, seed, partner=2), A, seed)
    db = new_events(grow(beta, A, seed, seed, partner=0), A, seed)
    return da, db


def f_g3a_retest(beta, seed):
    """Same early world (Pa) twice: same world and features, different early streams."""
    A = world(seed)
    d1 = new_events(grow(beta, A, seed, seed, partner=2), A, seed)
    d2 = new_events(grow(beta, A, seed, seed + 10**6, partner=2), A, seed)
    return d1, d2


def f_align(seed):
    A = world(seed); ag = FeatureAgent(0, seed)
    r1, r2 = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
    for _ in range(N_AE):
        ag.ae_step(sample(A, r1)[1])
    S = [sample(A, r2) for _ in range(2000)]
    a = np.array([s[0] for s in S]); h = np.array([ag.feat(s[1]) for s in S])
    C = np.array([[np.corrcoef(h[:, j], a[:, k])[0, 1] if h[:, j].std() > 1e-9 else 0.0 for k in range(K)] for j in range(H)])
    return np.nan_to_num(C).max(0)
