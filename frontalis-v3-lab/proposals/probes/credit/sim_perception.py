"""Evaluation -> choice -> exposure -> finer perception -> evaluation (DESIGN-perception, v12).

World (per seed): K aspects, each with a good / bad variant. x = sum a_k (A_k + (2v_k-1) B_k) + noise;
innate o = sum a_k (1 + 0.5 (2v_k-1)) + N(0, 0.1). Option k: aspect k present w.p. 0.9, others 0.3.
Agent: autoencoder features (plain reconstruction), 0/1 perception (> 0.5 x 90th percentile on neutral experience,
recalibrated every 500 steps with 300 samples), credit criterion c, prediction model M[k] (init 0.5, rate 0.05).
Choice: softmax(tau * c . M[k]).
Streams: world [s,0]; neutral & calibration [s,1]; init weights [s,5]; choice [s,7]; experience [s,3]; readout [s,11,k].
"""
import numpy as np

K, D, H, DELTA = 4, 16, 6, 0.5
WORLDS = {"main": (0.1, 0.5), "sub": (0.15, 0.3)}       # (B scale, raw noise)
TAU, N_NUDGE, N_FREE, N_NEUTRAL, ETA = 3.0, 400, 6000, 2000, 0.02
LR_AE, L1, CAL_EVERY, CAL_N, M_INIT, M_RATE, N_READ = 0.01, 0.01, 500, 300, 0.5, 0.05, 600


class World:
    def __init__(self, s, kind):
        b, self.noise = WORLDS[kind]
        r = np.random.default_rng([s, 0])
        self.A, self.B = r.normal(0, 1, (K, D)), r.normal(0, b, (K, D))

    def sample(self, rng, freq):
        a = (rng.random(K) < freq).astype(float)
        v = (rng.random(K) < 0.5).astype(float)
        x = (a[:, None] * (self.A + (2 * v - 1)[:, None] * self.B)).sum(0) + rng.normal(0, self.noise, D)
        return a, v, x, (a * (1 + DELTA * (2 * v - 1))).sum() + rng.normal(0, 0.1)


def opt_freq(k):
    f = np.full(K, 0.3)
    f[k] = 0.9
    return f


class Agent:
    def __init__(self, beta, s):
        r = np.random.default_rng([s, 5])
        self.W, self.b, self.U = r.normal(0, .3, (H, D)), np.zeros(H), r.normal(0, .3, (D, H))
        self.c, self.beta, self.s = np.zeros(H), beta, np.ones(H)

    def feat(self, x):
        return np.maximum(0, self.W @ x + self.b)

    def pres(self, x):
        return (self.feat(x) / self.s > 0.5).astype(float)

    def V(self, x):
        return float(self.c @ self.pres(x))

    def ae(self, x):
        h = self.feat(x)
        e = self.U @ h - x
        gh = (self.U.T @ e + L1 * np.sign(h)) * (h > 0)
        self.U -= LR_AE * np.outer(e, h)
        self.W -= LR_AE * np.outer(gh, x)
        self.b -= LR_AE * gh

    def cal(self, xs):
        self.s = np.maximum(np.percentile([self.feat(x) for x in xs], 90, axis=0), 1e-3)

    def credit(self, x, o):
        p = self.pres(x)
        if p.sum() == 0:
            return
        z = np.where(p == 1, self.beta * self.c, -np.inf)
        a = np.exp(z - z[p == 1].max())
        a /= a.sum()
        self.c += ETA * p * (a * o - self.c)


def readout(ag, w, k, s):
    """Held-out accuracy of a linear readout of aspect k's variant from the features (others at 0.3, with variants)."""
    r = np.random.default_rng([s, 11, k])
    X, y = [], []
    f = np.full(K, 0.3)
    f[k] = 1.0
    for _ in range(N_READ):
        a, v, x, _ = w.sample(r, f)
        X.append(np.r_[ag.feat(x), 1])
        y.append(2 * v[k] - 1)
    X, y = np.array(X), np.array(y)
    h = N_READ // 2
    wt = np.linalg.lstsq(X[:h], y[:h], rcond=None)[0]
    return float(np.mean(np.sign(X[h:] @ wt) == y[h:]))


def life(beta, s, kind, nudge):
    w = World(s, kind)
    ag = Agent(beta, s)
    rn, rc, re = np.random.default_rng([s, 1]), np.random.default_rng([s, 7]), np.random.default_rng([s, 3])
    neutral = np.full(K, 0.5)
    for _ in range(N_NEUTRAL):
        ag.ae(w.sample(rn, neutral)[2])
    ag.cal([w.sample(rn, neutral)[2] for _ in range(CAL_N)])
    M = np.full((K, H), M_INIT)
    picks, mid = [], None
    for t in range(N_NUDGE + N_FREE):
        if t < N_NUDGE:
            k = nudge
        else:
            q = TAU * (M @ ag.c)
            p = np.exp(q - q.max())
            p /= p.sum()
            k = int(rc.choice(K, p=p))
        a, v, x, o = w.sample(re, opt_freq(k))
        ag.credit(x, o)
        ag.ae(x)
        M[k] += M_RATE * (ag.pres(x) - M[k])
        picks.append(k)
        if t % CAL_EVERY == CAL_EVERY - 1:
            ag.cal([w.sample(rn, neutral)[2] for _ in range(CAL_N)])
        if t == N_NUDGE + N_FREE // 2 - 1:
            mid = (readout(ag, w, 0, s), readout(ag, w, 2, s))
    return ag, w, np.array(picks[N_NUDGE:]), mid


def vdiff(ag, w, k):
    """Value gap between the good and the bad variant of aspect k (alone, noiseless)."""
    return ag.V(w.A[k] + w.B[k]) - ag.V(w.A[k] - w.B[k])


def pair(beta, s, kind):
    pa, w, ka, ma = life(beta, s, kind, 0)
    pb, _, kb, mb = life(beta, s, kind, 2)
    ra = (readout(pa, w, 0, s), readout(pa, w, 2, s))
    rb = (readout(pb, w, 0, s), readout(pb, w, 2, s))
    swap_a, swap_b = Agent(beta, s), Agent(beta, s)          # Pa's features with Pb's c, and the reverse
    for dst, feat_src, c_src in ((swap_a, pa, pb), (swap_b, pb, pa)):
        dst.W, dst.b, dst.s, dst.c = feat_src.W, feat_src.b, feat_src.s, c_src.c
    h = N_FREE // 2
    return {
        "delta": (ra[0] - rb[0]) + (rb[1] - ra[1]),
        "delta_mid": (ma[0] - mb[0]) + (mb[1] - ma[1]),
        "p2": (vdiff(pa, w, 0) - vdiff(pb, w, 0)) + (vdiff(pb, w, 2) - vdiff(pa, w, 2)),
        "p2_featPa_cPb": vdiff(swap_a, w, 0) - vdiff(swap_a, w, 2),
        "p2_featPb_cPa": vdiff(swap_b, w, 0) - vdiff(swap_b, w, 2),
        "share_own_first": (float(np.mean(ka[:h] == 0)), float(np.mean(kb[:h] == 2))),
        "share_own_second": (float(np.mean(ka[h:] == 0)), float(np.mean(kb[h:] == 2))),
    }
