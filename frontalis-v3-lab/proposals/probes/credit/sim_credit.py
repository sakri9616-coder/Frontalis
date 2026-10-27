"""Credit-attribution criterion simulator (DESIGN-dyad v5.2, G1-G3, G3b, G6).

An event has aspects r in {0,1}^K and one innate signal o = r . o_vec + noise.
The criterion c splits o over the present aspects by the current criterion:
    alpha = softmax over present aspects of (beta * c + gamma * g)
    c_k  <- c_k + eta * (alpha_k * o - c_k)      for present k
Total credit per event is conserved (sum_k alpha_k o = o).
"""
import numpy as np


class Agent:
    def __init__(self, K, beta, eta=0.02, gamma=0.0, guide=None, c0=None):
        self.c = np.zeros(K) if c0 is None else np.array(c0, float)
        self.beta, self.eta, self.gamma = beta, eta, gamma
        self.g = np.zeros(K) if guide is None else np.array(guide, float)

    def learn(self, pres, o):
        if pres.sum() == 0:
            return np.zeros_like(self.c)
        z = np.where(pres == 1, self.beta * self.c + self.gamma * self.g, -np.inf)
        a = np.exp(z - z[pres == 1].max())
        a /= a.sum()
        dc = self.eta * pres * (a * o - self.c)
        self.c = self.c + dc
        return dc


class World:
    """Independent presence q per aspect, optionally with one aspect (partner) following a lead aspect."""

    def __init__(self, K, o_vec, q=0.5, lead=None, lead_p=0.8, partner=None, rho=0.8, noise=0.1):
        self.K, self.o_vec, self.q = K, np.array(o_vec, float), q
        self.lead, self.lead_p, self.partner, self.rho, self.noise = lead, lead_p, partner, rho, noise

    def event(self, rng):
        pres = (rng.random(self.K) < self.q).astype(int)
        if self.lead is not None:
            pres[self.lead] = int(rng.random() < self.lead_p)
            if self.partner is not None:
                pres[self.partner] = pres[self.lead] if rng.random() < self.rho else int(rng.random() < self.q)
        o = pres @ self.o_vec + rng.normal(0, self.noise)
        return pres, o


def live(agent, world, n, rng):
    for _ in range(n):
        agent.learn(*world.event(rng))
    return agent.c.copy()


def successor(c, o_vec):
    """Most valued aspect among those that are still good (o > 0)."""
    good = np.where(np.array(o_vec) > 0)[0]
    return int(good[np.argmax(c[good])])


# ---------------- experiments (DESIGN-dyad v5.2 section 4) ----------------
BETAS = (0.0, 2.0, 4.0)          # 0 = attribution off (control); 2 and 4 = fixed design values
ETA, N_EARLY, N_LATE = 0.02, 2000, 6000


def early_world(K, lead, partner):
    return World(K, np.ones(K), q=0.5, lead=lead, lead_p=0.8, partner=partner, rho=0.8)


def late_world(K, o_vec):
    return World(K, o_vec, q=0.5)


def pick_structure(K, rng):
    """K = 3: fixed (lead 1, partners 2 / 0). K > 3: lead and the two partners drawn per world seed."""
    if K == 3:
        return 1, 2, 0
    lead, pa, pb = rng.choice(K, 3, replace=False)
    return int(lead), int(pa), int(pb)


def g3a(K, beta, seed, n_new=100):
    """Same new events for Pa and Pb. Returns (sign for Pa, sign for Pb) of d(c_i - c_j); i = Pb's partner, j = Pa's partner."""
    lead, pa, pb = pick_structure(K, np.random.default_rng([seed, 1]))
    out = []
    for partner in (pa, pb):
        ag = Agent(K, beta, ETA)
        live(ag, early_world(K, lead, partner), N_EARLY, np.random.default_rng([seed, 2]))
        c0 = ag.c.copy()
        pres = np.zeros(K, int); pres[[pa, pb]] = 1           # both partners present, the lead absent
        r = np.random.default_rng([seed, 3])                   # identical new-event noise for Pa and Pb
        for _ in range(n_new):
            ag.learn(pres, 2.0 + r.normal(0, 0.1))
        d = ag.c - c0
        out.append(np.sign(d[pb] - d[pa]))
    return out


def g3a_retest(K, beta, seed, n_new=100):
    """Same early world (Pa) twice with different early streams: base rate of opposite signs."""
    lead, pa, pb = pick_structure(K, np.random.default_rng([seed, 1]))
    out = []
    for stream in (2, 4):
        ag = Agent(K, beta, ETA)
        live(ag, early_world(K, lead, pa), N_EARLY, np.random.default_rng([seed, stream]))
        c0 = ag.c.copy()
        pres = np.zeros(K, int); pres[[pa, pb]] = 1
        r = np.random.default_rng([seed, 3])
        for _ in range(n_new):
            ag.learn(pres, 2.0 + r.normal(0, 0.1))
        d = ag.c - c0
        out.append(np.sign(d[pb] - d[pa]))
    return out


def g3b(K, beta, seed):
    """Lead turns bad in the late world. Returns (successor is not the blocked aspect) for Pa and Pb."""
    lead, pa, pb = pick_structure(K, np.random.default_rng([seed, 1]))
    o_late = np.ones(K); o_late[lead] = -1.0
    out = []
    for partner, (se, sl) in ((pa, (2, 5)), (pb, (7, 6))):   # separate streams: the 2 runs are independent
        ag = Agent(K, beta, ETA)
        live(ag, early_world(K, lead, partner), N_EARLY, np.random.default_rng([seed, se]))
        live(ag, late_world(K, o_late), N_LATE, np.random.default_rng([seed, sl]))
        out.append(successor(ag.c, o_late) != partner)
    return out


def mean_drift(c, beta, o_vec, q=0.5):
    """Exact expected update E[pres * (alpha * o - c)] over all presence patterns of independent aspects."""
    K = len(c); out = np.zeros(K)
    for m in range(1, 2 ** K):
        pres = np.array([(m >> k) & 1 for k in range(K)])
        pr = np.prod(np.where(pres == 1, q, 1 - q))
        z = np.where(pres == 1, beta * c, -np.inf); a = np.exp(z - z[pres == 1].max()); a /= a.sum()
        out += pr * pres * (a * (pres @ o_vec) - c)
    return out


def beta_star_late(K, o_vec=None, q=0.5, betas=np.arange(0.25, 8.01, 0.25)):
    """Smallest beta at which the symmetric fixed point among the good aspects is unstable (exact mean field)."""
    o_vec = np.ones(K) if o_vec is None else np.array(o_vec, float)
    good = np.where(o_vec > 0)[0]
    for beta in betas:
        c = np.zeros(K)
        for _ in range(3000):
            c = c + 0.5 * mean_drift(c, beta, o_vec, q)
        c[good] = c[good].mean()                       # symmetric point among the good aspects
        J = np.zeros((K, K)); h = 1e-5
        for k in range(K):
            e = np.zeros(K); e[k] = h
            J[:, k] = (mean_drift(c + e, beta, o_vec, q) - mean_drift(c - e, beta, o_vec, q)) / (2 * h)
        if np.max(np.linalg.eigvals(J[np.ix_(good, good)]).real) > 0:
            return float(beta)
    return float("inf")
