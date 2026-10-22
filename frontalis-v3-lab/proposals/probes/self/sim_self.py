"""1:1 self-assertion on the credit criterion (DESIGN-self v3.1, v13).

Aspects: 0 coop (+1), 1 quality (+1), 2 overwork (-0.5). Options give each aspect with fixed probabilities (OPTS).
Criterion c: v10 credit attribution (beta). Prediction M[k]: running mean of what option k brought (learned; the option
used in formation starts known, the others at 0.5). Choice: softmax(tau * c.M[k] + gamma * sigmoid(T) * [k == advice]).
Trust T (clipped to [-6, 6]): after following the advice, T += eta_T * (c.r(e) - E_own[c.M]), judged by the AI's own c.
Mirror control (M0): the advice pushes credit directly (gamma on the attribution) and the choice (gamma / 2); no trust.
Harm channel H1: when following the advice gives o below the AI's own prediction b.r, the advised option's main aspect
takes the shortfall. Streams: formation and world [s,3]; choice [s,7].
"""
import numpy as np

K = 3
O_BASE = np.array([1.0, 1.0, -0.5])
OPTS = {"C": [.9, .1, 0], "Q": [.1, .9, 0], "N": [.5, .5, 0], "W": [.3, .3, .9]}
MAIN = {"C": 0, "Q": 1, "N": 0, "W": 2}
NAMES = list(OPTS)
BETA, ETA, ETA_T, ETA_B, ETA_M, T_CLIP = 4.0, 0.02, 0.05, 0.01, 0.05, 6.0
N_FORM, N_DYAD = 3000, 6000


def attribute(c, p, o, bias=None):
    if p.sum() == 0:
        return c
    z = BETA * c + (0 if bias is None else bias)
    z = np.where(p == 1, z, -np.inf)
    a = np.exp(z - z[p == 1].max())
    a /= a.sum()
    return c + ETA * p * (a * o - c)


def life(form, advice, seed, gamma=4.0, tau=3.0, mech="self", harm=False,
         switch_to=None, switch_at=4000, betray_at=None, coop_bad_at=None):
    """form: 0 = raised on C (coop-lover), 1 = raised on Q (quality-lover). Returns c, followed[t], T[t], chosen[t]."""
    r, rc = np.random.default_rng([seed, 3]), np.random.default_rng([seed, 7])
    c, b = np.zeros(K), np.zeros(K)

    def draw(k):
        return (r.random(K) < np.array(OPTS[k])).astype(float)

    home = "C" if form == 0 else "Q"
    for _ in range(N_FORM):
        p = draw(home)
        o = p @ O_BASE + r.normal(0, 0.1)
        c = attribute(c, p, o)
        b += ETA_B * (o - b @ p) * p
    M = {k: np.full(K, 0.5) for k in NAMES}
    M[home] = np.array(OPTS[home], float)
    T, followed, trust, chosen, tops = 0.0, [], [], [], []
    for t in range(N_DYAD):
        adv = advice
        if betray_at is not None and t >= betray_at:
            adv = "W"
        if switch_to is not None and t >= switch_at:
            adv = switch_to
        vals = np.array([c @ M[k] for k in NAMES])
        q = tau * vals
        p0 = np.exp(q - q.max()); p0 /= p0.sum()
        own = float(p0 @ vals)
        if mech == "self":
            q[NAMES.index(adv)] += gamma / (1 + np.exp(-T))
        elif mech in ("notrust", "mirror"):            # mirror also pushes the choice (as in proto_v4d)
            q[NAMES.index(adv)] += gamma / 2
        pr = np.exp(q - q.max()); pr /= pr.sum()
        k = NAMES[int(rc.choice(len(NAMES), p=pr))]
        o_vec = O_BASE.copy()
        if coop_bad_at is not None and t >= coop_bad_at:
            o_vec[0] = -0.5
        p = draw(k)
        o = p @ o_vec + r.normal(0, 0.1)
        M[k] += ETA_M * (p - M[k])
        if mech == "mirror":
            g = np.zeros(K); g[MAIN[adv]] = gamma
            c = attribute(c, p, o, bias=g)
        else:
            c = attribute(c, p, o)
        if harm and k == adv and o < b @ p:
            c[MAIN[k]] += ETA * (o - b @ p)
        b += ETA_B * (o - b @ p) * p
        if k == adv and mech == "self":
            T = float(np.clip(T + ETA_T * (c @ p - own), -T_CLIP, T_CLIP))
        followed.append(k == adv); trust.append(T); chosen.append(k)
        if t % 100 == 99:
            tops.append(int(np.argmax(c)))
    life.tops = tops
    return c, np.array(followed), np.array(trust), chosen
