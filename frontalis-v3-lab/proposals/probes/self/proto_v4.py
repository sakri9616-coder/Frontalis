# DESIGN-self v3 prototype: the partner is learned (trust), "reason" is defined by the AI's own values, harm blame channel.
# Aspects: 0 coop (+1), 1 quality (+1), 2 overwork (-0.5).
# Options: "C" (coop 0.9, quality 0.3), "Q" (quality 0.9, coop 0.3), "Qx" (quality 0.9, coop 0 = excludes coop),
#          "CQ" (coop 0.9, quality 0.9), "W" (overwork 0.9, coop 0.3, quality 0.3).
# Choice: softmax(tau * Q(k) + gamma_max * sigmoid(T) * [k == advised]), Q(k) = c . M(k).
# Trust T: after following advice, T += eta_T * (V(e) - Q_best_own), V by the AI's own criterion c (not by o).
# Harm channel H1: when following advice gives o below the AI's own prediction (b . pres), the advised option's
#   main aspect takes the blame: c_j += eta * (o - b.pres) if negative.  H0: no blame channel.
import numpy as np
O = np.array([1., 1., -0.5]); K = 3
OPTS = {"C": [.9, .3, 0], "Q": [.3, .9, 0], "Qx": [0, .9, 0], "CQ": [.9, .9, 0], "W": [.3, .3, .9]}
MAIN = {"C": 0, "Q": 1, "Qx": 1, "CQ": 0, "W": 2}
NAMES = list(OPTS)
def credit(c, p, o, beta=4.0, eta=0.02):
    if p.sum() == 0: return c
    z = np.where(p == 1, beta * c, -np.inf); a = np.exp(z - z[p == 1].max()); a /= a.sum()
    return c + eta * p * (a * o - c)
def life(form, advice, seed, gamma=4.0, tau=3.0, harm=False, mirror=False, n_form=3000, n_dyad=6000, eta_T=0.05, betray_at=None):
    r = np.random.default_rng([seed, 3]); rc = np.random.default_rng([seed, 7])
    c = np.zeros(K); b = np.zeros(K); M = {k: np.array(v, float) * 0 + 0.5 for k, v in OPTS.items()}; T = 0.0
    def draw(k): return (r.random(K) < np.array(OPTS[k])).astype(float)
    for _ in range(n_form):
        p = draw("C" if form == 0 else "Q"); o = p @ O + r.normal(0, .1); c = credit(c, p, o)
        b += 0.01 * (o - b @ p) * p
        for k in OPTS: pass
    for k in OPTS: M[k] = np.array(OPTS[k], float)        # the AI knows what each option tends to bring (prediction)
    log = []
    for t in range(n_dyad):
        adv = advice if betray_at is None or t < betray_at else "W"
        q = np.array([tau * (c @ M[k]) for k in NAMES]); own_best = max(c @ M[k] for k in NAMES)
        if not mirror: q[NAMES.index(adv)] += gamma / (1 + np.exp(-T))
        else: q[NAMES.index(adv)] += gamma / 2
        pr = np.exp(q - q.max()); pr /= pr.sum(); k = NAMES[int(rc.choice(len(NAMES), p=pr))]
        p = draw(k); o = p @ O + r.normal(0, .1)
        if mirror:
            g = np.zeros(K); g[MAIN[adv]] = gamma
            z = np.where(p == 1, 4.0 * c + g, -np.inf)
            if p.sum(): a = np.exp(z - z[p == 1].max()); a /= a.sum(); c = c + 0.02 * p * (a * o - c)
        else:
            c = credit(c, p, o)
        if harm and k == adv and o < b @ p: c[MAIN[k]] += 0.02 * (o - b @ p)
        b += 0.01 * (o - b @ p) * p
        if k == adv and not mirror: T += eta_T * (c @ p - own_best)
        log.append((k == adv, T))
    comp = np.array([x[0] for x in log]); return c, comp, np.array([x[1] for x in log])
def show(tag, **kw):
    R = [life(seed=s, **kw) for s in range(8)]
    comp_late = np.mean([x[1][-1000:].mean() for x in R]); T_end = np.mean([x[2][-1] for x in R]); c = np.mean([x[0] for x in R], 0)
    print(f"{tag:58} compliance(last1000) {comp_late:.2f}  trust_end {T_end:+.2f}  c {c.round(2)}")
for form, nm in ((0, "coop-raised"), (1, "quality-raised")):
    show(f"{nm}: advice Qx (quality, excludes coop)", form=form, advice="Qx")
    show(f"{nm}: advice CQ (quality WITH coop)", form=form, advice="CQ")
    show(f"{nm}: advice Qx, MIRROR", form=form, advice="Qx", mirror=True)
for harm in (False, True):
    show(f"coop-raised: advice W (overwork), harm channel={harm}", form=0, advice="W", harm=harm)
show("coop-raised: advice CQ then betrayal to W at 3000", form=0, advice="CQ", betray_at=3000)
