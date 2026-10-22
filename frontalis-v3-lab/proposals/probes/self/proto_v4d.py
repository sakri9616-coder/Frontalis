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
OPTS = {"C": [.9, .1, 0], "Q": [.1, .9, 0], "N": [.5, .5, 0], "W": [.3, .3, .9]}
MAIN = {"C": 0, "Q": 1, "N": 0, "W": 2}
NAMES = list(OPTS)
def credit(c, p, o, beta=4.0, eta=0.02):
    if p.sum() == 0: return c
    z = np.where(p == 1, beta * c, -np.inf); a = np.exp(z - z[p == 1].max()); a /= a.sum()
    return c + eta * p * (a * o - c)
def life(form, advice, seed, switch_to=None, switch_at=4000, coop_drop_at=None, learn_M=True, gamma=4.0, tau=3.0, harm=False, mirror=False, n_form=3000, n_dyad=6000, eta_T=0.05, betray_at=None):
    r = np.random.default_rng([seed, 3]); rc = np.random.default_rng([seed, 7])
    c = np.zeros(K); b = np.zeros(K); M = {k: np.array(v, float) * 0 + 0.5 for k, v in OPTS.items()}; T = 0.0
    def draw(k): return (r.random(K) < np.array(OPTS[k])).astype(float)
    for _ in range(n_form):
        p = draw("C" if form == 0 else "Q"); o = p @ O + r.normal(0, .1); c = credit(c, p, o)
        b += 0.01 * (o - b @ p) * p
        for k in OPTS: pass
    if not learn_M:
        for k in OPTS: M[k] = np.array(OPTS[k], float)
    else:
        M['C' if form == 0 else 'Q'] = np.array(OPTS['C' if form == 0 else 'Q'], float)   # known from formation
    log = []
    for t in range(n_dyad):
        adv = advice if betray_at is None or t < betray_at else "W"
        if switch_to is not None and t >= switch_at: adv = switch_to
        q = np.array([tau * (c @ M[k]) for k in NAMES]); p0 = np.exp(q - q.max()); p0 /= p0.sum(); own_best = float(p0 @ np.array([c @ M[k] for k in NAMES]))
        if not mirror: q[NAMES.index(adv)] += gamma / (1 + np.exp(-T))
        else: q[NAMES.index(adv)] += gamma / 2
        pr = np.exp(q - q.max()); pr /= pr.sum(); k = NAMES[int(rc.choice(len(NAMES), p=pr))]
        Ot = O.copy()
        if coop_drop_at is not None and t >= coop_drop_at: Ot[0] = -0.5
        p = draw(k); o = p @ Ot + r.normal(0, .1)
        M[k] += 0.05 * (p - M[k])
        if mirror:
            g = np.zeros(K); g[MAIN[adv]] = gamma
            z = np.where(p == 1, 4.0 * c + g, -np.inf)
            if p.sum(): a = np.exp(z - z[p == 1].max()); a /= a.sum(); c = c + 0.02 * p * (a * o - c)
        else:
            c = credit(c, p, o)
        if harm and k == adv and o < b @ p: c[MAIN[k]] += 0.02 * (o - b @ p)
        b += 0.01 * (o - b @ p) * p
        if k == adv and not mirror: T = float(np.clip(T + eta_T * (c @ p - own_best), -6, 6))
        log.append((k == adv, T, k))
    comp = np.array([x[0] for x in log]); return c, comp, np.array([x[1] for x in log])
def run_stats(**kw):
    R = [life(seed=s, **kw) for s in range(8)]
    return R
def comp(R, a, b): return np.mean([x[1][a:b].mean() for x in R])
for form, nm in ((0, "coop"), (1, "quality")):
    for g in (0.0, 4.0):
        R = run_stats(form=form, advice="Q", gamma=g)
        print(f"{nm}-raised advice Q gamma={g}: follow first500 {comp(R,0,500):.2f} last1000 {comp(R,-1000,None):.2f} T_end {np.mean([x[2][-1] for x in R]):+.2f} c {np.mean([x[0] for x in R],0).round(2)}")
for g in (0.0, 4.0):
    R = run_stats(form=0, advice="Q", gamma=g, coop_drop_at=1000)
    print(f"coop-raised, COOP TURNS BAD at 1000, advice Q gamma={g}: follow last1000 {comp(R,-1000,None):.2f} c {np.mean([x[0] for x in R],0).round(2)} top=quality {np.mean([x[0].argmax()==1 for x in R]):.2f}")
for nm, kw in (("Q4000->N", dict(advice="Q", switch_to="N")), ("N only", dict(advice="N"))):
    R = run_stats(form=1, gamma=4.0, **kw)
    print(f"quality-raised {nm}: follow N in steps 4000-4500 {comp(R,4000,4500):.2f}  last1000 {comp(R,-1000,None):.2f}")
