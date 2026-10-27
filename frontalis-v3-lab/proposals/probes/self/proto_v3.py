# DESIGN-self v2 prototype. The partner's advice "j matters" acts on what the AI TRIES (choice), never on credit.
# Whether the AI comes to value j is decided by its own credit attribution of what it then experiences (its current c).
# Aspects: 0 coop (+1), 1 quality (+1), 2 flattery (0; comes with coop w.p. 0.9 when coop is present), 3 novel (+1; absent in formation).
# Option k: aspect k present w.p. 0.9, others 0.3 (flattery follows coop). Choice: softmax(tau * c.M[k] + gamma * [k == j]).
import numpy as np
O = np.array([1., 1., 0., 1.]); K = 4
def ev(r, freq, allow_novel=True):
    p = (r.random(K) < freq).astype(float)
    if r.random() < 0.9: p[2] = p[0]
    if not allow_novel: p[3] = 0.0
    return p
def opt(k): f = np.full(K, 0.3); f[k] = 0.9; return f
def credit(c, p, o, beta, eta=0.02):
    if p.sum() == 0: return c
    z = np.where(p == 1, beta * c, -np.inf); a = np.exp(z - z[p == 1].max()); a /= a.sum()
    return c + eta * p * (a * o - c)
def life(form, target, gamma, seed, beta=4.0, tau=3.0, n_form=3000, n_dyad=6000, n_after=4000, mech="choice"):
    r = np.random.default_rng([seed, 3]); rc = np.random.default_rng([seed, 7]); c = np.zeros(K)
    f = np.full(K, 0.3); f[form] = 0.9
    for _ in range(n_form):
        p = ev(r, f, False); c = credit(c, p, p @ O + r.normal(0, .1), beta)
    cf = c.copy(); M = np.full((K, K), 0.5)
    def step(advice):
        nonlocal c
        q = tau * (M @ c)
        if advice is not None and mech == "choice": q[advice] += gamma
        pr = np.exp(q - q.max()); pr /= pr.sum(); k = int(rc.choice(K, p=pr))
        p = ev(r, opt(k)); o = p @ O + r.normal(0, .1)
        if advice is not None and mech == "M0":           # mirror control: advice pushes credit directly
            g = np.zeros(K); g[advice] = gamma
            z = np.where(p == 1, beta * c + g, -np.inf)
            if p.sum():
                a = np.exp(z - z[p == 1].max()); a /= a.sum(); c = c + 0.02 * p * (a * o - c)
        else:
            c = credit(c, p, o, beta)
        M[k] += 0.05 * (p - M[k]); return k
    for _ in range(n_dyad): step(target)
    cd = c.copy()
    for _ in range(n_after): step(None)
    return cf, cd, c.copy()
names = {1: "quality", 2: "flattery", 3: "novel"}
for mech in ("choice", "M0"):
    for t in (2, 3, 1):
        for g in (0, 2, 4):
            res = {}
            for form, nm in ((0, "coop-raised"), (1, "quality-raised")):
                R = [life(form, t, g, s, mech=mech) for s in range(8)]
                cd = np.array([x[1] for x in R]); ca = np.array([x[2] for x in R]); cf = np.array([x[0] for x in R])
                res[nm] = f"top-is-target {np.mean(cd.argmax(1) == t):.2f} after-leave {np.mean(ca.argmax(1) == t):.2f} keep-own-top {np.mean(cd.argmax(1) == cf.argmax(1)):.2f} c_t {cd[:, t].mean():+.2f}"
            print(f"{mech:6} advice={names[t]:8} g={g}: coop-raised [{res['coop-raised']}] | quality-raised [{res['quality-raised']}]")
