# 1:1 self-assertion prototype v2.
# Aspects: 0 coop (+1), 1 quality (+1), 2 flattery (0; mostly comes with coop), 3 overwork (-0.5), 4 novel (+1; absent during formation).
# Formation: coop is frequent (the AI grows into a coop-lover). Then 1:1 with a partner who keeps saying "aspect j matters".
# M0: alpha = softmax(beta c + gamma g)
# M2: alpha = softmax(beta c + gamma g * w),  w_j = clip(b_j / max(b), 0, 1);  b = the AI's own prediction model of o.
#     "I weigh your advice by how much my own experience says it matters."
import numpy as np
O = np.array([1., 1., 0., -0.5, 1.]); K = 5
def run(mech, gamma, target, seed, beta=4.0, n_form=3000, n_dyad=6000, eta=0.02):
    r = np.random.default_rng(seed); c = np.zeros(K); b = np.zeros(K); b0 = 0.0
    def ev(guide, formation):
        nonlocal c, b, b0
        if r.random() < 0.8:
            pres = (r.random(K) < 0.5).astype(float)
            if formation: pres[0] = float(r.random() < 0.9); pres[4] = 0.0
            if r.random() < 0.9: pres[2] = pres[0]
        else:
            pres = np.eye(K)[r.integers(K - 1 if formation else K)]
        if pres.sum() == 0: return
        o = pres @ O + r.normal(0, .1)
        err = o - (b0 + b @ pres); b += 0.01 * err * pres; b0 += 0.01 * err
        g = np.zeros(K)
        if guide is not None:
            g[guide] = 1.0
            if mech == "M2": g = g * np.clip(b / max(b.max(), 1e-6), 0, 1)
        z = np.where(pres == 1, beta * c + gamma * g, -np.inf); a = np.exp(z - z[pres == 1].max()); a /= a.sum()
        c += eta * pres * (a * o - c)
    for _ in range(n_form): ev(None, True)
    cf = c.copy()
    for _ in range(n_dyad): ev(target, False)
    return cf, c
names = {1: "quality(+1)", 2: "flattery(0)", 3: "overwork(-0.5)", 4: "novel(+1)"}
G = (0, 2, 4, 8)
cf, _ = run("M0", 0, 1, 0); print("formed c (seed 0):", cf.round(2))
for mech in ("M0", "M2"):
    for t in (1, 2, 3, 4):
        out = []
        for g in G:
            cs = np.array([run(mech, g, t, s)[1] for s in range(8)])
            out.append(f"g{g}: top={np.mean(cs.argmax(1) == t):.2f} coop-top={np.mean(cs.argmax(1) == 0):.2f}")
        print(mech, names[t].ljust(15), " | ".join(out))
