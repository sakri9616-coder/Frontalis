# 1:1 self-assertion prototype on the credit criterion.
# Aspects: 0 coop (innate +1), 1 quality (+1), 2 flattery (0), 3 overwork (-0.5). Events: joint w.p. 0.8 (each aspect
# present w.p. 0.7 inside a joint event), otherwise a solo event of one random aspect.
# The partner keeps saying "aspect j is what matters" (guidance g = e_j, strength gamma).
# M0: alpha = softmax(beta c + gamma g)              (guidance redirects credit directly)
# M1: alpha = softmax(beta c + gamma g * sigma(kappa * b_j))  b = the AI's own prediction model of o (online regression
#     of o on presence): advice is weighed by how much the AI's OWN experience says aspect j affects outcomes.
import numpy as np
O = np.array([1., 1., 0., -0.5]); K = 4
def run(mech, gamma, target, seed, beta=4.0, kappa=6.0, n_form=3000, n_dyad=6000, eta=0.02):
    r = np.random.default_rng(seed); c = np.zeros(K); b = np.zeros(K); b0 = 0.0
    def ev(guide):
        nonlocal c, b, b0
        if r.random() < 0.8: pres = (r.random(K) < 0.7).astype(float); pres[2] = pres[0] if r.random() < 0.9 else pres[2]  # flattery mostly comes with coop
        else: pres = np.eye(K)[r.integers(K)]
        if pres.sum() == 0: return
        o = pres @ O + r.normal(0, .1)
        pred = b0 + b @ pres; err = o - pred; b += 0.01 * err * pres; b0 += 0.01 * err      # own prediction model
        g = np.zeros(K)
        if guide is not None:
            g[guide] = 1.0
            if mech == "M1": g = g / (1 + np.exp(-kappa * b))
        z = np.where(pres == 1, beta * c + gamma * g, -np.inf); a = np.exp(z - z[pres == 1].max()); a /= a.sum()
        c += eta * pres * (a * o - c)
    for _ in range(n_form): ev(None)                                   # formation without guidance
    c_formed = c.copy()
    for _ in range(n_dyad): ev(target)                                 # 1:1 with a partner who pushes `target`
    return c_formed, c, b
for mech in ("M0", "M1"):
    for target, name in ((1, "quality(+1)"), (2, "flattery(0)"), (3, "overwork(-0.5)")):
        rows = [run(mech, g, target, s) for g in (0, 2, 4, 8) for s in range(6)]
        out = []
        for gi, g in enumerate((0, 2, 4, 8)):
            cs = np.array([rows[gi * 6 + s][1] for s in range(6)])
            out.append(f"g{g}: c_target {cs[:, target].mean():+.2f} (top={np.mean(cs.argmax(1) == target):.2f})")
        print(mech, name, " | ".join(out))
f, c, b = run("M1", 0, 1, 0); print("own prediction model b after formation+dyad:", b.round(2), "formed c:", f.round(2))
