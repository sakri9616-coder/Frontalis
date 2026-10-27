# Credit-attribution criterion: an event has aspects (present/absent) and one innate signal o.
# The innate o is split over the present aspects by the CURRENT criterion (attention):
#   alpha_k = softmax_k over present aspects of beta*c_k ;  target for c_k = alpha_k * o  (only if present)
# Total credit per event is conserved (sum alpha_k o = o): attention redistributes, never amplifies.
import numpy as np
def run(beta, p_joint, early, late, o_c=1.0, o_q=1.0, n=6000, eta=0.02, seed=0, c0=(0., 0.)):
    r = np.random.default_rng(seed); c = np.array(c0, float)
    def ev(phase):
        u = r.random()
        if phase == "early_coop":    pres = np.array([1, 1]) if u < p_joint else np.array([1, 0])
        elif phase == "early_qual":  pres = np.array([1, 1]) if u < p_joint else np.array([0, 1])
        else:                        pres = np.array([1, 1]) if u < p_joint else (np.array([1, 0]) if u < p_joint + (1 - p_joint) / 2 else np.array([0, 1]))
        return pres
    hist = []
    for phase, steps in ((early, n // 3), (late, n)):
        for _ in range(steps):
            pres = ev(phase)
            oc, oq = (o_c, o_q) if phase != "late_qbad" else (o_c, -1.0)
            o = pres[0] * oc + pres[1] * oq + r.normal(0, .1)
            z = np.where(pres == 1, beta * c, -np.inf); a = np.exp(z - z[pres == 1].max()); a = a / a.sum()
            c = c + eta * pres * (a * o - c)
        hist.append(c.round(2).tolist())
    return hist
print("beta  p_joint | early coop-heavy -> late mixed | early qual-heavy -> late mixed | no early")
for beta in (0, 1, 2, 4):
    for pj in (0.5, 0.8):
        a = run(beta, pj, "early_coop", "late"); b = run(beta, pj, "early_qual", "late"); z = run(beta, pj, "late", "late")
        print(f"{beta:4} {pj:6} | {a} | {b} | {z}")
print("world change: quality turns bad in late (o_q=-1), beta=4, pj=0.8")
for e in ("early_coop", "early_qual"):
    print(e, run(4, 0.8, e, "late_qbad"))
print("seed spread of late-end c (beta=4,pj=0.8) for retests of the same early:")
for e in ("early_coop", "early_qual"):
    print(e, [run(4, 0.8, e, "late", seed=s)[1] for s in range(5)])

# ---- part 2: partner guidance as an attention prior ----
import numpy as np
O3 = np.array([1., 1., 0.])
def run_dyad(beta, gamma, guide, early_c, n=6000, eta=0.02, seed=0, pj=0.8, solo=True):
    r = np.random.default_rng(seed); c = np.array(early_c, float); g = np.array(guide, float)
    for _ in range(n):
        u = r.random()
        if u < pj: pres = np.array([1, 1, 1])              # joint events carry all aspects
        elif solo: pres = np.eye(3, dtype=int)[r.integers(3)]
        else: pres = np.array([1, 1, 1])
        o = pres @ O3 + r.normal(0, .1)
        z = np.where(pres == 1, beta * c + gamma * g, -np.inf); a = np.exp(z - z[pres == 1].max()); a /= a.sum()
        c = c + eta * pres * (a * o - c)
    return c.round(2).tolist()
coop_lover = (1.9, 0.1, 0.0)
print("coop-lover, partner says quality matters (g=[0,1,0]):")
for gamma in (0, 1, 2, 4, 8): print("  gamma", gamma, run_dyad(4, gamma, [0, 1, 0], coop_lover))
print("coop-lover, partner says flattery matters (g=[0,0,1]):")
for gamma in (0, 2, 4, 8): print("  gamma", gamma, run_dyad(4, gamma, [0, 0, 1], coop_lover), "| no solo events:", run_dyad(4, gamma, [0, 0, 1], coop_lover, solo=False))
print("guidance removed after it flipped (gamma=8 then 0):")
c1 = run_dyad(4, 8, [0, 1, 0], coop_lover); print("  after guidance", c1, "-> after removal", run_dyad(4, 0, [0, 1, 0], c1))
