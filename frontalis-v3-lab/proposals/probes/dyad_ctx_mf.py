# Construction check for DESIGN-dyad v4: cue z's innate effect depends on context w.
# innate z-effect u(w) = w . U ; meaning m = o + lam*tanh(g); g target = V (own view), V target = m (q=1, mean field)
# Learners: (L) linear bilinear g_z(w) = a + b.w ; (M) small MLP on [z*w, w] trained by GD.
import numpy as np
rng = np.random.default_rng(0)
W = {"A": np.array([1, 0, 0.]), "B": np.array([0, 1, 0.]), "C": np.array([.5, .5, 0]),
     "D": np.array([.7, 0, .7]), "E": np.array([0, .7, .7])}
U = np.array([1., -1., 0.])          # innate: z good in A, bad in B, neutral in C; D,E differ only via dims 1,2
u = {k: float(v @ U) for k, v in W.items()}
def run(learner, early, late, lam, steps=4000, lr=0.05, seed=0):
    r = np.random.default_rng(seed)
    if learner == "L":
        th = np.zeros(4)
        f = lambda w, th: th[0] + th[1:] @ w
        grad = lambda w, th: np.r_[1, w]
    else:
        H = 16; th = [r.normal(0, .5, (H, 3)), np.zeros(H), r.normal(0, .5, H) / np.sqrt(H)]
    V = {}
    def gz(w):
        if learner == "L": return f(w, th)
        h = np.tanh(th[0] @ w + th[1]); return float(th[2] @ h)
    def step(ctxs):
        nonlocal th
        k = ctxs[r.integers(len(ctxs))]; w = W[k]
        m = u[k] + lam * np.tanh(gz(w))
        V[k] = V.get(k, 0.) + 0.1 * (m - V.get(k, 0.))      # own view of z in context k
        err = V[k] - gz(w)
        if learner == "L": th = th + lr * err * grad(w, th)
        else:
            h = np.tanh(th[0] @ w + th[1]); dh = th[2] * (1 - h**2)
            th = [th[0] + lr * err * np.outer(dh, w), th[1] + lr * err * dh, th[2] + lr * err * h]
    for _ in range(steps): step(early)
    s = {k: round(gz(W[k]), 2) for k in W}
    for _ in range(steps): step(late)
    e = {k: round(gz(W[k]), 2) for k in W}
    return s, e
for learner in ("L", "M"):
    for lam in (0.5,):
        for name, early in (("P1(A only)", ["A"]), ("P3(A,B)", ["A", "B"])):
            outs = [run(learner, early, ["A", "B", "C"], lam, seed=s) for s in range(5)]
            print(learner, lam, name, "early-end", outs[0][0])
            for s, (_, e) in enumerate(outs): print("   late-end seed", s, e)

# ---- part 2: P1-P3 differences at the end of the late phase ----
def show(tag, learner, lam, e1, e2, late, n=5):
    d = []
    for s in range(n):
        _, a = run(learner, e1, late, lam, seed=s); _, b = run(learner, e2, late, lam, seed=s)
        d.append({k: round(a[k] - b[k], 2) for k in W})
    print(tag, learner, lam, "P1-P3 late-end diff per seed:", d)
for learner in ("L", "M"):
    show("late A,B,C  ", learner, 1.5, ["A"], ["A", "B"], ["A", "B", "C"])
    show("late B,C only", learner, 0.5, ["A"], ["A", "B"], ["B", "C"])
    show("early D vs E, late A,B", learner, 0.5, ["A", "D"], ["A", "E"], ["A", "B"])
