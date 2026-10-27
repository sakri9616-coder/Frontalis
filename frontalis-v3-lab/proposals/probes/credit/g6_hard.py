# Same as g6_diag, in a world where variants are hard to tell apart: variant direction scale B_SCALE, raw noise NOISE.
import numpy as np, sys
B_SCALE, NOISE = float(sys.argv[1]), float(sys.argv[2])
src = open("g6_diag.py", encoding="utf-8").read().split("def decod")[0]
exec(src)
def world(ws):
    r = np.random.default_rng([ws, 0]); return r.normal(0, 1, (K, D)), r.normal(0, B_SCALE, (K, D))
def sample(W, rng, freq):
    A, B = W; a = (rng.random(K) < freq).astype(float); v = (rng.random(K) < 0.5).astype(float)
    x = (a[:, None] * (A + (2 * v - 1)[:, None] * B)).sum(0) + rng.normal(0, NOISE, D)
    o = (a * (1 + DELTA * (2 * v - 1))).sum() + rng.normal(0, 0.1)
    return a, v, x, o
def decod(ag, Wd, k, rng, n=600):
    A, B = Wd; X, y = [], []
    for _ in range(n):
        v = rng.random() < .5; others = (rng.random(K) < .3).astype(float); others[k] = 0
        x = A[k] + (1 if v else -1) * B[k] + (others[:, None] * A).sum(0) + rng.normal(0, NOISE, D)
        X.append(np.r_[ag.feat(x), 1]); y.append(1 if v else -1)
    X, y = np.array(X), np.array(y); tr, te = slice(0, n // 2), slice(n // 2, n)
    w = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; return float(np.mean(np.sign(X[te] @ w) == y[te]))
for beta in (0.0, 4.0):
    R = []
    for s in range(10):
        pa, Wd, ca = life2(beta, s, 0, True); pb, _, cb = life2(beta, s, 2, True); r = np.random.default_rng([s, 11])
        R.append([decod(pa, Wd, 0, r), decod(pb, Wd, 0, r), decod(pa, Wd, 2, r), decod(pb, Wd, 2, r), ca[0], cb[2]])
    m = np.array(R)
    print(f"B={B_SCALE} noise={NOISE} beta={beta}: aspect0 Pa {m[:,0].mean():.2f} Pb {m[:,1].mean():.2f} | aspect2 Pa {m[:,2].mean():.2f} Pb {m[:,3].mean():.2f} | "
          f"own-focus better in {np.mean((m[:,0]-m[:,1])+(m[:,3]-m[:,2])>0):.1f} | shares {m[:,4].mean():.2f},{m[:,5].mean():.2f}")
