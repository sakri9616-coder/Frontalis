# Diagnosis for P53: is the perception measure too crude? Continuous measure: how well the variant of aspect k
# can be read linearly from the features h (least-squares readout, held-out accuracy), Pa vs Pb, under g6.
import numpy as np
exec(open("g6_proto.py", encoding="utf-8").read().split("for beta in (0.0, 4.0):")[0])
def decod(ag, Wd, k, rng, n=400):
    A, B = Wd; X, y = [], []
    for _ in range(n):
        v = rng.random() < .5; others = (rng.random(K) < .3).astype(float); others[k] = 0
        x = A[k] + (1 if v else -1) * B[k] + (others[:, None] * A).sum(0) + rng.normal(0, .1, D)
        X.append(np.r_[ag.feat(x), 1]); y.append(1 if v else -1)
    X, y = np.array(X), np.array(y); tr, te = slice(0, n // 2), slice(n // 2, n)
    w = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]; return float(np.mean(np.sign(X[te] @ w) == y[te]))
for beta in (0.0, 4.0):
    R = []
    for s in range(10):
        pa, Wd, _ = life2(beta, s, 0, True); pb, _, _ = life2(beta, s, 2, True); r = np.random.default_rng([s, 11])
        R.append([decod(pa, Wd, 0, r), decod(pb, Wd, 0, r), decod(pa, Wd, 2, r), decod(pb, Wd, 2, r)])
    m = np.array(R)
    print(f"beta={beta}: variant decodability aspect0 Pa {m[:,0].mean():.2f} Pb {m[:,1].mean():.2f} | aspect2 Pa {m[:,2].mean():.2f} Pb {m[:,3].mean():.2f} | "
          f"own-focus better in {np.mean((m[:,0]-m[:,1])+(m[:,3]-m[:,2])>0):.1f} of seeds")
