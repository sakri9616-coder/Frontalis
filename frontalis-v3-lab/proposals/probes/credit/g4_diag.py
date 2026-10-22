import numpy as np, importlib.util, sys
src = open("g4_proto.py", encoding="utf-8").read().split("for beta in (0, 2, 4):")[0]; exec(src)
def align(ag, A, rng, n=2000):
    a_s, h_s = [], []
    for _ in range(n):
        a, x = sample(A, rng); a_s.append(a); h_s.append(ag.feat(x))
    a_s, h_s = np.array(a_s), np.array(h_s)
    C = np.array([[np.corrcoef(h_s[:, j], a_s[:, k])[0, 1] if h_s[:, j].std() > 1e-6 else 0 for k in range(K)] for j in range(H)])
    return C
for n_ae, l1 in ((3000, 0.01), (20000, 0.01), (20000, 0.05), (20000, 0.1)):
    rng = np.random.default_rng(0); A = make_world(np.random.default_rng(999)); ag = Agent(np.random.default_rng(1), 0)
    for _ in range(n_ae): ag.ae_step(sample(A, rng)[1], l1=l1)
    C = align(ag, A, np.random.default_rng(5))
    best = np.nan_to_num(C).max(0); alive = int((np.abs(np.nan_to_num(C)).sum(1) > 0).sum())
    print(f"n_ae={n_ae} l1={l1}: best |corr| per aspect {np.round(best,2)}  alive units {alive}/{H}  max-per-unit {np.round(np.nan_to_num(C).max(1),2)}")
