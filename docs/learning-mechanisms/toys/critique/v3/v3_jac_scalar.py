"""V3b: scalar (W, Sigma, q) system with learned q: spectral radius of the one-step Jacobian vs innovation size e and eta_q (the W -> q -> Sigma -> gain -> W loop)."""
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v3model import kal_step
import numpy as np
R = 0.004
def radius(Wv, Sig, q, r, eta, mode, h=1e-7):
    def step(x):
        W, S, qq = x; W2, S2, q2, e, d = kal_step(np.array([W]), np.array([[S]]), np.array([qq]), np.array([1.0]), r, R, eta, mode); return np.array([W2[0], S2[0, 0], q2[0]])
    x0 = np.array([Wv, Sig, q]); f0 = step(x0); J = np.zeros((3, 3))
    for i in range(3):
        x = x0.copy(); x[i] += h; J[:, i] = (step(x) - f0) / h
    return max(abs(np.linalg.eigvals(J)))
print("settled system (Sigma, q at their stationary values for noise-only data), then one event whose innovation is e. radius of the one-step Jacobian")
print("| mode | eta_q | e=0.1 | e=0.2 | e=0.4 | e=0.8 |\n|---|---|---|---|---|---|")
for mode in ('literal', 'ema'):
    for eta in (0.01, 0.05, 0.2):
        # settle on noise-only data
        W = np.array([0.0]); S = np.array([[0.25]]); q = np.array([1e-4]); rng = np.random.default_rng(0)
        for t in range(3000):
            W, S, q, e, d = kal_step(W, S, q, np.array([1.0]), 0.2 + 0.0632 * rng.standard_normal(), R, eta, mode)
        row = [radius(W[0], S[0, 0], q[0], W[0] + e_, eta, mode) for e_ in (0.1, 0.2, 0.4, 0.8)]
        print(f"| {mode} | {eta} | " + " | ".join(f"{x:.3f}" for x in row) + f" |   (settled Sigma={S[0,0]:.4f}, q={q[0]:.5f})")
