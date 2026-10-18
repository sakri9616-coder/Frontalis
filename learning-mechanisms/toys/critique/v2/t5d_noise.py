"""T5d: e_x = signed MF innovation is zero-mean at convergence (Kalman innovations are white). Does the commitment hold under pure innovation noise (no real counter-evidence)?"""
import numpy as np
def flips(g, ke, sig, ac=0.1, T=20000, seed=0):
    rng = np.random.default_rng(seed); c = 1.5; f = 0; sign = 1
    for t in range(T):
        c += ac * (-c + g * np.tanh(c) + ke * sig * rng.standard_normal())
        if sign > 0 and c < -0.3: sign = -1; f += 1
        elif sign < 0 and c > 0.3: sign = 1; f += 1
    return f
print("sign flips of c in 20000 updates under zero-mean innovation noise only (R=0.004 => sd 0.063 per active need; 3 active needs => 0.11)")
print("| g | k_e | sd=0.063 | sd=0.11 |\n|---|---|---|---|")
for g in (1.1, 1.3, 1.5):
    for ke in (0.5, 1.0, 2.0):
        print(f"| {g} | {ke} | {flips(g,ke,0.063)} | {flips(g,ke,0.11)} |")
