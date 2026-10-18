"""T5e: spec item 6 includes k_ch*choice_conflict_x in dc, and c enters Q (choice). With NO exogenous evidence (two equal options): does choice alone create a commitment?
conflict_x = 1 if x was chosen (undefined in spec; the simplest reading)."""
import numpy as np
def run(kch, g=1.3, lam_c=1.0, beta=10.0, ac=0.1, T=600, seed=0):
    rng = np.random.default_rng(seed); c = np.array([0.0, 0.0]); 
    for t in range(T):
        q = lam_c * c; p = np.exp(beta * (q - q.max())); p /= p.sum(); j = rng.choice(2, p=p)
        for x in range(2): c[x] += ac * (-c[x] + g * np.tanh(c[x]) + kch * (x == j))
    return c
print("two options with IDENTICAL exogenous value and no counter/pro evidence; 200 runs; fraction with |c_1 - c_2| > 0.8 at T=600 (one option 'committed' by choice alone)")
print("| k_ch | g | lam_c | fraction committed |\n|---|---|---|---|")
for kch in (0.0, 0.02, 0.05, 0.1):
    for g in (1.1, 1.3):
        for lc in (0.3, 1.0):
            cs = np.array([run(kch, g, lc, seed=s) for s in range(200)]); print(f"| {kch} | {g} | {lc} | {np.mean(np.abs(cs[:,0]-cs[:,1])>0.8):.2f} |")
