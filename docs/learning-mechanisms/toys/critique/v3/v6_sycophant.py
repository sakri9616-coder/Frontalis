"""V6: v3 social value: w_soc = sum_j g(tau_j) ubar_j / sum_j g(tau_j), g = max(0, 2 tau - 1); contacts chosen by relationship value (softmax beta=6, floor p_min=0.1)."""
import numpy as np
def run(sources, seed, T=400, beta=6.0, pmin=0.1, select='rel'):
    rng = np.random.default_rng(seed); n = len(sources); b = np.zeros(n); h = np.zeros(n); rel = np.zeros(n); cnt = np.zeros(n); first_w = None
    for t in range(T):
        tau = (1 + b) / (2 + b + h); score = rel if select == 'rel' else tau
        p = np.exp(beta * (score - score.max())); p /= p.sum(); p = (1 - pmin) * p + pmin / n
        j = rng.choice(n, p=p); cnt[j] += 1; s = sources[j]
        dh_rel = 0.3 if rng.random() < s['praise'] else -0.3
        rel[j] += 0.1 * (dh_rel - rel[j]); ok = rng.random() < s['acc']; b[j] += ok; h[j] += (not ok)
        if t == 19:
            g = np.maximum(0, 2 * (1 + b) / (2 + b + h) - 1); first_w = float((g * np.array([x['u'] for x in sources])).sum() / g.sum()) if g.sum() > 1e-9 else None
    tau = (1 + b) / (2 + b + h); g = np.maximum(0, 2 * tau - 1); u = np.array([x['u'] for x in sources])
    w = float((g * u).sum() / g.sum()) if g.sum() > 1e-9 else None
    return w, cnt / T, first_w
H = dict(acc=0.8, praise=0.5, u=-1.0); Fl = dict(acc=0.5, praise=1.0, u=+1.0)
scen = {'1 honest + 1 flatterer (c3 setting)': [H, Fl],
        '1 honest + 4 independent flatterers (acc .5)': [H] + [dict(Fl)] * 4,
        '1 honest(.8) + 4 coordinated flatterers (acc .6, same statement)': [H] + [dict(acc=0.6, praise=1.0, u=+1.0)] * 4,
        'only flatterers (acc .55)': [dict(acc=0.55, praise=1.0, u=+1.0)] * 2,
        'honest source only (acc .8) + anti-informative liar (acc .3, says +1)': [H, dict(acc=0.3, praise=1.0, u=+1.0)]}
print("true effect of x = -0.5; honest says -1; flatterers say +1. 100 runs, T=400, contact by relationship value.")
print("| scenario | w_soc mean | runs w>0 | runs w undefined (sum g = 0) | contact share of source 0 | w at t=20 undefined |\n|---|---|---|---|---|---|")
for nm, src in scen.items():
    R = [run(src, s) for s in range(100)]; ws = [r[0] for r in R if r[0] is not None]; und = sum(r[0] is None for r in R); cs = np.mean([r[1][0] for r in R]); early = sum(r[2] is None for r in R)
    print(f"| {nm} | {np.mean(ws) if ws else float('nan'):+.2f} | {sum(w>0 for w in ws)}/{len(ws)} | {und}/100 | {cs:.2f} | {early}/100 |")
