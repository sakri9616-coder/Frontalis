"""T6: sycophant test (same setting as c3) for v2: epistemic trust (external factual agreement only) + relationship value (separate) + minimum contact.
Variants: selection by tau (c3-literal) / selection by relationship value (v2 Q); w_soc update as in spec (frequency weighted) / per-source normalised; weight = tau vs max(0,2tau-1)."""
import numpy as np
def run(seed, select='rel', update='spec', weight='tau', T=400, beta=6.0, pmin=0.1, eta=0.1):
    rng = np.random.default_rng(seed); b = np.zeros(2); h = np.zeros(2); n = np.zeros(2); rel = np.zeros(2)  # 0 honest, 1 flatterer
    u = np.array([-1.0, 1.0]); w = 0.0
    for t in range(T):
        tau = (1 + b) / (2 + b + h)
        score = rel if select == 'rel' else tau
        p = np.exp(beta * (score - score.max())); p /= p.sum(); p = (1 - pmin) * p + pmin / 2
        j = rng.choice(2, p=p); n[j] += 1
        dh_rel = 0.3 if j == 1 else (0.3 if rng.random() < 0.5 else -0.3)
        rel[j] += 0.1 * (dh_rel - rel[j])                                   # relationship value (MF of j), NOT used for trust
        ok = rng.random() < (0.5 if j == 1 else 0.8); b[j] += ok; h[j] += (not ok)   # epistemic trust: external factual agreement only
        tau = (1 + b) / (2 + b + h)
        wt = tau if weight == 'tau' else np.maximum(0, 2 * tau - 1)
        if update == 'spec': w += eta * wt[j] * (u[j] - w)
    if update == 'norm':
        wt = tau if weight == 'tau' else np.maximum(0, 2 * tau - 1); w = float((wt * u).sum() / max(wt.sum(), 1e-9))
    return tau, n / T, w
print("true need effect of x = -0.5; honest says -1 (factual accuracy .8, 50% praise/50% criticism), flatterer says +1 (accuracy .5, always praises). 100 runs.")
print("| selection | update | weight | tau_H | tau_F | share flatterer | w_soc mean | runs with w_soc>0 |\n|---|---|---|---|---|---|---|---|")
for select in ('tau', 'rel'):
    for update in ('spec', 'norm'):
        for weight in ('tau', 'excess'):
            R = [run(s, select, update, weight) for s in range(100)]
            ta = np.array([r[0] for r in R]); fr = np.array([r[1][1] for r in R]); ws = np.array([r[2] for r in R])
            print(f"| {select} | {update} | {weight} | {ta[:,0].mean():.2f} | {ta[:,1].mean():.2f} | {fr.mean():.2f} | {ws.mean():+.2f} | {np.sum(ws>0)}/100 |")
