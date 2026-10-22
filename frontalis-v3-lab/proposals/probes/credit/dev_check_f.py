"""Dev trial for v11: learning sanity only (no F-G3a quantity is computed)."""
import sys, time
import numpy as np
import sim_features as F
t0 = time.time(); a, b = map(int, sys.argv[1:3]); zero_frac, cmax, probe_zero = [], [], 0
for s in range(a, b):
    A = F.world(s)
    for beta in (0.0, 2.0, 4.0):
        ag = F.grow(beta, A, s, s, partner=2)
        r = np.random.default_rng([s, 9]); xs = [F.sample(A, r)[1] for _ in range(300)]
        zero_frac.append(np.mean([ag.pres(x).sum() == 0 for x in xs])); cmax.append(np.abs(ag.c).max())
        probe_zero += int(ag.pres(np.eye(F.K)[0] @ A).sum() == 0) + int(ag.pres(np.eye(F.K)[2] @ A).sum() == 0)
print(f"agents={len(cmax)} mean share of events with no perceived feature={np.mean(zero_frac):.3f} max|c|={max(cmax):.2f} probes with no perceived feature={probe_zero}/{2*len(cmax)} time={time.time()-t0:.0f}s")
