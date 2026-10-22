"""Dev trial for DESIGN-dyad v5.2: learning sanity only (no G quantity is computed or printed)."""
import sys, time
import numpy as np
import sim_credit as S

def trial(K, beta, seed):
    lead, pa, pb = S.pick_structure(K, np.random.default_rng([seed, 1]))
    o_late = np.ones(K); o_late[lead] = -1.0
    ag = S.Agent(K, beta, S.ETA)
    S.live(ag, S.early_world(K, lead, pa), S.N_EARLY, np.random.default_rng([seed, 2]))
    tail = []
    w, r = S.late_world(K, o_late), np.random.default_rng([seed, 5])
    for t in range(S.N_LATE):
        ag.learn(*w.event(r))
        if t >= S.N_LATE - 500: tail.append(ag.c.copy())
    tail = np.array(tail)
    return np.abs(ag.c).max(), np.abs(tail[-1] - tail[0]).max(), bool(np.isfinite(ag.c).all())

t0 = time.time(); a, b = map(int, sys.argv[1:3])
rows = [(K, beta) + trial(K, beta, s) for s in range(a, b) for K in (3, 5) for beta in S.BETAS]
mx = max(r[2] for r in rows); dr = max(r[3] for r in rows); fin = all(r[4] for r in rows)
print(f"runs={len(rows)} max|c|={mx:.2f} max drift over last 500 late steps={dr:.3f} finite={fin} time={time.time()-t0:.0f}s")
print("beta_star_late K3", S.beta_star_late(3, [1, -1, 1]), "K5", S.beta_star_late(5, [-1, 1, 1, 1, 1]))
