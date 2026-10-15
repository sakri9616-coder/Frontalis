"""V2: sign of the value of friend A (a NEW individual of the 'collaborator' class) after the section-6 NEW event, R/K/X; A inherits the persona's class weight."""
import sys, copy; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v3model import *
np.set_printoptions(precision=3, suppress=True)
names = {'rescued-by-collaboration': 'R', 'credit-taken': 'K', 'explorer': 'X'}
print("| persona | inherited V(A) before | update direction sign(V_after - V_before) | V(A) after | class (others+people) before -> after | residual sum_k e_k | individual deviation of A (sum over needs) |\n|---|---|---|---|---|---|---|")
rows = {}
for kind in P.KINDS:
    a = run_persona(kind); v0 = a.inherited(); c0 = v0
    # predicted r-hat before the event (to expose the residual)
    d = a.zdict(P.NEW, fresh=False); n = a.hk.n; z = np.zeros(n)
    for nm, v in d.items(): z[a.hk.idx[nm]] = v
    rhat = np.array([z @ a.hk.W[:n, k] for k in range(K)])
    r, err = a.experience(P.NEW); v1 = a.value_of('friendA'); c1 = a.inherited(); dev = a.hk.val(['friendA'])
    rows[names[kind]] = (v0, v1, err)
    print(f"| {names[kind]} | {v0:+.3f} | {'+' if v1 - v0 > 0 else '-'} ({v1 - v0:+.3f}) | {v1:+.3f} | {c0:+.3f} -> {c1:+.3f} | {err.sum():+.3f} | {dev:+.3f} |")
print("\nper-need r (same for the persona's state), r-hat before, residual e (R, K, X):")
for kind in P.KINDS:
    a = run_persona(kind); d = a.zdict(P.NEW, fresh=False); n = a.hk.n; z = np.zeros(n)
    for nm, v in d.items(): z[a.hk.idx[nm]] = v
    rhat = np.array([z @ a.hk.W[:n, k] for k in range(K)]); r, err = a.experience(P.NEW)
    print(f"  {names[kind]}: r={np.round(r,3)} rhat={np.round(rhat,3)} e={np.round(err,3)}  V(A) by need after = {np.round(np.array([a.hk.W[a.hk.idx['friendA'],k]+a.hk.W[a.hk.idx['others'],k]+a.hk.W[a.hk.idx['people'],k] for k in range(K)]),3)}")
print("\nWhat would flip K negative? scale the authorship loss of the NEW event (out authorship = -0.8*s) and the number of credit-stolen events in K's history")
for s in (1, 2, 4, 8):
    ev = list(P.NEW); out = ev[2].copy(); out[IDX['authorship']] = -0.8 * s; ev[2] = out; ev = tuple(ev)
    a = run_persona('credit-taken'); v0 = a.inherited(); a.experience(ev); print(f"  authorship loss x{s}: V(A) before {v0:+.3f} after {a.value_of('friendA'):+.3f}")
print("K history variants (stolen events only, no later solo success):")
for nst in (8, 24):
    hist = [e for e in P.history('credit-taken') if e[0].startswith('credit-stolen')]
    a = V3Agent()
    for i in range(nst): a.experience(hist[i % len(hist)])
    v0 = a.inherited(); a.experience(P.NEW); print(f"  {nst} stolen events: V(A) before {v0:+.3f} after {a.value_of('friendA'):+.3f}")
