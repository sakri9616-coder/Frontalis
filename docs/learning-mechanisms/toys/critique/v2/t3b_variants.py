"""T3b: commitment scale/gain variants (post-hoc, declared) + state-corrected null ratio."""
import sys, io, contextlib, os
sys.dont_write_bytecode = True
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
C8 = os.path.join(HERE, '..', 'c8_simple_alt.py'); src = open(C8).read().split("rows=[S.AMV_L1")[0]
with contextlib.redirect_stdout(io.StringIO()):
    __file__ = C8; exec(src)
os.chdir(HERE)
import v2model as V
def mk(name, **kw):
    class X(V.V2):
        def __init__(self): super().__init__(**kw)
    X.name = name; return X
def null_corrected(cls):
    c = run_persona(cls, 'credit-taken'); h_save = c.h.copy(); c.h = np.full(V.K, V.PARAMS['h0']); p0 = c.probe()[PI].copy(); c.h = h_save
    null = ('null', vec(music=1), vec(music=1), None)
    for i in range(200): c.experience(null)
    c.h = np.full(V.K, V.PARAMS['h0']); p1 = c.probe()[PI]; ex = [j for j, k in enumerate(PROBES) if k != 'music']
    return round(float(np.linalg.norm(p1[ex]) / max(np.linalg.norm(p0[ex]), 1e-9)), 2)
rows = [mk('v2 core', commit=False)]
for lc in (1.0, 0.3, 0.1):
    for g in (1.1, 1.3):
        rows.append(mk(f'v2 + c (lam_c={lc}, g={g})', commit=True, lam_c=lc, g=g))
rows.append(mk('v2 + c (lam_c=0.1, g=1.3, k_e=1)', commit=True, lam_c=0.1, g=1.3, k_e=1.0))
rows.append(mk('v2 + c + EC (lam_c=0.1, g=1.3)', commit=True, ec=True, lam_c=0.1, g=1.3))
print("| model | cos | retention 8/32/128 | reversal 32/128 | null ratio | null ratio (state-corrected, h reset to baseline) |\n|---|---|---|---|---|---|")
for cls in rows:
    o = metrics(cls); print(f"| {cls.name} | {o['cos']:.2f} | {o['retention']} | {o['reversal']} | {o['null_ratio']} | {null_corrected(cls)} |")
