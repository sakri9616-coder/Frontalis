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
print("| model | cos | retention 8/32/128 | reversal 32/128 | null ratio |\n|---|---|---|---|---|")
for cls in (mk('v2 core RB=.4 q=0', commit=False, RB=0.4, q=0.0), mk('v2 core RB=.4 q=1e-3', commit=False, RB=0.4, q=1e-3), mk('v2 + c (lam_c=.3,g=1.3) RB=.4 q=0', commit=True, lam_c=0.3, RB=0.4, q=0.0)):
    o = metrics(cls); print(f"| {cls.name} | {o['cos']:.2f} | {o['retention']} | {o['reversal']} | {o['null_ratio']} |")
