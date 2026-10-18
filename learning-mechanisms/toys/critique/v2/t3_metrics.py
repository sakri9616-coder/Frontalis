"""T3: cos / retention / reversal / null ratio on the doc's harness. Same metrics function as c8 (copied by exec), same stream seeds."""
import sys, io, contextlib, os
sys.dont_write_bytecode = True
import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
C8 = os.path.join(HERE, '..', 'c8_simple_alt.py'); src = open(C8).read().split("rows=[S.AMV_L1")[0]
with contextlib.redirect_stdout(io.StringIO()):
    __file__ = C8; exec(src)                      # defines S (sim5), RW, RWC, metrics, run_persona (harness), np ...
os.chdir(HERE)
import v2model as V
def mk(name, **kw):
    class X(V.V2):
        def __init__(self): super().__init__(**kw)
    X.name = name; return X
def mkrwc(g):
    class X(RWC):
        def __init__(self): super().__init__(g=g)
    X.name = f"RW+commit g={g}"; return X
rows = [S.AMV_L1, S.AMV_L2, RW, mkrwc(1.5), mkrwc(1.3),
        mk('v2 core', commit=False, ec=False), mk('v2 + c', commit=True, ec=False), mk('v2 + c + EC', commit=True, ec=True),
        mk('v2 core RB=.4', commit=False, ec=False, RB=0.4), mk('v2 + c RB=.4', commit=True, ec=False, RB=0.4), mk('v2 + c + EC RB=.4', commit=True, ec=True, RB=0.4),
        mk('v2 core q=0', commit=False, ec=False, q=0.0), mk('v2 + c, k_e=1', commit=True, ec=False, k_e=1.0)]
print("| model | cos | retention 8/32/128 | reversal 32/128 | null ratio |\n|---|---|---|---|---|")
for cls in rows:
    o = metrics(cls); print(f"| {cls.name} | {o['cos']:.2f} | {o['retention']} | {o['reversal']} | {o['null_ratio']} |")
