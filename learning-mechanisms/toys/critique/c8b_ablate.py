"""C8b: ablate the pieces of the doc's L1/L2 toy (what does allocation rho, cache a_min/n_h, set point actually contribute?)"""
import sys, io, contextlib, os, numpy as np
sys.dont_write_bytecode=True
V=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'value'))
sys.path.insert(0,V); os.chdir(V)
with contextlib.redirect_stdout(io.StringIO()):
    import sim5_amv as S
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import importlib.util
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'c8_simple_alt.py')).read().split("rows=[S.AMV_L1")[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(src)
def mk(base,**kw):
    class X(base):
        def __init__(self): super().__init__(**kw)
    X.name=base.name+str(kw); return X
print("L1 (belief beta + cache + concern allocation): ablations")
for kw in (dict(),dict(rho=0.0),dict(rho=1.0),dict(rho=6.0),dict(amin=0.01),dict(amin=0.04),dict(nh=3.0),dict(nh=30.0)):
    o=metrics(mk(S.AMV_L1,**kw)); print(f"  {str(kw):18s} cos={o['cos']:.2f} retention={o['retention']} reversal={o['reversal']}")
print("L2 (needs + set point + cache): ablations")
for kw in (dict(),dict(rho=0.0),dict(eta=0.0),dict(eta=0.04),dict(rho_h=0.0025),dict(rho_h=0.01)):
    o=metrics(mk(S.AMV_L2,**kw)); print(f"  {str(kw):18s} cos={o['cos']:.2f} retention={o['retention']} reversal={o['reversal']} null={o['null_ratio']}")
