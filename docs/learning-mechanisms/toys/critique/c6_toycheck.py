"""C6: re-read the toy the doc cites for §6 signs (sim5_amv L1/L2): value of friendA before/after the NEW event per persona."""
import sys, io, contextlib, numpy as np
sys.dont_write_bytecode=True
import os; V=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'value'))
sys.path.insert(0,V)
import os; os.chdir(V)
with contextlib.redirect_stdout(io.StringIO()):
    import sim5_amv as S
import prototype as P
from sim1_candidates import run_persona, PROBES, PI
np.set_printoptions(precision=4, suppress=True)
print("probe order:",PROBES)
for cls in (S.AMV_L1, S.AMV_L2):
    print("==",cls.name)
    for k in P.KINDS:
        c=run_persona(cls,k); b=c.probe()[PI].copy(); c.experience(P.NEW); a=c.probe()[PI]
        print(f"  {k:26s} before {b}  after {a}  delta_friendA={a[4]-b[4]:+.4f} delta_authorship={a[1]-b[1]:+.4f}")
