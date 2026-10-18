import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
np.set_printoptions(precision=3, suppress=True)
for kind in P.KINDS:
    c=run_persona(lambda: V2(commit=True,ec=False),kind); print(kind,'h',c.h,'\n  probe',c.probe(),'\n  c',c.c[PI],'omega',[round(c.omega(i),2) for i in PI])
    b=c.probe().copy(); c.experience(P.NEW); print('  after NEW',c.probe(),'delta',c.probe()-b)
