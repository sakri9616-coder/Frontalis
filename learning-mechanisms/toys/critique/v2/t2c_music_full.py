"""T2c/T2e: music starvation with the FULL readout (belief-MB sees the same events). h reset to baseline before each event so r is deterministic."""
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
FR, MU = IDX['friendA'], IDX['music']
def ev(z): return ('t', vec(**{k: 1 for k in z}), vec(**{k: 1 for k in z}), None)
s = PARAMS['s']; h0 = PARAMS['h0']
def dh_for(r):   # competence-need change that yields hedonic r at baseline
    Dh = -s * np.log(1 - r / (s * np.exp(-h0 / s))); return np.array([0, Dh, 0, 0, 0.0])
def run(p_alone, T=400, pre=150, seed=0, **kw):
    c = V2(**kw); rng = np.random.default_rng(seed)
    def step(z, r):
        c.h = np.full(K, h0); c.experience(ev(z), dh=dh_for(r))
    for t in range(pre): step(['friendA'], .5)
    for t in range(T): step(['friendA'], .5) if rng.random() < p_alone else step(['friendA', 'music'], .8)
    return c
print("T2c full readout, p_alone=0.3 (true marginal: friend 0.5 / music 0.3; the marginal belief gives friend ~0.71 / music 0.8)")
for RB in (0.004, 0.04, 0.4):
    c = run(0.3, commit=False, q=1e-4, RB=RB)
    print(f"   RB={RB}: music V_MF={c.V_MF(MU):.3f} V_MB={c.V_MB(MU):.3f} omega={c.omega(MU):.3f} -> U_val={c.U_val(MU):.3f} | friend V_MF={c.V_MF(FR):.3f} V_MB={c.V_MB(FR):.3f} omega={c.omega(FR):.3f} U_val={c.U_val(FR):.3f}")
print("   always together (p_alone=0), no friend pre-training, q=1e-4:")
for RB in (0.004, 0.4):
    c = run(0.0, pre=0, commit=False, q=1e-4, RB=RB); print(f"   RB={RB}: music V_MF={c.V_MF(MU):.3f} V_MB={c.V_MB(MU):.3f} omega={c.omega(MU):.3f} U_val={c.U_val(MU):.3f}")
print("T2e EC ON, RB=0.4:")
c = run(0.3, commit=False, ec=True, q=1e-4, RB=0.4); print(f"   Q(music)={c.Q(MU):.3f} = U_val {c.U_val(MU):.3f} + lam_EC*vEC {0.5*c.vEC[MU]:.3f}")
