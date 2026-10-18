"""T2: music starvation test (same setting as c11) on the v2 MF Kalman module and on the full readout U_val (MB marginal belief)."""
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v2model import *
np.set_printoptions(precision=3, suppress=True)
FR, MU = IDX['friendA'], IDX['music']
def ev(z): return ('t', vec(**{k: 1 for k in z}), vec(**{k: 1 for k in z}), None)
def run(p_alone, together=0.8, alone=0.5, pre=150, T=400, seed=0, init_friend=True, **kw):
    c = V2(commit=False, **kw); rng = np.random.default_rng(seed)
    r_vec = lambda x: np.array([0, x, 0, 0, 0.0])               # teacher injected on the competence channel; utility-change vector
    if init_friend:
        for t in range(pre): c.experience(ev(['friendA']), r=r_vec(alone))   # friend established at 0.5
    for t in range(T):
        if rng.random() < p_alone: c.experience(ev(['friendA']), r=r_vec(alone))
        else: c.experience(ev(['friendA', 'music']), r=r_vec(together))
    return c
print("T2a  friend established (0.5, 150 pre-trials), music new; together->0.8, friend alone->0.5.  V_MF(friend), V_MF(music) [true 0.5 / 0.3]")
for p_alone in (0.0, 0.3):
    for q in (0.0, 1e-4, 1e-3):
        c = run(p_alone, q=q); print(f"   p_alone={p_alone} q={q:g}: V_MF friend={c.V_MF(FR):.3f} music={c.V_MF(MU):.3f}  sigma2_MF(friend)={c.sig_mf(FR):.4f} (music)={c.sig_mf(MU):.4f}  omega(music)={c.omega(MU):.3f}")
print("T2b  always together, both NEW (equal prior variance) -> split; and with prior-variance ratio (friend established)")
for pre in (0, 150):
    c = run(0.0, pre=pre, init_friend=pre > 0, q=1e-4); print(f"   pre-trials of friend alone={pre}: V_MF friend={c.V_MF(FR):.3f} music={c.V_MF(MU):.3f} (sum {c.V_MF(FR)+c.V_MF(MU):.3f}, target 0.8)")
print("T2c  full readout with belief-MB (event-level marginal belief), p_alone=0.3, q=1e-4: U_val vs V_MF vs V_MB for music, and omega; RB sweep")
for RB in (0.004, 0.04, 0.4):
    c = run(0.3, q=1e-4, RB=RB); print(f"   RB={RB}: music V_MF={c.V_MF(MU):.3f} V_MB={c.V_MB(MU):.3f} omega={c.omega(MU):.3f} U_val={c.U_val(MU):.3f}   friend V_MF={c.V_MF(FR):.3f} V_MB={c.V_MB(FR):.3f} U_val={c.U_val(FR):.3f}")
print("T2d  omega for always-co-occurring concepts as time grows (null-direction variance grows by q per step): sigma2_MF(music), omega")
for T in (100, 400, 1600):
    c = run(0.0, T=T, pre=0, init_friend=False, q=1e-4); print(f"   always together T={T}: sigma2_MF(music)={c.sig_mf(MU):.4f}  omega={c.omega(MU):.4f}  V_MF music={c.V_MF(MU):.3f} friend={c.V_MF(FR):.3f}")
print("T2e  EC ON (gamma_EC=.5, lam_EC=.5): music readout Q with p_alone=0.3")
c = V2(commit=False, ec=True, q=1e-4); rng = np.random.default_rng(0)
r_vec = lambda x: np.array([0, x, 0, 0, 0.0])
for t in range(150): c.experience(ev(['friendA']), r=r_vec(0.5))
for t in range(400):
    if rng.random() < 0.3: c.experience(ev(['friendA']), r=r_vec(0.5))
    else: c.experience(ev(['friendA', 'music']), r=r_vec(0.8))
print(f"   V_MF music={c.V_MF(MU):.3f}, vEC music={c.vEC[MU]:.3f} (=gamma*V_dir(friend)~{0.5*c.U_val(FR):.3f}), Q(music)={c.Q(MU):.3f}")
