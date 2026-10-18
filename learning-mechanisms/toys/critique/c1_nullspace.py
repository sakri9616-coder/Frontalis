"""C1: does concern-weighted allocation s_x ∝ (|U z|+eps)^rho stay in the null space? (scratchpad only)"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# ---- (a) exactly redundant, binary z, noisy exogenous teacher
def run_red(seed, rho, T=4000, alpha=0.1, noise=0.3, mu=1.0, eps=0.05, d0=0.02):
    rng = np.random.default_rng(seed)
    u = np.array([mu/2 + d0/2, mu/2 - d0/2])
    for t in range(T):
        r = mu + noise*rng.standard_normal()
        err = r - u.sum()
        c = np.abs(u) + eps
        s = c**rho / np.sum(c**rho)
        u = u + alpha*s*err
    return u
print("(a) exact redundancy, binary z, teacher mean 1.0 noise 0.3, tiny initial asymmetry d0=0.02")
for rho in (0.0, 1.0, 3.0):
    us=[run_red(s, rho) for s in range(40)]; sh = np.array([u[0]/u.sum() for u in us])
    print(f"  rho={rho}: share of u1 over 40 seeds: mean {sh.mean():.2f}, sd {sh.std():.2f}, frac(>0.9 or <0.1)={np.mean((sh>0.9)|(sh<0.1)):.2f}; sum u1+u2 ~ {us[0].sum():.2f}")

# ---- (b) deterministic version: result depends on rho and d0 only
print("(b) noiseless: final share of u1 vs initial asymmetry d0 and rho")
for rho in (0.0, 1.0, 3.0):
    row = []
    for d0 in (0.0, 0.01, 0.1, -0.1):
        u = run_red(0, rho, noise=0.0, d0=d0, T=3000)
        row.append(u[0]/u.sum())
    print(f"  rho={rho}: d0=(0,0.01,0.1,-0.1) -> share {np.round(row,3)}")

# ---- (c) non-binary z: total step of prediction depends on s
print("(c) non-binary z=(1,0.1): predicted-sum step per unit error = sum_x s_x z_x^2")
z = np.array([1.0, 0.1])
for s in ([.9,.1],[.5,.5],[.1,.9]):
    s = np.array(s); print("   s=",s," sum s z^2 =", round(float((s*z*z).sum()),3))
z = np.array([0.5, 0.5])   # responsibilities r summing to 1 used as z
print("   z=r=(0.5,0.5): sum s z^2 =", 0.25, "for every s (rate is 4x slower than binary case)")

# ---- (d) overlapping, non-realizable design: fixed point depends on rho
def run_ov(rho, A0, B0, T=20000, alpha=0.05, eps=0.02, seed=0):
    rng = np.random.default_rng(seed)
    u = np.array([A0, B0], float)
    ev = [((0,1), 1.0), ((0,), 0.3), ((1,), 0.3)]
    p = [0.5, 0.25, 0.25]
    for t in range(T):
        k = rng.choice(3, p=p); act, y = ev[k]; act = list(act)
        err = y - u[act].sum()
        c = np.abs(u[act]) + eps
        s = c**rho / np.sum(c**rho)
        u[act] += alpha*s*err
    return u
print("(d) overlapping non-realizable design: events {A,B}->1.0 (p.5), {A}->0.3, {B}->0.3. final (A,B)")
for rho in (0.0, 1.0, 3.0):
    print(f"  rho={rho}: init (0.05,0.0)->", np.round(run_ov(rho,0.05,0.0),3), " init (0,0.05)->", np.round(run_ov(rho,0.0,0.05),3),
          " init (0.0,0.0)->", np.round(run_ov(rho,0.0,0.0),3))
# gain: linearized log-share elasticity at symmetric point
print("  elasticity d log(s_A/s_B)/d log(A/B) = rho (so rho=3 > 1 => symmetric point is a pitchfork, rho<=1 not necessarily)")
# (e) G3-like: tag dropout 5% flips winner?
def run_ov_tag(rho, seed, drop=0.05, T=8000, alpha=0.05, eps=0.02):
    rng = np.random.default_rng(seed)
    u = np.array([0.02, 0.0])
    ev = [((0,1), 1.0), ((0,), 0.3), ((1,), 0.3)]; p=[.5,.25,.25]
    for t in range(T):
        k = rng.choice(3, p=p); act, y = ev[k]; act=list(act)
        if len(act)==2 and rng.random()<drop: act=[rng.choice(2)]   # LLM drops one tag
        err = y - u[act].sum(); c = np.abs(u[act])+eps; s=c**rho/np.sum(c**rho); u[act]+=alpha*s*err
    return u
for rho in (0.0, 3.0):
    sh = np.array([ (lambda u: u[0]/(u[0]+u[1]+1e-9))(run_ov_tag(rho, s)) for s in range(30)])
    print(f"(e) rho={rho}: share(A) under 5% tag dropout, 40 seeds: mean {sh.mean():.2f} sd {sh.std():.2f}, frac A-wins(>0.7)={np.mean(sh>0.7):.2f} B-wins(<0.3)={np.mean(sh<0.3):.2f}")
