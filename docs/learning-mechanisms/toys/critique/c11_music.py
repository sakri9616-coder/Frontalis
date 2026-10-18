"""C11: §16 'music becomes liked': a NEW concept (music, U=0) always co-occurs with an already valued concept (friend, u=0.5) in rewarding events.
Does concern-weighted allocation s ∝ (|U z|+eps)^rho let music acquire value? Teacher = exogenous Delta h. Scratchpad only."""
import numpy as np
def run(rho, p_alone=0.0, together_target=0.8, friend_alone=0.5, eps=0.05, T=6000, alpha=0.1, seed=0, friend0=0.5):
    rng=np.random.default_rng(seed); u=np.array([friend0,0.0])   # [friend, music]
    for t in range(T):
        if rng.random()<p_alone: act=[0]; y=friend_alone
        else: act=[0,1]; y=together_target
        err=y-u[act].sum(); c=np.abs(u[act])+eps; s=c**rho/np.sum(c**rho); u[act]+=alpha*s*err
    return u
print("friend established at 0.5; together-events pay 0.8 (music's true marginal effect = +0.3 given friend); friend-alone pays 0.5")
for p_alone in (0.0,0.3):
    for rho in (0.0,1.0,3.0):
        u=run(rho,p_alone); print(f"  p(friend alone)={p_alone}: rho={rho}: friend={u[0]:.2f} music={u[1]:.2f}   (true: friend .5, music .3)")
print("eps sweep (rho=3, p_alone=0.3):", {e: np.round(run(3.0,0.3,eps=e),2).tolist() for e in (0.02,0.05,0.2,0.5)})
