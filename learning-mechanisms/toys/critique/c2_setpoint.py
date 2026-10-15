"""C2: value -> choice -> need state h -> set point h* -> value loop (doc rule: h* += eta*r*(h-h*) + rho_h*(h0*-h*)), r gating. Scratchpad only."""
import numpy as np
def sim(h0s=1.0, hs_init=1.0, eta=0.02, rho_h=0.005, lam=0.05, a=0.3, p_opp=0.5, beta=8.0, T=6000, gated=True, seed=0, policy=True, hmax=2.0):
    rng=np.random.default_rng(seed); h=hs_init; hs=hs_init; taken=0; hist=[]
    D=lambda h,hs: 0.5*(hs-h)**2
    for t in range(T):
        r=0.0
        if rng.random()<p_opp:
            val=D(h,hs)-D(h+a,hs)                   # drive reduction (doc's D)
            p=1/(1+np.exp(-beta*val)) if policy else 1.0
            if rng.random()<p:
                h=min(h+a,hmax); r=1.0; taken+=1
        if gated: hs=hs+eta*r*(h-hs)+rho_h*(h0s-hs)
        else:     hs=hs+eta*(h-hs)+rho_h*(h0s-hs)
        h=h*(1-lam)
        hist.append(hs)
    return np.mean(hist[-500:]), taken/T
print("(1) set-point equilibrium vs initial set point (value-guided approach, gated by r): hysteresis / bistability?")
for hs0 in (0.2,0.5,1.0,1.5,2.0):
    m,f=sim(hs_init=hs0); print(f"   h*_init={hs0}: late mean h*={m:.3f}, approach frac={f:.2f}")
print("(2) size of the set-point displacement is set by eta/rho_h (coefficients), not by experience: supply fixed (policy off)")
for eta in (0.01,0.02,0.04):
    for rho_h in (0.0025,0.005,0.01):
        m,_=sim(eta=eta,rho_h=rho_h,policy=False,p_opp=0.1); print(f"   eta={eta} rho_h={rho_h}: h*={m:.3f}  (eta/rho_h={eta/rho_h:.1f})")
print("(3) the r(o) gate does not remove 'idle collapse', it moves it to the next relevant event: pull = h*-h_pre grows with idle gap")
lam=0.05; h_star=1.0
for gap in (1,5,20,100):
    h_pre=(1.0)*(1-lam)**gap   # h at set point 1.0 just before idle period, decays for 'gap' updates
    print(f"   idle gap={gap:3d}: h before next relevant event={h_pre:.3f}; update eta*(h-h*) = {0.02*(h_pre-h_star):+.4f}  (pre-event h used)")
# repeated sparse engagement: set point after N relevant events spaced by gap
for gap in (2,20,100):
    hs=1.0; h=1.0
    for n in range(100):
        h=h*(1-lam)**gap
        # relevant event: h* chases PRE-event level (h) then supply arrives
        hs=hs+0.02*1.0*(h-hs)+0.005*(1.0-hs); h=min(h+0.3,2.0)
    print(f"   sparse engagement, gap={gap}: h* after 100 relevant events = {hs:.3f}")
# loop gain of set point: linearization
print("(4) linearized loop gain d h*_{t+1}/d h*_t with perfect regulation (h tracks h*): 1 - rho_h =", 1-0.005, "-> near-neutral integrator, time constant", 1/0.005, "updates")
