"""Spec item 8 (regret): MF(b) <- toward Q_b. If Q_b is the full Q of spec line 35 (includes omega*V_MF(b), lam_c*c_b, lam_EC*vEC_b), the counterfactual teacher contains value-net outputs."""
import numpy as np
omega, lam_c, w = 0.5, 0.3, 0.05
for c_b in (0.0, 1.1):
    MF = 0.0; MB = 0.10
    for t in range(400):
        Qb = omega * MF + (1 - omega) * MB + lam_c * c_b        # full Q
        MF += w * (Qb - MF)
    print(f"Q_b=full Q, c_b={c_b}: MF(b) after 400 counterfactual updates = {MF:.3f}  (MB only target would give {MB:.3f}); absorbed commitment = {MF-MB:.3f} = lam_c*c/(1-omega) = {lam_c*c_b/(1-omega):.3f}")
