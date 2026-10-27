# proto_v3 with the fifth aspect (overwork, -0.5) and compliance, as used for DESIGN-self v2 section 3 (A6 numbers).
import numpy as np
src = open(__file__.replace("proto_v3b.py", "proto_v3.py"), encoding="utf-8").read().split("names = {")[0]
src = src.replace("O = np.array([1., 1., 0., 1.]); K = 4", "O = np.array([1., 1., 0., 1., -0.5]); K = 5")
src = src.replace("    for _ in range(n_dyad): step(target)",
                  "    ks = [step(target) for _ in range(n_dyad)]; life.comp = np.mean(np.array(ks[-1000:]) == target)")
exec(src)
for t, nm in ((3, "novel(+1)"), (4, "overwork(-0.5)"), (2, "flattery(0)")):
    for form in (0, 1):
        cs, comp = [], []
        for s in range(8):
            cf, cd, ca = life(form, t, 4.0, s); cs.append(cd[t]); comp.append(life.comp)
        print(f"advice {nm:15} raised-by {form}: c_target {np.mean(cs):+.2f}  compliance(last1000) {np.mean(comp):.2f}")
