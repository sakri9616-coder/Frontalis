import numpy as np, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import belief_sims as S
# Sim1a sensitivity: prior over concentration kappa (homogeneity assumption)
pm = 0.25
print('Sim1a prior sensitivity: transfer ratio = new-partner shift / same-partner shift (n=1,2,4,8 same | diverse)')
for name, rng_ in [('kappa~logU[0.5,50] (heterogeneous allowed)', (math.log(.5), math.log(50))),
                   ('kappa~logU[5,50]  (people alike)', (math.log(5), math.log(50))),
                   ('kappa~logU[0.5,5] (people differ)', (math.log(.5), math.log(5)))]:
    out = []
    for n in (1, 2, 4, 8):
        s, sf = S.hb_predict([(n, n)], logk_range=rng_); d, df = S.hb_predict([(1, 1)] * n, logk_range=rng_)
        out.append('%.2f|%.2f' % ((s - pm) / (sf - pm), (d - pm) / (df - pm)))
    print(' %-45s %s' % (name, '  '.join(out)))
# Sim1b sensitivity to the feature-scale prior (s0) -> narrowing strength
import belief_sims as S2
orig = S2.lmarg_nig
for s0 in (0.3**2, 0.15**2, 0.05**2):
    def f(xs, mu0=0.5, k0=1.0, nu0=2.0, s0_=s0, _o=orig): return _o(xs, mu0, k0, nu0, s0_)
    S2.lmarg_nig = f
    row = []
    for n in (1, 2, 4, 8):
        near = S2.lc_predict([0.5]*n, [1]*n, 0.5); far = S2.lc_predict([0.5]*n, [1]*n, 0.95)
        row.append('%.2f' % ((far-0.25)/(near-0.25)))
    row2 = []
    for n in (2, 4, 8):
        xs = list(np.linspace(0.3, 0.7, n))
        near = S2.lc_predict(xs, [1]*n, 0.5); far = S2.lc_predict(xs, [1]*n, 0.95)
        row2.append('%.2f' % ((far-0.25)/(near-0.25)))
    print('Sim1b s0=%.4f far/near same n=1,2,4,8: %s | diverse n=2,4,8: %s' % (s0, ' '.join(row), ' '.join(row2)))
S2.lmarg_nig = orig
