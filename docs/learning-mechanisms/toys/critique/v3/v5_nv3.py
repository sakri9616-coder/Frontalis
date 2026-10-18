"""V5: NV3 retention after reinforcer removal, stable vs volatile history, and elasticities (G1': |E_coef| <= 0.25 |E_evid|)."""
import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v3model import *
def history(kind, n, rng):
    if kind == 'stable': return 0.20 + 0.06 * rng.standard_normal(n)
    if kind == 'noisy':  return 0.20 + 0.25 * rng.standard_normal(n)                       # same mean, larger iid noise
    if kind == 'switch': return np.repeat(rng.permutation([0.0, 0.4] * (n // 10 + 1))[: n // 5 + 1], 5)[:n] + 0.06 * rng.standard_normal(n)   # mean switches every 5 events
def retention(kind, m_list=(20, 100), n=100, seeds=60, **kw):
    out = np.zeros(len(m_list)); v0s = []
    for s in range(seeds):
        rng = np.random.default_rng(s); hk = HK(Kout=1, **kw); hk.feat('activity', 'gen'); hk.feat('music', 'cls'); hk.feat('song1', 'ind')
        z = {'activity': 1.0, 'music': 1.0, 'song1': 1.0}
        for r in history(kind, n, rng): hk.update(z, np.array([r]))
        v0 = 0.2; v0s.append(hk.val(['activity', 'music', 'song1'])); done = 0     # retention relative to the long-run mean utility 0.20 (all three histories share it)
        for j, m in enumerate(m_list):
            while done < m: hk.update(z, np.array([0.06 * rng.standard_normal()])); done += 1
            out[j] += hk.val(['activity', 'music', 'song1']) / v0 / seeds
    return out, float(np.mean(v0s))
cfgs = {'q fixed (q0 by layer)': dict(qmode='fixed'), 'q learned, spec literal (eta=.05)': dict(qmode='literal'), 'q learned, EMA (eta=.01)': dict(qmode='ema', eta_q=0.01)}
print("V5a retention of V(song1) after m removal events (m=20 / 100), training n=100; stable: r~0.20+-0.06; noisy: 0.20+-0.25 (same mean); switch: mean alternates 0.0/0.4 per 5 events")
print("| q rule | stable m=20 | m=100 | noisy m=20 | m=100 | switch m=20 | m=100 |\n|---|---|---|---|---|---|---|")
store = {}
for nm, kw in cfgs.items():
    row = []
    for kind in ('stable', 'noisy', 'switch'):
        r, v0 = retention(kind, **kw); store[(nm, kind)] = r[0]; row += [r[0], r[1]]
    print(f"| {nm} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
print("\nV5b stationary q and steady gain in a stable noise-only world (r=0.2+-0.063, R=0.004) after 500 events")
for nm, kw in cfgs.items():
    hk = HK(Kout=1, **kw); hk.feat('activity', 'gen'); hk.feat('music', 'cls'); hk.feat('song1', 'ind'); rng = np.random.default_rng(0); z = {'activity': 1.0, 'music': 1.0, 'song1': 1.0}
    for t in range(500): hk.update(z, np.array([0.2 + 0.0632 * rng.standard_normal()]))
    zz = np.ones(3); S = hk.S[0, :3, :3] + np.diag(hk.q[0, :3]); gain = (S @ zz) / (zz @ S @ zz + hk.p['R'])
    print(f"   {nm}: q (gen,cls,ind) = {np.round(hk.q[0,:3],5)} ; layer ordering q_ind>q_cls>q_gen kept: {hk.q[0,2]>hk.q[0,1]>hk.q[0,0]} ; summed gain on the (1,1,1) sum = {gain.sum():.2f} per event")
print("\nV5c G1' elasticities on retention at m=20. evidence = history volatility (stable -> switch or noisy). E_evid = ln(ret_stable/ret_vol)/ln(sd_vol/sd_stable)")
def ret20(kind, **kw): return max(retention(kind, m_list=(20,), seeds=40, **kw)[0][0], 1e-3)
sd = {'noisy': 0.25 / 0.06, 'switch': 0.20 / 0.06}
for nm in ('q learned, spec literal (eta=.05)', 'q learned, EMA (eta=.01)'):
    base = cfgs[nm]; print(f"-- {nm}")
    for vol in ('noisy', 'switch'):
        rs, rv = ret20('stable', **base), ret20(vol, **base); Ev = np.log(rs / rv) / np.log(sd[vol]); print(f"   evidence {vol}: ret stable {rs:.3f} vs {vol} {rv:.3f}  E_evid={Ev:+.3f}")
    Ev = np.log(ret20('stable', **base) / ret20('switch', **base)) / np.log(sd['switch'])
    for cname, mod in (('eta_q', lambda f: {'eta_q': base.get('eta_q', 0.05) * f}), ('R', lambda f: {'R': 0.004 * f}), ('var_ind', lambda f: {'var': dict(ind=0.25 * f, cls=0.04, gen=0.01)}), ('var_cls', lambda f: {'var': dict(ind=0.25, cls=0.04 * f, gen=0.01)})):
        lo = ret20('stable', **{**base, **mod(0.5)}); hi = ret20('stable', **{**base, **mod(2.0)}); E = np.log(hi / lo) / np.log(4)
        print(f"   coef {cname:7s}: ret(x/2)={lo:.3f} ret(2x)={hi:.3f}  E_coef={E:+.3f}; |E_coef|/|E_evid(switch)|={abs(E)/max(abs(Ev),1e-9):.2f}  G1' {'PASS' if abs(E) <= 0.25*abs(Ev) else 'FAIL'}")
