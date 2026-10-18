import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, 'v5_nv3.py')).read().split("cfgs = {")[0])
rows = {'q fixed 1e-6 (x0.01 of default)': dict(qmode='fixed', q0=dict(ind=1e-6, cls=2e-7, gen=5e-8)),
        'q fixed default (1e-4)': dict(qmode='fixed'),
        'EMA eta=.002': dict(qmode='ema', eta_q=0.002), 'EMA eta=.005': dict(qmode='ema', eta_q=0.005), 'EMA eta=.01': dict(qmode='ema', eta_q=0.01)}
print("| q rule | stable m=20 | noisy m=20 | switch m=20 | stable m=100 | switch m=100 | stationary q (cls) in stable world | q in switch world |\n|---|---|---|---|---|---|---|---|")
def qstat(kind, **kw):
    hk = HK(Kout=1, **kw); hk.feat('activity', 'gen'); hk.feat('music', 'cls'); hk.feat('song1', 'ind'); rng = np.random.default_rng(0); z = {'activity': 1.0, 'music': 1.0, 'song1': 1.0}
    for r in history(kind, 300, rng): hk.update(z, np.array([r]))
    return hk.q[0, 1]
for nm, kw in rows.items():
    a = retention('stable', seeds=40, **kw)[0]; b = retention('noisy', m_list=(20,), seeds=40, **kw)[0]; c = retention('switch', seeds=40, **kw)[0]
    print(f"| {nm} | {a[0]:.2f} | {b[0]:.2f} | {c[0]:.2f} | {a[1]:.2f} | {c[1]:.2f} | {qstat('stable', **kw):.5f} | {qstat('switch', **kw):.5f} |")
print("\nG1' for EMA eta=.002 (retention m=20, stable vs switch; coefficient elasticities)")
base = dict(qmode='ema', eta_q=0.002)
def ret20(kind, **kw): return max(retention(kind, m_list=(20,), seeds=60, **kw)[0][0], 1e-3)
rs, rv = ret20('stable', **base), ret20('switch', **base); Ev = np.log(rs / rv) / np.log(0.2 / 0.06); print(f"   ret stable {rs:.3f} vs switch {rv:.3f}: E_evid={Ev:+.2f}")
for cname, mod in (('eta_q', lambda f: {'eta_q': 0.002 * f}), ('R', lambda f: {'R': 0.004 * f}), ('var_ind', lambda f: {'var': dict(ind=0.25 * f, cls=0.04, gen=0.01)})):
    lo = ret20('stable', **{**base, **mod(0.5)}); hi = ret20('stable', **{**base, **mod(2.0)}); E = np.log(hi / lo) / np.log(4)
    print(f"   coef {cname:7s}: ret(x/2)={lo:.3f} ret(2x)={hi:.3f} E_coef={E:+.2f} ratio {abs(E)/max(abs(Ev),1e-9):.2f} {'PASS' if abs(E) <= .25*abs(Ev) else 'FAIL'}")
