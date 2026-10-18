import sys; import os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, 'v5_nv3.py')).read().split("cfgs = {")[0])
print("fixed q (all three layers = q_ind; cls = q/5, gen = q/20): retention at m=20 / 100 for the stable history; steady gain")
for q in (1e-7, 1e-6, 1e-5, 1e-4):
    kw = dict(qmode='fixed', q0=dict(ind=q, cls=q / 5, gen=q / 20)); r = retention('stable', seeds=40, **kw)[0]; print(f"   q_ind={q:g}: ret20={r[0]:.3f} ret100={r[1]:.3f}")
