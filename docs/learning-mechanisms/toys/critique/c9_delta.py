"""C9: delta = delta_V + delta_W identity with the doc's Q = sum_k P(y_k=1) Vhat_k when felt/forecast is tanh-nonadditive."""
import numpy as np, itertools
v=np.array([1.5,1.5]); p=np.array([0.5,0.5])
Vhat=lambda y: np.tanh(np.dot(y,v))
Q_doc=sum(p[k]*Vhat(np.eye(2)[k]) for k in range(2))
EP=0.0
for y in itertools.product([0,1],repeat=2):
    P=np.prod([p[k] if y[k] else 1-p[k] for k in range(2)]); EP+=P*Vhat(np.array(y))
print(f"Q (doc, marginal sum) = {Q_doc:.3f};  E_{{o~P}}[Vhat(o)] (joint) = {EP:.3f};  gap = {EP-Q_doc:+.3f}")
f=0.9
print(f"delta = f-Q = {f-Q_doc:+.3f};  delta_V+delta_W = f-E_P[Vhat] = {f-EP:+.3f}; residual (neither value nor world) = {(f-Q_doc)-(f-EP):+.3f}")
