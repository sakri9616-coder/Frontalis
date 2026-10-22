# NPN prototype: the evaluation is a neural network over the TOKENS of an experience (raw cue vectors, no labels).
# Credit principle inside the network: attention over tokens = softmax(beta * v_i) from the network's own current
# token values (detached); the training target of each token's value is alpha_i * o. The encoder is trained by this loss,
# so what the network learns to see is shaped by its own evaluation.
import numpy as np, torch, torch.nn as nn, sys
torch.set_num_threads(1)
K, D, H = 4, 16, 32
class NPN(nn.Module):
    def __init__(self, seed):
        super().__init__(); g = torch.Generator().manual_seed(seed)
        self.enc = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.v = nn.Linear(H, 1)
        for p in self.parameters():
            with torch.no_grad(): p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.dim() > 1 else 0.0))
    def token_values(self, toks): return self.v(self.enc(toks)).squeeze(-1)
def world(ws): return torch.tensor(np.random.default_rng([ws, 0]).normal(0, 1, (K, D)), dtype=torch.float32)
def event(A, rng, lead=None, partner=None, q=0.5):
    a = (rng.random(K) < q).astype(float)
    if lead is not None:
        a[lead] = float(rng.random() < 0.8)
        if partner is not None: a[partner] = a[lead] if rng.random() < 0.8 else float(rng.random() < q)
    idx = np.where(a == 1)[0]
    toks = A[idx] + torch.tensor(rng.normal(0, 0.1, (len(idx), D)), dtype=torch.float32)
    return toks, a.sum() + rng.normal(0, 0.1)
def learn(net, opt, toks, o, beta):
    if len(toks) == 0: return
    v = net.token_values(toks)
    a = torch.softmax(beta * v.detach(), 0)
    loss = ((v - a * o) ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
def grow(beta, A, seed, early_seed, partner, n=2000):
    net = NPN(seed); opt = torch.optim.SGD(net.parameters(), lr=0.02); r = np.random.default_rng([early_seed, 3])
    for _ in range(n):
        toks, o = event(A, r, lead=1, partner=partner); learn(net, opt, toks, o, beta)
    return net, opt
def same_new(net, opt, A, seed, beta, n_new=100):
    with torch.no_grad(): v0 = net.token_values(A).numpy().copy()
    r = np.random.default_rng([seed, 4])
    for _ in range(n_new):
        toks = A[[0, 2]] + torch.tensor(r.normal(0, 0.1, (2, D)), dtype=torch.float32); learn(net, opt, toks, 2 + r.normal(0, .1), beta)
    with torch.no_grad(): v1 = net.token_values(A).numpy()
    d = v1 - v0; return d[0] - d[2], v0
for beta in (0.0, 2.0, 4.0):
    opp = rt = 0; ex = None
    for s in range(int(sys.argv[1]), int(sys.argv[2])):
        A = world(s)
        na, oa = grow(beta, A, s, s, 2); da, va = same_new(na, oa, A, s, beta)
        nb, ob = grow(beta, A, s, s, 0); db, vb = same_new(nb, ob, A, s, beta)
        nr, orr = grow(beta, A, s, s + 10**6, 2); dr, _ = same_new(nr, orr, A, s, beta)
        opp += da * db < 0; rt += da * dr < 0
        if ex is None: ex = (va.round(2), vb.round(2))
    n = int(sys.argv[2]) - int(sys.argv[1])
    print(f"beta={beta}: opposite Pa-vs-Pb {opp}/{n}  retest opposite {rt}/{n}  | token values after early (seed0) Pa {ex[0]} Pb {ex[1]}")
