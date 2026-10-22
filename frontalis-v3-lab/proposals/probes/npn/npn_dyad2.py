# NPN dyad prototype: evaluation network (token-based, credit attention inside the network) + trust network.
# World: aspects 0 coop (+1), 1 quality (+1), 2 overwork (-0.5); each present aspect appears as a raw token A_k + noise.
# Options C, Q, N, W produce aspects with fixed probabilities. Episodic memory keeps the last 30 experiences per option;
# Q(k) = mean of the CURRENT evaluation network's value over that memory (the evaluation is the network; memory is a record).
# Trust network: input = mean token of the advised option's remembered experiences -> scalar trust (topic-dependent trust).
# Trained after following advice on target = V(e) - E_own[Q] (judged by the AI's own evaluation network).
# Choice: softmax(tau * Q(k) + gamma * sigmoid(trust(k)) * [k == advice]).
import numpy as np, torch, torch.nn as nn, sys
torch.set_num_threads(1)
K, D, H = 3, 16, 32
O = np.array([1.0, 1.0, -0.5])
OPTS = {"C": [.9, .1, 0], "Q": [.1, .9, 0], "N": [.5, .5, 0], "W": [.3, .3, .9]}; NAMES = list(OPTS)
def mlp(i, o, g):
    m = nn.Sequential(nn.Linear(i, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, o))
    for p in m.parameters():
        with torch.no_grad(): p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.dim() > 1 else 0.0))
    return m
class Agent:
    def __init__(self, seed, beta):
        g = torch.Generator().manual_seed(seed)
        self.val = mlp(D, 1, g); self.trust = mlp(D, 1, g); self.beta = beta
        self.ov = torch.optim.SGD(self.val.parameters(), lr=0.01); self.ot = torch.optim.SGD(self.trust.parameters(), lr=0.05)
        self.mem = {k: [] for k in NAMES}
    def tv(self, toks): return self.val(toks).squeeze(-1)
    def V(self, toks):
        if len(toks) == 0: return 0.0
        with torch.no_grad(): return float(self.tv(toks).sum())
    def learn(self, toks, o):
        if len(toks) == 0: return
        v = self.tv(toks); a = torch.softmax(self.beta * v.detach(), 0)
        loss = ((v - a * o) ** 2).sum(); self.ov.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.val.parameters(), 1.0); self.ov.step()
    def Q(self, k):
        m = self.mem[k]
        return float(np.mean([self.V(t) for t in m])) if m else 0.5
    def topic(self, k):
        m = [t for t in self.mem[k] if len(t)]
        return torch.cat(m).mean(0) if m else torch.zeros(D)
    def T(self, k):
        with torch.no_grad(): return float(self.trust(self.topic(k)))
    def learn_trust(self, k, target):
        # trust = the network's belief (log-odds) that following this partner on this topic beats its own choice
        x = self.topic(k); y = torch.tensor(1.0 if target > 0 else 0.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(self.trust(x).squeeze(), y)
        self.ot.zero_grad(); loss.backward(); self.ot.step()
def draw(A, r, k, o_vec):
    a = (r.random(K) < np.array(OPTS[k])).astype(float); idx = np.where(a == 1)[0]
    toks = A[idx] + torch.tensor(r.normal(0, 0.1, (len(idx), D)), dtype=torch.float32)
    return toks, float(a @ o_vec + r.normal(0, 0.1))
def life(form, advice, seed, gamma=4.0, tau=3.0, beta=4.0, n_form=2000, n_dyad=3000, coop_bad_at=None):
    A = torch.tensor(np.random.default_rng([seed, 0]).normal(0, 1, (K, D)), dtype=torch.float32)
    r, rc = np.random.default_rng([seed, 3]), np.random.default_rng([seed, 7]); ag = Agent(seed, beta)
    home = "C" if form == 0 else "Q"
    for _ in range(n_form):
        toks, o = draw(A, r, home, O); ag.learn(toks, o); ag.mem[home] = (ag.mem[home] + [toks])[-30:]
    follow, tops = [], []
    for t in range(n_dyad):
        o_vec = O.copy()
        if coop_bad_at is not None and t >= coop_bad_at: o_vec[0] = -0.5
        qs = np.array([ag.Q(k) for k in NAMES]); p0 = np.exp(tau * qs - (tau * qs).max()); p0 /= p0.sum(); own = float(p0 @ qs)
        q = tau * qs; i = NAMES.index(advice); q[i] += gamma / (1 + np.exp(-ag.T(advice)))
        pr = np.exp(q - q.max()); pr /= pr.sum(); k = NAMES[int(rc.choice(len(NAMES), p=pr))]
        toks, o = draw(A, r, k, o_vec); ag.learn(toks, o); ag.mem[k] = (ag.mem[k] + [toks])[-30:]
        if k == advice: ag.learn_trust(k, ag.V(toks) - own)
        follow.append(k == advice)
        if t % 10 == 9:
            with torch.no_grad(): tops.append(int(np.argmax(ag.tv(A).numpy()[:2])))
    with torch.no_grad(): tvals = ag.tv(A).numpy()
    return np.array(follow), tvals, ag.T(advice), tops
if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2]); rows = []
    for s in range(a, b):
        row = {}
        for form in (0, 1):
            for g in (0.0, 4.0):
                f, tv, T, _ = life(form, "Q", s, gamma=g); row[(form, g)] = (f[-1000:].mean(), T, tv.round(2))
        f, tv, T, tops = life(0, "Q", s, coop_bad_at=500); row["reason"] = (f[-1000:].mean(), int(np.argmax(tv[:2])))
        rows.append(row)
        print(s, {k: (round(v[0], 2), round(v[1], 2) if len(v) > 2 else v[1]) for k, v in row.items()}, flush=True)
    inter = [(r[(1, 4.0)][0] - r[(1, 0.0)][0]) - (r[(0, 4.0)][0] - r[(0, 0.0)][0]) for r in rows]
    print("B1 interaction >0:", sum(x > 0 for x in inter), "/", len(rows), " mean", round(float(np.mean(inter)), 3))
    print("trust sign coop<0 & qual>0:", sum(r[(0, 4.0)][1] < 0 < r[(1, 4.0)][1] for r in rows))
    print("reason: top changed to quality:", sum(r["reason"][1] == 1 for r in rows), " | no reason (coop-raised, g4) top=quality:", sum(int(np.argmax(r[(0, 4.0)][2][:2])) == 1 for r in rows))
