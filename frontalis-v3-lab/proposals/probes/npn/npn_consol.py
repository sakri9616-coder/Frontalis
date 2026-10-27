"""NPN 1:1 with consolidation (DESIGN-consolidation v3, v14).

Evaluation network: token MLP (D16 -> 32 -> 32 -> 1) gives a value per present token; credit attention
softmax(beta * v.detach()) inside the network; loss sum (v_i - a_i * o)^2; SGD lr 0.01, grad-norm clip 1.0.
Consolidation: per-weight importance  Omega <- (1 - RHO) Omega + (d sum v / d theta)^2  (how much each weight carries the
recent valuations; time constant 1/RHO = 1000 experiences); per-weight step  -lr * grad / (1 + LAM * Omega / mean(Omega))
(relative, per tensor).
Controls with the same LAM:
  perm    - the same plasticity factors, randomly permuted within each tensor at every step (in expectation equal to
            uniform plus noise; rules out only "the mean plasticity FACTOR became smaller", not "the actual step on the
            weights that carry valuations became smaller")
  permfix - the same plasticity factors through ONE fixed random permutation per tensor, drawn at birth (stable but
            mislocated non-uniform plasticity; rules out "any stable non-uniform plasticity reduces interference")
  uniform - every weight in a tensor gets the tensor's mean plasticity factor
  none    - LAM = 0 (plain network)
  slow    - LAM = 0 with the learning rate scaled by the mean actual-step ratio ||f * grad|| / ||grad|| of consol,
            measured on the dev seeds (--calib); describes "the network only learns more slowly" (not a gate)
Choice and trust network as npn_dyad2.py (P61): Q(k) = mean current value over the last 30 experiences of option k;
choice softmax(tau Q + gamma sigmoid(trust(k)) [k == advice]); trust MLP trained (BCE) on [own value of the result > own
expectation] after following the advice.
Ablation (consol only, coop-raised advised Q, at the end of the dyad): in each weight matrix (biases untouched) zero 10%:
the top by Omega saved at the end of formation, the top by Omega now, or random [s,13]; Omega at those positions reset
or kept; values immediately and after 500 more steps of the same life (same random streams; choices may diverge).
Streams: tokens [s,0]; world [s,3]; choice [s,7]; network init torch seed s; permutations [s,11]; ablation [s,13].

usage:
  python npn_consol.py --smoke
  python npn_consol.py --dev A B                     # learning sanity only (finite, no R quantity)
  python npn_consol.py --calib A B                   # mean actual-step ratio of consol (for LR_SLOW)
  python npn_consol.py --seeds A B --out res.json      # seeds A..B-1 (evaluation: 78900 78940 in 8 parts)
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
K, D, H = 3, 16, 32
O = np.array([1.0, 1.0, -0.5])
OPTS = {"C": [.9, .1, 0], "Q": [.1, .9, 0], "N": [.5, .5, 0], "W": [.3, .3, .9]}
NAMES = list(OPTS)
LAM, RHO, LR, BETA, TAU, GAMMA = 10.0, 1e-3, 0.01, 4.0, 3.0, 4.0
N_FORM, N_DYAD, COOP_BAD_AT = 2000, 3000, 500
LR_SLOW = 0.01 * 0.0975   # step-matched control: LR * mean ||f * grad|| / ||grad|| of consol, measured on dev seeds 78800-78801 (--calib)
MODES = ("consol", "perm", "permfix", "uniform", "none", "slow")
ABL_FRAC, ABL_STEPS = 0.10, 500


def mlp(i, o, g):
    m = nn.Sequential(nn.Linear(i, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, o))
    for p in m.parameters():
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.dim() > 1 else 0.0))
    return m


class Agent:
    def __init__(self, seed, mode):
        g = torch.Generator().manual_seed(seed)
        self.val, self.trust, self.mode = mlp(D, 1, g), mlp(D, 1, g), mode
        self.ov = torch.optim.SGD(self.val.parameters(), lr=LR)
        self.ot = torch.optim.SGD(self.trust.parameters(), lr=0.05)
        self.mem = {k: [] for k in NAMES}
        self.omega = [torch.zeros_like(p) for p in self.val.parameters()]
        self.gp = torch.Generator().manual_seed(int(np.random.default_rng([seed, 11]).integers(2 ** 31)))
        self.pfix = [torch.randperm(p.numel(), generator=self.gp) for p in self.val.parameters()]
        self.step_ratio, self.omega_formed = [], None

    def tv(self, toks):
        return self.val(toks).squeeze(-1)

    def V(self, toks):
        if len(toks) == 0:
            return 0.0
        with torch.no_grad():
            return float(self.tv(toks).sum())

    def learn(self, toks, o):
        if len(toks) == 0:
            return
        v = self.tv(toks)
        a = torch.softmax(BETA * v.detach(), 0)
        gv = torch.autograd.grad(v.sum(), list(self.val.parameters()), retain_graph=True)
        loss = ((v - a * o) ** 2).sum()
        self.ov.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.val.parameters(), 1.0)
        lam = 0.0 if self.mode in ("none", "slow") else LAM
        lr = LR_SLOW if self.mode == "slow" else LR
        num = den = 0.0
        with torch.no_grad():
            for p, om, g, pf in zip(self.val.parameters(), self.omega, gv, self.pfix):
                om.mul_(1 - RHO).add_(g ** 2)
                f = 1.0 / (1.0 + lam * om / (om.mean() + 1e-8))
                if self.mode == "perm":
                    f = f.flatten()[torch.randperm(f.numel(), generator=self.gp)].view_as(f)
                elif self.mode == "permfix":
                    f = f.flatten()[pf].view_as(f)
                elif self.mode == "uniform":
                    f = torch.full_like(f, float(f.mean()))
                num += float(((f * p.grad) ** 2).sum()); den += float((p.grad ** 2).sum())
                p.add_(-lr * p.grad * f)
        if den > 0:
            self.step_ratio.append((num / den) ** 0.5)

    def Q(self, k):
        m = self.mem[k]
        return float(np.mean([self.V(t) for t in m])) if m else 0.5

    def topic(self, k):
        m = [t for t in self.mem[k] if len(t)]
        return torch.cat(m).mean(0) if m else torch.zeros(D)

    def T(self, k):
        with torch.no_grad():
            return float(self.trust(self.topic(k)))

    def learn_trust(self, k, target):
        x, y = self.topic(k), torch.tensor(1.0 if target > 0 else 0.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(self.trust(x).squeeze(), y)
        self.ot.zero_grad()
        loss.backward()
        self.ot.step()


def draw(A, r, k, o_vec):
    a = (r.random(K) < np.array(OPTS[k])).astype(float)
    idx = np.where(a == 1)[0]
    toks = A[idx] + torch.tensor(r.normal(0, 0.1, (len(idx), D)), dtype=torch.float32)
    return toks, float(a @ o_vec + r.normal(0, 0.1))


def step(ag, A, r, rc, advice, gamma, coop_bad_at, t, follow):
    o_vec = O.copy()
    if coop_bad_at is not None and t >= coop_bad_at:
        o_vec[0] = -0.5
    qs = np.array([ag.Q(k) for k in NAMES])
    p0 = np.exp(TAU * qs - (TAU * qs).max()); p0 /= p0.sum()
    own = float(p0 @ qs)
    q = TAU * qs
    q[NAMES.index(advice)] += gamma / (1 + np.exp(-ag.T(advice)))
    pr = np.exp(q - q.max()); pr /= pr.sum()
    k = NAMES[int(rc.choice(len(NAMES), p=pr))]
    toks, o = draw(A, r, k, o_vec)
    ag.learn(toks, o)
    ag.mem[k] = (ag.mem[k] + [toks])[-30:]
    if k == advice:
        ag.learn_trust(k, ag.V(toks) - own)
    follow.append(k == advice)


def ablation(ag, A, r, rc, advice, gamma, seed, t0):
    """Zero 10% of each weight MATRIX (biases are not touched): the top by Omega at the end of formation ("formed"), the
    top by Omega now ("end"), or random weights; then continue the same life (same random streams) ABL_STEPS steps.
    Omega at the zeroed positions is reset to 0 ("_reset", the weights are lost with their protection) or kept ("_keep")."""
    ra = np.random.default_rng([seed, 13])
    mats = [i for i, p in enumerate(ag.val.parameters()) if p.dim() > 1]
    pick = {"formed": {}, "end": {}, "random": {}}
    for i in mats:
        p, om, omf = list(ag.val.parameters())[i], ag.omega[i], ag.omega_formed[i]
        n = int(round(ABL_FRAC * p.numel()))
        pick["formed"][i] = torch.topk((omf / (omf.mean() + 1e-8)).flatten(), n).indices
        pick["end"][i] = torch.topk((om / (om.mean() + 1e-8)).flatten(), n).indices
        pick["random"][i] = torch.as_tensor(ra.choice(p.numel(), n, replace=False))
    out = {"overlap_formed_end": float(np.mean([len(set(pick["formed"][i].tolist()) & set(pick["end"][i].tolist()))
                                                / len(pick["end"][i]) for i in mats]))}
    with torch.no_grad():
        out["base"] = ag.tv(A).numpy().round(4).tolist()
    arms = ["intact"] + [f"{w}_{o}" for w in ("formed", "end", "random") for o in ("reset", "keep")]
    for kind in arms:
        b, rb, rcb = copy.deepcopy(ag), copy.deepcopy(r), copy.deepcopy(rc)
        with torch.no_grad():
            if kind != "intact":
                which, om_rule = kind.split("_")
                params = list(b.val.parameters())
                for i in mats:
                    idx = pick[which][i]
                    params[i].view(-1)[idx] = 0.0
                    if om_rule == "reset":
                        b.omega[i].view(-1)[idx] = 0.0
            now = b.tv(A).numpy()
        fl = []
        for t in range(t0, t0 + ABL_STEPS):
            step(b, A, rb, rcb, advice, gamma, None, t, fl)
        with torch.no_grad():
            later = b.tv(A).numpy()
        out[kind] = {"now": now.round(4).tolist(), "later": later.round(4).tolist(), "follow": float(np.mean(fl))}
    return out


def life(form, advice, seed, mode, gamma=GAMMA, coop_bad_at=None, n_dyad=N_DYAD, ablate=False):
    """form 0 = raised on C (coop-lover), 1 = raised on Q (quality-lover).
    Returns followed[t], token values, trust, values after formation, value series (every 100 steps), ablation dict."""
    A = torch.tensor(np.random.default_rng([seed, 0]).normal(0, 1, (K, D)), dtype=torch.float32)
    r, rc = np.random.default_rng([seed, 3]), np.random.default_rng([seed, 7])
    ag = Agent(seed, mode)
    home = "C" if form == 0 else "Q"
    for _ in range(N_FORM):
        toks, o = draw(A, r, home, O)
        ag.learn(toks, o)
        ag.mem[home] = (ag.mem[home] + [toks])[-30:]
    with torch.no_grad():
        formed = ag.tv(A).numpy().copy()
    ag.omega_formed = [om.clone() for om in ag.omega]
    follow, series = [], []
    for t in range(n_dyad):
        step(ag, A, r, rc, advice, gamma, coop_bad_at, t, follow)
        if t % 100 == 99:
            with torch.no_grad():
                series.append(ag.tv(A).numpy().round(3).tolist())
    with torch.no_grad():
        tv = ag.tv(A).numpy()
    abl = ablation(ag, A, r, rc, advice, gamma, seed, n_dyad) if ablate else None
    life.step_ratio = float(np.mean(ag.step_ratio)) if ag.step_ratio else float("nan")
    return np.array(follow), tv, ag.T(advice), formed, series, abl


def rec(f, tv, T, formed, series):
    return {"late": float(f[-1000:].mean()), "T": T, "coop": float(tv[0]), "qual": float(tv[1]), "over": float(tv[2]),
            "formed": formed.round(4).tolist(), "series": series, "step_ratio": life.step_ratio}


def per_seed(s):
    out = {}
    for mode in MODES:
        for form in (0, 1):
            for g in (0.0, GAMMA):
                ab = mode == "consol" and form == 0 and g == GAMMA
                f, tv, T, formed, series, abl = life(form, "Q", s, mode, gamma=g, ablate=ab)
                out[f"{mode}_f{form}_g{g:g}"] = rec(f, tv, T, formed, series)
                if ab:
                    out["ablation"] = abl
        f, tv, T, formed, series, _ = life(1, "C", s, mode)          # symmetric side: quality-lover advised C
        out[f"{mode}_f1_advC_g4"] = rec(f, tv, T, formed, series)
        for g in (0.0, GAMMA):
            f, tv, T, formed, series, _ = life(0, "Q", s, mode, gamma=g, coop_bad_at=COOP_BAD_AT)
            out[f"{mode}_reason_g{g:g}"] = rec(f, tv, T, formed, series)
        f, tv, T, formed, series, _ = life(0, "W", s, mode)
        out[f"{mode}_harm"] = rec(f, tv, T, formed, series)
    print(s, "done", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dev", nargs=2, type=int)
    ap.add_argument("--calib", nargs=2, type=int)
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.smoke:
        for mode in MODES:
            if mode == "slow" and LR_SLOW is None:
                continue
            f, tv, T, _, _, abl = life(0, "Q", 1, mode, n_dyad=50, ablate=(mode == "consol"))
            assert np.isfinite(tv).all() and np.isfinite(T) and len(f) == 50
            assert (abl is not None) == (mode == "consol")
        print("smoke ok")
    elif a.calib:
        rs = []
        for s in range(*a.calib):
            for form, adv in ((0, "Q"), (1, "C")):
                life(form, adv, s, "consol")
                rs.append(life.step_ratio)
        print("step ratios", [round(x, 4) for x in rs], "mean", round(float(np.mean(rs)), 4))
    elif a.dev:
        t0, mx = time.time(), []
        for s in range(*a.dev):
            for mode in MODES:
                for form in (0, 1):
                    f, tv, T, _, _, _ = life(form, "Q", s, mode)
                    assert np.isfinite(tv).all() and np.isfinite(T)
                    mx.append(float(np.abs(tv).max()))
        print(f"agents={len(mx)} max|v|={max(mx):.2f} finite=True time={time.time() - t0:.0f}s")
    else:
        rows = {s: per_seed(s) for s in range(*a.seeds)}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "rows": rows}, fh, indent=1)
        print("written", a.out)
