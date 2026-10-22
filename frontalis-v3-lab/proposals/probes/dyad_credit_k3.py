# K=3 check for CRITIQUE-dyad-5 condition 2: after a world change, does the destination depend on the early form?
# Aspects: 0 coop, 1 quality, 2 novelty. Each event: every aspect present independently with prob q_present.
import numpy as np
def run(beta, early_bias, o_late, n_early=2000, n_late=6000, eta=0.02, seed=0, q=0.6):
    r = np.random.default_rng(seed); c = np.zeros(3)
    def step(o_vec, bias):
        nonlocal c
        p = np.clip(q + bias, 0.05, 0.95); pres = (r.random(3) < p).astype(int)
        if pres.sum() == 0: return
        o = pres @ o_vec + r.normal(0, .1)
        z = np.where(pres == 1, beta * c, -np.inf); a = np.exp(z - z[pres == 1].max()); a /= a.sum()
        c = c + eta * pres * (a * o - c)
    for _ in range(n_early): step(np.ones(3), early_bias)
    c_early = c.round(2).tolist()
    for _ in range(n_late): step(np.array(o_late, float), np.zeros(3))
    return c_early, c.round(2).tolist()
E = {"coop-heavy": np.array([.3, -.3, -.3]), "qual-heavy": np.array([-.3, .3, -.3]),
     "qual>nov": np.array([-.3, .3, 0.]), "qual>coop": np.array([0., .3, -.3])}
for beta in (1, 4):
    for change, o_late in (("no change", [1, 1, 1]), ("quality turns bad", [1, -1, 1])):
        print(f"beta={beta} late world: {change}")
        for name, b in E.items():
            outs = [run(beta, b, o_late, seed=s) for s in range(4)]
            print(f"   {name:10} early-end {outs[0][0]}  late-end by seed {[o[1] for o in outs]}")

# ---- part 2: does the early SECONDARY ranking decide the winner after quality turns bad? (20 seeds) ----
print("\nbeta | P(qual>nov early -> novelty wins) | P(qual>coop early -> novelty wins) | early-end c (seed 0) a / b")
for beta in (1.5, 2, 2.5, 3, 4):
    wa = wb = 0
    for s in range(20):
        ea, la = run(beta, E["qual>nov"], [1, -1, 1], seed=s); eb, lb = run(beta, E["qual>coop"], [1, -1, 1], seed=s)
        wa += la[2] > la[0]; wb += lb[2] > lb[0]
    print(f"{beta:4} | {wa/20:.2f} | {wb/20:.2f} | {run(beta, E['qual>nov'], [1,-1,1])[0]} / {run(beta, E['qual>coop'], [1,-1,1])[0]}")

# ---- part 3: early worlds differ in CO-OCCURRENCE (which aspect tends to come with quality), not frequency ----
def run_corr(beta, partner, o_late, n_early=2000, n_late=6000, eta=0.02, seed=0, q=0.5, rho=0.8):
    r = np.random.default_rng(seed); c = np.zeros(3)
    def step(o_vec, corr):
        nonlocal c
        pres = (r.random(3) < q).astype(int)
        if corr is not None:                 # early: quality is frequent, and `partner` follows quality with prob rho
            pres[1] = int(r.random() < 0.8)
            pres[corr] = pres[1] if r.random() < rho else int(r.random() < q)
        if pres.sum() == 0: return
        o = pres @ o_vec + r.normal(0, .1)
        z = np.where(pres == 1, beta * c, -np.inf); a = np.exp(z - z[pres == 1].max()); a /= a.sum()
        c = c + eta * pres * (a * o - c)
    for _ in range(n_early): step(np.ones(3), partner)
    ce = c.round(2).tolist()
    for _ in range(n_late): step(np.array(o_late, float), None)
    return ce, c.round(2).tolist()
print("\nco-occurrence: P_a = novelty comes with quality, P_b = coop comes with quality; late: quality turns bad (20 seeds)")
print("beta | P_a: coop wins / novelty wins | P_b: coop wins / novelty wins | early-end c seed0 P_a / P_b")
for beta in (0, 2, 4):
    res = {}
    for name, partner in (("a", 2), ("b", 0)):
        wins = [run_corr(beta, partner, [1, -1, 1], seed=s)[1] for s in range(20)]
        res[name] = (sum(w[0] > w[2] for w in wins) / 20, sum(w[2] > w[0] for w in wins) / 20)
    print(f"{beta:4} | {res['a']} | {res['b']} | {run_corr(beta, 2, [1,-1,1])[0]} / {run_corr(beta, 0, [1,-1,1])[0]}")

# ---- part 4 (CRITIQUE-dyad-5.1 cond. 2): the SAME new events (coop + novelty, o = 2) given right after the early phase ----
def same_events(beta, partner, seed, n_new=100, eta=0.02):
    ce, _ = run_corr(beta, partner, [1, 1, 1], n_late=0, seed=seed)
    c = np.array(ce, float); c0 = c.copy(); r = np.random.default_rng(10_000 + seed)   # identical new-event noise for Pa and Pb
    pres = np.array([1, 0, 1])
    for _ in range(n_new):
        o = 2 + r.normal(0, .1)
        z = np.where(pres == 1, beta * c, -np.inf); a = np.exp(z - z[pres == 1].max()); a /= a.sum()
        c = c + eta * pres * (a * o - c)
    d = c - c0
    return d[0] - d[2]          # >0: the new events taught "coop" more than "novelty"
print("\nsame new events after the early phase: share of seeds where they taught coop more than novelty (20 seeds)")
for beta in (0, 2, 4):
    a = [same_events(beta, 2, s) for s in range(20)]; b = [same_events(beta, 0, s) for s in range(20)]
    print(f"beta={beta}: Pa {np.mean(np.array(a) > 0):.2f} (mean {np.mean(a):+.3f}) | Pb {np.mean(np.array(b) > 0):.2f} (mean {np.mean(b):+.3f})")
