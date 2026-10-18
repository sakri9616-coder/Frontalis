"""Toy sanity checks for belief-net candidates (toy sketch; not an experiment).
Sim1a: hierarchical Beta-Bernoulli (HB) -- transfer of a 'credit-stolen' belief to a brand-new partner.
Sim1b: CRP latent-cause model with partner features -- generalization gradient vs distance, n, same/diverse partners.
Sim2 : hot stove -- wrong-negative vs wrong-positive initial belief under greedy / Thompson / epistemic-bonus policies.
Sim3 : persistence vs plasticity -- HB vs flat discounted counting vs flat Bayesian counting.
Sim4 : interpretation of uninformative ambiguous events -- exact Bayes vs mean-field VB vs hard assignment.
Sim5 : value-only / belief-only histories in a bilinear (successor-feature, gamma=0) model.
"""
import numpy as np, itertools, math
from math import lgamma, log, exp
rng = np.random.default_rng(0)

def lbeta(a, b): return lgamma(a) + lgamma(b) - lgamma(a + b)

# ---------------------------------------------------------------- Sim1a
def hb_predict(partner_counts, mu_prior=(1.5, 4.5), logk_range=(np.log(0.5), np.log(50.0)), ng=60):
    """partner_counts: list of (k_successes, n_trials) per observed partner.
    Returns posterior-mean P(steal) for a NEW partner, and mean for the first observed partner."""
    mus = (np.arange(ng) + .5) / ng
    ks = np.exp(np.linspace(*logk_range, ng))           # log-uniform prior over concentration kappa
    M, K = np.meshgrid(mus, ks, indexing='ij')
    lp = (mu_prior[0] - 1) * np.log(M) + (mu_prior[1] - 1) * np.log(1 - M)   # Beta prior on mu (unnormalised)
    for (k, n) in partner_counts:
        a, b = M * K, (1 - M) * K
        # beta-binomial marginal (up to binomial coeff which is constant)
        lp = lp + np.vectorize(lbeta)(a + k, b + n - k) - np.vectorize(lbeta)(a, b)
    w = np.exp(lp - lp.max()); w /= w.sum()
    new_partner = float((w * M).sum())
    # first observed partner: posterior mean of theta_e given (mu,kappa)
    k0, n0 = partner_counts[0]
    a, b = M * K, (1 - M) * K
    first = float((w * (a + k0) / (a + b + n0)).sum())
    return new_partner, first

def sim1a():
    prior_mean = 1.5 / 6.0
    print('Sim1a  HB beta-bernoulli. prior P(steal)=%.3f ; value shown = P(steal) predicted for a brand-new partner (transfer ratio in brackets)' % prior_mean)
    print(' n | same partner n times      | n different partners (1 each) | flat counting Beta(1.5,4.5)')
    for n in (1, 2, 4, 8):
        same, same_first = hb_predict([(n, n)])
        div, div_first = hb_predict([(1, 1)] * n)
        flat = (1.5 + n) / (6 + n)
        tr = lambda p: (p - prior_mean) / ((1.5 + n) / (6 + n) - prior_mean)
        print(' %d | %.3f (first partner %.3f) [%.2f] | %.3f (a partner %.3f) [%.2f]   | %.3f' % (n, same, same_first, tr(same), div, div_first, tr(div), flat))
    print('  transfer ratio = (new-partner shift)/(experienced-partner shift):')
    for n in (1, 2, 4, 8):
        s, sf = hb_predict([(n, n)]); d, df = hb_predict([(1, 1)] * n)
        print('   n=%d same: %.2f   diverse: %.2f' % (n, (s - prior_mean) / (sf - prior_mean), (d - prior_mean) / (df - prior_mean)))

# ---------------------------------------------------------------- Sim1b
def lmarg_nig(xs, mu0=0.5, k0=1.0, nu0=2.0, s0=0.3 ** 2):
    """log marginal likelihood of 1-D data under Normal with Normal-Inverse-Chi2 prior (mu0,k0,nu0,s0)."""
    n = len(xs)
    if n == 0: return 0.0
    xbar = float(np.mean(xs)); ss = float(np.sum((np.array(xs) - xbar) ** 2))
    kn = k0 + n; nun = nu0 + n
    nus = nu0 * s0 + ss + k0 * n / kn * (xbar - mu0) ** 2
    return (lgamma(nun / 2) - lgamma(nu0 / 2) + 0.5 * (log(k0) - log(kn)) + (nu0 / 2) * log(nu0 * s0) - (nun / 2) * log(nus) - (n / 2) * log(math.pi))

def lmarg_bern(k, n, a=1.5, b=4.5): return lbeta(a + k, b + n - k) - lbeta(a, b)

def partitions(items):
    if not items: yield []; return
    first, rest = items[0], items[1:]
    for p in partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p

def lcrp(part, alpha):
    n = sum(len(c) for c in part)
    return len(part) * log(alpha) + sum(lgamma(len(c)) for c in part) - (lgamma(alpha + n) - lgamma(alpha))

def lc_predict(xs, outs, xnew, alpha=0.5):
    """exact enumeration of partitions (n<=8). returns P(steal | xnew)."""
    idx = list(range(len(xs)))
    logw = []; preds = []
    for part in partitions(idx):
        lp = lcrp(part, alpha)
        num = 0.0; den = 0.0
        tot = 0
        terms = []
        n = len(xs)
        for c in part:
            cx = [xs[i] for i in c]; co = [outs[i] for i in c]
            lp += lmarg_nig(cx) + lmarg_bern(sum(co), len(co))
            # predictive density for xnew in this cluster = marginal ratio
            px = exp(lmarg_nig(cx + [xnew]) - lmarg_nig(cx))
            theta = (1.5 + sum(co)) / (6 + len(co))
            terms.append((len(c) / (n + alpha) * px, theta))
        px_new = exp(lmarg_nig([xnew]) - lmarg_nig([]))
        terms.append((alpha / (n + alpha) * px_new, 1.5 / 6))
        s = sum(t[0] for t in terms)
        logw.append(lp + log(s)); preds.append(sum(t[0] * t[1] for t in terms) / s)
    logw = np.array(logw); w = np.exp(logw - logw.max()); w /= w.sum()
    return float((w * np.array(preds)).sum())

def sim1b():
    print('\nSim1b  CRP latent-cause (alpha=0.5), cause generates partner feature x ~ N(m,s^2) and outcome ~ Bern(theta).')
    print('       experienced partner(s) around x=0.5, all outcomes = steal. P(steal) for a new partner at distance d (prior = %.3f)' % (1.5 / 6))
    ds = [0.0, 0.1, 0.2, 0.3, 0.45]
    print(' condition                  | ' + ' '.join('d=%.2f' % d for d in ds) + ' | far/near ratio (d=.45/d=0)')
    for n in (1, 2, 4, 8):
        xs = [0.5] * n
        row = [lc_predict(xs, [1] * n, 0.5 + d) for d in ds]
        print(' same partner x%-2d            | ' % n + ' '.join('%.3f' % r for r in row) + ' | %.2f' % ((row[-1] - 0.25) / (row[0] - 0.25)))
    for n in (2, 4, 8):
        xs = list(np.linspace(0.3, 0.7, n))
        row = [lc_predict(xs, [1] * n, 0.5 + d) for d in ds]
        print(' diverse partners x%-2d (.3-.7)| ' % n + ' '.join('%.3f' % r for r in row) + ' | %.2f' % ((row[-1] - 0.25) / (row[0] - 0.25)))

# ---------------------------------------------------------------- Sim2
def hot_stove(p_true=0.5, init=(1, 4), policy='greedy', beta=5.0, forced=False, T=300, trials=2000, lam=0.0, tol=0.1):
    """Approach -> observe good(+1)/bad(-1) with prob p_true; avoid -> value 0, no observation.
    returns fraction of runs whose posterior mean is within tol of p_true at T, and mean time-to-correct (cap T)."""
    ok = 0; times = []
    for _ in range(trials):
        a, b = init; fixed = None
        t_fix = T
        for t in range(T):
            m = a / (a + b); sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
            if forced: go = True
            elif policy == 'greedy':
                ev = 2 * m - 1 + lam * sd
                go = rng.random() < 1 / (1 + math.exp(-beta * ev))
            elif policy == 'thompson':
                go = rng.beta(a, b) > 0.5
            if go:
                if rng.random() < p_true: a += 1
                else: b += 1
            if abs(a / (a + b) - p_true) < tol and t_fix == T: t_fix = t
        ok += abs(a / (a + b) - p_true) < tol
        times.append(t_fix)
    return ok / trials, float(np.mean(times))

def hot_stove_inflate(init=(1, 4), rho=0.03, lam=2.0, beta=5.0, T=300, trials=1000, p_true=0.5, tol=0.1, floor=2.0):
    """greedy+epistemic bonus; while AVOIDING, counts shrink toward total=floor keeping the mean (variance inflation)."""
    times = []
    for _ in range(trials):
        a, b = init; t_fix = T
        for t in range(T):
            m = a / (a + b); sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
            go = rng.random() < 1 / (1 + math.exp(-beta * (2 * m - 1 + lam * sd)))
            if go:
                if rng.random() < p_true: a += 1
                else: b += 1
            else:
                tot = a + b; new_tot = max(floor, tot * (1 - rho))
                a, b = m * new_tot, (1 - m) * new_tot
            if abs(a / (a + b) - p_true) < tol and t_fix == T: t_fix = t
        times.append(t_fix)
    return float(np.mean(times))

def sim2():
    print('\nSim2  hot stove: truth P(good)=0.5, V(good)=+1,V(bad)=-1. wrong-negative init Beta(1,4) (mean .2) vs wrong-positive Beta(4,1) (mean .8).')
    print(' policy                           | negative: P(corrected by T=300) / mean t | positive: ... | ratio of mean t (neg/pos)')
    for name, kw in [('forced exposure (no choice)', dict(forced=True)),
                     ('greedy+softmax beta=5', dict(policy='greedy', beta=5.0)),
                     ('greedy+epistemic bonus lam=2', dict(policy='greedy', beta=5.0, lam=2.0)),
                     ('Thompson sampling', dict(policy='thompson'))]:
        n = hot_stove(init=(1, 4), trials=1000, **kw); p = hot_stove(init=(4, 1), trials=1000, **kw)
        print(' %-32s | %.2f / %6.1f                       | %.2f / %6.1f | %.2f' % (name, n[0], n[1], p[0], p[1], n[1] / p[1]))
    for rho in (0.01, 0.03, 0.1):
        n = hot_stove_inflate(init=(1, 4), rho=rho); p = hot_stove_inflate(init=(4, 1), rho=rho)
        print(' bonus lam=2 + variance inflation rho=%.2f | mean t neg %.1f pos %.1f ratio %.2f' % (rho, n, p, n / p))

# ---------------------------------------------------------------- Sim3
def sim3():
    print('\nSim3  persistence vs plasticity. P1 history: 8 partners x1 trial, all steal. P2: 8 partners x1 trial, none steal.')
    print('      then 8 new experiences with ONE new partner N (all "kind", i.e. 0 steals). Measure for P1-P2 difference:')
    h1 = [(1, 1)] * 8; h2 = [(0, 1)] * 8
    base1, _ = hb_predict(h1); base2, _ = hb_predict(h2)
    h1b = h1 + [(0, 8)]; h2b = h2 + [(0, 8)]
    pop1, _ = hb_predict(h1b); pop2, _ = hb_predict(h2b)
    # partner N's own estimate: posterior mean of theta_N
    def own(h):
        mus = (np.arange(60) + .5) / 60; ks = np.exp(np.linspace(np.log(.5), np.log(50), 60)); M, K = np.meshgrid(mus, ks, indexing='ij')
        lp = 0.5 * np.log(M) + 3.5 * np.log(1 - M)
        for (k, n) in h:
            a, b = M * K, (1 - M) * K
            lp = lp + np.vectorize(lbeta)(a + k, b + n - k) - np.vectorize(lbeta)(a, b)
        w = np.exp(lp - lp.max()); w /= w.sum(); a, b = M * K, (1 - M) * K
        k0, n0 = h[-1]
        return float((w * (a + k0) / (a + b + n0)).sum())
    print('  HB: new-partner (unseen) prediction diff before=%.3f after 8 exps=%.3f (retained %.0f%%)' % (base1 - base2, pop1 - pop2, 100 * (pop1 - pop2) / (base1 - base2)))
    print('      partner N own estimate: P1 %.3f  P2 %.3f  (plastic: both drop toward 0 from prior %.2f)' % (own(h1b), own(h2b), 0.25))
    # persistence as a function of history DIVERSITY (same 8 trials, split over P partners)
    print('  HB retention of P1-P2 difference for an UNSEEN partner vs number of distinct partners in history (8 trials total):')
    for P in (1, 2, 4, 8):
        per = 8 // P
        a1, _ = hb_predict([(per, per)] * P); a2, _ = hb_predict([(0, per)] * P)
        b1, _ = hb_predict([(per, per)] * P + [(0, 8)]); b2, _ = hb_predict([(0, per)] * P + [(0, 8)])
        print('    P=%d partners x %d trials: diff %.3f -> %.3f (retained %.0f%%)' % (P, per, a1 - a2, b1 - b2, 100 * (b1 - b2) / (a1 - a2)))
    # flat discounted (alpha=.25) tracking single probability
    for alpha in (0.25, 0.1):
        v1 = 0.9; v2 = 0.1
        for t in range(8): v1 += alpha * (0 - v1); v2 += alpha * (0 - v2)
        print('  flat delta rule alpha=%.2f: diff 0.80 -> %.3f (retained %.0f%%)' % (alpha, v1 - v2, 100 * (v1 - v2) / 0.8))
    # flat bayesian counting with Beta(1.5,4.5)+counts
    a1, b1 = 1.5 + 8, 4.5; a2, b2 = 1.5, 4.5 + 8
    d0 = a1 / (a1 + b1) - a2 / (a2 + b2)
    b1 += 8; b2 += 8
    d1 = a1 / (a1 + b1) - a2 / (a2 + b2)
    print('  flat Bayesian counting: diff %.3f -> %.3f (retained %.0f%%) ; own-estimate move is just the pooled estimate (no partner-level plasticity): P1 %.3f P2 %.3f' % (d0, d1, 100 * d1 / d0, a1 / (a1 + b1), a2 / (a2 + b2)))

# ---------------------------------------------------------------- Sim4
def psi(x):
    return float(np.log(x) - 1 / (2 * x) - 1 / (12 * x ** 2)) if x > 6 else None
from math import log as _log
def digamma(x):
    r = 0.0
    while x < 6: r -= 1 / x; x += 1
    return r + _log(x) - 1 / (2 * x) - 1 / (12 * x ** 2) + 1 / (120 * x ** 4)

def sim4(n_amb=20, init=(6.0, 2.0)):
    print('\nSim4  %d uninformative ambiguous events (likelihood ratio 1) after prior counts %s (belief %.2f):' % (n_amb, init, init[0] / sum(init)))
    a, b = init
    print('  exact Bayes (mixture posterior, mean):  %.3f' % (a / (a + b)))
    # mean-field VB responsibilities
    a2, b2 = init
    for _ in range(n_amb):
        ra, rb = exp(digamma(a2)), exp(digamma(b2)); r = ra / (ra + rb); a2 += r; b2 += 1 - r
    print('  mean-field VB / soft counts:            %.3f' % (a2 / (a2 + b2)))
    a3, b3 = init
    for _ in range(n_amb):
        if a3 >= b3: a3 += 1
        else: b3 += 1
    print('  hard assignment (self-training):        %.3f' % (a3 / (a3 + b3)))
    a4, b4 = init
    for _ in range(n_amb):
        r = a4 / (a4 + b4); a4 += r; b4 += 1 - r
    print('  naive soft counts (r=current mean):     %.3f  (counts grow proportionally: no drift in mean)' % (a4 / (a4 + b4)))
    # symmetric: belief 0.5 start-> noise-driven
    print('  hard assignment from a 0.50 start, 20 events with weakly informative 55/45 evidence in random direction: ', end='')
    finals = []
    for _ in range(2000):
        a, b = 4.0, 4.0
        for _ in range(20):
            true_steal = rng.random() < 0.5
            lik_s = 0.55 if true_steal else 0.45; lik_n = 1 - lik_s
            r = a * lik_s / (a * lik_s + b * lik_n)
            if r >= 0.5: a += 1
            else: b += 1
        finals.append(a / (a + b))
    finals = np.array(finals); print('mean |belief-0.5|=%.3f ; exact Bayes (martingale) mean |belief-0.5| for comparison computed below' % np.mean(np.abs(finals - 0.5)))
    fin2 = []
    for _ in range(2000):
        a, b = 4.0, 4.0
        for _ in range(20):
            true_steal = rng.random() < 0.5
            lik_s = 0.55 if true_steal else 0.45; lik_n = 1 - lik_s
            r = a * lik_s / (a * lik_s + b * lik_n)
            a += r; b += 1 - r
        fin2.append(a / (a + b))
    print('      exact-Bayes soft: mean |belief-0.5|=%.3f' % np.mean(np.abs(np.array(fin2) - 0.5)))

# ---------------------------------------------------------------- Sim5
def sim5():
    print('\nSim5  bilinear separation Q = w . psi (gamma=0 successor features): belief-only and value-only histories')
    d = 4  # outcome features: [credit_lost, help_received, success, novelty]
    w = np.array([-1.0, 0.8, 0.5, 0.2]); psi_ = np.array([0.6, 0.2, 0.7, 0.1])
    Q0 = w @ psi_
    # belief-only: partner now steals less but gives equally-valued mix: change psi along null space of w
    null = np.array([0.8, 1.0, 0.0, 0.0]); null = null - (null @ w) / (w @ w) * w   # orthogonal to w
    psi_b = psi_ + 0.3 * null
    print('  belief-only: dpsi=%.3f dV(w.dpsi)=%.3f  dQ=%.3f' % (np.linalg.norm(psi_b - psi_), (psi_b - psi_) @ w, w @ psi_b - Q0))
    # value-only: reweight w (reappraisal) with psi saturated/unchanged
    w2 = w.copy(); w2[0] = -0.3; w2[1] = 1.2
    print('  value-only : dpsi=0   dw=%.3f dQ=%.3f  (Q changes without any new belief)' % (np.linalg.norm(w2 - w), w2 @ psi_ - Q0))
    print('  => Q changes in both cases; readouts (psi-probes, w-probes) are separately controllable iff d>=2 and psi free in null(w).')

if __name__ == '__main__':
    sim1a(); sim1b(); sim2(); sim3(); sim4(); sim5()
