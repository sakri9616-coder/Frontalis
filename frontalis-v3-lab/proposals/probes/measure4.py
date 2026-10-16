"""Draft-4 measurements (PREREG draft 4, sections 4-6). Dev seeds only.
A View is one personality as seen by the measurements: V (vector evaluation), PB (belief), H (value representation, NN only),
head(h), and its evaluative state. Planted views (calibration) are built from real views."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

import sim4
from sim4 import DEV, KB, TOL, D_IN, N_CTX

SETS = ("tr", "un", "nc")


def _t(a):
    return torch.tensor(a, device=DEV)


class View:
    def __init__(self, state, i, world, wi):
        self.st = state; self.i = i; self.w = world; self.wi = wi
        net = state["net"]; self.net = net; self.nn = hasattr(net, "p")
        self.p = net.slice(i) if self.nn else None
        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        c = state["cfg"]; self.cfg = c
        self.state = dict(w=state["w_use"][i].clone(), fscale=float(state["fscale"][i]), kappa=float(state["kappa"][i]) if int(state["kmode_vec"][i]) != 0 else 0.0,
                          sig2=state["sig2"][i].clone(), Ef=state["Ef"][i].clone(), Es=state["Es"][i].clone())

    def rows(self, s, p, c):
        return _t(sim4.rows_np(self.w, s, p, c))[None]

    def meta(self, s, p, c):
        return dict(s=_t(s)[None], p=_t(p)[None], c=_t(c)[None], wi=torch.tensor([self.wi], device=DEV))

    def _ridge_W(self):
        return self.net.W[self.i:self.i + 1]

    def H(self, s, p, c):
        with torch.no_grad():
            return self.net.vrep(self.rows(s, p, c), self.p)[0]

    def head(self, h):
        return (h @ self.p["h.Wv"][0] + self.p["h.bv"][0])

    def V(self, s, p, c):
        with torch.no_grad():
            if self.nn:
                return self.net.value_vec(self.rows(s, p, c), self.p)[0]
            return self.net.predict(self.rows(s, p, c), self.meta(s, p, c), self._ridge_W())[1][0]

    def PB(self, s, p, c):
        with torch.no_grad():
            if self.nn:
                return self.net.belief(self.rows(s, p, c), self.p)[0]
            return self.net.predict(self.rows(s, p, c), self.meta(s, p, c), self._ridge_W())[0][0]

    def get(self, which):
        if which not in self.cache:
            s, p, c = sim4.probe_set(self.w, which)
            d = dict(V=self.V(s, p, c).cpu().numpy(), PB=self.PB(s, p, c).cpu().numpy())
            if self.nn:
                d["H"] = self.H(s, p, c)
            self.cache[which] = d
        return self.cache[which]


class Planted(View):
    """B := base (A) with alpha * u * phi(x) added in the value representation h (NN only), same head and belief as base."""

    def __init__(self, base: View, u: torch.Tensor, phi_fn, alpha: float):
        self.__dict__.update({k: v for k, v in base.__dict__.items() if k != "cache"})
        self.cache = {}; self.base = base; self.u = u; self.phi_fn = phi_fn; self.alpha = alpha

    def H(self, s, p, c):
        return self.base.H(s, p, c) + self.alpha * _t(self.phi_fn(s, p, c)).float()[:, None] * self.u[None]

    def get(self, which):
        if which not in self.cache:
            b = self.base.get(which)
            s, p, c = sim4.probe_set(self.w, which)
            phi = _t(self.phi_fn(s, p, c)).float()
            H = b["H"] + self.alpha * phi[:, None] * self.u[None]
            with torch.no_grad():
                V = self.head(H).cpu().numpy()
            self.cache[which] = dict(V=V, PB=b["PB"], H=H)
        return self.cache[which]


# --------------------------------------------------------------------------------------------------- quantities
def cent(a):
    return a - a.mean(0, keepdims=True)


def dist(a, b, scale):
    return float(np.abs(a - b).sum(1).mean() / scale)


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-12))


def ghat_resid(view: View, innate: str) -> Dict[str, np.ndarray]:
    """rho = V - ghat(own belief through a FIXED basis), ghat fitted on P_tr (ridge, per channel)."""
    def basis(which):
        d = view.get(which); s, p, c = sim4.probe_set(view.w, which)
        pb_s = d["PB"]
        pb_p = view_pb_partner(view, which)
        cols = [np.ones((len(s), 1))]
        if innate in ("IH", "IHE"):          # the FIXED innate rule applied to the own belief (hinge), per-channel coefficients
            cols += [np.clip(np.abs(pb_s) - TOL, 0, None) ** 2, np.clip(np.abs(pb_p) - TOL, 0, None) ** 2]
        return np.concatenate(cols, 1)
    Btr = basis("tr"); Vtr = view.get("tr")["V"]
    lam = 1e-3 * len(Btr)
    coef = np.linalg.solve(Btr.T @ Btr + lam * np.eye(Btr.shape[1]), Btr.T @ Vtr)
    out = {}
    for which in ("un", "nc"):
        out[which] = cent(view.get(which)["V"] - basis(which) @ coef)
    return out


def view_pb_partner(view: View, which):
    key = "_pbp_" + which
    if key not in view.cache:
        s, p, c = sim4.probe_set(view.w, which)
        view.cache[key] = (view.PB(p, s, c).cpu().numpy() if not isinstance(view, Planted) else view.base.PB(p, s, c).cpu().numpy())
    return view.cache[key]


def new_experiences(w, seed):
    rng = np.random.default_rng([seed, 61])
    un = np.concatenate([w["un0"], w["un1"]])
    e1 = rng.choice(un, 30); e2 = np.array([rng.choice(un[un != a]) for a in e1]); c = rng.integers(0, sim4.N_CTX_TR, 30)
    act = np.where(rng.random(30) < 1 / (1 + np.exp(-(w["prop"][e1] - w["prop"][e2]))), e1, e2)
    oth = np.where(act == e1, e2, e1)
    db = sim4.outcome_np(w, act, oth, c) + 0.1 * rng.normal(size=(30, KB))
    return e1, e2, c, db.astype(np.float32)


def swap_update(receiver: View, donor: View, w, seed, innate, cyc, ref=0.5, steps=20, lr=0.05):
    """Receiver's network learns 30 new experiences with targets made from the DONOR's evaluative state; returns the
    change of the receiver's evaluation on P_un (flattened)."""
    e1, e2, c, db = new_experiences(w, seed)
    S = np.stack([e1, e2], 1).reshape(-1); Pp = np.stack([e2, e1], 1).reshape(-1); C = np.repeat(c, 2)
    dbt = _t(db)
    with torch.no_grad():
        pb_r = receiver.PB(S, Pp, C).reshape(30, 2, KB)
        v_d = donor.V(S, Pp, C).reshape(30, 2, KB) if not isinstance(donor, Planted) else donor.base.V(S, Pp, C).reshape(30, 2, KB)
        stt = donor.state
        d = torch.clamp(dbt.abs() - TOL, min=0) ** 2
        ll = -((dbt[:, None, :] - pb_r) ** 2 / (2 * receiver.state["sig2"][None, None, :])).sum(-1)
        if innate in ("I0", "IHE"):
            pmix = (torch.softmax(ll, 1)[..., None] * pb_r).sum(1)
            err = (dbt - pmix) ** 2
            Ef = stt["Ef"][_t(c)]; Es = stt["Es"][_t(c)]
            Ef = Ef + 0.1 * (err - Ef); Es = Es + 0.01 * (err - Es)
            lp = (Es - Ef) / (Es + 1e-3)
        wv = stt["w"][None] if cyc else torch.ones(1, KB, device=DEV)
        if innate == "IH":
            inn = -(wv * d)
        elif innate == "I0":
            inn = wv * lp
        else:
            inn = torch.cat([-(wv[:, :4] * d[:, :4]), (wv[:, 4] * torch.clamp(dbt[:, 4], min=0))[:, None]], 1) + 0.5 * wv * lp
        if cyc:
            f_k = inn + ref * (inn - v_d.sum(1))
            fs = max(stt["fscale"], 1e-3); f_k = f_k / fs; f = f_k.sum(1)
            a_ = torch.sign(f)[:, None] * v_d.sum(-1) / fs
            r = torch.softmax(ll + stt["kappa"] * a_, 1)
        else:
            f_k = inn / max(receiver.state["fscale"], 1e-3); r = torch.full((30, 2), 0.5, device=DEV)
        tgt = (r[:, :, None] * f_k[:, None, :]).reshape(60, KB)
    s, p, cc = sim4.probe_set(w, "un")
    if receiver.nn:
        q = {k: v.clone().requires_grad_(True) for k, v in receiver.p.items()}
        X = receiver.rows(S, Pp, C); Xp = receiver.rows(s, p, cc)
        with torch.no_grad():
            v0 = receiver.net.value_vec(Xp, q)[0]
        o = torch.optim.SGD([q[k] for k in q if k.startswith("h.") or receiver.net.kind not in ("nn0", "rf")], lr=lr)
        for _ in range(steps):
            loss = ((receiver.net.value_vec(X, q)[0] - tgt) ** 2).sum(-1).mean()
            o.zero_grad(); loss.backward()
            gs = [g for g in q.values() if g.grad is not None]
            gn = float(torch.sqrt(sum((g.grad ** 2).sum() for g in gs)))
            for g in gs:
                g.grad.mul_(min(1.0, 1.0 / (gn + 1e-12)))
            o.step()
        with torch.no_grad():
            return (receiver.net.value_vec(Xp, q)[0] - v0).cpu().numpy().ravel()
    net = receiver.net; i = receiver.i
    with torch.no_grad():
        phi = net.feats(receiver.rows(S, Pp, C), receiver.meta(S, Pp, C))[0]
        Gv = net.Gv[i] + phi.T @ phi; Rv = net.R[i, :, KB:] + phi.T @ tgt
        Wv = torch.cholesky_solve(Rv, torch.linalg.cholesky(Gv))
        ph = net.feats(receiver.rows(s, p, cc), receiver.meta(s, p, cc))[0]
        return (ph @ Wv - ph @ net.W[i, :, KB:]).cpu().numpy().ravel()


def das_centered(A: View, B: View, k=4, steps=300, seed=0, shuffle=False):
    """Optimise U (k-dim orthonormal) on P_tr so that moving B's h toward A's inside U minimises the CENTERED gap."""
    Ha = A.get("tr")["H"]; Hb = B.get("tr")["H"]; Va = _t(cent(A.get("tr")["V"]))
    if shuffle:
        perm = torch.randperm(Ha.shape[0], generator=torch.Generator().manual_seed(int(seed) + 99)).to(DEV)
        Ha = Ha[perm]; Va = Va[perm]
    g = torch.Generator().manual_seed(int(seed))
    Z = torch.randn(Ha.shape[1], k, generator=g).to(DEV).requires_grad_(True)
    o = torch.optim.Adam([Z], lr=0.05)
    for _ in range(steps):
        U, _ = torch.linalg.qr(Z)
        vp = B.head(Hb + (Ha - Hb) @ U @ U.T)
        loss = (vp - vp.mean(0, keepdim=True) - Va).abs().sum(1).mean()
        o.zero_grad(); loss.backward(); o.step()
    return torch.linalg.qr(Z.detach())[0]


def feat_xpred(src: View, tgt: View, which, lam_frac=1e-2):
    """Feature cross-prediction (J4 in PREREG draft 4 rev. a): ridge from the SOURCE personality's value representation h on
    P_tr to the TARGET's centered evaluation on P_tr; relative L1 error of the prediction on `which`."""
    Htr = src.get("tr")["H"].cpu().numpy(); Hq = src.get(which)["H"].cpu().numpy()
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-6
    Z = (Htr - mu) / sd; Zq = (Hq - mu) / sd
    Y = cent(tgt.get("tr")["V"])
    coef = np.linalg.solve(Z.T @ Z + lam_frac * len(Z) * np.eye(Z.shape[1]), Z.T @ Y)
    Yq = cent(tgt.get(which)["V"]); pr = cent(Zq @ coef)
    return float(np.abs(Yq - pr).sum(1).mean() / (np.abs(Yq).sum(1).mean() + 1e-12))


def closure(A: View, B: View, U, which):
    Ha = A.get(which)["H"]; Hb = B.get(which)["H"]
    with torch.no_grad():
        vp = B.head(Hb + (Ha - Hb) @ U @ U.T).cpu().numpy()
    Va = cent(A.get(which)["V"]); Vb = cent(B.get(which)["V"])
    return float(1 - np.abs(Va - cent(vp)).sum(1).mean() / (np.abs(Va - Vb).sum(1).mean() + 1e-12))


def persistence(A: View, B: View, U, steps=200, lr=3e-3):
    """Remove U from B's h, retrain only B's value head on B's own recent record (distractors set to 0), measure change."""
    base = B.base if isinstance(B, Planted) else B
    st = base.st; i = base.i
    n = st["t"]; lo = max(0, n - 2000)
    sb = st["buf"]["s"][i, lo:n].cpu().numpy(); cb = st["buf"]["c"][i, lo:n].cpu().numpy(); tv = st["buf"]["TV"][i, lo:n].reshape(-1, KB)
    S = sb.reshape(-1); Pp = sb[:, ::-1].reshape(-1); C = np.repeat(cb, 2)
    with torch.no_grad():
        Hr = B.H(S, Pp, C)
        if isinstance(B, Planted):      # a planted net was never trained: its own evaluation on its record is the target
            tv = B.head(Hr)
        Hr = Hr - Hr @ U @ U.T
    Wv = B.p["h.Wv"][0].clone().requires_grad_(True); bv = B.p["h.bv"][0].clone().requires_grad_(True)
    o = torch.optim.Adam([Wv, bv], lr=lr)
    for _ in range(steps):
        loss = ((Hr @ Wv + bv - tv) ** 2).sum(-1).mean()
        o.zero_grad(); loss.backward(); o.step()
    Hu = B.get("un")["H"]
    with torch.no_grad():
        after = ((Hu - Hu @ U @ U.T) @ Wv + bv).cpu().numpy()
    Vb = B.get("un")["V"]; Va = A.get("un")["V"]
    return float(np.abs(cent(after) - cent(Vb)).sum(1).mean() / (np.abs(cent(Va) - cent(Vb)).sum(1).mean() + 1e-12))


def own_r2(view: View, which, remap=False):
    s, p, c = sim4.probe_set(view.w, which)
    return r2(view.get(which)["PB"], sim4.outcome_np(view.w, s, p, c, remap=remap))


def conflict_share(V, scale):
    big = np.abs(V) > 0.1 * scale
    return float((((V > 0) & big).any(1) & ((V < 0) & big).any(1)).mean())


def interaction_share_vec(dv, n_e_rows):
    """Per-channel share of non-additive variance on the (entity-row x context) grid, averaged over channels."""
    out = []
    for k in range(dv.shape[1]):
        G = dv[:, k].reshape(n_e_rows, -1)
        tot = ((G - G.mean()) ** 2).sum()
        res = G - G.mean(1, keepdims=True) - G.mean(0, keepdims=True) + G.mean()
        out.append((res ** 2).sum() / max(tot, 1e-12))
    return float(np.mean(out))


# ------------------------------------------------------------------------------------------------ one world group
def measure_group(vw: Dict[str, View], w, seed, innate, cyc, das=True, k=4, extra=None) -> Dict[str, Any]:
    """vw: role -> View with roles A, A2 (A'), B, B2 (B'). Returns all draft-4 quantities for one world group."""
    A, A2, B, B2 = vw["A"], vw["A2"], vw["B"], vw["B2"]
    out: Dict[str, Any] = {}
    scale = np.mean([np.abs(v.get("un")["V"]).mean() for v in (A, A2, B, B2)]) * KB + 1e-12
    for which in ("un", "nc"):
        Va, Va2, Vb, Vb2 = (v.get(which)["V"] for v in (A, A2, B, B2))
        out[f"D_AB_{which}"] = dist(Va, Vb, scale); out[f"D_AA_{which}"] = dist(Va, Va2, scale)
        out[f"Dc_AB_{which}"] = dist(cent(Va), cent(Vb), scale); out[f"Dc_AA_{which}"] = dist(cent(Va), cent(Va2), scale)
        x1 = (cent(Va) - cent(Vb)).ravel(); x2 = (cent(Va2) - cent(Vb2)).ravel()
        out[f"repro_{which}"] = float(np.corrcoef(x1, x2)[0, 1]) if x1.std() > 0 and x2.std() > 0 else 0.0
    rho = {r: ghat_resid(v, innate) for r, v in (("A", A), ("A2", A2), ("B", B))}
    for which in ("un", "nc"):
        out[f"Drho_AB_{which}"] = dist(rho["A"][which], rho["B"][which], scale)
        out[f"Drho_AA_{which}"] = dist(rho["A"][which], rho["A2"][which], scale)
    # J3 evaluation-state swap, receiver = B'
    uA = swap_update(B2, A, w, seed, innate, cyc); uB = swap_update(B2, B, w, seed, innate, cyc); uA2 = swap_update(B2, A2, w, seed, innate, cyc)
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    out["swap_AB"] = 1 - cos(uA, uB); out["swap_AA"] = 1 - cos(uA, uA2)
    out["swap_identical"] = int(np.allclose(uA, uB) and np.allclose(uA, uA2))
    nrm = np.abs(uA).sum() + 1e-12
    out["swapd_AB"] = float(np.abs(uA - uB).sum() / nrm); out["swapd_AA"] = float(np.abs(uA - uA2).sum() / nrm)
    # J5 world-feature transfer
    out["r2_un"] = own_r2(A, "un"); out["r2_nc"] = own_r2(A, "nc")
    # representation (NN only)
    if A.nn and "H" in A.get("tr"):
        for which in ("un", "nc"):
            e = {("A", "B"): feat_xpred(A, B, which), ("B", "A"): feat_xpred(B, A, which),
                 ("B2", "B"): feat_xpred(B2, B, which), ("A2", "A"): feat_xpred(A2, A, which)}
            out[f"xp_other_{which}"] = e[("A", "B")] + e[("B", "A")]
            out[f"xp_same_{which}"] = e[("B2", "B")] + e[("A2", "A")]
    if A.nn and das:
        U_ab = das_centered(A, B, k, seed=seed); U_sh = das_centered(A, B, k, seed=seed, shuffle=True); U_aa = das_centered(A, A2, k, seed=seed + 1)
        for which in ("un", "nc"):
            out[f"c_AB_{which}"] = closure(A, B, U_ab, which); out[f"c_sh_{which}"] = closure(A, B, U_sh, which)
            out[f"c_AA_{which}"] = closure(A, A2, U_aa, which)
        out["p_AB"] = persistence(A, B, U_ab); out["p_sh"] = persistence(A, B, U_sh)
    # descriptives
    out["conflict_A"] = conflict_share(A.get("un")["V"], scale / KB); out["conflict_B"] = conflict_share(B.get("un")["V"], scale / KB)
    n_rows = len(np.concatenate([w["un0"], w["un1"]])) * 2
    out["inter_AB"] = interaction_share_vec(A.get("un")["V"] - B.get("un")["V"], n_rows)
    out["inter_AA"] = interaction_share_vec(A.get("un")["V"] - A2.get("un")["V"], n_rows)
    out["kappa"] = [v.state["kappa"] for v in (A, A2, B, B2)]
    out["w_A"] = A.state["w"].cpu().numpy().round(3).tolist(); out["w_B"] = B.state["w"].cpu().numpy().round(3).tolist()
    out["scale"] = float(scale)
    if extra:
        out.update(extra)
    return out


def views(state, pers, worlds) -> Dict[Any, Dict[str, View]]:
    wk = state["wkeys"]; wi = {k: j for j, k in enumerate(wk)}
    g: Dict[Any, Dict[str, View]] = {}
    for i, q in enumerate(pers):
        g.setdefault(q["group"], {})[q["role"]] = View(state, i, worlds[q["world"]], wi[q["world"]])
    return g


# ----------------------------------------------------------------------------------------------- judgement rule
def judge(diffs: List[float], floor: float, n_req_frac=0.75):
    d = np.array([x for x in diffs if np.isfinite(x)])
    n_pos = int((d > 0).sum()); med = float(np.median(d)) if len(d) else float("nan")
    need = int(np.ceil(n_req_frac * len(diffs)))
    return dict(n_pos=n_pos, n=len(diffs), median=med, floor=floor, passed=bool(n_pos >= need and med >= floor))
