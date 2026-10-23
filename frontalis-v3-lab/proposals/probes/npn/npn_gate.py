"""NPN 1:1 with surprise-gated consolidation (DESIGN-gate, v15). Builds on the frozen v14 code (npn_consol.py, unchanged).

The evaluation network, credit attention, choice, trust network and world are those of npn_consol.py.
Consolidation as in v14 (importance Omega, relative per tensor, LAM = 10), but the protection is loosened by surprise:
  prediction p = sum of the current token values (before learning), error e = o - p,
  expected surprise m <- (1 - RHO_S) m + RHO_S e^2   (the network's running record of how wrong it usually is),
  surprise s = e^2 / (m + eps)  (computed before m is updated),
  plasticity factor  f = 1 / (1 + LAM * Omega / mean(Omega) * exp(-s / KAPPA)).
When an experience is about as predictable as usual, the weights that carried recent valuations stay protected; when it
is far more wrong than usual, the protection is released for that step.
Prototype history (seeds 0-7, disclosed in the design): KAPPA 1, 4, 16 were tried; a second gate multiplied s by the
share of the error gradient on protected weights (phi) and was dropped because it released the protection more often.
Modes:
  gated  - the mechanism above
  gshuf  - the same, but s is replaced by a value drawn at random from the agent's own past s [s,17]
           (same distribution of loosening, unrelated to the present experience)
  match  - plain network (no consolidation) with the learning rate LR_MATCH chosen on the dev seeds so that its median
           change time after a reason equals that of gated (speed-matched control; --calib looks at change times only)
  consol - v14 (no gate), slow - v14 slow (LAM = 0, LR_SLOW), none - plain network
usage:
  python npn_gate.py --smoke
  python npn_gate.py --proto A B KAPPA            # prototype print (not evaluation seeds)
  python npn_gate.py --dev A B                    # learning sanity only (finite)
  python npn_gate.py --seeds A B --out res.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import npn_consol as C  # noqa: E402

KAPPA = float(os.environ.get("NPN_GATE_KAPPA", "1.0"))
RHO_S = 0.01
LR_MATCH = float(os.environ.get("NPN_GATE_LR_MATCH", "nan"))
MODES = ("gated", "gshuf", "match", "consol", "slow", "none")


class GAgent(C.Agent):
    def __init__(self, seed, mode):
        super().__init__(seed, {"gated": "consol", "gshuf": "consol", "match": "none"}.get(mode, mode))
        self.gmode, self.m, self.past = mode, None, []
        self.rs = np.random.default_rng([seed, 17])

    def learn(self, toks, o):
        if self.gmode == "match":
            if len(toks) == 0:
                return
            v = self.tv(toks)
            a = torch.softmax(C.BETA * v.detach(), 0)
            loss = ((v - a * o) ** 2).sum()
            self.ov.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.val.parameters(), 1.0)
            with torch.no_grad():
                for p in self.val.parameters():
                    p.add_(-LR_MATCH * p.grad)
            return
        if self.gmode not in ("gated", "gshuf"):
            return super().learn(toks, o)
        if len(toks) == 0:
            return
        v = self.tv(toks)
        e2 = (o - float(v.detach().sum())) ** 2
        s = e2 / (self.m + 1e-4) if self.m is not None else 0.0
        self.m = e2 if self.m is None else (1 - RHO_S) * self.m + RHO_S * e2
        self.past.append(s)
        if self.gmode == "gshuf":
            s = self.past[int(self.rs.integers(len(self.past)))]
        h = math.exp(-s / KAPPA)
        a = torch.softmax(C.BETA * v.detach(), 0)
        gv = torch.autograd.grad(v.sum(), list(self.val.parameters()), retain_graph=True)
        loss = ((v - a * o) ** 2).sum()
        self.ov.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.val.parameters(), 1.0)
        num = den = 0.0
        with torch.no_grad():
            for p, om, g in zip(self.val.parameters(), self.omega, gv):
                om.mul_(1 - C.RHO).add_(g ** 2)
                f = 1.0 / (1.0 + C.LAM * om / (om.mean() + 1e-8) * h)
                num += float(((f * p.grad) ** 2).sum()); den += float((p.grad ** 2).sum())
                p.add_(-C.LR * p.grad * f)
        if den > 0:
            self.step_ratio.append((num / den) ** 0.5)
        self.h_log.append(h)


def life(form, advice, seed, mode, gamma=C.GAMMA, coop_bad_at=None, n_dyad=C.N_DYAD):
    A = torch.tensor(np.random.default_rng([seed, 0]).normal(0, 1, (C.K, C.D)), dtype=torch.float32)
    r, rc = np.random.default_rng([seed, 3]), np.random.default_rng([seed, 7])
    ag = GAgent(seed, mode)
    ag.h_log = []
    home = "C" if form == 0 else "Q"
    for _ in range(C.N_FORM):
        toks, o = C.draw(A, r, home, C.O)
        ag.learn(toks, o)
        ag.mem[home] = (ag.mem[home] + [toks])[-30:]
    with torch.no_grad():
        formed = ag.tv(A).numpy().copy()
    n_form_h = len(ag.h_log)
    follow, series = [], []
    for t in range(n_dyad):
        C.step(ag, A, r, rc, advice, gamma, coop_bad_at, t, follow)
        if t % 10 == 9:
            with torch.no_grad():
                series.append(ag.tv(A).numpy().round(3).tolist())
    with torch.no_grad():
        tv = ag.tv(A).numpy()
    hd = ag.h_log[n_form_h:]
    return {"late": float(np.mean(follow[-1000:])), "T": ag.T(advice), "coop": float(tv[0]), "qual": float(tv[1]),
            "over": float(tv[2]), "formed": formed.round(4).tolist(), "series10": series,
            "step_ratio": float(np.mean(ag.step_ratio)) if ag.step_ratio else float("nan"),
            "h_mean_dyad": float(np.mean(hd)) if hd else float("nan"),
            "h_mean_first100": float(np.mean(hd[:100])) if hd else float("nan")}


def change_time(series10, start, k=0):
    """First step (10-step grid, counted from `start`) at which aspect k's value is below 0; len if never."""
    for i, x in enumerate(series10):
        t = (i + 1) * 10
        if t > start and x[k] < 0:
            return t - start
    return C.N_DYAD - start


def per_seed(s):
    out = {}
    for mode in MODES:
        out[f"{mode}_f0_g4"] = life(0, "Q", s, mode)
        out[f"{mode}_f1_g4"] = life(1, "Q", s, mode)
        out[f"{mode}_f1_advC_g4"] = life(1, "C", s, mode)
        for g in (0.0, C.GAMMA):
            out[f"{mode}_reason_g{g:g}"] = life(0, "Q", s, mode, gamma=g, coop_bad_at=C.COOP_BAD_AT)
    print(s, "done", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--proto", nargs=2, type=int)
    ap.add_argument("--dev", nargs=2, type=int)
    ap.add_argument("--calib", nargs=3)            # A B LR: change times of gated and of match at LR (one seed range)
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.smoke:
        LR_MATCH = 0.005 if math.isnan(LR_MATCH) else LR_MATCH
        for mode in MODES:
            row = life(0, "Q", 1, mode, n_dyad=50, coop_bad_at=20)
            assert np.isfinite(row["coop"]) and len(row["series10"]) == 5
        print("smoke ok")
    elif a.proto:
        for s in range(*a.proto):
            row = {}
            for mode in ("gated", "gshuf", "slow", "consol", "none"):     # match needs LR_MATCH
                n = life(0, "Q", s, mode)
                rr = life(0, "Q", s, mode, gamma=0.0, coop_bad_at=C.COOP_BAD_AT)
                row[mode] = (round(n["coop"], 2), round(n["qual"], 2), round(n["late"], 2),
                             change_time(rr["series10"], C.COOP_BAD_AT), round(rr["coop"], 2),
                             round(n["h_mean_first100"], 2), round(n["h_mean_dyad"], 2), round(n["step_ratio"], 3))
            print(s, "KAPPA", KAPPA, json.dumps(row), flush=True)
    elif a.calib:
        LR_MATCH = float(a.calib[2])
        for s in range(int(a.calib[0]), int(a.calib[1])):
            ct = {m: change_time(life(0, "Q", s, m, gamma=0.0, coop_bad_at=C.COOP_BAD_AT)["series10"], C.COOP_BAD_AT)
                  for m in ("gated", "match")}
            print(s, "LR", LR_MATCH, "KAPPA", KAPPA, json.dumps(ct), flush=True)
    elif a.dev:
        t0, mx = time.time(), []
        for s in range(*a.dev):
            for mode in MODES:
                row = life(0, "Q", s, mode, coop_bad_at=C.COOP_BAD_AT)
                assert np.isfinite(row["coop"]) and np.isfinite(row["T"])
                mx.append(abs(row["coop"]))
        print(f"agents={len(mx)} max|coop|={max(mx):.2f} finite=True time={time.time() - t0:.0f}s")
    else:
        rows = {s: per_seed(s) for s in range(*a.seeds)}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"seeds": a.seeds, "kappa": KAPPA, "rows": rows}, fh, indent=1)
        print("written", a.out)
