"""Probe P0 (audit evidence; runs the existing npnv3 code read-only, no state saved).

Checks, against the code rather than the docs:
  1. For every outcome type and every reachable need state h, the sign of each need's teacher contribution equals the sign
     of the developer's effect-table row (magnitude in [0, 1], utility monotone) -> the direction of what each outcome
     teaches is fixed for all personalities and all histories.
  2. Outcome types with identical rows in M (innately indistinguishable for every personality).
  3. Additivity of the MF value: V_MF(person p, context c) = f(p) + g(c, concepts) exactly, for a trained NPN
     (person x context interaction is not representable).
  4. Conditioned value C of an unseen cue is exactly 0 (no generalisation across cue names).
Run: PYTHONUTF8=1 python -B p0_audit_checks.py  (from this folder)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB))
from npnv3 import innate  # noqa: E402
from npnv3.npn import NPN  # noqa: E402

out = {}
rng = np.random.default_rng([77399, 0])
H = rng.uniform(0, 6, size=(20000, 5))
mags = rng.uniform(0, 1, size=20000)
viol = {}
for t in innate.OUTCOME_TYPES:
    row = innate.M[innate.TYPE_INDEX[t]]
    rows = mags[:, None] * row[None, :]
    du = innate.delta_u(H, rows)
    bad = ((du > 1e-12) & (row[None, :] < 0)) | ((du < -1e-12) & (row[None, :] > 0)) | ((np.abs(du) > 1e-12) & (row[None, :] == 0))
    viol[t] = int(bad.sum())
out["sign_violations_per_type"] = viol
out["types_all_nonpositive"] = [t for t in innate.OUTCOME_TYPES if (innate.M[innate.TYPE_INDEX[t]] <= 0).all() and (innate.M[innate.TYPE_INDEX[t]] < 0).any()]
out["types_all_nonnegative"] = [t for t in innate.OUTCOME_TYPES if (innate.M[innate.TYPE_INDEX[t]] >= 0).all() and (innate.M[innate.TYPE_INDEX[t]] > 0).any()]
out["types_zero_row"] = [t for t in innate.OUTCOME_TYPES if (innate.M[innate.TYPE_INDEX[t]] == 0).all()]
out["identical_rows"] = [(a, b) for a, b in itertools.combinations(innate.OUTCOME_TYPES, 2)
                         if np.array_equal(innate.M[innate.TYPE_INDEX[a]], innate.M[innate.TYPE_INDEX[b]])]

# 3. additivity of V_MF in person x context, on an NPN trained on a short history with person x context structure
npn = NPN(seed=77399)
ctxs = ["studio", "project_meeting"]
for i in range(30):
    # A helps in studio, harms in project_meeting; B the reverse (a person x context interaction in the world)
    for pid, good_ctx in (("A", "studio"), ("B", "project_meeting")):
        ctx = ctxs[i % 2]
        typ = "support" if ctx == good_ctx else "betrayal"
        npn.experience({"id": f"e{i}{pid}", "context": ctx, "concepts": ["collaboration"],
                        "participants": [{"id": pid, "class": "collaborator"}],
                        "outcomes": [{"type": typ, "actor": pid, "magnitude": 0.8}]})
V = {}
for pid in ("A", "B"):
    for ctx in ctxs:
        mf = npn.value.mf([{"id": pid, "class": "collaborator"}], ctx, ["collaboration"], npn.concern.a)
        V[(pid, ctx)] = mf["V_MF"]
inter = V[("A", "studio")] - V[("A", "project_meeting")] - V[("B", "studio")] + V[("B", "project_meeting")]
out["V_MF"] = {f"{k[0]}|{k[1]}": v for k, v in V.items()}
out["V_MF_person_x_context_interaction"] = inter
# 4. conditioned value of an unseen cue
out["C_unseen_cue"] = npn.cond.value(["concept:never_seen_before"], "studio")
out["C_seen_cues"] = {c: npn.cond.value([c], "studio") for c in ("cls:collaborator", "ind:A", "ind:B", "type:support", "type:betrayal")}
out["n_mlp_heads_in_default_npn"] = sum(1 for k in npn.belief.heads if k == "mlp")
out["default_mixture_heads"] = list(npn.cfg.mixture_heads)
print(json.dumps(out, indent=1, ensure_ascii=False))
(Path(__file__).resolve().parent / "p0_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
