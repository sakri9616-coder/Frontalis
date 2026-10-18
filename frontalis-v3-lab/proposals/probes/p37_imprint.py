"""P37: can an individual's history leave an imprint on the LEARNED evaluation that exceeds path noise? Dev seeds 77380-77389,
mid scale, F1, I-H, C-SLp, no choice scenes. Pre-note in LOG-dev.md (P37), written before running.
Quadruple of roles (same init): A (early 0, env 1, sgd 3), B (early 1, env 1, sgd 3: history differs only),
A_sgd (early 0, env 1, sgd 4: SGD order differs only), A_env (early 0, env 2, sgd 3: outcome samples differ only).
H = D(A,B), S = D(A,A_sgd), E = D(A,A_env) on the centered evaluation on P_un (draft-4 dist scale).
  base / O1  NN-SET; O1 = mean evaluation of K = 4 nets that learn the same history with 4 SGD streams; base = member 0.
  O2         shared slow weights + individual fast weights: an NN-SET encoder pre-trained per world on a common stream (both
             clusters, its own env and sgd streams, C-off; no individual's history), frozen; each individual learns only the head.
  NN0        reference: frozen random encoder, head learned (draft-4 NN-0).
Writes p37_imprint.json."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "diag5"))
import sim5, sim4, measure4 as m4
from cells import SCALE, pers_list

SEEDS = list(range(77380, 77390))
SG = {"A": (3, 5, 6, 7), "B": (3, 5, 6, 7), "A_env": (3, 5, 6, 7), "A_sgd": (4, 8, 9, 0)}
QUAD = (("A", 0, 1), ("B", 1, 1), ("A_sgd", 0, 1), ("A_env", 0, 2))
LR = json.loads((HERE / "p15_lr.json").read_text(encoding="utf-8"))["chosen"]
BASE = dict(innate="IH", **SCALE["mid"], lr=LR, choice="none")


def roles_for(K):
    out = []
    for name, early, env in QUAD:
        for m in range(K):
            out.append((f"{name}#{m}", early, env, SG[name][m] if K > 1 else SG[name][0]))
    return tuple(out)


def evals(st, pers, worlds):
    E = {}
    for (s, f, c), vw in m4.views(st, pers, worlds).items():
        E[s] = {r: dict(V=v.get("un")["V"], r2_un=m4.own_r2(v, "un") if r.endswith("#0") else None) for r, v in vw.items()}
    return E


def table(E, K):
    rows = []
    for s, d in E.items():
        V = {name: np.mean([d[f"{name}#{m}"]["V"] for m in range(K)], 0) for name, _, _ in QUAD}
        scale = np.mean([np.abs(v).mean() for v in V.values()]) * sim4.KB + 1e-12
        D = lambda a, b: m4.dist(m4.cent(V[a]), m4.cent(V[b]), scale)
        rows.append(dict(seed=s, H=D("A", "B"), S=D("A", "A_sgd"), E=D("A", "A_env"), r2_un_A=d["A#0"]["r2_un"]))
    return rows


def verdict(rows):
    med = lambda k: float(np.median([r[k] for r in rows]))
    nHS = sum(r["H"] > r["S"] for r in rows); nHE = sum(r["H"] > r["E"] for r in rows)
    return dict(H=med("H"), S=med("S"), E=med("E"), H_over_S=med("H") / max(med("S"), 1e-12), n_H_gt_S=nHS, n_H_gt_E=nHE,
                r2_un_A=med("r2_un_A"), success=bool(med("H") >= 2 * med("S") and nHS >= 8 and nHE >= 8 and med("r2_un_A") >= 0.5))


def o3():
    """O3 (added after seeing O1/O2, disclosed in LOG-dev P37): O2 with K = 4 heads per individual (same history, 4 SGD streams)."""
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in SEEDS}
    pre_p = pers_list(SEEDS, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim5.run(pre_p, worlds, dict(learner="set", **BASE))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    pers = pers_list(SEEDS, ("F1",), roles_for(4), "", ["C-SLp"])
    idx = torch.tensor([wi[q["world"][0]] for q in pers], device=sim4.DEV)
    st = sim5.run(pers, worlds, dict(learner="nn0", partner_scale=0.0, **BASE), init_params={k: v[idx].clone() for k, v in shared.items()})
    rows = table(evals(st, pers, worlds), 4)
    out = json.loads((HERE / "p37_imprint.json").read_text(encoding="utf-8"))
    out["O3_shared_encoder_head_K4"] = dict(verdict=verdict(rows), rows=rows, note="added after seeing O1/O2")
    (HERE / "p37_imprint.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("O3", json.dumps(out["O3_shared_encoder_head_K4"]["verdict"]))


def main():
    worlds = {(s, "F1"): sim5.make_world(s, "F1") for s in SEEDS}
    out, t0 = {}, time.time()
    # base and O1 (one run, K = 4)
    pers = pers_list(SEEDS, ("F1",), roles_for(4), "", ["C-SLp"])
    st = sim5.run(pers, worlds, dict(learner="set", **BASE)); E = evals(st, pers, worlds); del st; torch.cuda.empty_cache()
    for nm, K in (("base_NNSET_K1", 1), ("O1_ensemble_K4", 4)):
        rows = table({s: {k: v for k, v in d.items()} for s, d in E.items()}, K)
        out[nm] = dict(verdict=verdict(rows), rows=rows)
    # O2: shared pre-trained encoder (common stream per world), individual head
    pre_p = pers_list(SEEDS, ("F1",), (("S", -1, 9, 9),), "", ["C-off"])
    stp = sim5.run(pre_p, worlds, dict(learner="set", **BASE))
    shared = {k: v.detach() for k, v in stp["net"].p.items()}; wi = {q["world"][0]: i for i, q in enumerate(pre_p)}
    del stp; torch.cuda.empty_cache()
    pers1 = pers_list(SEEDS, ("F1",), roles_for(1), "", ["C-SLp"])
    idx = torch.tensor([wi[q["world"][0]] for q in pers1], device=sim4.DEV)
    init = {k: v[idx].clone() for k, v in shared.items()}
    for nm, ip in (("O2_shared_encoder_head", init), ("NN0_random_encoder_head", None)):
        st = sim5.run(pers1, worlds, dict(learner="nn0", partner_scale=0.0, **BASE), init_params=ip)
        rows = table(evals(st, pers1, worlds), 1); del st; torch.cuda.empty_cache()
        out[nm] = dict(verdict=verdict(rows), rows=rows)
    out["secs"] = time.time() - t0
    (HERE / "p37_imprint.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, json.dumps(v["verdict"]))


if __name__ == "__main__":
    o3() if sys.argv[1:] == ["o3"] else main()
