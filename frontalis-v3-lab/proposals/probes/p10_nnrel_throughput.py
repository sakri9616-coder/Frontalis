"""P10: early measurement of NN-REL throughput (lead decision 5 on draft 3). Kernel-level benchmark: per-personality
weights, forward + backward + AdamW on minibatches of 32, all personalities batched on the GPU. No world, no seeds of the
evaluation range. Three learners:
  MLP64  : the P5/P9 learner core (two encoders 64 x 2, belief + multi-channel value heads) -> calibrates against P9's 46k
  SET128 : NN-SET as in DESIGN s5 (two encoders 128 x 2)
  REL64  : NN-REL as in DESIGN s5 (tokens -> 2 layers of multi-head self-attention, d = 64, 4 heads, MLP 128; two encoders)
  REL64m : the same NN-REL with attention written as plain matmuls (same function; faster for T = 6)
Tokens for REL: actor entity, other entity, context, body state, 2 recent-event tokens (T = 6), each projected from 24 dims.
Estimated full-simulator throughput of X = 46,000 * kernel(MLP64) / kernel(X) (assumes the non-kernel overhead scales alike;
stated as an assumption)."""
import json, sys, time
from pathlib import Path
import torch

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
BS, T, DT, K = 32, 6, 24, 3


def init(P, shapes, g):
    return {n: (torch.randn(P, *s, generator=g, device=DEV) * (2 / s[0]) ** 0.5).requires_grad_(True) if n[0] == "W"
            else torch.zeros(P, *s, device=DEV).requires_grad_(True) for n, s in shapes.items()}


def mlp_params(P, h, g):
    sh = {}
    for e in ("enc", "venc"):
        sh.update({f"W1{e}": (T * DT, h), f"b1{e}": (1, h), f"W2{e}": (h, h), f"b2{e}": (1, h)})
    sh.update({"Wb": (h, K), "bb": (1, K), "Wv": (h, K), "bv": (1, K)})
    return init(P, sh, g)


def mlp_fwd(p, x):
    x = x.flatten(2)
    out = []
    for e in ("enc", "venc"):
        h = torch.relu(torch.bmm(x, p[f"W1{e}"]) + p[f"b1{e}"])
        out.append(torch.relu(torch.bmm(h, p[f"W2{e}"]) + p[f"b2{e}"]))
    return torch.bmm(out[0], p["Wb"]) + p["bb"], torch.bmm(out[1], p["Wv"]) + p["bv"]


def rel_params(P, d, g, layers=2):
    sh = {}
    for e in ("enc", "venc"):
        sh[f"Win{e}"] = (DT, d); sh[f"bin{e}"] = (1, d)
        for l in range(layers):
            for n in ("q", "k", "v", "o"):
                sh[f"W{n}{e}{l}"] = (d, d)
            sh[f"Wf1{e}{l}"] = (d, 2 * d); sh[f"bf1{e}{l}"] = (1, 2 * d)
            sh[f"Wf2{e}{l}"] = (2 * d, d); sh[f"bf2{e}{l}"] = (1, d)
    sh.update({"Wb": (d, K), "bb": (1, K), "Wv": (d, K), "bv": (1, K)})
    return init(P, sh, g)


def rel_fwd(p, x, d=64, H=4, layers=2, manual=False):
    P = x.shape[0]
    out = []
    for e in ("enc", "venc"):
        h = (torch.bmm(x.reshape(P, BS * T, DT), p[f"Win{e}"]) + p[f"bin{e}"])           # P, BS*T, d
        for l in range(layers):
            hn = torch.nn.functional.layer_norm(h, (d,))
            q, k_, v = (torch.bmm(hn, p[f"W{n}{e}{l}"]).view(P * BS, T, H, d // H).transpose(1, 2) for n in "qkv")
            if manual:   # T = 6 is tiny; the fused SDPA kernels pad to 64 x 64 tiles and are ~20x slower here (profiled)
                a = torch.softmax(q @ k_.transpose(-1, -2) / (d // H) ** 0.5, -1) @ v
            else:
                a = torch.nn.functional.scaled_dot_product_attention(q, k_, v)               # P*BS, H, T, d/H
            h = h + torch.bmm(a.transpose(1, 2).reshape(P, BS * T, d), p[f"Wo{e}{l}"])
            hn = torch.nn.functional.layer_norm(h, (d,))
            h = h + torch.bmm(torch.relu(torch.bmm(hn, p[f"Wf1{e}{l}"]) + p[f"bf1{e}{l}"]), p[f"Wf2{e}{l}"]) + p[f"bf2{e}{l}"]
        out.append(h.view(P, BS, T, d).mean(2))
    return torch.bmm(out[0], p["Wb"]) + p["bb"], torch.bmm(out[1], p["Wv"]) + p["bv"]


def bench(name, P, steps=60):
    g = torch.Generator(device=DEV); g.manual_seed(77395)
    if name == "MLP64":
        p, f = mlp_params(P, 64, g), mlp_fwd
    elif name == "SET128":
        p, f = mlp_params(P, 128, g), mlp_fwd
    elif name == "REL64":
        p, f = rel_params(P, 64, g), rel_fwd
    else:
        p, f = rel_params(P, 64, g), (lambda p_, x_: rel_fwd(p_, x_, manual=True))
    opt = torch.optim.AdamW(list(p.values()), lr=3e-4, weight_decay=0.0, foreach=True)
    x = torch.randn(P, BS, T, DT, generator=g, device=DEV)
    yb = torch.randn(P, BS, K, generator=g, device=DEV); yv = torch.randn(P, BS, K, generator=g, device=DEV)
    for i in range(steps + 10):
        if i == 10:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        b, v = f(p, x)
        loss = ((b - yb) ** 2).mean() + ((v - yv) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / steps
    mem = torch.cuda.max_memory_allocated() / 2 ** 20
    torch.cuda.reset_peak_memory_stats()
    del p, opt, x
    torch.cuda.empty_cache()
    return dict(P=P, sec_per_step=sec, kernel_pers_steps_per_sec=P / sec, peak_mib=mem)


out = {}
for name in ("MLP64", "SET128", "REL64", "REL64m"):
    for P in (360, 720, 1440):
        try:
            r = bench(name, P)
        except torch.cuda.OutOfMemoryError:
            r = dict(P=P, oom=True); torch.cuda.empty_cache()
        out[f"{name}@{P}"] = r
        print(name, r, file=sys.stderr)
best = {n: max((v.get("kernel_pers_steps_per_sec", 0) for k, v in out.items() if k.startswith(n + "@"))) for n in ("MLP64", "SET128", "REL64", "REL64m")}
est = {n: 46000 * best[n] / best["MLP64"] for n in best}
budget = 1.43e9
out["estimate"] = dict(best_kernel=best, est_full_sim_pers_steps_per_sec=est,
                       main_judgement_hours={n: budget / est[n] / 3600 for n in est},
                       note="11,200 personalities x 128,000 steps per learner family; est = 46k (P9) * kernel ratio to MLP64")
Path(__file__).with_name("p10_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out["estimate"], indent=1))
