"""P11: engineering speed-up of the NN-REL kernel for few tokens (lead decision on draft 3.1, item 1(c)).
The function (architecture, parameters, loss) is unchanged; only how it is computed changes. Each variant is checked
against the P10 reference (REL64m, fp32) for equality of outputs and gradients on the same parameters and inputs
(max relative difference; fp32 rounding only), then timed (forward + backward + AdamW, minibatch 32).
No world and no seeds of the evaluation range (init seed 77396, in the probe range)."""
import json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False   # equality checks in true fp32
BS, T, DT, K, D, H, L = 32, 6, 24, 3, 64, 4, 2
DH = D // H
ENC = ("enc", "venc")


def params(P, g):
    sh = {}
    for e in ENC:
        sh[f"Win{e}"] = (DT, D); sh[f"bin{e}"] = (1, D)
        for l in range(L):
            for n in "qkvo":
                sh[f"W{n}{e}{l}"] = (D, D)
            sh[f"Wf1{e}{l}"] = (D, 2 * D); sh[f"bf1{e}{l}"] = (1, 2 * D)
            sh[f"Wf2{e}{l}"] = (2 * D, D); sh[f"bf2{e}{l}"] = (1, D)
    sh.update({"Wb": (D, K), "bb": (1, K), "Wv": (D, K), "bv": (1, K)})
    return {n: (torch.randn(P, *s, generator=g, device=DEV) * (2 / s[0]) ** 0.5).requires_grad_(True) if n[0] == "W"
            else (0.1 * torch.randn(P, *s, generator=g, device=DEV)).requires_grad_(True) for n, s in sh.items()}


def ref_fwd(p, x):
    """P10 REL64m (reference)."""
    P = x.shape[0]
    out = []
    for e in ENC:
        h = torch.bmm(x.reshape(P, BS * T, DT), p[f"Win{e}"]) + p[f"bin{e}"]
        for l in range(L):
            hn = F.layer_norm(h, (D,))
            q, k_, v = (torch.bmm(hn, p[f"W{n}{e}{l}"]).view(P * BS, T, H, DH).transpose(1, 2) for n in "qkv")
            a = torch.softmax(q @ k_.transpose(-1, -2) / DH ** 0.5, -1) @ v
            h = h + torch.bmm(a.transpose(1, 2).reshape(P, BS * T, D), p[f"Wo{e}{l}"])
            hn = F.layer_norm(h, (D,))
            h = h + torch.bmm(torch.relu(torch.bmm(hn, p[f"Wf1{e}{l}"]) + p[f"bf1{e}{l}"]), p[f"Wf2{e}{l}"]) + p[f"bf2{e}{l}"]
        out.append(h.view(P, BS, T, D).mean(2))
    return torch.bmm(out[0], p["Wb"]) + p["bb"], torch.bmm(out[1], p["Wv"]) + p["bv"]


def attn_bcast(q, k_, v):
    """q, k, v: (N, T, H, DH). Scores by broadcast multiply-sum (no tiny GEMMs, no transposes)."""
    s = (q.unsqueeze(2) * k_.unsqueeze(1)).sum(-1) / DH ** 0.5          # N, Tq, Tk, H
    a = torch.softmax(s, 2)
    return (a.unsqueeze(-1) * v.unsqueeze(1)).sum(2)                     # N, Tq, H, DH


def v1_fwd(p, x):
    """Fused QKV projection (one bmm) + broadcast attention."""
    P = x.shape[0]
    out = []
    for e in ENC:
        h = torch.bmm(x.reshape(P, BS * T, DT), p[f"Win{e}"]) + p[f"bin{e}"]
        for l in range(L):
            hn = F.layer_norm(h, (D,))
            qkv = torch.bmm(hn, torch.cat([p[f"Wq{e}{l}"], p[f"Wk{e}{l}"], p[f"Wv{e}{l}"]], -1)).view(P * BS, T, 3, H, DH)
            a = attn_bcast(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
            h = h + torch.bmm(a.reshape(P, BS * T, D), p[f"Wo{e}{l}"])
            hn = F.layer_norm(h, (D,))
            h = h + torch.bmm(torch.relu(torch.bmm(hn, p[f"Wf1{e}{l}"]) + p[f"bf1{e}{l}"]), p[f"Wf2{e}{l}"]) + p[f"bf2{e}{l}"]
        out.append(h.view(P, BS, T, D).mean(2))
    return torch.bmm(out[0], p["Wb"]) + p["bb"], torch.bmm(out[1], p["Wv"]) + p["bv"]


def v2_fwd(p, x):
    """V1 + both encoders computed in one batch of 2P (stacked weights): half the kernel launches, larger GEMMs."""
    P = x.shape[0]
    st = lambda n: torch.cat([p[n.format(e="enc")], p[n.format(e="venc")]], 0)
    xx = x.reshape(P, BS * T, DT).repeat(2, 1, 1)
    h = torch.bmm(xx, st("Win{e}")) + st("bin{e}")
    for l in range(L):
        hn = F.layer_norm(h, (D,))
        W = torch.cat([st("Wq{e}%d" % l), st("Wk{e}%d" % l), st("Wv{e}%d" % l)], -1)
        qkv = torch.bmm(hn, W).view(2 * P * BS, T, 3, H, DH)
        a = attn_bcast(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        h = h + torch.bmm(a.reshape(2 * P, BS * T, D), st("Wo{e}%d" % l))
        hn = F.layer_norm(h, (D,))
        h = h + torch.bmm(torch.relu(torch.bmm(hn, st("Wf1{e}%d" % l)) + st("bf1{e}%d" % l)), st("Wf2{e}%d" % l)) + st("bf2{e}%d" % l)
    z = h.view(2 * P, BS, T, D).mean(2)
    return torch.bmm(z[:P], p["Wb"]) + p["bb"], torch.bmm(z[P:], p["Wv"]) + p["bv"]


VARIANTS = {"ref": ref_fwd, "v1_fusedqkv_bcast": v1_fwd, "v2_stacked_enc": v2_fwd}


def check(P=64):
    g = torch.Generator(device=DEV); g.manual_seed(77396)
    p = params(P, g); x = torch.randn(P, BS, T, DT, generator=g, device=DEV)
    res = {}
    base = None
    for n, f in VARIANTS.items():
        for t in p.values():
            t.grad = None
        b, v = f(p, x)
        loss = (b ** 2).mean() + (v ** 3).mean()
        loss.backward()
        flat = torch.cat([b.flatten(), v.flatten()] + [p[k].grad.flatten() for k in sorted(p)]).detach().clone()
        if base is None:
            base = flat
        res[n] = float(((flat - base).abs().max() / base.abs().max()).item())
    return res


def bench(f, P, steps=40, tf32=True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    g = torch.Generator(device=DEV); g.manual_seed(77396)
    p = params(P, g)
    opt = torch.optim.AdamW(list(p.values()), lr=3e-4, foreach=True)
    x = torch.randn(P, BS, T, DT, generator=g, device=DEV)
    yb = torch.randn(P, BS, K, generator=g, device=DEV); yv = torch.randn(P, BS, K, generator=g, device=DEV)
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 8):
        if i == 8:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        b, v = f(p, x)
        loss = ((b - yb) ** 2).mean() + ((v - yv) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / steps
    r = dict(P=P, kernel_pers_steps_per_sec=P / sec, peak_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
    del p, opt, x
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    return r


if __name__ == "__main__":
    out = {"equality_max_rel_diff_vs_ref_fp32": check()}
    print(out, file=sys.stderr)
    for n, f in VARIANTS.items():
        for P in (360, 720):
            out[f"{n}@{P}"] = bench(f, P)
            print(n, out[f"{n}@{P}"], file=sys.stderr)
    Path(__file__).with_name("p11_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
