"""Build diag5/sim5.py = frozen sim4.py + diagnostic / candidate-mechanism switches (all default OFF, so sim5 with default
cfg is sim4). sim4.py itself is never modified.
  bel_r      NN belief loss uses the cycle's attribution posterior r (stored per experience) as responsibilities,
             instead of the fixed-temperature mixture loss that ignores r.
  krr_flat_b ridge learners: belief rows weighted 0.5 / 0.5 instead of r (diagnostic for the KRR > NN-SET gap).
  wgate      candidate mechanism: the CURRENT evaluative state w_use gates the plasticity of every replayed value update
             per channel (loss weight w_k), so a change of state changes how the whole record is (re)learned from then on,
             not only the targets of new experiences. w_use is 1 in C-off, so C-off is unchanged.
  krr_self_b ridge learners: belief rows weighted by softmax(-|db - pb_j|^2 / 0.2) over the two candidates j, computed
             from the learner's OWN belief, in every condition. This is exactly the responsibility that the NN-SET belief
             loss -logsumexp(-|db - pb_j|^2 / 0.2) implies (its gradient weights each candidate by this softmax), so KRR and
             NN-SET use the same attribution rule and the cycle no longer changes KRR's belief input. (2026-10-18 rev.: the
             first version used softmax(ll) with the calibrated sig2, which is not the same temperature; review A4.)
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
src = (HERE.parent / "sim4.py").read_text(encoding="utf-8")


def rep(a, b, n=1):
    global src
    assert src.count(a) == n, (a, src.count(a))
    src = src.replace(a, b)


rep('partner_scale=1.0, chunk=1000, sdpa=0, tf32=0, fused_opt=0)',
    'partner_scale=1.0, chunk=1000, sdpa=0, tf32=0, fused_opt=0, bel_r=0, krr_flat_b=0)')
rep('SAL=torch.ones(P, N, device=DEV), DIS=',
    'SAL=torch.ones(P, N, device=DEV), R=torch.full((P, N, 2), 0.5, device=DEV), DIS=')
rep('buf["SAL"][:, t] = sal\n', 'buf["SAL"][:, t] = sal; buf["R"][:, t] = r\n')
rep('net.add(phi, db[:, None, :].expand(-1, 2, -1), r, TVt,',
    'net.add(phi, db[:, None, :].expand(-1, 2, -1), torch.full_like(r, 0.5) if c["krr_flat_b"] else r, TVt,')
rep('                mix = -torch.logsumexp(-((dbb[:, :, None, :] - pbb) ** 2).sum(-1) / 0.2, dim=2).mean(1)\n',
    '                if c["bel_r"]:\n'
    '                    rb = buf["R"][ar[:, None], idx]\n'
    '                    mix = (rb * ((dbb[:, :, None, :] - pbb) ** 2).sum(-1)).sum(-1).mean(1) / 0.2\n'
    '                else:\n'
    '                    mix = -torch.logsumexp(-((dbb[:, :, None, :] - pbb) ** 2).sum(-1) / 0.2, dim=2).mean(1)\n')
rep('bel_r=0, krr_flat_b=0)', 'bel_r=0, krr_flat_b=0, wgate=0, krr_self_b=0)')
rep('torch.full_like(r, 0.5) if c["krr_flat_b"] else r, TVt,', 'torch.full_like(r, 0.5) if c["krr_flat_b"] else (torch.softmax(-((db[:, None, :] - pb) ** 2).sum(-1) / 0.2, 1) if c["krr_self_b"] else r), TVt,')
rep('                lv = (sw[:, :, None, None] * (vv - tv) ** 2).sum(-1).mean((1, 2))\n',
    '                gw = state["w_use"][:, None, None, :] if c["wgate"] else 1.0\n'
    '                lv = (sw[:, :, None, None] * gw * (vv - tv) ** 2).sum(-1).mean((1, 2))\n')
src = src.replace('"""Draft-4 batched simulator', '"""sim5 = sim4 + switches (diag5/make_sim5.py). Draft-4 batched simulator', 1)
(HERE / "sim5.py").write_text(src, encoding="utf-8")
print("ok")
