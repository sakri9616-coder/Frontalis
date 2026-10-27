import numpy as np, torch
exec(open("npn_proto.py", encoding="utf-8").read().split("for beta in (0.0, 2.0, 4.0):")[0])
beta = 4.0
for s in range(6):
    A = world(s)
    na, oa = grow(beta, A, s, s, 2); nb, ob = grow(beta, A, s, s, 0)
    with torch.no_grad(): va, vb = na.token_values(A).numpy(), nb.token_values(A).numpy()
    # attention the same new event gets in each network
    with torch.no_grad():
        aa = torch.softmax(beta * na.token_values(A[[0, 2]]), 0).numpy(); ab = torch.softmax(beta * nb.token_values(A[[0, 2]]), 0).numpy()
    da, _ = same_new(na, oa, A, s, beta); db, _ = same_new(nb, ob, A, s, beta)
    print(f"seed {s}: Pa v {va.round(2)} att(0,2) {aa.round(2)} d {da:+.2f} | Pb v {vb.round(2)} att(0,2) {ab.round(2)} d {db:+.2f}")
