import numpy as np, torch
exec(open("npn_dyad.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0])
for s in range(3):
    for form in (0, 1):
        A = torch.tensor(np.random.default_rng([s, 0]).normal(0, 1, (K, D)), dtype=torch.float32)
        r = np.random.default_rng([s, 3]); ag = Agent(s, 4.0); home = "C" if form == 0 else "Q"
        for _ in range(2000):
            toks, o = draw(A, r, home, O); ag.learn(toks, o); ag.mem[home] = (ag.mem[home] + [toks])[-30:]
        with torch.no_grad(): tv = ag.tv(A).numpy()
        qs = [round(ag.Q(k), 2) for k in NAMES]
        # value of typical experiences of each option (if the AI had tried them)
        rr = np.random.default_rng(99); typ = [round(float(np.mean([ag.V(draw(A, rr, k, O)[0]) for _ in range(200)])), 2) for k in NAMES]
        print(f"seed {s} form {form}: token values coop/qual/overwork {tv.round(2)} | Q from memory {dict(zip(NAMES, qs))} | value of typical experiences {dict(zip(NAMES, typ))}")
