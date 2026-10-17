"""P22: how many evaluation worlds (seeds) are needed? Power and false-positive rate of the judgement rule
(>= 75 % of seeds in the predicted direction AND median >= floor; 8 of 10 for n <= 10) as a function of the number of seeds n,
from the production-scale dev data (dev seeds 77380-77389; no evaluation seed). CPU only.
Model (stated before computing): for a judgement J, the per-seed contrast at planted amplitude m is
    d_i(m) = c_J(m) + r_i,   r_i = d_i(PL-struct, m=1) - median_i d_i(PL-struct, m=1)       (P15 calibration rows, 10 seeds)
with c_J(m) the median contrast measured in P19 at m in {0.5, 1, 1.5, 2, 3}; the null is the PL-noise rows themselves.
The floor is resampled with the seeds (0.25 x median of the retest quantity of the resampled seeds).
Bootstrap: n seeds drawn with replacement from the 10, 20,000 draws. Limitation: 10 dev seeds give the noise shape only
roughly; tails beyond the 10 observed values are not represented. Also reported: the exact sign-test false-positive rate
under a symmetric null (no floor), and the real NN-SET contrasts (C-off and C-SLp, F1) for information only (n is NOT chosen
from them). Writes p22_results.json."""
import json
from math import comb, ceil
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
NS = (10, 15, 20, 25, 30, 40)
J = {"J1-a": ("D_AB_un", "D_AA_un", "D_AA_un"), "J2-a": ("Drho_AB_un", "Drho_AA_un", "Drho_AA_un"),
     "J2-b": ("Drho_AB_nc", "Drho_AA_nc", "Drho_AA_nc")}
rng = np.random.default_rng([77150, 22])


def need(n):
    return ceil((0.8 if n <= 10 else 0.75) * n)


def sign_fp(n, p=0.5):
    k = need(n)
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def boot(d, base, n, B=20000):
    idx = rng.integers(0, len(d), size=(B, n))
    dd = d[idx]; fl = 0.25 * np.median(base[idx], axis=1)
    return float(((dd > 0).sum(1) >= need(n)) & (np.median(dd, axis=1) >= fl)).mean() if False else \
        float((((dd > 0).sum(1) >= need(n)) & (np.median(dd, axis=1) >= fl)).mean())


def main():
    cells = json.loads((HERE / "p15_cells_set_IH.json").read_text(encoding="utf-8"))
    off = [r for r in cells["set_IH_C-off_prod"]["rows"] if r["fam"] == "F1"]
    slp = [r for r in cells["set_IH_C-SLp_prod"]["rows"] if r["fam"] == "F1"]
    p19 = json.loads((HERE / "p19_results.json").read_text(encoding="utf-8"))
    out = {"sign_test_false_positive_symmetric_null": {n: sign_fp(n) for n in NS}}
    for j, (ab, aa, base_k) in J.items():
        st = np.array([r["calib"]["PL-struct"][ab] - r["calib"]["PL-struct"][aa] for r in off])
        nz = np.array([r["calib"]["PL-noise"][ab] - r["calib"]["PL-noise"][aa] for r in off])
        base = np.array([r["calib"]["PL-struct"][base_k] for r in off])
        resid = st - np.median(st)
        res = {"null(PL-noise)": {n: boot(nz, base, n) for n in NS}}
        for m in ("0.5", "1.0", "1.5", "2.0", "3.0"):
            c = p19[m]["PL-struct"][j]["median"]
            res[f"planted m={m}"] = {n: boot(c + resid, base, n) for n in NS}
        for nm, rows in (("real NN-SET C-off F1", off), ("real NN-SET C-SLp F1", slp)):
            d = np.array([r[ab] - r[aa] for r in rows]); b = np.array([r[base_k] for r in rows])
            res[nm + " (information only)"] = {n: boot(d, b, n) for n in NS}
        out[j] = res
    (HERE / "p22_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for j, res in out.items():
        print(j)
        for k, v in res.items() if isinstance(res, dict) and j != "sign_test_false_positive_symmetric_null" else [("sign", res)]:
            print("  ", k.ljust(42), " ".join(f"n{n}:{v[n]:.3f}" for n in NS))


if __name__ == "__main__":
    main()
