"""Judge v19 (PREREG-v19) from result JSON files of npn_dyad_transfer.py.

usage: python judge_dyad_transfer.py --out judgment.json res1.json [res2.json ...]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "credit"))
from run_g import binom_two_sided  # noqa: E402

DOMS = ("low", "blueN", "red", "round", "greenN")
RUNS = (("X2_net", "W"), ("X2_net", "Q"), ("X2r_net", "W"), ("N2_net", "W"), ("X2_lin", "W"))


def three_way(x):
    x = np.asarray(x, float)
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    p = binom_two_sided(pos, pos + neg) if pos + neg else 1.0
    g = {"pos": pos, "neg": neg, "p": p, "mean": float(x.mean()),
         "pass": bool(pos >= 30 and p < 0.01), "reverse_pass": bool(neg >= 30 and p < 0.01)}
    g["verdict"] = "pass" if g["pass"] else ("reverse" if g["reverse_pass"] else "neither")
    return g


def desc(x):
    x = np.asarray(x, float)
    return {"pos": int((x > 0).sum()), "neg": int((x < 0).sum()), "mean": float(x.mean())}


def M(r, name, cond, dom="low", t="1500", kind="boost"):
    ms = r[name][cond]["meas"]
    d = lambda k: ms[t][f"{kind}_{k}"] - ms["0"][f"{kind}_{k}"]
    return -(d(dom) - d("blueN"))


def reading(res):
    v = {k: res[k]["verdict"] for k in ("D1", "D2", "D3", "D4")}
    if not res["sanity"]["ok"]:
        return "健全性を満たさないので、D1〜D4 を解釈しない。"
    if v["D2"] == "pass" and v["D3"] == "pass" and v["D4"] == "pass":
        return ("相手が外れた経験から学んだ言葉の割り引きが、学んだ『悪さ』の表し方を通って、相手が一度も勧めていない、"
                "自分が悪いと評価する分野へ移った（評価の水準での準備）。")
    if v["D2"] == "pass":
        miss = [s for k, s in (("D3", "学んだ評価の寄与"), ("D4", "悪さへの固有性")) if v[k] != "pass"]
        return ("相手が外れた経験は、低の分野の言葉の効きを、対照と比べて下げた。それが自分の悪さの表し方を通ったかは示せなかった（"
                + "・".join(miss) + "が示せなかった）。")
    if v["D2"] == "reverse":
        return "相手が外れた経験の後、低の分野の言葉の効きは、対照と比べてむしろ上がった。"
    return "相手が外れた経験による、低の分野への割り引きの移転は示せなかった。D1 は入力の構造による広がりと区別できないので、移転の根拠にしない。"


def judge(rows):
    a = lambda f: np.array([f(r) for r in rows])
    res = {"n": len(rows)}
    sv = int(sum(r["X2_net"]["sane"][0] > 0 and r["X2_net"]["sane"][1] < 0 and r["X2_net"]["sane"][2] < 0 for r in rows))
    rf = int(sum(r["X2_net"]["W"]["meas"]["0"]["boost_round"] - r["X2_net"]["W"]["meas"]["1500"]["boost_round"] > 0
                 for r in rows))
    res["sanity"] = {"V": sv, "round_discount": rf, "ok": bool(sv >= 36 and rf >= 30)}
    res["D1"] = three_way(a(lambda r: M(r, "X2_net", "W")))
    res["D2"] = three_way(a(lambda r: M(r, "X2_net", "W") - M(r, "X2_net", "Q")))
    res["D3"] = three_way(a(lambda r: M(r, "X2_net", "W") - M(r, "X2r_net", "W")))
    res["D4"] = three_way(a(lambda r: M(r, "X2_net", "W") - M(r, "X2_net", "W", "red")))
    res["reading"] = reading(res)
    d = {"M_low_mean": {f"{n}.{c}": float(a(lambda r: M(r, n, c)).mean()) for n, c in RUNS},
         "M_red_mean": {f"{n}.{c}": float(a(lambda r: M(r, n, c, "red")).mean()) for n, c in RUNS},
         "N2_W_M": desc(a(lambda r: M(r, "N2_net", "W"))),
         "X2_minus_N2": desc(a(lambda r: M(r, "X2_net", "W") - M(r, "N2_net", "W"))),
         "lin_M_abs_max": float(np.abs(a(lambda r: M(r, "X2_lin", "W"))).max()),
         "D2_on_e": desc(a(lambda r: M(r, "X2_net", "W", kind="e") - M(r, "X2_net", "Q", kind="e"))),
         "D1_on_e": desc(a(lambda r: M(r, "X2_net", "W", kind="e"))),
         "overlap": {n: float(a(lambda r: r[n]["overlap"]).mean()) for n in ("X2_net", "X2r_net", "N2_net")},
         "overlap_M_corr_X2W": float(np.corrcoef(a(lambda r: r["X2_net"]["overlap"]), a(lambda r: M(r, "X2_net", "W")))[0, 1]),
         "means": {f"{n}.{c}": {t: {f"{k}_{dm}": float(a(lambda r: r[n][c]["meas"][t][f"{k}_{dm}"]).mean())
                                    for k in ("boost", "V", "VE", "e") for dm in DOMS} for t in ("0", "1500")}
                   for n, c in RUNS},
         "follow": {f"{n}.{c}": {k: np.sum([r[n][c][k] for r in rows], 0).tolist() for k in ("follow_conf", "follow_quiet")}
                    for n, c in RUNS}}
    res["describe"] = d
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")     # prototype seeds only; not a judgment
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    rows = {}
    for fn in a.files:
        rows.update(json.load(open(fn, encoding="utf-8"))["rows"])
    seeds = sorted(rows, key=int)
    if len(seeds) != 40 and not a.check:
        sys.exit(f"{len(seeds)} seeds: PREREG-v19 judges only when all 40 are present (use --check for prototype seeds)")
    res = judge([rows[s] for s in seeds])
    res["seeds"] = [int(seeds[0]), int(seeds[-1])]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: (v.get("verdict", v.get("ok")) if isinstance(v, dict) else v)
                      for k, v in res.items() if k not in ("describe", "reading")}))
