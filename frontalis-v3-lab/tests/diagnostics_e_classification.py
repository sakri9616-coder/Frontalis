"""診断（テストではない）: E 系と W6 の不合格を、割当の記録・応答の内訳から「計算機構の破綻」と「モデル化された性質」に読み分ける（SPEC-v3.3 第 8.1 節の原則: 数値的な故障と、モデルが設計どおりに作る性質を区別する）。
事前登録した判定基準・結果は変えない。これは読み方の注記の根拠になる数値を作る。

  python3 tests/diagnostics_e_classification.py     → tests/results/diag_e_classification.json

分類の基準（数値を見る前に書いた）:
  計算機構の破綻 = 状態が有界でない / 学習規則が自分の定義どおりに動かない（記録の恒等式が破れる）/ 変化の原因を記録から説明できない。
  モデル化された性質 = 結果が、そのモデルの規則の帰結として記録から説明できる（規則どおりに動いた結果、基準に届かない・基準を破る）。

ここで測る恒等式（どれも破れていなければ、破綻ではない）:
  (1) 応答 C = Σ_k p(k) C_k（新しい原因の価値は 0）が explain_response で成り立つ
  (2) 応答の変化 = 重みの変化の項 Σ_k Δp(k) C_k + 値の変化の項 Σ_k p'(k) ΔC_k に分けられる
  E1: 獲得の後、手がかりを使わない 50 経験 → 応答は何で縮んだか（重みの項か、値の項か）
  E8(b): 値を持つ別の手がかり other_fear の C の変化は、x のプロトコル（干渉）か、時間だけか（同じ経験数の無関係な経験だけの対照と比べる）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import e_protocol as EP  # noqa: E402
from helpers import RESULTS, dump, pmap  # noqa: E402

SEEDS = list(range(20))


def e1_worker(args):
    seed, design = args
    col = {}
    EP.run_variant(seed, full=True, collect=col, lc_design=design)
    a, b = col["acq"], col["E1"]
    ea = a.explain_response(EP.CUE_NAME, EP.CTX_A)
    eb = b.explain_response(EP.CUE_NAME, EP.CTX_A)
    ua = {c["uid"]: c for c in ea["causes"]}
    ub = {c["uid"]: c for c in eb["causes"]}
    id_err = max(abs(ea["response"] - sum(c["contribution"] for c in ea["causes"])), abs(eb["response"] - sum(c["contribution"] for c in eb["causes"])))
    d_weight = sum((ub[u]["p"] - ua[u]["p"]) * ua[u]["value"] for u in ua if u in ub)
    d_value = sum(ub[u]["p"] * (ub[u]["value"] - ua[u]["value"]) for u in ua if u in ub)
    return {"seed": seed, "design": design, "ratio": eb["response"] / ea["response"], "response_acq": ea["response"], "response_E1": eb["response"],
            "p_new_acq": ea["p_new"], "p_new_E1": eb["p_new"], "sum_p_old_acq": float(sum(c["p"] for c in ea["causes"])), "sum_p_old_E1": float(sum(c["p"] for c in eb["causes"])),
            "identity_error": id_err, "max_abs_dC_k": max([abs(ub[u]["value"] - ua[u]["value"]) for u in ua if u in ub] or [0.0]),
            "delta_response": eb["response"] - ea["response"], "delta_from_weights": d_weight, "delta_from_values": d_value,
            "n_causes_acq": len(ea["causes"]), "n_causes_E1": len(eb["causes"])}


def leak_worker(args):
    seed, design = args
    r = EP.valued_cue_leak(seed, lc_design=design)
    return {"seed": seed, "design": design, "before": r["before"], "after_protocol": r["after_protocol"], "after_control": r["after_control"]}


def mean(xs):
    return float(np.mean(list(xs)))


def main():
    out = {}
    for design in ("spec_literal", "revised"):
        e1 = pmap(e1_worker, [(s, design) for s in SEEDS])
        lk = pmap(leak_worker, [(s, design) for s in SEEDS])
        out[design] = {
            "E1": {"ratio_mean": mean(x["ratio"] for x in e1), "identity_error_max": max(x["identity_error"] for x in e1), "max_abs_dC_k_over_seeds": max(x["max_abs_dC_k"] for x in e1),
                   "p_new_acq_mean": mean(x["p_new_acq"] for x in e1), "p_new_E1_mean": mean(x["p_new_E1"] for x in e1),
                   "sum_p_old_acq_mean": mean(x["sum_p_old_acq"] for x in e1), "sum_p_old_E1_mean": mean(x["sum_p_old_E1"] for x in e1),
                   "delta_response_mean": mean(x["delta_response"] for x in e1), "delta_from_weights_mean": mean(x["delta_from_weights"] for x in e1),
                   "delta_from_values_mean": mean(x["delta_from_values"] for x in e1), "n_causes_E1_max": max(x["n_causes_E1"] for x in e1)},
            "E8b_other_fear": {"before_mean": mean(x["before"] for x in lk), "change_after_protocol_mean": mean(x["after_protocol"] - x["before"] for x in lk),
                               "change_after_time_only_control_mean": mean(x["after_control"] - x["before"] for x in lk),
                               "max_abs_change_after_protocol": max(abs(x["after_protocol"] - x["before"]) for x in lk),
                               "max_abs_difference_protocol_minus_control": max(abs(x["after_protocol"] - x["after_control"]) for x in lk)},
        }
    dump("diag_e_classification", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
