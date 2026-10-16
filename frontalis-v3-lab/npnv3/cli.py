"""コマンドライン。

python -m npnv3 init  STATE_DIR [--seed 0]
python -m npnv3 feed  STATE_DIR experiences.jsonl      学習し、各経験の要約を表示し、保存
python -m npnv3 probe STATE_DIR probes.json            学習せずに評価
python -m npnv3 show  STATE_DIR                         価値の上位・信念の要約
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from .innate import NEED_LABELS, NEEDS
from .npn import NPN


def _fmt_needs(d: Dict[str, float]) -> str:
    return " ".join(f"{n}{d[n]:+.2f}" for n in NEEDS)


def _top_emotions(emo: Dict[str, float], k: int = 3) -> str:
    items = sorted(emo.items(), key=lambda kv: -kv[1])[:k]
    return ", ".join(f"{n} {v:.2f}" for n, v in items if v > 0.01) or "なし"


def _summarize(ap: Dict[str, Any]) -> str:
    head = (f"[{ap['id']}] 予期 Q={ap['Q']:+.3f} (U_val={ap['Q_val']:+.3f} + 条件づけ C_Q={ap['C_Q']:+.3f}; "
            f"V_MB={ap['V_MB']:+.3f}, V_MF={ap['V_MF']:+.3f}, ω={ap['omega']:.2f})")
    if ap.get("optional"):
        head += f"  接近確率 {ap['p_approach']:.2f} → " + ("接近" if ap["approached"] else "回避（学習しない）")
    if ap["f"] is None:
        return head
    fp = ap["f_parts"]
    lines = [head,
             f"    感じた価値 f={ap['f']:+.3f}（生得 {fp['innate']:+.2f} + 条件づけ {fp['conditioned']:+.2f} + 参照点 {fp['reference']:+.2f}）"
             f"  驚き δ={ap['delta']:+.3f}  気分 {ap['mood']:+.3f}→{ap['mood_after']:+.3f}",
             f"    効用変化 Δu: {_fmt_needs(ap['du'])}  (合計 {ap['du_total']:+.3f})",
             f"    感情: {_top_emotions(ap['emotions'])}"]
    for it in ap.get("interpretations") or []:
        lines.append("    曖昧な結果の解釈（" + it["actor"] + "）: " + ", ".join(f"{t} {w:.2f}" for t, w in it["weights"].items())
                     + "（LLM の素の読み " + ", ".join(f"{t} {w:.2f}" for t, w in it["llm"].items()) + "）")
    for w in ap.get("warnings") or []:
        lines.append("    警告: " + w)
    vc = ap["value_change"]
    for pid, v in vc["entities"].items():
        lines.append(f"    {pid} への評価 {v['before']:+.3f} → {v['after']:+.3f} ({v['delta']:+.3f})")
    s = vc["situation"]
    lines.append(f"    状況の評価 {s['before']:+.3f} → {s['after']:+.3f} ({s['delta']:+.3f})")
    return "\n".join(lines)


def cmd_init(args: argparse.Namespace) -> int:
    n = NPN(seed=args.seed)
    h = n.save(args.state_dir)
    print(f"初期化しました: {args.state_dir}（種 {args.seed}、内容ハッシュ {h[:12]}…）")
    return 0


def cmd_feed(args: argparse.Namespace) -> int:
    n = NPN.load(args.state_dir)
    count = 0
    with open(args.experiences, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                exp = json.loads(line)
                ap = n.experience(exp)
            except ValueError as e:
                print(f"{args.experiences}:{ln}: エラー: {e}", file=sys.stderr)
                return 2
            print(_summarize(ap))
            count += 1
    h = n.save(args.state_dir)
    print(f"{count} 件を処理し、保存しました（経験 {n.n_experiences} 件、回避 {n.n_avoided} 件、ハッシュ {h[:12]}…）")
    return 0


def _probe_line(p: Dict[str, Any]) -> str:
    k = p["kind"]
    if k == "situation":
        return (f"状況: Q={p['Q']:+.3f} (V_MB={p['V_MB']:+.3f}, V_MF={p['V_MF']:+.3f}, ω={p['omega']:.2f}); "
                f"参加者 " + ", ".join(f"{x['id']} {x['value']:+.3f}" for x in p["persons"]))
    if k == "entity":
        return (f"エンティティ {p['entity']}（{p['class']}）: like={p['like']:+.3f} "
                f"({_fmt_needs(p['like_by_need'])}) 既知={p['known']}; Q={p['Q']:+.3f}; "
                f"P(手柄を取る)={p['p_person']['credit_taken']:.2f}; 条件づけ C={p['conditioned']:+.3f}")
    if k == "cue":
        ctx = f"（文脈 {p['context']}）" if p.get("context") else "（文脈を問わない）"
        return f"手がかり {p['cue']}{ctx}: 条件づけられた価値 C={p['C']:+.3f} 原因の数={p['n_causes']} 既知={p['known']}"
    return (f"概念 {p['concept']}: like={p['like']:+.3f} ({_fmt_needs(p['like_by_need'])}) 既知={p['known']}"
            f" 条件づけ C={p['conditioned']:+.3f}")


def cmd_probe(args: argparse.Namespace) -> int:
    n = NPN.load(args.state_dir)
    with open(args.probes, "r", encoding="utf-8") as f:
        doc = json.load(f)
    probes: List[Dict[str, Any]] = doc if isinstance(doc, list) else [doc]
    for p in probes:
        try:
            out = n.probe(p)
        except ValueError as e:
            print(f"エラー: {e}", file=sys.stderr)
            return 2
        print(json.dumps(out, ensure_ascii=False, indent=1) if args.json else _probe_line(out))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    n = NPN.load(args.state_dir)
    s = n.summary()
    print(f"経験 {s['n_experiences']} 件（回避 {s['n_avoided']} 件）  気分 {s['mood']:+.3f}")
    print("欲求の状態 h: " + " ".join(f"{NEED_LABELS[k]}={v:.2f}" for k, v in s["h"].items()))
    print("関心の重み a: " + " ".join(f"{NEED_LABELS[k]}={v:.2f}" for k, v in s["a"].items()))
    print("\n人モデルの価値（上位）")
    for r in s["person_top"]:
        print(f"  {r['feature']:<28s} {r['value']:+.3f}  ({_fmt_needs(r['by_need'])})")
    print("\n状況モデルの価値（上位）")
    for r in s["situation_top"]:
        print(f"  {r['feature']:<28s} {r['value']:+.3f}  ({_fmt_needs(r['by_need'])})")
    print("\n条件づけられた価値 C（手がかり。|C| の大きい順。C は潜在原因の価値の、計数による加重平均。読み出しの応答は文脈と時刻に依存する）")
    for r in s["conditioned_top"]:
        print(f"  {r['cue']:<28s} {r['C']:+.3f}  （原因 {r['n_causes']} 個）")
    b = s["belief"]
    print(f"\n信念: 個体 {b['n_individuals']}、文脈 {b['n_contexts']}、観測 {b['n_obs']}、事前ヘッド {b['head_kind']}（ヘッドの寄与 ρ_h={b['rho_h']:.2f}）")
    if b["head_kind"] == "mixture":
        print("  ヘッドの重み（類を除いたモデルの転移性能の softmax）: "
              + ", ".join(f"{h} {w:.2f}" for h, w in b["head_weights"].items() if h in b["mixture_heads"]))
    for c, rates in sorted(b["class_rate"].items()):
        top = sorted(rates.items(), key=lambda kv: -kv[1])[:4]
        print(f"  類 {c}（κ_e={b['kappa_e'][c]:.2f}）: " + ", ".join(f"{t} {v:.2f}" for t, v in top))
    for w in s["warnings"]:
        print("  警告: " + w)
    pi = sorted(b["pi_person"].items(), key=lambda kv: -kv[1])[:4]
    print("  「人のせい」の事前 π_person（高い順）: " + ", ".join(f"{t} {v:.2f}" for t, v in pi))
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m npnv3", description="NPN v3 最小実装")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="新しい NPN を作って保存"); p.add_argument("state_dir")
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_init)
    p = sub.add_parser("feed", help="経験（JSONL）を学習して保存"); p.add_argument("state_dir")
    p.add_argument("experiences"); p.set_defaults(fn=cmd_feed)
    p = sub.add_parser("probe", help="学習せずに評価"); p.add_argument("state_dir"); p.add_argument("probes")
    p.add_argument("--json", action="store_true", help="詳細を JSON で出力"); p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("show", help="価値の上位と信念の要約"); p.add_argument("state_dir"); p.set_defaults(fn=cmd_show)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
