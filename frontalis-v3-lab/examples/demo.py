"""デモ: 3 人格が異なる経験史を経て、同じ新しい経験をどう評価し、何を学ぶか。保存→復元→継続も見せる。

実行: python3 examples/demo.py [--seed 0]

3 人格は同じ初期状態・同じ係数から始まる。違うのは経験史だけ（data/histories/{R,K,X}.jsonl）。
  R 協働に救われた / K 手柄を取られた / X 探索者
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npnv3 import NPN  # noqa: E402
from npnv3.innate import NEED_LABELS, NEEDS  # noqa: E402

HIST = ROOT / "data" / "histories"
LABEL = {"R": "R 協働に救われた", "K": "K 手柄を取られた", "X": "X 探索者"}
EMO = {"joy": "喜び", "disappointment": "失望", "relief": "安堵", "anger": "怒り", "sadness": "悲しみ", "interest": "興味"}
PROBE_AGAIN = {"context": "music_session", "concepts": ["music", "collaboration"],
               "participants": [{"id": "friendA", "class": "collaborator"}]}
UNSEEN = {"entity": "friendB", "class": "collaborator", "context": "music_session"}


def load_jsonl(name: str):
    with open(HIST / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def needs(d, fmt="{:+.3f}"):
    return "  ".join(f"{n}{fmt.format(d[n])}" for n in NEEDS)


def hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    new = json.load(open(HIST / "new_event.json", encoding="utf-8"))
    hr("0. 新しい経験（3 人格に同じものを与える）")
    print("初めて会う友人 A に誘われて曲を作った。曲は好評で、A から新しい技法も学んだが、自分の案はほとんど採用されなかった。")
    print(json.dumps(new, ensure_ascii=False))

    # --- 1. 形成 ---
    npns = {}
    for k in "RKX":
        n = NPN(seed=args.seed)                 # 全人格、同じ初期状態・同じ係数
        for e in load_jsonl(f"{k}.jsonl"):
            n.experience(e)
        npns[k] = n
    hr("1. 形成: 各人格の経験史の後の状態（同じ初期状態から、経験史だけが違う）")
    print(f"{'人格':<14}{'経験数':>6}  {'欲求の状態 h (R C A S Sa)':<34}{'気分':>7}  未見の共同作業者への評価 like")
    for k, n in npns.items():
        like = n.probe({"entity": "friendB", "class": "collaborator"})
        print(f"{LABEL[k]:<14}{n.n_experiences:>6}  {' '.join(f'{x:5.2f}' for x in n.h):<34}{n.mood:>+7.3f}  {like['like']:+.3f}")
    print("  （like = 価値ネット MF の人モデルの予測。類「共同作業者」から継承した価値。経験史で符号まで変わる）")

    # --- 2. 同じ新しい経験の評価 ---
    hr("2. 同じ新しい経験の評価（結果を見る前の予期 → 結果を見たあとの評価）")
    aps = {k: n.appraise(new) for k, n in npns.items()}
    print(f"{'人格':<14}{'Q':>8}{'V_MB':>8}{'V_MF':>8}{'ω':>6}{'f':>8}{'δ':>8}{'気分':>8}")
    for k, a in aps.items():
        print(f"{LABEL[k]:<14}{a['Q']:>+8.3f}{a['V_MB']:>+8.3f}{a['V_MF']:>+8.3f}{a['omega']:>6.2f}{a['f']:>+8.3f}{a['delta']:>+8.3f}{a['mood']:>+8.3f}")
    print("  Q: 予期  V_MB: 信念から計算した価値  V_MF: 学習した価値  ω: MF への重み  f: 感じた価値  δ: 驚き = f - tanh(Q/s_f)")
    print("\n  効用変化 Δu（欲求ごと。欲求の状態 h が違うので、同じ出来事でも人格で違う）")
    for k, a in aps.items():
        print(f"    {LABEL[k]:<14}{needs(a['du'])}   合計 {a['du_total']:+.3f}")
    print("\n  結果ごとの寄与（合計 = Δu）と帰属 φ（人のせい / 場面のせい）。人格の信念で φ が違う")
    for k, a in aps.items():
        print(f"    {LABEL[k]}")
        for c in a["contributions"]:
            who = c["actor"]
            phi = f"φ人={c['phi_person']:.2f} φ場面={c['phi_ctx']:.2f}" if c["phi_person"] is not None else "（自分・世界）"
            tot = sum(c["contrib"].values())
            print(f"      {c['type']:<14}{who:<8}大きさ{c['magnitude']:.1f}  寄与 {tot:+.3f}  {phi}  予期していた確率 {c['p_predicted']:.2f}")
    print("\n  感情の読み出し（言語化用の手がかり。学習には使わない）")
    for k, a in aps.items():
        e = a["emotions"]
        print(f"    {LABEL[k]:<14}" + "  ".join(f"{EMO[n]} {e[n]:.2f}" for n in EMO))

    # --- 3. 学ぶもの ---
    hr("3. 学ぶもの（価値の更新）。教師は Δu を帰属で分けたもの。自分の出力は教師に入れない")
    before_probe = {k: n.probe(PROBE_AGAIN)["Q"] for k, n in npns.items()}
    before_cls = {k: n.value.person.W[n.value.person.idx["cls:collaborator"]].sum() for k, n in npns.items()}
    results = {k: n.experience(new) for k, n in npns.items()}
    print(f"{'人格':<14}{'A への評価':>22}{'更新の向き':>10}{'類（共同作業者）の価値':>26}{'同じ誘いの Q':>20}")
    for k, n in npns.items():
        v = results[k]["value_change"]["entities"]["friendA"]
        cls_after = n.value.person.W[n.value.person.idx["cls:collaborator"]].sum()
        after_probe = n.probe(PROBE_AGAIN)["Q"]
        print(f"{LABEL[k]:<14}{v['before']:>+9.3f} → {v['after']:>+7.3f}{('上げた' if v['delta'] > 0 else '下げた'):>10}"
              f"{before_cls[k]:>+14.3f} → {cls_after:>+7.3f}{before_probe[k]:>+11.3f} → {after_probe:>+7.3f}")
    print("  学習の向きが人格の過去で逆になる。A への評価は、継承した予期と比べた予測誤差の符号で決まる。")
    print("  類の価値は、個体の数に比例してしか動かない（新しい個体 A が誤差の大半を引き受ける）。")
    print("\n  価値の予測誤差（Δu - 継承した予期。欲求ごと）と、動いた特徴の上位")
    print("  （カルマン更新は共分散を通じて、この経験に現れない特徴も動かす。過去に一緒に現れた特徴は、説明を分け合う）")
    for k, r in results.items():
        vc = r["value_change"]
        print(f"    {LABEL[k]:<14}誤差 {needs(vc['prediction_error']['person']['friendA'])}")
        print("      動いた特徴: " + ", ".join(f"{t['feature']} {t['dW_sum']:+.3f}" for t in vc["top_features"][:4]))

    # --- 4. 未見の相手への影響 ---
    hr("4. 経験の後: 未見の共同作業者 friendB への評価（持続と個別化）")
    for k, n in npns.items():
        p = n.probe(UNSEEN)
        print(f"{LABEL[k]:<14}like={p['like']:+.3f}  Q={p['Q']:+.3f}  P(手柄を取る)={p['p_person']['credit_taken']:.2f}")
    print("  A 個人への評価は大きく動いても、未見の共同作業者への評価は、形成期の経験史の違いを残している。")

    # --- 5. 保存 → 復元 → 継続 ---
    hr("5. 保存 → 復元 → 継続（K）")
    follow = copy.deepcopy(new)
    follow["id"] = "new-song-with-friendA-2"
    follow["outcomes"] = [{"type": "connection", "actor": "friendA", "magnitude": 1.0},
                          {"type": "credit_given", "actor": "friendA", "magnitude": 0.7},
                          {"type": "achievement", "actor": "self", "magnitude": 0.8}]
    cont = copy.deepcopy(npns["K"])               # 連続して学ぶ側
    a_cont = cont.experience(follow)
    with tempfile.TemporaryDirectory() as d:
        digest = npns["K"].save(d)
        print(f"保存: {d}/state.json + arrays.npz（内容ハッシュ {digest[:16]}…）")
        restored = NPN.load(d)
        a_rest = restored.experience(follow)
    same = json.dumps(a_cont, sort_keys=True) == json.dumps(a_rest, sort_keys=True)
    print(f"復元してから次の経験を学ぶ。連続して学んだ場合と、評価・更新が完全に一致: {same}")
    print(f"  K の 2 回目の A との経験: 予期 Q={a_rest['Q']:+.3f}（1 回目の後なので、A への評価が上がっている）  "
          f"f={a_rest['f']:+.3f}  δ={a_rest['delta']:+.3f}")
    print(f"  A への評価 {a_rest['value_change']['entities']['friendA']['before']:+.3f} → "
          f"{a_rest['value_change']['entities']['friendA']['after']:+.3f}")
    print("\n  台帳（K の類「共同作業者」の価値を動かした経験）:")
    hist = restored.ledger.history_of("person", "cls:collaborator")
    for h in hist[:4]:
        print(f"    {h['id']:<28} ΔW(5 欲求の和) {sum(h['dW']):+.4f}")
    print(f"    … 計 {len(hist)} 件。W は台帳の総和に等しい（不変量）。")


if __name__ == "__main__":
    main()
