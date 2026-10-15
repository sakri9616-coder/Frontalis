"""デモ（v3.2）: 3 人格が異なる経験史を経て、同じ新しい経験をどう評価し、何を学ぶか。保存→復元→継続も見せる。

実行: python3 examples/demo.py [--seed 0]

3 人格は同じ初期状態・同じ係数から始まる。違うのは経験史だけ（data/histories/{R,K,X}.jsonl）。
  R 協働に救われた / K 手柄を取られた / X 探索者
v3.1 で見せること:
  - 欲求の状態 h と気分 m を揃えても、受け止め方 f が人格で違う（関心の重み a、条件づけられた価値 C、予期との差）
  - 生得の効果を持たない praise の価値が、随伴関係（後に何が来たか）で形成される
  - 曖昧な出来事の解釈が、人格の信念で違い、その解釈のまま学ぶ
v3.2 で見せること（評価基準が、学習と行動を変える）:
  - 褒められた後に害を受けた人格は、褒め言葉を伴う申し出を避ける（条件づけられた価値 C が予期 Q に入る）
  - 関心の重み a が違うと、同じ経験から学ぶ量が変わる（欲求ごとの観測雑音 R/a_k）。a の方向（sensitize / habituate）は実験条件
  - 事前ヘッドの重みは予測性能で決まる（前向きの対数尤度）
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
sys.path.insert(0, str(ROOT / "data"))

import numpy as np  # noqa: E402
import make_histories as mh  # noqa: E402
from npnv3 import NPN  # noqa: E402
from npnv3.innate import NEED_LABELS, NEEDS, TYPE_INDEX, M  # noqa: E402

HIST = ROOT / "data" / "histories"
LABEL = {"R": "R 協働に救われた", "K": "K 手柄を取られた", "X": "X 探索者"}
EMO = {"joy": "喜び", "disappointment": "失望", "relief": "安堵", "anger": "怒り", "sadness": "悲しみ", "interest": "興味"}
PROBE_AGAIN = {"context": "music_session", "concepts": ["music", "collaboration"],
               "participants": [{"id": "friendA", "class": "collaborator"}]}
UNSEEN = {"entity": "friendB", "class": "collaborator", "context": "music_session"}
AMBIGUOUS = {"id": "ambiguous-idea-unused", "context": "project_meeting", "concepts": ["collaboration"],
             "participants": [{"id": "partnerQ", "class": "collaborator"}],
             "outcomes": [{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "partnerQ",
                           "magnitude": 0.8}], "optional": False}


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
    print(f"{'人格':<14}{'経験数':>6}  {'欲求の状態 h (R C A S Sa)':<30}{'気分':>7}  {'関心の重み a (R C A S Sa)':<30}未見の共同作業者 like")
    for k, n in npns.items():
        like = n.probe({"entity": "friendB", "class": "collaborator"})
        print(f"{LABEL[k]:<14}{n.n_experiences:>6}  {' '.join(f'{x:5.2f}' for x in n.h):<30}{n.mood:>+7.3f}  "
              f"{' '.join(f'{x:5.2f}' for x in n.a):<30}{like['like']:+.3f}")
    print("  a: 人生で大きく揺さぶられた欲求ほど大きい（和は 5）。like = 価値ネット MF の人モデルの予測を a で重み付けた合計。")

    # --- 2. 同じ新しい経験の評価 ---
    hr("2. 同じ新しい経験の評価（結果を見る前の予期 → 結果を見たあとの評価）")
    aps = {k: n.appraise(new) for k, n in npns.items()}
    print(f"{'人格':<14}{'Q':>8}{'V_MB':>8}{'V_MF':>8}{'ω':>6}{'f':>8}{'δ':>8}{'気分':>8}")
    for k, a in aps.items():
        print(f"{LABEL[k]:<14}{a['Q']:>+8.3f}{a['V_MB']:>+8.3f}{a['V_MF']:>+8.3f}{a['omega']:>6.2f}{a['f']:>+8.3f}{a['delta']:>+8.3f}{a['mood']:>+8.3f}")
    print("  Q: 予期  V_MB: 信念から計算した価値  V_MF: 学習した価値  ω: MF への重み  f: 感じた価値  δ: 驚き = f - tanh(Q/s_f)")
    print("\n  効用変化 Δu（欲求ごと。欲求の状態 h が違うので、同じ出来事でも人格で違う）")
    for k, a in aps.items():
        print(f"    {LABEL[k]:<14}{needs(a['du'])}   a で重み付けた合計 {a['du_weighted']:+.3f}")
    print("\n  結果ごとの寄与（合計 = Δu）と帰属 φ（人のせい / 場面のせい）。人格の信念で φ が違う")
    for k, a in aps.items():
        print(f"    {LABEL[k]}")
        for c in a["contributions"]:
            phi = f"φ人={c['phi_person']:.2f} φ場面={c['phi_ctx']:.2f}" if c["phi_person"] is not None else "（自分・世界）"
            print(f"      {c['type']:<14}{c['actor']:<8}大きさ{c['magnitude']:.1f}  寄与 {sum(c['contrib'].values()):+.3f}  {phi}  予期していた確率 {c['p_predicted']:.2f}")
    print("\n  感情の読み出し（言語化用の手がかり。学習には使わない）")
    for k, a in aps.items():
        e = a["emotions"]
        print(f"    {LABEL[k]:<14}" + "  ".join(f"{EMO[n]} {e[n]:.2f}" for n in EMO))

    # --- 2b. h と m を揃えても、受け止め方は人格で違う ---
    hr("2b. 欲求の状態 h と気分 m を基準値に揃えても、受け止め方 f は人格で違う（v3.1）")
    aligned = {}
    for k, n in npns.items():
        m = copy.deepcopy(n)
        m.h[:] = 1.0
        m.mood = 0.0
        aligned[k] = m
    empty = NPN(seed=args.seed)
    print(f"{'人格':<14}{'f':>8}   = tanh( ( 生得(a で重み付け)  + 条件づけ C  + 参照点 η(生得−Q) ) / 0.5 )   Q")
    rows = {k: m.appraise(new) for k, m in aligned.items()}
    rows["空"] = empty.appraise(new)
    for k, a in rows.items():
        p = a["f_parts"]
        name = LABEL.get(k, "学習なし（空の NPN）")
        print(f"{name:<14}{a['f']:>+8.3f}   =        {p['innate']:>+8.3f}         {p['conditioned']:>+8.3f}       {p['reference']:>+8.3f}"
              f"                   {a['Q']:>+8.3f}")
    gap = max(rows[k]["f"] for k in "RKX") - min(rows[k]["f"] for k in "RKX")
    print(f"  h と m が同じなのに、3 人格の f の最大差は {gap:.3f}（v3.0 では 0.000）。")
    # 学習した状態だけを入れ替える
    r, kk = copy.deepcopy(aligned["R"]), aligned["K"]
    r.emb, r.belief, r.value, r.concern, r.cond = copy.deepcopy((kk.emb, kk.belief, kk.value, kk.concern, kk.cond))
    f_swap = r.appraise(new)["f"]
    print(f"  入れ替え: R の殻（h, m）に K の学習した状態を入れると f = {f_swap:+.3f}"
          f"（R の f {rows['R']['f']:+.3f}、K の f {rows['K']['f']:+.3f}）→ f は学習した状態と一緒に移る。")

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
    print("\n  価値の予測誤差（Δu - 継承した予期。欲求ごと）と、動いた特徴の上位")
    print("  （カルマン更新は共分散を通じて、この経験に現れない特徴も動かす。過去に一緒に現れた特徴は、説明を分け合う）")
    for k, r_ in results.items():
        vc = r_["value_change"]
        print(f"    {LABEL[k]:<14}誤差 {needs(vc['prediction_error']['person']['friendA'])}")
        print("      動いた特徴: " + ", ".join(f"{t['feature']} {t['dW_sum']:+.3f}" for t in vc["top_features"][:4]))
        print("      関心の重み a: " + " ".join(f"{x:.2f}" for x in vc["a_after"]))

    # --- 4. praise: 随伴で価値が形成される ---
    hr("4. 生得の効果を持たない praise の価値は、随伴関係で形成される（条件づけ C、TD(0)）")
    print("効果表 M の praise の行:", [float(x) for x in M[TYPE_INDEX['praise']]], "（全欲求で 0 = 生得の効果なし）")
    pr = {}
    for var, text in (("H1", "褒められた後に害が起きる人生"), ("H2", "褒められた後に支えが来る人生")):
        n = NPN(seed=args.seed)
        for e in mh.generate_praise_history(var, args.seed):
            n.experience(e)
        pr[var] = n
        c = n.probe({"cue": "type:praise"})["C"]
        praise_only = {"id": "praise-only", "context": "cafe_chat", "concepts": ["conversation"],
                       "participants": [{"id": "newfriend", "class": "friend"}],
                       "outcomes": [{"type": "praise", "actor": "newfriend", "magnitude": 0.8}], "optional": False}
        a = n.appraise(praise_only)
        print(f"  {var}（{text}）: C(praise) = {c:+.3f}   新しく褒められたときの感じた価値 f = {a['f']:+.3f}"
              f"（生得 {a['f_parts']['innate']:+.2f} + 条件づけ {a['f_parts']['conditioned']:+.2f} + 参照点 {a['f_parts']['reference']:+.2f}）")
    print("  同じ「褒められる」でも、その後に何が来たかで、好悪が正にも負にもなる。どの手がかりがどんな価値を帯びるかは、経験の随伴構造が決める。")
    print("  C(praise) を動かした経験（H1、台帳。最初の 4 回）:")
    hist = [(e["id"], e["td"]["dC"]["type:praise"]) for e in pr["H1"].ledger.entries
            if e["kind"] == "experience" and "type:praise" in e["td"]["dC"]][:4]
    for eid, d in hist:
        print(f"    {eid:<18} ΔC = {d:+.4f}")


    # --- 4b. C が選択を変える ---
    hr("4b. 褒められた後に害を受けた人格は、褒め言葉を伴う申し出を避ける（C → Q → 接近・回避、v3.2）")
    offers = mh.generate_praise_offers(args.seed, 40)
    print("申し出（optional、announced = ['praise']）: 参加者なし、履歴に現れない文脈 studio、結果は褒め言葉だけ（生得の効果なし）。")
    print("Q = U_val + λ_CQ · ΣC(手がかり)。手がかりに type:praise が入る。H1 と H2 に同じ 40 件を与える（接近は p = σ(4(Q + 0.05)) で確率的）。")
    print(f"  {'人格':<26}{'C(praise)':>10}{'最初の申し出の Q':>16}{'  = U_val':>10}{' + C_Q':>9}{'p(接近)':>9}{'  接近率(40件)':>14}")
    for var, text in (("H1", "H1 褒められた後に害"), ("H2", "H2 褒められた後に支え")):
        for label, kw in (("", {}), (" [C を切る]", {"no_conditioning": True})):
            n = NPN(seed=args.seed, **kw)
            for e in mh.generate_praise_history(var, args.seed):
                n.experience(e)
            a0 = n.appraise(offers[0])
            c = n.probe({"cue": "type:praise"})["C"]
            rate = sum(bool(n.experience(e)["approached"]) for e in offers) / len(offers)
            print(f"  {text + label:<26}{c:>+10.3f}{a0['Q']:>+16.3f}{a0['Q_val']:>+10.3f}{a0['C_Q']:>+9.3f}{a0['p_approach']:>9.2f}{rate:>14.2f}")
    print("  H1 は褒め言葉を伴う申し出を避け、H2 は受ける。C を切ると差はほとんど消える（欲求の状態 h・関心の重み a を通る分だけ残る）。")
    print("  申し出を受けて何も起きないと C は 0 に戻る（消去）。避けると消去が止まる = 標本の偏りによる有界な正帰還（W6 で有界を検査）。")

    # --- 4c. a が学習の量を変える ---
    hr("4c. 関心の重み a が違うと、同じ経験から学ぶ量と確信が変わる（欲求ごとの観測雑音 R_k = R / a_k、v3.2）")
    base = npns["K"]
    probe_exp = json.loads(json.dumps(new).replace("friendA", "friendC"))       # 新しい個体（友人 C）とのもの。経験の中身は new と同じ
    probe_exp["id"] = "same-experience-for-learning"
    print("K の学習した状態のまま、a だけを入れ替えて（実験: a を固定）、同じ経験（初めて会う友人 C とのもの。中身は新しい経験と同じ）を与える。")
    print("友人 C の価値 W[ind:friendC]（欲求ごと）の変化と、その事後の標準偏差（確信。小さいほど確信が高い）:")
    print(f"  {'a の与え方':<22}{'a (R C A S Sa)':<28}{'ΔW[ind:friendC] (R C A S Sa)':<42}{'事後の標準偏差 (R C A S Sa)':<34}")
    cases = (("K 自身の a", None), ("R の a に替える", npns["R"].concern.a),
             ("R を重視 (3,1,.5,.25,.25)", [3.0, 1.0, 0.5, 0.25, 0.25]), ("R を軽視 (.25,.25,.5,1,3)", [0.25, 0.25, 0.5, 1.0, 3.0]))
    for name, a_ in cases:
        c = copy.deepcopy(base)
        if a_ is not None:
            c.cfg.no_concern = True                       # a を固定する（実験条件）
            c.concern.a = np.array(a_, dtype=float)
        c.experience(probe_exp)
        dW = c.ledger.entries[-1]["dW"]["person"]["ind:friendC"]
        kal = c.value.person
        sd = kal.S[:, kal.idx["ind:friendC"], kal.idx["ind:friendC"]] ** 0.5
        print(f"  {name:<22}{' '.join(f'{x:5.2f}' for x in c.a):<28}{' '.join(f'{x:+7.3f}' for x in dW):<42}{' '.join(f'{x:6.3f}' for x in sd):<34}")
    print("  a の大きい欲求は観測雑音が小さく（R/a_k）、同じ予測誤差からより大きく学び、確信も高くなる（標準偏差が小さい）。")
    print("  ただし、新しい個体の事前分散（0.25）が観測雑音 R（0.01）よりずっと大きいので、W の変化は極端な a でも 1 割ほどしか変わらない（確信は分散で 2 倍ほど変わる）。")
    print("  a の目標は |Δu|（外生）で W に依存しない（a → W の一方向）。")
    print("\n  関心の重みの方向は実験条件（concern_direction）: 同じ K の履歴を、sensitize / habituate で育てる")
    for direction in ("sensitize", "habituate"):
        n = NPN(seed=args.seed, concern_direction=direction)
        for e in load_jsonl("K.jsonl"):
            n.experience(e)
        c = copy.deepcopy(n)
        c.experience(probe_exp)
        dW = c.ledger.entries[-1]["dW"]["person"]["ind:friendC"]
        print(f"    {direction:<10} a = {' '.join(f'{x:5.2f}' for x in n.a)}   ΔW[ind:friendC] = {' '.join(f'{x:+7.3f}' for x in dW)}")
    print("  主張は『a が経験から形成され、受け止め方と学習を変える』まで。どちらの方向が人に近いかは主張しない。")

    # --- 4d. 事前ヘッドの重み ---
    hr("4d. 事前ヘッドの重みは予測性能で決まる（前向きの対数尤度の softmax、v3.2）")
    print(f"  {'人格':<14}" + "".join(f"{h:>9}" for h in ("mlp", "linear", "knn1", "none")) + "   （none = 一般の率。観測前の予測の対数尤度で重み付け）")
    for k, n in npns.items():
        w = n.belief.head_weights_dict()
        print(f"  {LABEL[k]:<14}" + "".join(f"{w[h]:>9.2f}" for h in ("mlp", "linear", "knn1", "none")))
    print("  短い履歴では、最も近い既知の類の率を使う knn1 が最も予測が良く、重みが大きい。MLP が必ず勝つわけではない（W7 の数値を参照）。")

    # --- 5. 曖昧な出来事の解釈 ---
    hr("5. 曖昧な出来事の解釈は人格の信念で違い、その解釈のまま学ぶ（v3.1）")
    print("出来事: 『自分の案が使われなかった』（LLM の素の読み: 手柄を取られた 0.5 / 共有された 0.5。人格を知らない）")
    for k in "KR":
        a = npns[k].appraise(AMBIGUOUS)
        w = a["interpretations"][0]["weights"]
        t = a["teachers"]
        print(f"  {LABEL[k]:<14}解釈の重み: 取られた {w['credit_taken']:.2f} / 共有された {w['credit_shared']:.2f}"
              f"   Δu 合計 {a['du_weighted']:+.3f}   人への教師 {sum(t['person']['partnerQ'].values()):+.3f}")
    k = copy.deepcopy(npns["K"])
    k.experience(AMBIGUOUS)
    p0 = k.probe({"entity": "partnerQ", "class": "collaborator", "context": "project_meeting"})["p_person"]["credit_taken"]
    for i in range(8):
        k.experience({"id": f"clear-share-{i}", "context": "project_meeting", "concepts": ["collaboration"],
                      "participants": [{"id": "partnerQ", "class": "collaborator"}],
                      "outcomes": [{"type": "credit_shared", "actor": "partnerQ", "magnitude": 0.7},
                                   {"type": "connection", "actor": "partnerQ", "magnitude": 0.5}], "optional": False})
    p1 = k.probe({"entity": "partnerQ", "class": "collaborator", "context": "project_meeting"})["p_person"]["credit_taken"]
    a2 = k.appraise(AMBIGUOUS)["interpretations"][0]["weights"]["credit_taken"]
    print(f"  K に明確な反証（手柄を共有された）を 8 回: P(手柄を取る | partnerQ) {p0:.2f} → {p1:.2f}"
          f"、同じ曖昧な出来事の『取られた』の重み → {a2:.2f}")
    print("  曖昧な結果は信念を更新しない（解釈が信念を自己訓練するループを作らない）。明確な反証だけが信念を変える。")

    # --- 6. 未見の相手への影響 ---
    hr("6. 経験の後: 未見の共同作業者 friendB への評価（持続と個別化）")
    for k, n in npns.items():
        p = n.probe(UNSEEN)
        print(f"{LABEL[k]:<14}like={p['like']:+.3f}  Q={p['Q']:+.3f}  P(手柄を取る)={p['p_person']['credit_taken']:.2f}")
    print("  A 個人への評価は大きく動いても、未見の共同作業者への評価は、形成期の経験史の違いを残している。")

    # --- 7. 保存 → 復元 → 継続 ---
    hr("7. 保存 → 復元 → 継続（K）")
    follow = copy.deepcopy(new)
    follow["id"] = "new-song-with-friendA-2"
    follow["outcomes"] = [{"type": "connection", "actor": "friendA", "magnitude": 1.0},
                          {"type": "credit_given", "actor": "friendA", "magnitude": 0.7},
                          {"type": "achievement", "actor": "self", "magnitude": 0.8}]
    cont = copy.deepcopy(npns["K"])               # 連続して学ぶ側
    a_cont = cont.experience(follow)
    with tempfile.TemporaryDirectory() as d:
        digest = npns["K"].save(d)
        print(f"保存: {d}/state.json + arrays.npz（内容ハッシュ {digest[:16]}…。埋め込み・a・C・効果表を含む）")
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
