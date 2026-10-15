# frontalis-v3-lab: NPN 最小実装 v3.1

経験によって人格が形成される外部ニューラルネットワーク（NPN）の、**実際に動く最小実装**です。
経験の入力 → 評価 → 学習 → 次の評価への反映、までを実行し、テストで検証します。

- 仕様: [`SPEC.md`](SPEC.md)（v3.0。判定基準 T1〜T10）と [`SPEC-v3.1.md`](SPEC-v3.1.md)（v3.1 の差分。判定基準 U1〜U9）。どちらも実装前に固定
- 設計の根拠: `docs/04-learning-mechanisms.md`（精度配分・階層型フレーム v3）
- 結果: [`REPORT.md`](REPORT.md)　仕様からの変更の記録: [`CHANGELOG.md`](CHANGELOG.md)

## 目的

研究思想の中心は、次の循環です。

1. 既存の人格が経験を評価する。
2. その評価と経験した結果によって人格が更新される。
3. 更新後の人格が、次の経験を以前とは異なる仕方で受け止める。

この実装は、その循環を最小の部品で作り、次を満たすかを調べるためのものです（研究思想 第1・4・6節）。

- 人格の正本は NPN の状態です。入力は、LLM の意味抽出の出力に相当する**構造化された経験 JSON**で、数値や感情の判断は LLM に任せません。
- **価値ネットの更新目標に、価値ネット自身の出力を入れません。** 教師は「欲求の効用変化 Δu を、信念の責任 φ で分けたもの」だけです（docs/04 補題1）。
- 結果の予測（**信念**）と結果の評価（**価値**）は、別のパラメータ・別の誤差で学びます。
- 人格の差は、ラベルや初期値ではなく**経験史**から生じます。全人格は同じ初期状態・同じ係数から始まります（`NPN(seed)` に人格を指定する引数はありません）。

### v3.1 で変わったこと

v3.0 は、理念を優先する批評で「NPN の最小フレームとは呼べない」と判定されました（受け止め方 f が学習状態に依存しない／価値が効果表 M の写しになっている／ニューラルネットが使われていない）。
v3.1 は、自己増幅を起こさずに、**経験が評価基準そのもの（何をどれだけ良い・悪いと感じるか）を変え、その変わった基準が次の経験の受け止め方と学習を変える**ことを目指します。

| 節 | 変更 | 何が変わるか |
| --- | --- | --- |
| 1.1 | 欲求の関心の重み **a** | 人生で大きく揺さぶられた欲求ほど重くなる（感作）。感じた価値・MB 価値・好悪の読み出しを重み付ける |
| 1.2 | 条件づけられた価値 **C**（TD(0)） | 生得の効果を持たない手がかり（`praise` など）が、経験の随伴関係から価値を得る |
| 1.3 | 受け止め方 **f** | `f = tanh(((Σ a_k Δu_k) + λ_C ΣC + η_ref(Σ a_k Δu_k − Q)) / s_f + β m)`。学習した状態に依存する |
| 1.4 | 曖昧な結果の**解釈** | `w_I ∝ q_LLM(I) · P_belief(I \| actor, context)`。同じ出来事を、人格の信念で違うように受け止める |
| 1.5 | 事前ヘッドの**実際の利用** | 50 件の解禁を廃止し `prior_cls = ρ_h·head + (1−ρ_h)·m_gen`、ρ_h = n/(n+30)。基線（linear / knn1 / none） |
| 1.6 | **埋め込みは NPN が所有** | 初めて現れた語の埋め込みを状態に写して保存。LLM や埋め込みモデルを替えても既存の人格の座標は変わらない |
| 1.7 | **κ_e の経験ベイズ推定**（改訂: 類 × 結果型ごと） | 「上位の層の有効標本数は相手の数」を固定の κ でなく推定で。どの型が個体間でばらつくかは型ごとに違う（Kemp et al. 2007） |
| 1.8 | **確率的な経験史の生成器** | 乱数種 20 本が独立な標本になる（`data/make_histories.py`） |

> この実装が確かめるのは学習の機構です。「人格らしい発言」ができるかは対象外で、LLM との接続も含みません。
> **「人格が形成された」「循環が成立した」とは主張しません。** 何が確かめられ、何が確かめられていないかは REPORT.md を見てください。

## 構成

```
frontalis-v3-lab/
  npnv3/
    config.py     設定（係数・アブレーションの切り替え）。保存・復元の対象
    schema.py     経験・プローブの検証（未知の語は警告つきで許す。曖昧な結果、session）
    innate.py     欲求 5 つ・効果表 M・効用（固定、全人格共通）
    lexicon.py    埋め込みの供給源（data/lexicon.json）と、NPN が所有する埋め込みの保管庫
    mlp.py        numpy の小さな MLP と線形（ロジスティック回帰）の基線
    belief.py     信念ネット: スコープ帰属つき階層ベイズ ＋ 事前ヘッド ＋ κ_e の推定
    value.py      価値ネット: 階層カルマン MF、MB、調停、関心の重み a、条件づけ C、気分、感情
    ledger.py     更新台帳
    npn.py        統合: experience() / appraise() / probe() / save() / load()
    cli.py, __main__.py
  data/
    lexicon.json          手で意味を持たせた 8 次元ベクトル（供給源）
    make_histories.py     経験史の生成（固定の R/K/X と、確率的な生成器 generate_*）
    histories/            R, K, X（固定。デモ用）、T4_*, T7_*, T8_train、new_event.json、gen_*（生成器の見本）
  tests/                  unittest（T1〜T10、U1〜U9、機構の単体検査）、results/ に数値、diagnostics_m_scale.py（診断）
  examples/demo.py        3 人格が同じ新しい経験をどう評価し、何を学ぶか（v3.1 の機構つき）
```

`npnv3/config.py` は SPEC の構成にない追加ファイルです（係数を 1 か所に集め、保存・復元の対象にするため）。

## 1 回の経験で起きること

`experience(exp)` は次の順に進みます（SPEC 第6節＋v3.1）。

1. スキーマ検証。
2. **予期**（結果を見る前）: 信念の予測 → `V_MB`, `V_MF`（a で重み付け）, `ω`, `Q`。曖昧な結果は、ここで信念により解釈の重みが決まる。
3. `optional: true` なら、接近確率 `σ(4 (Q + 0.05))` で接近・回避を選ぶ。**回避したら、回避したことだけ台帳に記録して終わり**（学習しない。信念の計数は不変）。
4. 欲求の変化 Δh、効用変化 Δu（5 欲求のベクトル）、結果ごとの寄与、感じた価値 `f`（生得＋条件づけ＋参照点）、驚き `δ`、感情。
5. **信念の更新**: 責任 φ（人のせい／場面のせい）→ 計数、スコープの事前 π、型の大きさ、リプレイ緩衝 → 事前ヘッドの SGD（目標は観測そのもの）。**曖昧な結果の型は観測なし**（計数しない）。
6. **価値の更新**: Δu を φ で人モデルと状況モデルに分け、それぞれカルマン更新。**条件づけ C を TD(0) で更新**（直前の事象の手がかりを、この事象の `r + γ C_frozen` へ）。
7. **関心の重み a**、欲求の状態 `h`、気分 `m` の更新。
8. 台帳に記録（経験 id、各特徴の ΔW、ΔC、a、φ、Δu、Q、f、δ）。

`appraise(exp)` は 2〜4 だけを計算し、**状態（乱数を含む）を一切変えません**。`probe(p)` も学習しません（未登録の語の埋め込みを読んでも、登録はしません）。

## 入力形式

### 経験（LLM 抽出の出力に相当）

```json
{
  "id": "e0001",
  "context": "music_session",
  "concepts": ["music", "collaboration"],
  "participants": [{"id": "friendA", "class": "collaborator"}],
  "outcomes": [
    {"type": "connection", "actor": "friendA", "magnitude": 1.0},
    {"type": "achievement", "actor": "self", "magnitude": 0.8},
    {"type": "credit_taken", "actor": "friendA", "magnitude": 0.8},
    {"type": "taught", "actor": "friendA", "magnitude": 0.6}
  ],
  "optional": false,
  "session": "day-12"
}
```

- `actor` は `"self"`、`"world"`、または参加者の id。`magnitude` は 0〜1。`type` は下の効果表の語彙に限り、未知の型はエラーです。
- **曖昧な結果**（v3.1）: 単一の型の代わりに解釈の候補を持てます。確率は LLM の素の読みです（人格を知らない）。
  `{"interpretations": {"credit_taken": 0.5, "credit_shared": 0.5}, "actor": "friendA", "magnitude": 0.8}`
- `session`（任意）: 文脈の連続。条件づけの TD は、同じ session（またはどちらも未指定）の連続する事象の間だけつなぎます。途切れたら次の価値を 0 とします。
- `context`・`concepts`・参加者の `class` は、lexicon にない語でも受け付けます。**ゼロベクトルとして状態に登録し、警告を記録します**（結果の型と actor は未知ならエラー）。個体の `class` は途中で変えられません。
- `optional: true` のとき、NPN が予期 Q に基づいて接近・回避を選びます。
- `outcomes` は空でもよい（何も起きなかった経験。欲求の状態が基準へ戻るだけ）。

### プローブ（学習しない評価）

```json
{"context": "music_session", "concepts": ["music"], "participants": [{"id": "friendA", "class": "collaborator"}]}
{"entity": "friendA", "class": "collaborator", "context": "music_session"}   // context は省略可
{"concept": "music"}
{"cue": "type:praise"}                                                       // 条件づけられた価値 C（v3.1）
```

状況プローブは `Q, V_MB, V_MF, ω` と予測確率を、エンティティプローブは好悪 `like`（人モデルの予測を a で重み付けた合計）・`p_person`・`Q`・`conditioned`（C）を、
概念プローブは `like` と `conditioned` を、手がかりプローブは `C` を返します。手がかりの名前は `concept:<x>`、`ind:<id>`、`cls:<c>`、`type:<結果型>`。

## 効果表 M（生得・固定）

欲求は 5 つ: **R** 関係性、**C** 有能感、**A** 自律、**S** 刺激、**Sa** 安全。
結果 1 つの欲求の変化は `Δh = magnitude × M[type]`、事象の欲求の変化は `Δh = Σ_o magnitude_o · M[type_o]` です（曖昧な結果は解釈の重みでの混合）。
基本の大きさ 0.30 は docs/04 の玩具（gM = 0.3）と同じ。安全の効用は不足側の二乗罰で傾きが小さいので、安全の行は大きめ（0.60）にしてあります。

| 結果の型 | R（関係性） | C（有能感） | A（自律） | S（刺激） | Sa（安全） |
| --- | ---: | ---: | ---: | ---: | ---: |
| `connection` | +0.30 | 0 | 0 | 0 | 0 |
| `rejection` | -0.30 | -0.05 | 0 | 0 | -0.10 |
| `support` | +0.15 | +0.15 | 0 | 0 | +0.15 |
| `betrayal` | -0.30 | 0 | 0 | 0 | -0.30 |
| `harm` | -0.05 | -0.05 | 0 | 0 | -0.60 |
| `praise` | 0 | 0 | 0 | 0 | 0 |
| `criticism` | -0.05 | -0.20 | 0 | 0 | -0.05 |
| `credit_taken` | 0 | 0 | -0.30 | 0 | 0 |
| `credit_given` | +0.05 | +0.10 | +0.20 | 0 | 0 |
| `credit_shared` | +0.10 | +0.10 | +0.10 | 0 | 0 |
| `taught` | 0 | 0 | 0 | +0.30 | 0 |
| `achievement` | 0 | +0.30 | 0 | 0 | 0 |
| `failure` | 0 | -0.30 | 0 | 0 | -0.10 |
| `autonomy_gain` | 0 | 0 | +0.30 | 0 | 0 |
| `autonomy_loss` | 0 | 0 | -0.30 | 0 | 0 |
| `novelty` | 0 | 0 | 0 | +0.30 | 0 |
| `monotony` | 0 | 0 | 0 | -0.30 | 0 |
| `safety_threat` | 0 | 0 | 0 | 0 | -0.60 |
| `comfort` | +0.05 | 0 | 0 | 0 | +0.30 |

v3.0 からの差: `praise` は**生得の効果を持たない**（全欲求 0。社会的承認の価値は条件づけで形成される。U2 の前提）。`credit_shared`（手柄を共有された）を加えた（曖昧な出来事の候補）。
表は `npnv3.innate.m_table_markdown()` が生成します。最上位の価値の軸＝欲求の種類・M・効用は設計者の所与で、経験からは生まれません（docs/04 第8節 1）。
M は NPN ごとに引数 `effect_table` で差し替えられます（U3 の頑健性検査用。保存・復元の対象）。

- 欲求の状態: `h0 = 1`、`h ← clip(h + Δh, 0, 6)`、続いて `h ← h + 0.1 (1 − h)`。
- 効用: 成長欲求 R, C, A, S は `u(h) = 3 (1 − exp(−h/3))`。安全は `u(h) = −½ max(1 − h, 0)²`。
- 事象の効用変化（ベクトル）: `Δu_k = u_k(h_k + Δh_k) − u_k(h_k)`。結果 o の寄与は `Δu_k · Δh_{k,o} / Δh_k`（`Δh_k = 0` なら 0）。寄与の合計は `Δu_k` に等しい。

## 数式の要約

### 信念（`belief.py`）

- 単位: 個体 ind ← 類 cls ← 一般 gen（人の層）、文脈 ctx ← 世界 world（文脈の層）。各単位・各型 k に計数 (n1, n0)。
- 個体 `p_ind = (κ_e m_cls + n1) / (κ_e + n)`、類 `m_cls = (κ_c prior_cls + Σ w_i r_i) / (κ_c + Σ w_i)`、`w_i = n_i / (n_i + κ_e)`。
  **類の有効標本数は個体の数で上限が決まる。** 一般 `m_gen` は、類の生の率 `Σ w_i r_i / Σ w_i` を重み `W_cls/(W_cls + κ_c)` で集計し、事前 Beta(1, 4) を `κ_g` 分加える。文脈の層も同じ形。`κ_c = κ_g = 4`（固定）。
- **κ_e（v3.1。設計の改訂でオーケストレーター承認）**: 格子（0.3〜50 の対数 12 点）上の Beta–Binomial 周辺尤度の事後平均。類全体の値κ_c は全型をまとめた尤度、**型 k の値は log κ の事後 ∝ 型 k の尤度 × N(log κ_c, 1.0²)**（データの少ない型は類全体の値へ縮約）。正の計数が 1 つもない型は κ_c。個体が 2 人未満の類は κ_e = 2。文脈の層の κ_e は 2 のまま。`kappa_per_type=False` で類ごとに 1 つ（改訂前）。
- **事前ヘッド（v3.1）**: `prior_cls = ρ_h · head + (1 − ρ_h) · m_gen`、`ρ_h = n_head/(n_head + 30)`、n_head は学習した参加者の観測数。ヘッドは 8+8 次元 → 16 tanh → 19 ロジット（`head_kind = "mlp"`）。
  基線: `linear`（ロジスティック回帰）、`knn1`（既知の類のうち埋め込みが最も近い類の率）、`none`（`no_head=True` と同じ）。学習は直近 500 件のリプレイ緩衝から毎経験 4 ミニバッチ（各 16 件）、SGD 学習率 0.05、**目標は観測 y であり事後平均ではない**。
- スコープ帰属: `φ_person ∝ π_{k,person} Bern(y; p_ind)`、`φ_ctx ∝ π_{k,ctx} Bern(y; p_ctx^person)`。計数は φ で重み付けして加え、π（Dirichlet(1,1)）の計数にも φ を加える。
  予測 `P(y_{e,k}=1) = π̄_person p_ind + π̄_ctx p_ctx^person`。
- **曖昧な結果（v3.1）**: 解釈の重み `w_I ∝ q_LLM(I) · P_belief(I | actor, context)`。効果（Δh, Δu, 帰属, 教師）は重みでの混合で計算する。**信念の計数は曖昧な結果の型では更新しない**（観測なしとしてマスク。解釈が信念を自己訓練するループを作らない）。実験条件 `interpretation_counts=True` で、重みで計数する。`no_interpretation=True` は w_I = q_LLM(I)（信念を使わない）。

### 価値（`value.py`）

- 人モデル: 特徴 `people`(0.01) + `cls:<c>`(0.04) + `ind:<id>`(0.25) + 類の埋め込み 8 次元(各 0.02)。
  状況モデル: `world`(0.01) + `ctx:<c>`(0.1) + `concept:<x>`(各 0.1) + 概念埋め込みの平均 8 次元(各 0.02)。（括弧は事前分散）
- カルマン更新（観測雑音 R = 0.01、欲求 5 つで共分散を共有）: `K = Σz / (zᵀΣz + R)`、`W += K (Δu − zᵀW)`、`Σ ← Σ − K zᵀΣ`、続いて関与した特徴の対角に過程雑音 q（ind 1e-4、cls・ctx・concept 2e-5、people・world・埋め込み 5e-6）。**教師は重みなしの Δu のまま**（a は読み出しだけに使う）。
- 教師の分け方: 参加者 e には `Σ_{o: actor=e} φ_person(o) · contrib(o)`、状況には `Σ_{o: actor∈{self, world}} contrib(o) + Σ_{o: actor=参加者} φ_ctx(o) · contrib(o)`。人と状況の教師の和は事象の Δu に等しい。
- **新しい個体は類の価値を継承する**（加法の階層。個体の偏差の事前平均は 0）。
- **関心の重み a（v3.1）**: `a ← normalize( a + η_a (|Δu| / mean(|Δu|_run) − a) )`、η_a = 0.02、和は 5。`|Δu|` は生得の効用の変化の大きさ（外生）で、a の目標に価値ネットの出力は入らない。
  人生で大きく揺さぶられた欲求ほど重くなる（正負を問わない感作。**この方向は設計上の仮説**）。`mean(|Δu|_run)` は全欲求の `|Δu|` の平均の移動平均（速さ 0.05。SPEC に値の指定がなく、実装時に決めた）。
- **条件づけられた価値 C（v3.1）**: 手がかり z = 事象の概念・参加者の個体と類・結果型（0/1）への線形の重み。事象 t+1 を見たとき、事象 t の手がかりを
  `y_t = r_{t+1} + γ C_frozen(z_{t+1})`、`C += α_C (y_t − C·z_t) z_t / (z_tᵀ z_t)` で更新（α_C = 0.1、γ = 0.6、`r = Σ_k a_k Δu_k`）。目標の r は C に依存せず、γ < 1 のブートストラップは縮小。
- **MB 価値**: 信念の予測確率で結果の集合を 64 回サンプル（乱数は経験 id と設定の種から作る）。各サンプルで `Σ_k a_k (u_k(h + Δh) − u_k(h))` を平均。
  分散 `σ²_MB = Var_サンプル / 64 + Σ_{e,k} g_k² Var_k(e) + Σ_k g_k² Var_k(文脈)`。`Var` は個体・類・一般・文脈・世界の Beta 事後分散 `m(1−m)/(κ + n_eff + 1)` の和、`g_k` は型 k の結果が平均の大きさで 1 回起きたときの（a で重み付けた）効用変化（デルタ法で確率の不確実性を価値の尺度に直す）。
- **調停**: `ω = σ²_MB / (σ²_MB + σ²_MF)`、`Q = U_val = ω V_MF + (1 − ω) V_MB`。`σ²_MF` は `zᵀΣz` の和。確信の高い方に重みが寄る。
- **感じた価値（読み出し。目標にはしない）**: `f = tanh( ((Σ_k a_k Δu_k) + λ_C ΣC + η_ref (Σ_k a_k Δu_k − Q)) / 0.5 + 0.3 m )`、λ_C = 1、η_ref = 0.5。
  予期との差の項は、主観的な幸福が報酬だけでなく期待と予測誤差に依存することに基づく（Rutledge et al. 2014, PNAS 111(33), 12252–12257。文献の照合は未了）。
  驚き `δ = f − tanh(Q / 0.5)`、気分 `m ← 0.8 m + 0.2 δ`。
- **感情**（言語化用の手がかり。強度は 0〜1）:
  喜び = `max(f, 0)`、失望 = `min(1, −δ)`（δ < 0 かつ Q > 0）、安堵 = `min(1, δ)`（δ > 0 かつ Q < 0）、
  怒り = `tanh(Σ_{行為者が参加者} φ_person · max(−寄与の和, 0) / 0.5)`、悲しみ = `tanh(Σ max(−寄与_関係性, 0) / 0.5)`、興味 = `tanh(Σ max(寄与_刺激, 0) / 0.5)`。

### 埋め込みの所有（`lexicon.py`）

`data/lexicon.json` は、類・文脈・概念それぞれの **8 次元の属性ベクトル**（social, cooperation, competition, creativity, hierarchy, warmth, risk, novelty）を、**手で意味を持たせて**置いた**供給源**です。
v3.1 では、語が初めて学習に現れたとき（`experience()`）に、供給源の値を**NPN の状態へ写して保存**します。以後は状態の値だけを使います。供給源（lexicon、実運用では LLM 側の凍結した文埋め込み）を
別の座標に替えても、登録済みの語の評価は変わりません（`NPN.load(dir, lexicon=別の座標)`、U9）。lexicon にない語はゼロベクトルとして登録し、警告を `npn.warnings` に残します。
保存ファイルには lexicon の中身は入らず、所有する埋め込みの値だけが入ります。T8 の転移は、設計者が置いた類似度（「バンド仲間」は「共同作業者」に近い）に依存します。

## 使い方

### Python API

```python
import json
from npnv3 import NPN

npn = NPN(seed=0)                                  # 全人格は同じ初期状態・同じ係数
for line in open("data/histories/K.jsonl"):
    npn.experience(json.loads(line))               # 学習する。appraisal dict が返る

new = json.load(open("data/histories/new_event.json"))
a = npn.appraise(new)                              # 状態を変えない評価
print(a["Q"], a["V_MB"], a["V_MF"], a["omega"], a["f"], a["f_parts"], a["delta"], a["du"], a["emotions"])

r = npn.experience(new)                            # 学習する
print(r["value_change"]["entities"]["friendA"])    # A への評価の before / after / delta
print(npn.a, npn.probe({"cue": "type:praise"}))    # 関心の重み、条件づけられた価値

npn.probe({"entity": "friendA", "class": "collaborator", "context": "music_session"})
npn.save("state_dir")                              # state.json + arrays.npz（版とハッシュ付き）
npn = NPN.load("state_dir")                        # 改ざん・版違いは例外。lexicon=... で供給源を替えられる
```

確率的な経験史（SPEC-v3.1 1.8）:

```python
import sys; sys.path.insert(0, "data")
import make_histories as mh
history = mh.generate_history("K", seed=7, n=36)          # 型 R / K / X の世界から、種ごとに別の履歴
offers = mh.generate_offers("K", seed=7, n=40)            # optional な申し出
h1 = mh.generate_praise_history("H1", seed=7)              # 褒められた後に害（H2: 支え）
```

アブレーション（テスト用、既定はすべて OFF）: `flat_value`、`fixed_teacher`、`no_head`（= `head_kind="none"`）、`no_scope`、
**`no_concern`**（a = 1 固定）、**`no_conditioning`**（C を使わない）、**`no_reference`**（η_ref = 0）、**`no_interpretation`**（曖昧な結果を LLM の確率のまま混合）。
そのほか `head_kind="mlp"|"linear"|"knn1"|"none"`、`estimate_kappa=False`、`interpretation_counts=True`（実験条件）、`record_ledger=False`（長い履歴の検査用）。
`appraise` / `experience` の戻り値の主なキー: `Q, V_MB, V_MF, omega, sigma2_MB, sigma2_MF, f, f_parts, delta, mood, du, du_weighted, a, dh, contributions, phi, interpretations, teachers, emotions, cues, warnings`（`experience` のみ `value_change`, `mood_after`）。

### CLI

```
python -m npnv3 init  STATE_DIR [--seed 0]
python -m npnv3 feed  STATE_DIR experiences.jsonl      # 学習し、各経験の要約を表示し、保存
python -m npnv3 probe STATE_DIR probes.json [--json]   # 学習せずに評価
python -m npnv3 show  STATE_DIR                         # 価値の上位・関心の重み・条件づけ・信念の要約
```

### デモ

```
python3 examples/demo.py
```

3 人格（R 協働に救われた、K 手柄を取られた、X 探索者）が同じ新しい経験を、評価の内訳（Q, V_MB, V_MF, ω, f, δ, Δu, 帰属 φ, 感情）とともにどう評価し、何を学ぶかを表示します。
さらに、**h と m を揃えても受け止め方が人格で違うこと**、**`praise` の価値が随伴で形成されること**、**曖昧な出来事の解釈が人格で違うこと**、保存→復元→継続を見せます。

## テストの実行

```
python3 -m unittest discover -s tests -v          # 全体で約 3.5 分（numpy のみ。pytest は使わない。種ごとの計算は 4 プロセスで並列）
python3 -m unittest discover -s tests -p "test_u04*.py" -v
python3 tests/diagnostics_m_scale.py              # 診断（T2(c) の原因の切り分け。テストではない）
python3 tests/diagnostics_u8.py                   # 診断（U8 の持続が不合格になった原因の切り分け）
python3 tests/diagnostics_u8_scope.py             # 診断（U8: 信念の人の部分と文脈の部分に分ける）
python3 tests/diagnostics_head.py                 # 診断（R/K/X でヘッドが評価と帰属を動かしているか）
python3 data/make_histories.py                    # 経験史を再生成（決定的）
```

- **v3.0 の判定基準 T1〜T10**: `tests/test_t01_*.py` 〜 `test_t10_*.py`（固定の履歴。閾値は SPEC 第8節のまま）。T2(a) は記録用（状態を測っていた）、T2(c) は expectedFailure。
- **v3.1 の判定基準 U1〜U9**: `tests/test_u01_*.py` 〜 `test_u09_*.py`。種 0〜19 でそれぞれ別の経験史を生成する。測り方の細部は各ファイルの冒頭に実装前に書いた。
- 機構の単体検査: `test_v31_mechanisms.py`（a・C・参照点・解釈・埋め込み・κ_e・生成器・ヘッドの種類）、`test_innate.py`、`test_schema.py`、`test_ledger_and_purity.py`、`test_structure.py`、`test_cli.py`。
- 測った数値は `tests/results/*.json` に書かれます。`expectedFailure` の 3 件は「既知の制約」の記録です（REPORT 参照）。

## 実装時の解釈（SPEC に明記がない点）

CHANGELOG の「仕様の解釈」にも書きました。主なもの:

- 参加者に帰属される結果が 1 つもない経験でも、その参加者の人モデルは教師 0 で更新する（「この相手は今回何ももたらさなかった」という観測）。
- `fixed_teacher`（3-F 型）では、参加者の教師は帰属 φ を使わず「行為者がその参加者の結果の寄与の和」とし、`W[ind:参加者] += 0.1·教師`、状況は `W[ctx] += 0.1·教師`。
- `flat_value` では、人モデルは `ind:` 特徴のみ（0.25）、状況モデルは `ctx:` と `concept:` の特徴のみ（各 0.25）。
- 文脈の層は個体と同じ κ_e（2、推定しない）、世界は一般と同じ κ_g（4）と事前 Beta(1, 4)。
- 条件づけ C の手がかりは、概念・参加者の個体と類・結果型（曖昧な結果は候補の型すべて）。回避した申し出は TD の連鎖を動かさない。
- エンティティのプローブに文脈がないときは、事前ヘッドの文脈入力に、登録済みの文脈の埋め込みの平均（なければゼロ）を使う。
- 事前ヘッドは参加者のいない経験でも、リプレイ緩衝が空でなければ毎経験 4 ミニバッチ学習する（SPEC の「毎経験」どおり）。

## 未実装の項目・既知の限界

docs/04 の全体フレームのうち、この最小実装は次を**含みません**。`Q` は `U_val` だけです（社会・関心・評価条件づけの項を足した予期は未実装）。

- **社会的な価値**（他者の表明の取り込み）と**認識的信頼**
- **関心**（学習進捗）と、それによる有能感への寄与。（v3.1 の関心の重み a は、docs の「関心 ι」とは別物で、欲求の重みです）
- **評価条件づけ**（共起からの好悪の移り。v3.1 の C は随伴関係による条件づけで、共起の転移ではない）
- **後悔**、**コミットメント**、**変動性 q の学習**（q は層ごとに固定）
- **予報ヘッド V̂** と、驚きの δ_V / δ_W への分解
- 価値→信念の経路のうち、曖昧な結果の解釈以外（望ましさによる更新の非対称、自己奉仕的な帰属）
- **想起・反すう**（オフラインの再帰属）
- **κ_c・κ_g・π の事前の経験ベイズ学習**（κ_e だけ推定）、**行為者タグの誤りの混同行列 Λ**、**割引（変動性）**
- **LLM 抽出の接続**（入力は構造化 JSON を直接与える）、凍結した文埋め込み（手作りの 8 次元で代用）
- 個体の類を途中で変えること（関係の発達）、状態サイズ・計算量の最適化（新しい個体ごとに特徴が増え、Σ は密行列）、保存の原子性と真正性（ハッシュは完全性だけ）
- 感情は固定の 6 語の読み出しで、学習や選択に因果的な役割を持たない（気分 m は f にだけ入る）
- 感じた価値 f、驚き δ、感情は、どの学習の目標にも入らない（設計）。評価が更新に戻る経路は、接近・回避（Q）、解釈（信念 → 効果の混合）、関心の重み a（|Δu|）、条件づけ C（r = a·Δu）に限られる
