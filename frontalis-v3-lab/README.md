# frontalis-v3-lab: NPN 最小実装 v3.0

経験によって人格が形成される外部ニューラルネットワーク（NPN）の、**実際に動く最小実装**です。
経験の入力 → 評価 → 学習 → 次の評価への反映、までを実行し、テストで検証します。

- 仕様: [`SPEC.md`](SPEC.md)（判定基準 T1〜T10 は実装前に固定）
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

> この実装が確かめるのは学習の機構です。「人格らしい発言」ができるかは対象外で、LLM との接続も含みません。

## 構成

```
frontalis-v3-lab/
  npnv3/
    config.py     設定（係数・アブレーションの切り替え）。保存・復元の対象
    schema.py     経験・プローブの検証
    innate.py     欲求 5 つ・効果表 M・効用（固定、全人格共通）
    lexicon.py    概念・類・文脈の 8 次元埋め込み（data/lexicon.json を読む）
    mlp.py        numpy の小さな MLP（順伝播・逆伝播・SGD）
    belief.py     信念ネット: スコープ帰属つき階層ベイズ ＋ ニューラル事前ヘッド
    value.py      価値ネット: 階層カルマン MF、MB、調停、気分、感情
    ledger.py     更新台帳
    npn.py        統合: experience() / appraise() / probe() / save() / load()
    cli.py, __main__.py
  data/
    lexicon.json          手で意味を持たせた 8 次元ベクトル
    make_histories.py     経験史の生成スクリプト（決定的）
    histories/            R, K, X（T2/T3/T6/T9/T1）、T4_*, T7_*, T8_train、new_event.json
  tests/                  unittest（T1〜T10 と補助の検査）、results/ に数値、diagnostics_m_scale.py（診断）
  examples/demo.py        3 人格が同じ新しい経験をどう評価し、何を学ぶか
```

`npnv3/config.py` は SPEC の構成にない追加ファイルです（係数を 1 か所に集め、保存・復元の対象にするため）。

## 1 回の経験で起きること

`experience(exp)` は次の順に進みます（SPEC 第6節）。

1. スキーマ検証。
2. **予期**（結果を見る前）: 信念の予測 → `V_MB`, `V_MF`, `ω`, `Q`。
3. `optional: true` なら、接近確率 `σ(4 (Q + 0.05))` で接近・回避を選ぶ。**回避したら、回避したことだけ台帳に記録して終わり**（学習しない）。
4. 欲求の変化 Δh、効用変化 Δu（5 欲求のベクトル）、結果ごとの寄与、感じた価値 `f`、驚き `δ`、感情。
5. **信念の更新**: 責任 φ（人のせい／場面のせい）→ 計数、スコープの事前 π、型の大きさ、リプレイ緩衝 → 事前ヘッドの SGD（目標は観測そのもの）。
6. **価値の更新**: Δu を φ で人モデルと状況モデルに分け、それぞれカルマン更新。
7. 欲求の状態 `h` と気分 `m` の更新。
8. 台帳に記録（経験 id、各特徴の ΔW、φ、Δu、Q、f、δ）。

`appraise(exp)` は 2〜4 だけを計算し、**状態（乱数を含む）を一切変えません**。`probe(p)` も学習しません。

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
  "optional": false
}
```

- `actor` は `"self"`、`"world"`、または参加者の id。`magnitude` は 0〜1。`type` は下の効果表の語彙に限り、未知の型はエラーです。
- `context`・`concepts`・参加者の `class` は `data/lexicon.json` にある語彙に限ります。個体の `class` は途中で変えられません。
- `optional: true` のとき、NPN が予期 Q に基づいて接近・回避を選びます。
- `outcomes` は空でもよい（何も起きなかった経験。欲求の状態が基準へ戻るだけ）。

### プローブ（学習しない評価）

```json
{"context": "music_session", "concepts": ["music"], "participants": [{"id": "friendA", "class": "collaborator"}]}
{"entity": "friendA", "class": "collaborator", "context": "music_session"}   // context は省略可
{"concept": "music"}
```

状況プローブは `Q, V_MB, V_MF, ω` と予測確率を、エンティティプローブは好悪 `like`（人モデルの予測の 5 欲求の和）・
`p_person`（各型を起こす確率）・`Q` を、概念プローブは `like` を返します。

## 効果表 M（生得・固定）

欲求は 5 つ: **R** 関係性、**C** 有能感、**A** 自律、**S** 刺激、**Sa** 安全。
結果 1 つの欲求の変化は `Δh = magnitude × M[type]`、事象の欲求の変化は `Δh = Σ_o magnitude_o · M[type_o]` です。
基本の大きさ 0.30 は docs/04 の玩具（gM = 0.3）と同じ。安全の効用は不足側だけの二乗罰で傾きが小さいので、安全の行は大きめ（0.60）にしてあります。

| 結果の型 | R（関係性） | C（有能感） | A（自律） | S（刺激） | Sa（安全） |
| --- | ---: | ---: | ---: | ---: | ---: |
| `connection` | +0.30 | 0 | 0 | 0 | 0 |
| `rejection` | -0.30 | -0.05 | 0 | 0 | -0.10 |
| `support` | +0.15 | +0.15 | 0 | 0 | +0.15 |
| `betrayal` | -0.30 | 0 | 0 | 0 | -0.30 |
| `harm` | -0.05 | -0.05 | 0 | 0 | -0.60 |
| `praise` | +0.10 | +0.20 | 0 | 0 | 0 |
| `criticism` | -0.05 | -0.20 | 0 | 0 | -0.05 |
| `credit_taken` | 0 | 0 | -0.30 | 0 | 0 |
| `credit_given` | +0.05 | +0.10 | +0.20 | 0 | 0 |
| `taught` | 0 | 0 | 0 | +0.30 | 0 |
| `achievement` | 0 | +0.30 | 0 | 0 | 0 |
| `failure` | 0 | -0.30 | 0 | 0 | -0.10 |
| `autonomy_gain` | 0 | 0 | +0.30 | 0 | 0 |
| `autonomy_loss` | 0 | 0 | -0.30 | 0 | 0 |
| `novelty` | 0 | 0 | 0 | +0.30 | 0 |
| `monotony` | 0 | 0 | 0 | -0.30 | 0 |
| `safety_threat` | 0 | 0 | 0 | 0 | -0.60 |
| `comfort` | +0.05 | 0 | 0 | 0 | +0.30 |

（表は `npnv3.innate.m_table_markdown()` が生成します。最上位の価値の軸＝欲求の種類・M・効用は設計者の所与で、経験からは生まれません。docs/04 第8節 1。）

- 欲求の状態: `h0 = 1`、`h ← clip(h + Δh, 0, 6)`、続いて `h ← h + 0.1 (1 − h)`。
- 効用: 成長欲求 R, C, A, S は `u(h) = 3 (1 − exp(−h/3))`。安全は `u(h) = −½ max(1 − h, 0)²`。
- 事象の効用変化（ベクトル）: `Δu_k = u_k(h_k + Δh_k) − u_k(h_k)`。結果 o の寄与は `Δu_k · Δh_{k,o} / Δh_k`（`Δh_k = 0` なら 0）。寄与の合計は `Δu_k` に等しい。

## 数式の要約

### 信念（`belief.py`）

- 単位: 個体 ind ← 類 cls ← 一般 gen（人の層）、文脈 ctx ← 世界 world（文脈の層）。各単位・各型 k に計数 (n1, n0)。
- 個体 `p_ind = (κ_e m_cls + n1) / (κ_e + n)`、類 `m_cls = (κ_c prior_cls + Σ w_i r_i) / (κ_c + Σ w_i)`、`w_i = n_i / (n_i + κ_e)`。
  **類の有効標本数は個体の数で上限が決まる。** 一般 `m_gen` は、類の生の率 `Σ w_i r_i / Σ w_i` を重み `W_cls/(W_cls + κ_c)` で集計し、事前 Beta(1, 4) を `κ_g` 分加える。文脈の層も同じ形。`κ_e = 2, κ_c = 4, κ_g = 4`（固定）。
- `prior_cls` は事前ヘッドの出力（8+8 次元 → 16 tanh → 18 ロジット）。学習した観測が 50 件未満なら `m_gen`。
- スコープ帰属: `φ_person ∝ π_{k,person} Bern(y; p_ind)`、`φ_ctx ∝ π_{k,ctx} Bern(y; p_ctx^person)`。計数は φ で重み付けして加え、π（Dirichlet(1,1)）の計数にも φ を加える。
  予測 `P(y_{e,k}=1) = π̄_person p_ind + π̄_ctx p_ctx^person`。
- 事前ヘッドの学習: 直近 500 件のリプレイ緩衝から毎経験 4 ミニバッチ（各 16 件）、SGD 学習率 0.05。**目標は観測 y であり、事後平均ではない。**

### 価値（`value.py`）

- 人モデル: 特徴 `people`(0.01) + `cls:<c>`(0.04) + `ind:<id>`(0.25) + 類の埋め込み 8 次元(各 0.02)。
  状況モデル: `world`(0.01) + `ctx:<c>`(0.1) + `concept:<x>`(各 0.1) + 概念埋め込みの平均 8 次元(各 0.02)。（括弧は事前分散）
- カルマン更新（観測雑音 R = 0.01、欲求 5 つで共分散を共有）: `K = Σz / (zᵀΣz + R)`、`W += K (Δu − zᵀW)`、`Σ ← Σ − K zᵀΣ`、続いて関与した特徴の対角に過程雑音 q（ind 1e-4、cls・ctx・concept 2e-5、people・world・埋め込み 5e-6）。
- 教師の分け方: 参加者 e には `Σ_{o: actor=e} φ_person(o) · contrib(o)`、状況には `Σ_{o: actor∈{self, world}} contrib(o) + Σ_{o: actor=参加者} φ_ctx(o) · contrib(o)`。人と状況の教師の和は事象の Δu に等しい。
- **新しい個体は類の価値を継承する**（加法の階層。個体の偏差の事前平均は 0）。
- **MB 価値**: 信念の予測確率で結果の集合を 64 回サンプル（乱数は経験 id と設定の種から作る）。各サンプルで `U(h + Δh) − U(h)` の総和を平均。
  分散 `σ²_MB = Var_サンプル / 64 + Σ_{e,k} g_k² Var_k(e) + Σ_k g_k² Var_k(文脈)`。`Var` は個体・類・一般・文脈・世界の Beta 事後分散 `m(1−m)/(κ + n_eff + 1)` の和、`g_k` は型 k の結果が平均の大きさで 1 回起きたときの効用変化（デルタ法で確率の不確実性を価値の尺度に直す）。
- **調停**: `ω = σ²_MB / (σ²_MB + σ²_MF)`、`Q = U_val = ω V_MF + (1 − ω) V_MB`。`σ²_MF` は `zᵀΣz` の和。確信の高い方に重みが寄る。
- **読み出し**（目標にはしない）: `f = tanh(Σ_k Δu_k / 0.5 + 0.3 m)`、驚き `δ = f − tanh(Q / 0.5)`、気分 `m ← 0.8 m + 0.2 δ`。
- **感情**（言語化用の手がかり。強度は 0〜1）:
  喜び = `max(f, 0)`、失望 = `min(1, −δ)`（δ < 0 かつ Q > 0）、安堵 = `min(1, δ)`（δ > 0 かつ Q < 0）、
  怒り = `tanh(Σ_{行為者が参加者} φ_person · max(−寄与の和, 0) / 0.5)`、悲しみ = `tanh(Σ max(−寄与_関係性, 0) / 0.5)`、興味 = `tanh(Σ max(寄与_刺激, 0) / 0.5)`。

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
print(a["Q"], a["V_MB"], a["V_MF"], a["omega"], a["f"], a["delta"], a["du"], a["emotions"])

r = npn.experience(new)                            # 学習する
print(r["value_change"]["entities"]["friendA"])    # A への評価の before / after / delta

npn.probe({"entity": "friendA", "class": "collaborator", "context": "music_session"})
npn.save("state_dir")                              # state.json + arrays.npz（版とハッシュ付き）
npn = NPN.load("state_dir")                        # 改ざん・版違いは例外
```

アブレーション（テスト用、既定はすべて OFF）: `NPN(seed, flat_value=True)`、`fixed_teacher=True`、`no_head=True`、`no_scope=True`。
`appraise` / `experience` の戻り値の主なキー: `Q, V_MB, V_MF, omega, sigma2_MB, sigma2_MF, f, delta, mood, du, dh, contributions, phi, teachers, emotions, value_change`（`experience` のみ `value_change`, `mood_after`）。

### CLI

```
python -m npnv3 init  STATE_DIR [--seed 0]
python -m npnv3 feed  STATE_DIR experiences.jsonl      # 学習し、各経験の要約を表示し、保存
python -m npnv3 probe STATE_DIR probes.json [--json]   # 学習せずに評価
python -m npnv3 show  STATE_DIR                         # 価値の上位・信念の要約
```

### デモ

```
python3 examples/demo.py
```

3 人格（R 協働に救われた、K 手柄を取られた、X 探索者）が、同じ新しい経験を、評価の内訳（Q, V_MB, V_MF, ω, f, δ, Δu, 帰属 φ, 感情）とともにどう評価し、何を学ぶかを表示し、最後に保存→復元→継続を見せます。

## テストの実行

```
python3 -m unittest discover -s tests -v          # 全体で約 2 分（numpy のみ。pytest は使わない）
python3 -m unittest discover -s tests -p "test_t04*.py" -v
python3 tests/diagnostics_m_scale.py              # 診断（T2(c) の原因の切り分け。テストではない）
python3 data/make_histories.py                    # 経験史を再生成（決定的）
```

T1〜T10 は `tests/test_t01_*.py` 〜 `test_t10_*.py`。閾値は SPEC 第8節のままです。測った数値は `tests/results/*.json` に書かれます。
補助の検査: `test_innate.py`（寄与の分解）、`test_schema.py`、`test_ledger_and_purity.py`（台帳の不変量・appraise の純粋性・教師が価値ネットの出力に依存しないこと）、`test_structure.py`（継承・帰属・ヘッド・アブレーション）、`test_cli.py`。
`expectedFailure` の 2 件は「既知の制約」の記録です（REPORT 参照）。

## 語彙（lexicon）について

`data/lexicon.json` は、類・文脈・概念それぞれの **8 次元の属性ベクトル**（social, cooperation, competition, creativity, hierarchy, warmth, risk, novelty）を、**手で意味を持たせて**置いたものです。
**実運用では、LLM 側の凍結した文埋め込みに置き換える想定**です（次元は 8 に射影するか、`lexicon.EMB_DIM` と事前ヘッドの入力幅を変える）。この手作りの埋め込みは、
事前ヘッドの転移（T8 の「バンド仲間」は「共同作業者」に近いベクトル）と、価値の埋め込み特徴にだけ使われます。そのため T8 の転移は、設計者が置いた類似度に依存します。

## 実装時の解釈（SPEC に明記がない点）

CHANGELOG の「仕様の解釈」にも書きました。主なもの:

- 参加者に帰属する結果が 1 つもない経験でも、その参加者の人モデルは教師 0 で更新する（「この相手は今回何ももたらさなかった」という観測）。
- `fixed_teacher`（3-F 型）では、参加者の教師は帰属 φ を使わず「行為者がその参加者の結果の寄与の和」とし、`W[ind:参加者] += 0.1·教師`、状況は `W[ctx] += 0.1·教師`。
- `flat_value` では、人モデルは `ind:` 特徴のみ（0.25）、状況モデルは `ctx:` と `concept:` の特徴のみ（各 0.25）。
- 文脈の層は個体と同じ κ_e（2）、世界は一般と同じ κ_g（4）と事前 Beta(1, 4)。
- エンティティのプローブに文脈がないときは、事前ヘッドの文脈入力に全文脈の埋め込みの平均を使う。
- 事前ヘッドは参加者のいない経験でも、リプレイ緩衝が空でなければ毎経験 4 ミニバッチ学習する（SPEC の「毎経験」どおり）。

## 未実装の項目（v3.0 では実装していない）

docs/04 の全体フレームのうち、この最小実装は次を**含みません**。`Q` は `U_val` だけです（SPEC 5.2）。

- **社会的な価値**（他者の表明の取り込み）と**認識的信頼**
- **関心**（学習進捗）と、それによる有能感への寄与
- **評価条件づけ**（共起からの好悪の移り）
- **後悔**（捨てた選択肢の更新）
- **コミットメント**（実験条件）と**変動性 q の学習**（q は層ごとに固定）
- **予報ヘッド V̂** と、驚きの δ_V / δ_W への分解
- **曖昧な解釈**（解釈の硬さ γ_h）、価値→信念の経路のうち接近・回避以外（望ましさによる更新の非対称、自己奉仕的な帰属）
- **想起・反すう**（オフラインの再帰属）
- **事前（κ_e, κ_c, κ_g, π）の経験ベイズ学習**（固定）、**行為者タグの誤りの混同行列 Λ**、**割引（変動性）**
- **LLM 抽出の接続**（入力は構造化 JSON を直接与える）、凍結した文埋め込み（手作りの 8 次元で代用）
- 状況単位のモデルベース価値は動くが、**社会・関心・評価条件づけを含む統合の検証は未実施**
