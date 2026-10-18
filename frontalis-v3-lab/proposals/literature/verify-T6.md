# T6 検証メモ (LITERATURE-v4.md の #27-32 と補完)

全文を読めた出典は 0 件。検証は WebFetch による arXiv 摘要ページの確認が中心で、PDF 本文とIEEE ページは開けなかった。

## 既存項目の検証

| # | 出典 (訂正後) | 開いた URL | 検証レベル | 主張の要点 (1文) | 確認できた数値 | 注意 |
|---|---|---|---|---|---|---|
| 27 | Christiano et al. (2017) Deep RL from Human Preferences, arXiv:1706.03741 | https://arxiv.org/abs/1706.03741 | abstract only | 人間の選好比較から報酬モデルを学び、それで RL を行う。 | 選好フィードバックは全相互作用の 1% 未満 (摘要) | arXiv ページに掲載先の記載なし。NeurIPS 2017 は未確認。版: v1 2017-06-12, v4 2023-02-17 |
| 28 | Ouyang et al. (2022) InstructGPT, arXiv:2203.02155 | https://arxiv.org/abs/2203.02155 | abstract only | 教師あり微調整、順位付けによる報酬モデル、RL の3段階で意図追従を学ぶ。 | 1.3B モデルが 175B GPT-3 より好まれる (100倍少ないパラメータ) | 摘要に PPO の語はなく未確認。報酬モデルの固定は本文未読のため未確認。著者20名、v1のみ |
| 29 | Pathak et al. (2017) Curiosity-driven Exploration, arXiv:1705.05363 | https://arxiv.org/abs/1705.05363 | abstract only | 行動結果の予測誤差を、学習した特徴空間で内発的報酬とする。 | なし (VizDoom, Super Mario Bros. の環境名のみ) | arXiv コメント欄は ICML 2017。PMLR 70:2778-2787 のページ番号は未確認 |
| 30 | Ng & Russell (2000) Algorithms for IRL, ICML 2000 pp.663-670 | https://aima.eecs.berkeley.edu/~russell/papers/ml00-irl.pdf | could not open | 観測行動から報酬関数を推定する (縮退の指摘は未読)。 | 未読 | PDF は圧縮バイナリで本文不可。書誌 (pp.663-670) は DBLP/mlanthology の検索結果のみ。file の DOI 不使用は妥当 (mlanthology の DOI は別誌のもの) |
| 31 | Park et al. (2023) Generative Agents, arXiv:2304.03442 | https://arxiv.org/abs/2304.03442 | abstract only | 経験を自然言語で記憶し、反省で要約して計画に戻す。 | 25 体のエージェント、ablation は定量結果なし (摘要) | arXiv ページに掲載先の記載なし。UIST 2023 は未確認。評価特徴は学習されない点は file の注意どおり |
| 32 | Chen et al. (2025) Persona Vectors, arXiv:2507.21509 | https://arxiv.org/abs/2507.21509 | abstract only | 活性化空間の方向で人格特性を監視・制御し、学習中の変化を予測する。 | 数値なし (摘要) | 著者一致。v3 2025-09-05。対象モデルは摘要に記載なし。モデル内部の方向であり、外部評価器の分離ではない |

## 新規に確認した出典 (a) 凍結 LLM への条件づけ

| 出典 | URL | 検証レベル | 要点 | 数値・設定 | 注意 |
|---|---|---|---|---|---|
| Li & Liang (2021) Prefix-Tuning, arXiv:2101.00190 | https://arxiv.org/abs/2101.00190 | abstract only | モデルを凍結し、接頭の連続ベクトルのみ学習する。 | 学習するのは全パラメータの 0.1% (摘要)。GPT-2, BART | 学習済み接頭は各モデルに紐づく。別モデルへの転移は未検証 |
| Lester et al. (2021) Prompt Tuning, arXiv:2104.08691 | https://arxiv.org/abs/2104.08691 | abstract only | 凍結モデルに対して学習した soft prompt で条件づける。 | 「数十億パラメータ」を超えると全重み調整と差が縮まる (摘要) | 単一モデル内の議論。モデル間転移は扱わない |
| Zou et al. (2023) Representation Engineering, arXiv:2310.01405 | https://arxiv.org/abs/2310.01405 | abstract only | 集団的表現を監視・操作する枠組み (透明性研究)。 | 数値なし。作者21名 | 摘要は LLM 一般の記述。特定モデル間の転移は未確認 |
| Turner et al. (2023/2024) Activation Addition, arXiv:2308.10248 | https://arxiv.org/abs/2308.10248 | abstract only | 2つのプロンプトの活性化差から操舵ベクトルを作り、推論時に加える。学習は不要。 | 摘要は LLaMA-3, OPT を挙げる。版題は v5 で改題 | 操舵ベクトルは各モデル自身の活性化に依存。持続的外部状態ではない |
| Wu, Wu, Mou, Zero-Shot Continuous Prompt Transfer, arXiv:2310.01691 | https://arxiv.org/abs/2310.01691 | abstract only | 連続プロンプトを相対空間に写し、対象モデルで探索して移す。 | 数値なし | 共通語彙や対象モデルの学習の要否は摘要では不明。最も近い候補だが未読 |

## (b) 逆強化学習・選好学習

Ng & Russell (2000) と Christiano et al. (2017) が該当する。どちらも摘要レベルの確認にとどまり、報酬モデルの本文での検証は未実施。

## (c) Singh, Lewis, Barto, Sorg (2010)

存在は確認。IEEE Transactions on Autonomous Mental Development 2(2):70-82, DOI 10.1109/TAMD.2010.2051031 (IEEE Xplore の検索結果より。URL: https://ieeexplore.ieee.org/document/5471106)。検証レベル: could not open (IEEE ページは空で読めず、本文は未読)。

## 残るギャップ

- 「固定の外部評価器状態を、別の基盤 LLM が読んで従う」を直接扱う出典は未発見。既存の方法はいずれもモデルごとの学習または活性化に依存する。
- Wu et al. (2023) が近いが、本文未読のため採用不可。
- 数値は摘要由来のものに限る。本文の設定値は未確認。
