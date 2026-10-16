# verify-T4: 表現の測定と介入 (原典確認)

検証日: 2026-10-17 / 対象: LITERATURE-v4.md の T4 (#14-#20) と追加候補

全体の注意: WebFetch で arXiv の /pdf と transformer-circuits の本文は取得できなかった (バイナリ応答またはサイズ上限)。以下はすべて abs ページまでの確認であり、full text read は 0 件。節番号・図表番号・数値は一つも確認していない。

| # | 訂正後の書誌 | 開いたURL | 検証 | 要旨 (一文) | 方法・数値 | 留意 |
|---|---|---|---|---|---|---|
| 14 | Hewitt, J., Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. EMNLP 2019 | https://arxiv.org/abs/1909.03368 | abstract only | 高いプローブ精度だけでは表現の証拠にならず、単語型にランダム出力を対応させた control task で記憶の寄与を測る。 | 節・数値は未確認。abstract の "we propose control tasks, which associate word types with random outputs" のみ引用。 | プローブは読み出しの検査で、ablation/swap による因果的な使用は示さない。 |
| 15 | Lampinen, A. K., Chan, S. C. Y., Hermann, K. (2024). Learned feature representations are biased by complexity, learning order, position, and more. TMLR | https://arxiv.org/abs/2405.05847 | abstract only | 同程度に学習された特徴でも、複雑さや学習順で表現の強さに偏りが出る。 | 節・数値は未確認。 | 介入の効果量をそのまま特徴の強さとみなす前の警告。ablation の有効性を直接検証した論文ではない。 |
| 16 | Geiger, A., Lu, H., Icard, T., Potts, C. (2021). Causal Abstractions of Neural Networks. NeurIPS 2021 | https://arxiv.org/abs/2106.02997 | abstract only | 構造解析 (probing 等) を因果抽象の理論で基礎づける手法を提案する。 | abstract に interchange intervention の語は見当たらず、定義・節は未読。 | 収集者の「interchange intervention で因果役割を検証」は本文未確認。書誌 (著者順・NeurIPS 2021) は一致。 |
| 17 | Geiger, A., Wu, Z., Potts, C., Icard, T., Goodman, N. D. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. CLeaR 2024 | https://arxiv.org/abs/2303.02536 | abstract only | 勾配降下で、解釈可能な因果変数を分散表現の方向 (部分空間) に対応づける (DAS)。 | 節・式番号は未確認。abs ページには CLeaR の記載なし。 | 対応が見つかることと、その方向を模型が実際に使うことは別。介入で検証が必要。 |
| 18 | Bricken, T., Templeton, A., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread | https://transformer-circuits.pub/2023/monosemantic-features/ (index.html) | could not open | 未確認。 | 未確認。 | 本文取得に失敗。数値は採用しない。 |
| 19 | Locatello, F., et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. ICML 2019 (PMLR 97) | https://arxiv.org/abs/1811.12359 | abstract only | 教師なし disentanglement は帰納バイアスなしに同定できず、分離度が下流の効率を保証しない。 | 定理番号・実験数は未確認。abs に award の記載なし。 | 「best paper」は確認できず、採用しない。 |
| 20 | Lake, B. M., Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. ICML 2018 (PMLR 80) | https://arxiv.org/abs/1711.00350 | abstract only | seq2seq RNN は既知要素の組合せには汎化するが、体系的な構成には失敗する。 | 節・表の数値は未確認。ICML 2018 は journal-ref で確認。 | 構成的汎化の課題。モデル内部の特徴を因果的に検定するものではない。 |

## 追加候補

- 交換介入の精度 (IIA) の出典: Wu, Z., Geiger, A., Icard, T., Potts, C., Goodman, N. D. (2023/2024). Interpretability at Scale: Identifying Causal Mechanisms in Alpaca. arXiv:2305.08809 (abs only; comments に NeurIPS 2023)。abstract に精度数値はない。IIA の定義と数値は未読。
- プローブの対照検定: Hewitt & Liang (#14) が原典。補助として Belinkov, Y. (2022) Probing Classifiers: Promises, Shortcomings, and Advances. arXiv:2102.12452 (abs only。comments に Computational Linguistics の squib)。

## 訂正事項

- Ibeling 系の因果抽象論文: arXiv:2301.04709 の現行版の題名は "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability" (Geiger, Ibeling, Zur, Chaudhary, Chauhan, Huang, Arora, Wu, Goodman, Potts, Icard)。v1 は 2023-01-11、現行 v4 は 2025-05-08。v1 の題名は未確認。Geiger 2021 (#16) には含まれない点は収集者の記述どおり。
- #16 の著者・年・venue は一致。#17 の CLeaR 2024 は abs ページでは確認できなかった。
