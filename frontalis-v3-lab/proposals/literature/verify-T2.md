# T2 検証メモ (2026-10-17)

検証レベル: full text read は0件。abstract only = arXiv abs ページを WebFetch で確認。could not open = 原典に到達できず。

主張の基準: 「NN は線形・固定軸モデルが表せない相互作用特徴を獲得する。公平な比較が必要」

| # | 文献 (訂正後の書誌) | 開いたURL | 検証 | 支持する内容 / 確認した数値 / 注意 |
|---|---|---|---|---|
| 1 | Minsky & Papert (1969) Perceptrons, MIT Press | なし | could not open (二次: 講義資料) | 単層パーセプトロンはXORを表せない。注意: 多層は対象外。 |
| 2 | Barron (1993) IEEE Trans. Inf. Theory 39(3):930-945 | EPFL PDF (読めず) | could not open (検索スニペットのみ) | 1隠れ層シグモイドの誤差が1/n。固定基底の線形係数調整は1/n^(2/d)。注意: 本文未確認。 |
| 3 | Eldan & Shamir, "The Power of Depth for Feedforward Neural Networks", arXiv:1512.03965 (COLT 2016 と記載) | arxiv.org/abs/1512.03965 | abstract only | 3層では小さく表せ、2層は幅が次元dに対し指数的に必要。注意: 相互作用の具体例ではない。 |
| 4 | Telgarsky, "Representation Benefits of Deep Feedforward Networks", arXiv:1509.08101 (2015) | arxiv.org/abs/1509.08101 | abstract only | 浅い網は指数的なノード数が必要、深い網は2ノード×2k層で誤差0。注意: 2016 は依頼の年。venue は未確認。 |
| 5 | Chizat, Oyallon, Bach (2019) "On Lazy Training in Differentiable Programming", NeurIPS 2019, arXiv:1812.07956 | arxiv.org/abs/1812.07956 | abstract only | 遅延学習は線形化=カーネルと等価。非線形深層畳み込みは遅延領域で性能が落ちる。注意: 著者は Chizat & Bach ではなく三名。 |
| 6 | Jacot, Gabriel, Hongler (2018) NTK, NeurIPS 2018, arXiv:1806.07572 | arxiv.org/abs/1806.07572 | abstract only | 無限幅極限で学習は固定カーネルに収束。注意: カーネル領域の基準であり、相互作用の証拠ではない。 |
| 7 | Yang & Hu (2021) "Feature Learning in Infinite-Width Neural Networks", arXiv:2011.14522 (ICML 2021 と記載) | arxiv.org/abs/2011.14522 | abstract only | SP/NTK パラメトリゼーションは特徴学習の無限幅極限を持たず、修正版を提案。 |
| 8 | Damian, Lee, Soltanolkotabi (2022) "Neural Networks can Learn Representations with Gradient Descent", arXiv:2206.15144 (COLT 2022) | arxiv.org/abs/2206.15144 | abstract only | f*(x)=g(Ux)、d≫r、次数p。カーネル法 n≍d^p に対し GD は n≍d²r+dr^p。注意: XOR・幅の記述なし。 |
| 9 | Ghorbani, Mei, Misiakiewicz, Montanari (2020) "When Do Neural Networks Outperform Kernel Methods?", arXiv:2006.13409 | arxiv.org/abs/2006.13409 | abstract only | 入力が目標と同じ低次元構造を持つとき、次元の呪いが緩和され NN が有利。注意: 数値なし。 |
| 10 | Ghorbani et al. "Linearized two-layers neural networks in high dimension", arXiv:1904.12191 (arXiv 2019, 改訂2020) | arxiv.org/abs/1904.12191 | abstract only | RF/NT の線形化モデルはデータ数が d^(ℓ+δ) から d^(ℓ+1-δ) の間で次数ℓの多項式しか当てはまらない。注意: 依頼の「2021」は改訂年の可能性。掲載誌は未確認。 |
| 11 | Zhang, Bengio, Hardt, Recht, Vinyals (2017) ICLR, arXiv:1611.03530 | arxiv.org/abs/1611.03530 | abstract only | 大きな NN はランダムラベルを当てはめられる。注意: 記憶の話で、相互作用の話ではない。数値なし。 |
| 12 | Nakkiran et al. (2020) "Deep Double Descent", arXiv:1912.02292 | arxiv.org/abs/1912.02292 | abstract only | 誤差はモデル規模・学習エポックに対し二重降下を示す。サンプルを4倍にすると悪化しうる。注意: 学習ステップと標本数の分離設計の根拠としては弱い。 |
| 13 | Kaplan et al. (2020) "Scaling Laws for Neural Language Models", arXiv:2001.08361 | arxiv.org/abs/2001.08361 | abstract only | 損失はモデル規模・データ量・計算量のべき則。注意: 言語モデルのみ。指数の数値は未確認。 |
| 14 | Nagarajan, Liang, Lafferty (2021) | 検索3回で該当なし | could not find | 削除。書誌が誤っている可能性が高い。 |

## 近い代替候補 (未読・検索で発見)

- Warmuth, Kotłowski, Amid, ALT 2021 (talk: https://slideslive.com/38952832/a-case-where-a-spindly-twolayer-linear-network-decisively-outperforms-any-neural-network-with-a-fully-connected-input-layer). 検証: abstract only (talk ページのみ、論文本文は未読)。疎な目標で、全結合入力層の NN は疎な線形網より標本効率が悪いという主張。注意: 「NN が有利」と逆向きの例で、公平比較の反例として使える。

## まとめ

- 検証できた abstract は 13 件 (#3-#13 の大半)。全文は読めていない。
- Barron と Minsky & Papert は原典未確認。
- 依頼文の Nagarajan 2021 は見つからず削除。
- 相互作用 (XOR 型) を直接扱う原典は今回確保できていない。#3, #4 は深さの分離、#8 は多項式の相互作用に近いが、直接の証拠とは言えない。
