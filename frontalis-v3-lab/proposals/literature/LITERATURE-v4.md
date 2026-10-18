# Project frontalis: 文献収集 v4 (Collect-first)

収集日: 2026-10-17 / 収集者: literature collector (Haiku 5.5)
方針: 設計・議論はせず、出典の収集と検証状態の記録のみ行う。

検証レベルの定義:
- full text read: 全文を読んだ (今回は該当なし)
- abstract only: arXiv等のabstractページを WebFetch で取得して確認した
- secondary source only: 検索結果の要約・二次資料・索引ページのみで確認した (原典は未読)
- could not open: 原典に到達できなかった

## 索引 (トピック別)

- T1 (価値・特徴の表現学習): #1, #2, #3, #4, #5, #6, #7
- T2 (非線形性・NN対線形の比較): ギャップ (#8 は関連のみ)
- T3 (appraisal・構成的選好・再評価): #9, #10, #11, #12, #13 / 情動的条件づけ専用の出典は未発見
- T4 (表現の測定・介入): #14, #15, #16, #17, #18, #19, #20 (#21 は組合せ一般化)
- T5 (継続学習・消去/再出現): #22, #23, #24, #25, #26
- T6 (価値学習・人格的選好・LLM拡張): #27, #28, #29, #30, #31, #32
- T7 (階層ベイズ・カルマン型学習モデル): #33, #34, #35

## 文献一覧

### T1: 経験からの報酬・価値特徴の学習

1. Barreto, A., Dabney, W., Munos, R., Hunt, J. J., Schaul, T., van Hasselt, H., Silver, D. (2017). Successor Features for Transfer in Reinforcement Learning. NeurIPS 2017 (arXiv:1606.05312, v1 2016-06, 改訂 2018-04).
   - URL: https://arxiv.org/abs/1606.05312
   - 支持する内容: 報酬だけが変わる課題間で、力学と報酬を分離した表現を転移させる枠組み。
   - 設計上の問い: 「価値特徴」を遷移の予測と報酬の重みに分けて持つべきか。報酬特徴が固定か学習されるかで転移性がどう変わるか。
   - 検証: abstract only (WebFetch で確認)

2. Schaul, T., Horgan, D., Gregor, K., Silver, D. (2015). Universal Value Function Approximators. ICML 2015, PMLR 37:1312-1320.
   - URL: https://proceedings.mlr.press/v37/schaul15.html (検索結果から確認。PDF は未取得)
   - 支持する内容: 目標を入力に含む価値関数を、状態と目標の埋め込みの積として学び、未見の目標へ汎化させる。
   - 設計上の問い: 評価の「軸」を事前定義せず、埋め込みとして獲得させる場合の汎化の測り方。
   - 検証: secondary source only

3. Alexander, W. H., Gershman, S. J. (2022). Representation Learning with Reward Prediction Errors. Neurons, Behavior, Data Analysis, and Theory 1 (arXiv:2108.12402).
   - URL: https://arxiv.org/abs/2108.12402 (journal: https://nbdt.scholasticahq.com/article/37270)
   - 支持する内容: 報酬予測誤差が連合学習だけでなく、状態表現そのものの獲得を駆動しうるという主張。
   - 設計上の問い: 評価的特徴の獲得を、予測誤差 (信念の誤差) からどこまで導けるか。価値と信念の分離との関係。
   - 検証: secondary source only

4. Stachenfeld, K. L., Botvinick, M. M., Gershman, S. J. (2017). The hippocampus as a predictive map. Nature Neuroscience (preprint: bioRxiv 10.1101/097170).
   - URL: https://www.biorxiv.org/content/10.1101/097170v4.full (掲載誌の記載は検索結果の二次情報)
   - 支持する内容: 状態を将来の後継状態で表す successor representation が、報酬予測と神経基盤に結びつくという主張。
   - 設計上の問い: 経験から「次に何が起こるか」の表現を作ることで、評価特徴を派生させる設計の根拠。
   - 検証: secondary source only

5. Gershman, S. J. (2018). The Successor Representation: Its Computational Logic and Neural Substrates. Journal of Neuroscience 38(33):7193-7200. doi:10.1523/JNEUROSCI.0151-18.2018.
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6096039 (PMC の記録のみ確認。本文未読)
   - 支持する内容: successor representation を効率と柔軟性のトレードオフとして整理したレビュー。
   - 設計上の問い: 評価特徴を遷移表現と報酬の組合せとしてどの程度分解して持つか。
   - 検証: secondary source only

6. Oh, J., Hessel, M., Czarnecki, W. M., Xu, Z., van Hasselt, H., Singh, S., Silver, D. (2020). Discovering Reinforcement Learning Algorithms. NeurIPS 2020 (arXiv:2007.08794).
   - URL: https://arxiv.org/abs/2007.08794 (NeurIPS: https://proceedings.neurips.cc/paper/2020/hash/0b96d81f0494fde5428c7aea243c9157-Abstract.html)
   - 支持する内容: 予測対象と学習規則を含む更新則をメタ学習し、価値関数に代わる予測表現を発見させる研究。
   - 設計上の問い: 評価の枠組みそのものを開発者が固定せず、経験から発見させることの先例。ただし学習は玩具環境で行われる点に注意。
   - 検証: secondary source only (依頼文の「Mnih, Wayne, Kulkarni 2020」は誤りで、実際はこの論文に該当)

7. Zheng, Z., Oh, J., Hessel, M., Xu, Z., Kroiss, M., van Hasselt, H., Silver, D., Singh, S. (2020). What Can Learned Intrinsic Rewards Capture? ICML 2020.
   - URL: 検索結果のみ (論文ページの URL は未取得)
   - 支持する内容: 学習された内発的報酬が、どのような長期的価値の要素を捉えうるかの分析。
   - 設計上の問い: 報酬特徴を学習させた場合、何が「価値」として獲得されるのかの評価方法。
   - 検証: secondary source only

### T2: 非線形価値近似・特徴相互作用・比較方法論

8. Ghorbani, B., Mei, S., Misiakiewicz, T., Montanari, A. (2020). When Do Neural Networks Outperform Kernel Methods? NeurIPS 2020 (arXiv:2006.13409).
   - URL: https://arxiv.org/abs/2006.13409 (検索結果からの確認。本文未読)
   - 支持する内容: 入力分布が等方的でない、低次元構造がある場合に、ネットワークがカーネル法より有利になりうるという理論・実験。
   - 設計上の問い: NN が線形 Kalman モデルより有利になる条件 (データ構造の仮定) を明示できるか。
   - 検証: secondary source only

   T2 の注記: 依頼文にあった「Nagarajan, Liang, Lafferty 2021, learning curves ... interaction terms」は検索で見つからなかった (下記「見つからなかったもの」参照)。XOR 様の構造や深さの利点を直接扱う出典は今回の検索では確保できていない。

### T3: appraisal・情動・選好構成

9. Moors, A., Ellsworth, P. C., Scherer, K. R., Frijda, N. H. (2013). Appraisal Theories of Emotion: State of the Art and Future Development. Emotion Review 5(2):119-124. doi:10.1177/1754073912468165.
   - URL: https://journals.sagepub.com/doi/full/10.1177/1754073914534477 (旧 DOI 系の記載。冒頭部分のみ確認)
   - 支持する内容: 情動を環境の評価 (appraisal) に応じた適応反応とみなす立場の共通前提の整理。
   - 設計上の問い: 評価特徴を「appraisal 次元」として事前指定するか、経験から獲得させるか。
   - 検証: secondary source only (冒頭部のみ確認)

10. Wilson, T. D., Gilbert, D. T. (2005). Affective Forecasting: Knowing What to Want. Current Directions in Psychological Science 14(3):131-134.
   - URL: https://sites.lifesci.ucla.edu/psych-sanlab/wp-content/uploads/sites/31/2016/03/Wilson-Gilbert-2005-CDPsychSci.pdf
   - 支持する内容: 予測される感情反応の強度と持続の過大評価 (impact bias) と focalism の指摘。
   - 設計上の問い: 信念 (予測) と価値 (実際の感じ方) のずれを、エージェントが自分の内部で検出・補正できるか。
   - 検証: secondary source only (依頼文の venue「Advances in Experimental Social Psychology」は誤りで、2005年論文は CDPS。AESP 掲載は別の 2003 年レビューの可能性があり未確認)

11. Gross, J. J. (1998). The Emerging Field of Emotion Regulation: An Integrative Review. Review of General Psychology 2(3):271-299. doi:10.1037/1089-2680.2.3.271.
   - URL: https://pubmed.ncbi.nlm.nih.gov/12212647/ 系の書誌。本文は未取得。
   - 支持する内容: 感情調整を状況選択・状況修正・注意配分・認知的変化・反応調整の5点で整理するプロセスモデル。再評価 (cognitive change) の位置づけ。
   - 設計上の問い: 過去経験の再評価 (reappraisal) を、どの段階の処理の更新として実装するか。
   - 検証: secondary source only

12. Dayan, P., Niv, Y. (2008). Reinforcement learning: The good, the bad and the ugly. Current Opinion in Neurobiology 18(2):185-196. doi:10.1016/j.conb.2008.08.003.
   - URL: https://collaborate.princeton.edu/en/publications/reinforcement-learning-the-good-the-bad-and-the-ugly/
   - 支持する内容: 強化学習の定性・定量的枠組みのレビュー。価値と予測 (信念) を分けて扱う枠組みの前提の出典として使いやすい。
   - 設計上の問い: 「価値 (良い/悪い)」と「信念 (何が起こるか)」の分離を、RL のどの量に対応させるか。
   - 検証: secondary source only (摘要のみ)

13. De Houwer, J., Barnes-Holmes, D., Moors, A. (2013). What is learning? On the nature and merits of a functional definition of learning. Psychonomic Bulletin & Review 20(4):631-642. doi:10.3758/s13423-013-0386-3.
   - URL: https://pubmed.ncbi.nlm.nih.gov/23359420
   - 支持する内容: 学習を機能的に定義し直す議論 (ontogenetic adaptation)。「学習とは何か」の定義を、評価特徴の獲得の判定基準にどう使うかの参照点。
   - 設計上の問い: 「獲得された評価特徴」が学習の結果だと判定する機能的基準は何か。
   - 検証: secondary source only (依頼文の「evaluative conditioning review」という記述とは内容が異なる。評価的条件づけの専用レビューは未発見)

### T4: 学習表現の測定と介入

14. Hewitt, J., Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. EMNLP 2019 (arXiv:1909.03368).
   - URL: https://arxiv.org/abs/1909.03368
   - 支持する内容: ランダムラベルの control task を使い、プローブが表現の情報を読んでいるのか、プローブ自体が学習しているのかを切り分ける「選択性 (selectivity)」指標。
   - 設計上の問い: 「評価特徴が表現されている」と主張する前に、control task で読み出しの選択性を示すべきかどうか。
   - 検証: abstract only (WebFetch で確認)

15. Lampinen, A. K., Chan, S. C. Y., Hermann, K. (2024). Learned feature representations are biased by complexity, learning order, position, and more. TMLR (arXiv:2405.05847).
   - URL: https://arxiv.org/abs/2405.05847 (検索で著者に McClelland の名は見当たらなかった)
   - 支持する内容: 同程度に学習されても、単純な特徴や早く学習された特徴が強く表現される偏りがあり、解釈・介入手法もその偏りに影響される。
   - 設計上の問い: 「どの特徴が強いか」を、介入 (ablation/swap) の効果の大きさだけで評価してよいか。
   - 検証: abstract only (WebFetch で確認)

16. Geiger, A., Lu, H., Icard, T., Potts, C. (2021). Causal Abstractions of Neural Networks. NeurIPS 2021 (arXiv:2106.02997).
   - URL: https://arxiv.org/abs/2106.02997
   - 支持する内容: 神経表現を解釈可能な因果変数に対応づけ、interchange intervention で因果的役割を検証する方法。
   - 設計上の問い: NPN の内部の評価特徴が、swap 介入で期待どおりの判断変化を起こすかの検定。
   - 検証: secondary source only (検索要約)

17. Geiger, A., Wu, Z., Potts, C., Icard, T., Goodman, N. (2024). Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. CLeaR 2024, PMLR 236 (arXiv:2303.02536).
   - URL: https://arxiv.org/abs/2303.02536
   - 支持する内容: 分散表現に対して、勾配で因果変数との対応 (DAS) を探す方法。ニューロン単位ではなく部分空間で特徴を捉える。
   - 設計上の問い: 評価特徴が単一ユニットではなく部分空間に分散する場合の介入設計。
   - 検証: secondary source only

18. Bricken, T., Templeton, A., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.
   - URL: https://transformer-circuits.pub/2023/monosemantic-features/ (原典は未取得。二次資料の要約のみ)
   - 支持する内容: スパースオートエンコーダで、単一概念に対応しうる特徴を辞書学習として抽出する手法。
   - 設計上の問い: 獲得された評価特徴を、スパース辞書として事後的に取り出す検査の方法。
   - 検証: secondary source only (原典未読。数値は二次資料由来で未確認)

19. Locatello, F., Bauer, S., Lucic, M., Rätsch, G., Gelly, S., Schölkopf, B., Bachem, O. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. ICML 2019, PMLR 97 (arXiv:1811.12359).
   - URL: https://arxiv.org/abs/1811.12359
   - 支持する内容: 教師なしの disentanglement は帰納バイアスなしには識別不能で、分離度の向上が下流の sample complexity を下げるとは限らないという報告。
   - 設計上の問い: 「特徴が綺麗に分離されている」ことを成功基準にするのは不十分。下流の汎化で評価すべき。
   - 検証: secondary source only (検索結果の要約。best paper 受賞は二次資料の記述で未確認)

20. Lake, B. M., Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. ICML 2018, PMLR 80:2873-2882 (arXiv:1711.00350).
   - URL: https://arxiv.org/abs/1711.00350
   - 支持する内容: 既知要素の組合せでは汎化するが、新しい要素を体系的に組み合わせる汎化は失敗する、という SCAN での報告。
   - 設計上の問い: 「記憶ではなく転移する特徴」の検定として、要素の再組合せ課題を用いる設計。
   - 検証: secondary source only

### T5: 継続学習・安定性と可塑性・消去/再出現

22. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS 114(13):3521-3526 (arXiv:1612.00796).
   - URL: https://arxiv.org/abs/1612.00796 (PNAS: https://pmc.ncbi.nlm.nih.gov/articles/PMC5380101)
   - 支持する内容: 重要な重みの変化を弾性的に抑える elastic weight consolidation による忘却抑制。
   - 設計上の問い: 獲得された評価特徴を保持しつつ新経験で可塑的に更新する、重要度ベースの安定化の候補。
   - 検証: secondary source only (PNAS の書誌は検索結果で確認)

23. McClelland, J. L., McNaughton, B. L., O'Reilly, R. C. (1995). Complementary learning systems (Psychological Review)。書誌の題名と巻号は検索で未確認 (記憶ベース)。
   - URL: 検索結果 (ScienceDaily 等の要約) のみ。原典 URL は未確認。
   - 支持する内容: 新規経験の高速学習 (海馬) と構造化知識の緩徐学習 (新皮質) の二系統分離。
   - 設計上の問い: 評価特徴の緩徐な獲得層と、個別経験の高速な保持層を分けるか。
   - 検証: could not open (原典未読。要約のみ)

24. Kumaran, D., Hassabis, D., McClelland, J. L. (2016). What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated. Trends in Cognitive Sciences 20(7):512-534.
   - URL: https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf
   - 支持する内容: 海馬による経験のリプレイが、目標依存の重み付けを可能にする、という CLS の更新。
   - 設計上の問い: 評価特徴の再学習 (再評価) をリプレイで行うか。リプレイが過去経験の解釈を変える仕組みになりうるか。
   - 検証: secondary source only (PDF の URL は確認。本文未読)

25. Gershman, S. J., Blei, D. M., Niv, Y. (2010). Context, learning, and extinction. Psychological Review 117(1):197-209.
   - URL: https://www.cs.columbia.edu/~blei/papers/GershmanBleiNiv2010.pdf
   - 支持する内容: 文脈ごとの潜在原因推論により、消去が獲得記憶を消すのではなく新しい潜在原因として学ばれ、再出現 (renewal) が説明されるという計算モデル。
   - 設計上の問い: 消去後も獲得された評価を保持し、文脈による想起の切替えとして扱う枠組み。
   - 検証: secondary source only (PDF の URL を確認。本文未読)
   - 注: 依頼文の「Gershman, Radulescu, Norman, Niv 2017, context-dependent ...」は見つからなかった (下記参照)。

26. Bouton, M. E. (2004). Context and behavioral processes in extinction. Learning & Memory 11:485-494 (依頼文の書誌は検索で未確認)。
   - URL: 原典未取得。関連の二次資料 (PMC 2841965 等) のみ確認。
   - 支持する内容: 消去は文脈依存で、文脈の変化が元の反応の回復 (renewal, spontaneous recovery) を生む。
   - 設計上の問い: 文脈を明示的な入力とした想起の切替え設計。
   - 検証: could not open (原典未読。二次資料での引用のみ)

### T6: 価値学習・人格的選好・LLM 拡張

27. Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., Amodei, D. (2017). Deep Reinforcement Learning from Human Preferences. NeurIPS 2017 (arXiv:1706.03741; v4 2023-02).
   - URL: https://arxiv.org/abs/1706.03741
   - 支持する内容: 人間の選好比較から報酬モデルを学び、それで RL を行う方法。
   - 設計上の問い: 外部の評価者 (人間) からの選好を、内部の評価特徴の獲得にどう対応させるか。
   - 検証: secondary source only (arXiv 書誌は検索結果で確認。NeurIPS 掲載の記述は未確認)

28. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. arXiv:2203.02155.
   - URL: https://arxiv.org/abs/2203.02155
   - 支持する内容: 教師あり微調整、人間の順位付けによる報酬モデル、PPO による最適化、の3段階で意図追従を学習する手法。
   - 設計上の問い: 報酬モデルが固定の評価器として機能する点と、言語モデル本体が更新される点の分離。
   - 検証: secondary source only

29. Pathak, D., Agrawal, P., Efros, A. A., Darrell, T. (2017). Curiosity-driven Exploration by Self-supervised Prediction. ICML 2017, PMLR 70:2778-2787 (arXiv:1705.05363).
   - URL: https://arxiv.org/abs/1705.05363
   - 支持する内容: 自己の行動の結果の予測誤差を内発的報酬とし、その予測を学習された特徴空間で行う方法。
   - 設計上の問い: 予測誤差 (信念の誤差) を、特徴の学習に使う場合と、評価として使う場合の区別。
   - 検証: secondary source only

30. Ng, A. Y., Russell, S. (2000). Algorithms for Inverse Reinforcement Learning. ICML 2000, pp. 663-670.
   - URL: https://aima.eecs.berkeley.edu/~russell/papers/ml00-irl.pdf
   - 支持する内容: 観測された最適行動から報酬関数を推定する問題。報酬が一意に定まらない (degeneracy) ことの指摘。
   - 設計上の問い: 行動から評価特徴を推定する場合、評価の同定不能性をどう扱うか。
   - 検証: secondary source only (論文 PDF の URL は確認。DOI の記載は不整合のため不使用)

31. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. UIST 2023 (arXiv:2304.03442).
   - URL: https://arxiv.org/abs/2304.03442
   - 支持する内容: 経験を自然言語で記憶し、反省 (reflection) で上位の要約を作り、それを計画に戻す LLM エージェント構造。アブレーションで各要素の寄与を示す。
   - 設計上の問い: 外部モジュールが蓄積した経験・反省を LLM のプロンプトに注入する状態注入の先例。ただし評価特徴そのものは学習されない点に注意。
   - 検証: abstract only (WebFetch で確認。venue は UIST 2023 として検索結果から確認)

32. Chen, R., Arditi, A., Sleight, H., Evans, O., Lindsey, J. (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.
   - URL: https://arxiv.org/abs/2507.21509
   - 支持する内容: 活性化空間に人格特性に対応する方向があり、それで学習中の人格の変化を監視・制御できるという報告。
   - 設計上の問い: 「持続的な評価傾向」を LLM の内部方向として測るか、外部の評価器として分離するかの比較材料。
   - 検証: abstract only (WebFetch で確認。査読前の preprint)

### T7: 階層ベイズ・カルマン型モデル

33. Behrens, T. E. J., Woolrich, M. W., Walton, M. E., Rushworth, M. F. S. (2007). Learning the value of information in an uncertain world. Nature Neuroscience 10(9):1214-1221. doi:10.1038/nn1954.
   - URL: https://pubmed.ncbi.nlm.nih.gov/17676057/
   - 支持する内容: 人間が環境の変動性 (volatility) を推定し、それに応じて学習率を調整すること。ACC の信号と個人差の対応。
   - 設計上の問い: 学習率を固定せず、環境の変動の推定に応じて可塑性を変える設計の根拠。
   - 検証: secondary source only (PubMed の書誌と摘要の要約のみ)

34. Mathys, C., Daunizeau, J., Friston, K. J., Stephan, K. E. (2011). A Bayesian foundation for individual learning under uncertainty. Frontiers in Human Neuroscience 5:39.
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3096853
   - 支持する内容: 階層ガウス過程 (HGF) による個人差を含む学習モデル。更新式は RL に似ており、精度重み付き予測誤差を持つ。
   - 設計上の問い: カルマン型・HGF 型のベースラインが、固定された特徴空間の上でのみ動くことの限界の比較対象。
   - 検証: secondary source only (摘要のみ)

35. Dayan, P., Kakade, S., Montague, P. R. (2000). Learning and selective attention. Nature Neuroscience 3(S11):1218-1223. doi:10.1038/81504.
   - URL: https://homes.cs.washington.edu/~sham/papers/neuro/dkm00.pdf
   - 支持する内容: 不確実性と注意の統計的側面からの選択的注意の議論。
   - 設計上の問い: 注意による特徴選択を、固定の特徴空間ではなく不確実性から導く発想の出典。
   - 検証: secondary source only (依頼文の「Kalman filter/TD」の記述は検索結果で確認できず、未確認)

## 見つからなかったもの・未確認

- Mnih / Wayne / Kulkarni 2020 (依頼文): 該当論文は検索で見つからず。近い論文は #6 (Oh et al. 2020) で、著者が異なる。
- Singh 2010 (内発的報酬の発見): 見つからず。近い論文として #7 (Zheng et al. 2020) のみ確認。
- Gershman, Niv, Dayan 2023 (報酬特徴のベイズ学習): 見つからず。関連として Alexander & Gershman 2022 (#3) のみ。
- Nagarajan, Liang, Lafferty 2021 (NN 対線形、interaction terms): 見つからず。依頼文の書誌が誤っている可能性がある。NN 対線形の関連は #8 のみ。XOR 的構造や深さの利点は未確保。
- Gershman, Radulescu, Norman, Niv 2017 (文脈依存の消去/再出現): 見つからず。関連として #25 (Gershman, Blei, Niv 2010) のみ。
- Lampinen 論文の共著者 McClelland: 誤りで、実際の著者は Lampinen, Chan, Hermann (#15)。
- Ibeling: 2021 の Geiger et al. (#16) には含まれない。2023 版 (arXiv:2301.04709) に含まれる。今回は未収録。
- 評価的条件づけ (evaluative conditioning) の専用レビュー: 未発見。#13 は学習の機能的定義で、内容が一致しない。
- 持続的な「評価器」と差し替え可能な LLM の分離 (T6 の最後の論点): 今回の検索では直接扱う出典を確保できず。#31, #32 は関連の参照にとどまる。
- 固定特徴空間のカルマン型モデルへの批判 (T7): 専用の出典は未発見。#34 が HGF の枠組みを示すのみ。
- Bouton 2004 原典、McClelland 1995 原典、Bricken 2023 原典、Ng & Russell 原典 PDF の本文: 原典を開けず、二次資料での確認のみ。
- Dayan, Kakade, Montague 2000 が Kalman filter / TD を扱うかどうか: 未確認。
- 数値 (Medium 等の二次資料にあった相関係数など): 原典未確認のため本ファイルでは採用していない。

## 出典の質の注意

- 査読済み (peer-reviewed) の会議・雑誌: #1 (NeurIPS), #2 (ICML), #4-5 (Nature Neuroscience / J Neurosci), #8 (NeurIPS), #9 (Emotion Review), #11, #12, #13, #14 (EMNLP), #16 (NeurIPS), #17 (CLeaR), #19 (ICML), #20 (ICML), #22 (PNAS), #23-24 (Psych Rev / TiCS), #25 (Psych Rev), #27 (NeurIPS), #29 (ICML), #30 (ICML), #33 (Nat Neurosci), #34 (Front Hum Neurosci), #35 (Nat Neurosci)。
- 査読済みだが掲載の書誌が未完全: #3 (arXiv 版と Neural Theory 版が両方あり、版の差は未確認)。
- preprint / arXiv のみ: #15 (TMLR は査読誌だが、今回はarXiv版で確認), #28 (arXiv 2203.02155、NeurIPS 2022 掲載の記述は未確認), #31 (UIST 2023 とあるが arXiv 版で確認), #32 (arXiv のみ、査読前)。
- 原典未読で要約に依存: #18 (Transformer Circuits。数値は二次資料由来)、#6, #7, #8, #16, #17, #19, #20, #22, #24, #27 などの大半。
- 二次資料・書誌の不整合: #13 (依頼文の内容と不一致)、#10 (依頼文の venue と不一致)、#35 (Kalman の記述未確認)、#15 (依頼文の共著者誤り)。
- 検証レベル集計: abstract only 5件 (#1, #14, #15, #31, #32)、secondary source only 大半、could not open 2件 (#23, #26) 相当。全文を読んだ項目は 0 件。
- 本ファイルの書誌は、検索結果と WebFetch で確認できた範囲に限る。掲載年・巻号の細部は採用前に原典で確認すること。
