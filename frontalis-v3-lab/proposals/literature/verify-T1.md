# T1 検証記録 (表現・報酬特徴の学習)

検証日: 2026-10-17 / 検証者: literature verifier (Haiku 5.5)
対象: LITERATURE-v4.md の T1 (#1-#7)
方法: 原典の URL を WebFetch で開いた。WebFetch は本文を要約器で処理するため、「全文を読んだ」とは断定しない。本文の一部だけ返った場合は「部分的に本文」と明記する。

検証レベルの表記:
- abstract only: 要旨・書誌ページのみ
- 部分的に本文: 本文の節・図表が返ったが、全文ではない (または切り詰めあり)
- could not open: 原典に到達できず

---

## #1 Barreto et al. 2017 (Successor Features)

- 訂正した書誌: Barreto, A., Dabney, W., Munos, R., Hunt, J. J., Schaul, T., van Hasselt, H., Silver, D. (2017). Successor Features for Transfer in Reinforcement Learning. NeurIPS (NIPS) 2017. arXiv:1606.05312 (v1 2016-06-16, v2 2018-04-12)
- 開いた URL: https://arxiv.org/abs/1606.05312 (PDF https://arxiv.org/pdf/1606.05312 は取得できず)
- 検証レベル: abstract only
- 論文の主張: 報酬関数だけが変わり力学は同じ課題間で、後継特徴により価値関数を再利用して転移させる。
- 具体的数値・設定: 未確認 (本文未読)。arXiv のコメント欄に "Published at NIPS 2017" と記載があることは確認した。
- 注意: 「報酬特徴を学習させるか固定するか」の点は要旨からは確認できない。collector の設計上の問いは未検証。

## #2 Schaul et al. 2015 (Universal Value Function Approximators)

- 書誌 (確認済): Schaul, T., Horgan, D., Gregor, K., Silver, D. (2015). Universal Value Function Approximators. ICML 2015, PMLR 37:1312-1320.
- 開いた URL: https://proceedings.mlr.press/v37/schaul15.html
- 検証レベル: abstract only (PMLR のページは要旨と書誌)
- 論文の主張: 目標を入力とする価値関数 V(s, g) を学び、状態と目標の埋め込みへの分解により未見の目標へ汎化できるとする。
- 具体的数値・設定: 本ページには環境名・数値なし。
- 注意: 「評価の軸を埋め込みとして獲得させる」点は、埋め込みが経験から学習されるという点で支持される。ただし目標は外部から与えられる入力であり、「エージェントが自ら評価特徴を獲得する」とまでは言えない。

## #3 Alexander & Gershman 2021/2022 (Representation Learning with Reward Prediction Errors)

- 訂正した書誌: Alexander, W. H., Gershman, S. J. (2021; v3 改訂 2022-07). Representation learning with reward prediction errors. arXiv:2108.12402 (q-bio.NC).
- 開いた URL: https://arxiv.org/abs/2108.12402
- 検証レベル: abstract only
- 論文の主張: 報酬予測誤差が連合学習だけでなく状態表現の獲得も駆動しうると主張し、時間・空間表現、運動反応、抽象的カテゴリ化などのシミュレーションを提示する。
- 具体的数値・設定: 本ページにはパラメータ・数値なし。
- 注意: collector の書誌 "Neurons, Behavior, Data Analysis, and Theory 1" は、このページでは確認できなかった (arXiv 版のみ記載)。掲載誌は未確認。

## #4 Stachenfeld, Botvinick, Gershman 2017 (The hippocampus as a predictive map)

- 書誌: Stachenfeld, K. L., Botvinick, M. M., Gershman, S. J. (2017). The hippocampus as a predictive map. bioRxiv 10.1101/097170 (v4)。
- 開いた URL: https://www.biorxiv.org/content/10.1101/097170v4.full
- 検証レベル: 部分的に本文 (本文 170,707 字のうち先頭 100,000 字まで読んだ。残りは未読)
- 掲載誌: ページには記載なし。collector の "Nature Neuroscience" は未確認。
- 論文の主張: 海馬の場所細胞は空間地図ではなく後継表現 (SR) による予測地図として理解できる。価値は報酬と SR の内積で求まり、報酬が変わっても再学習が不要。
- 具体的数値・設定 (Methods/Results 節より): 学習率 η = 0.1 (既定値)、SR は単位行列で初期化、割引率 γ が場の広がりを決める (図 S4, S6)。
- 注意: 「報酬の変化への転移」は確認。「未見の経験への汎化」は本文の読んだ範囲では主題ではない。collector の「経験から評価特徴を派生させる根拠」は、計算論的な主張として限定的に支持される。

## #5 Gershman 2018 (The Successor Representation: Its Computational Logic and Neural Substrates)

- 書誌 (確認済): Gershman, S. J. (2018). J Neurosci 38(33):7193-7200. doi:10.1523/JNEUROSCI.0151-18.2018 (単著)。
- 開いた URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6096039
- 検証レベル: 部分的に本文 (節ごとの内容が返ったが、全文の確認は行っていない)
- 論文の主張: SR は効率と柔軟性の中間に位置する。報酬変化は価値に素早く伝わるが、遷移の変化には SR の再学習が必要。
- 具体的設定: 3 状態迷路の図 (Figure 2)。Momennejad et al. (2017) の再価値づけ実験の図 (Figure 3) を引用。
- 注意: collector の「効率と柔軟性のトレードオフとして整理したレビュー」は支持される。

## #6 Oh et al. 2020 (Discovering Reinforcement Learning Algorithms)

- 訂正した書誌: Oh, J., Hessel, M., Czarnecki, W. M., Xu, Z., van Hasselt, H., Singh, S., Silver, D. (2020). Discovering Reinforcement Learning Algorithms. arXiv:2007.08794. 著者は collector の記載どおり。
- 開いた URL: https://arxiv.org/abs/2007.08794 (PDF https://arxiv.org/pdf/2007.08794 は取得できず)
- 検証レベル: abstract only
- 論文の主張: 予測対象と更新則を含む RL アルゴリズム全体をメタ学習し、Learned Policy Gradient (LPG) を得る。LPG は価値関数に代わる独自の予測を持つ。玩具環境だけで学習したものが Atari に汎化したと要旨に記載。
- 具体的数値・設定: 要旨に記載の環境は「玩具環境」「Atari ゲーム」のみ。数値なし。
- 注意: venue は本ページでは arXiv のみ記載。collector の "NeurIPS 2020" は本検証では未確認。

## #7 Zheng et al. 2020 (What Can Learned Intrinsic Rewards Capture?)

- 訂正した書誌: Zheng, Z., Oh, J., Hessel, M., Xu, Z., Kroiss, M., van Hasselt, H., Silver, D., Singh, S. (2020). What Can Learned Intrinsic Rewards Capture? ICML 2020, PMLR 119:11436-11446. arXiv:1912.05500.
- 開いた URL: https://proceedings.mlr.press/v119/zheng20b.html (書誌の確認) と https://ar5iv.arxiv.org/html/1912.05500 (本文 HTML)
- 検証レベル: 部分的に本文 (ar5iv の本文から節・図を確認。切り詰めの有無は不明)
- 論文の主張: 報酬関数そのものに学習された知識を持たせ、複数の生涯 (lifetime) にわたって内発的報酬をメタ勾配で学習する。学習された報酬は長期的な探索と活用の知識を捉え、新しい行動空間や別の学習アルゴリズム (Q 学習) にも転移する。
- 具体的設定 (本文より): Empty Rooms 環境 (Section 5.2, Figure 2a)。ゴールは各生涯内で固定かつ不可視、1 生涯 200 エピソード。Figure 3 は 30 シード平均。
- 注意: collector の「長期的価値の要素を捉えうる」は支持される。転移の主張 (新しい行動空間・Q 学習への転移) は本文から確認したが、設定は Empty Rooms の 1 種類のみ。

---

## 集計と collector の誤り

- abstract only: #1, #2, #3, #6
- 部分的に本文: #4, #5, #7
- full text read: 0 件。(WebFetch は要約器経由のため、全文を読んだと断定しない)
- could not open (PDF のみ): #1 と #6 の PDF 本文は取得できず。

書誌・主張の訂正点:
- #3: 掲載誌 "Neurons, Behavior, Data Analysis, and Theory" は未確認。arXiv 版のみ確認。
- #4: 掲載誌 "Nature Neuroscience" は本ページでは未確認。
- #6: venue "NeurIPS 2020" は本検証では未確認 (arXiv のみ)。
- #7: 元の URL 欄が未取得だったが、PMLR (v119/zheng20b) と arXiv (1912.05500) を確認。
- #1: venue NIPS 2017 を arXiv のコメント欄で確認。

設計上の注意: T1 の出典は「報酬特徴の固定 vs 学習」「経験からの評価特徴の獲得と未見経験への転移」を直接実験で示すものは、今回の範囲では確認できなかった。#2 (目標の埋め込み) と #7 (学習された報酬の転移) が最も近いが、いずれも課題設定が限定的である。
