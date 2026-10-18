# T5・T7 出典検証 (2026-10-17)

検証レベル: full text read / abstract only / could not open のみ使用。検索要約しか得られなかったものは could not open とした。

| 項目 | 訂正後の書誌 | 開いたURL | 検証 | 要点 | 数値 (節) | 注意 |
|---|---|---|---|---|---|---|
| Kirkpatrick 2017 (#22) | arXiv:1612.00796。PNAS 114(13):3521-3526 は DOI 10.1073/pnas.1611835114 で整合 (掲載誌は未確認) | arxiv.org/abs/1612.00796 | abstract only | 旧タスクで重要な重みの学習を選択的に遅らせ忘却を抑える | なし | 要旨に "elastic weight consolidation" の語なし。消去・再出現は扱わない |
| McClelland, McNaughton, O'Reilly 1995 (#23) | Psychol Rev 102(3):419-457, DOI 10.1037/0033-295X.102.3.419, PMID 7624455 | Stanford PDF (読めず), PubMed (cookie 壁) | could not open | 海馬の高速記憶と新皮質の緩徐な干渉的学習 (検索要約のみ) | なし | 本文未読。書誌は検索結果由来 |
| Bouton 2004 (#26) | Learn Mem 11:485-494 (検索結果で確認) | doi.org/10.1101/lm.78804 → 403 | could not open | 消去は元の学習を残し文脈依存の新記憶を加える (検索要約) | なし | 本文未読。PMC4012323 は Bouton & Todd 2014 で別論文 |
| Gershman, Blei, Niv 2010 (#25) | Psychol Rev 117(1):197-209, DOI 10.1037/a0017808 | collaborate.princeton.edu (Columbia PDF は読めず) | abstract only (言い換え要旨) | 無限潜在原因のベイズ推論で再出現・潜在抑制を説明 | なし | 文脈依存の消去は潜在原因推論の枠組み。忘却抑制の論文ではない |
| Gershman, Jones, Norman, Monfils, Niv 2013 (依頼文 Science 誤り) | Front Behav Neurosci 7, 2013, DOI 10.3389/fnbeh.2013.00164 | frontiersin.org/.../fnbeh.2013.00164/full | full text read | 徐々に強化確率を下げる消去は、急な消去に比べ恐怖の再出現を防ぐ | 75匹のラット (Subjects) | 2021 年の訂正記事あり (検索結果のみ、未読)。ラットの恐怖条件づけで一般化には注意 |
| Gershman, Radulescu, Norman, Niv 2017 (依頼文) | 該当なし。近い論文は PLoS Comput Biol 2014, 10(11):e1003939, PMC4222636 | PMC は CAPTCHA で未取得 | could not open | 切替え型カルマン的モデルで、緩やかな変化は記憶更新、急な変化は新記憶 (検索要約) | なし | 2017 年の文脈依存消去の論文は未発見。文脈依存消去の論文ではない |
| Parisi et al. 2019 (新規リード) | Neural Netw 113:54-71 (arXiv:1802.07569 ほか。掲載誌名は未確認) | arxiv.org/abs/1802.07569 | abstract only | 非定常データで破局的忘却が起き、その抑制手法を概観 | なし | 要旨に安定性と可塑性の語なし |
| Dayan, Kakade, Montague 2000 (#35) | Nat Neurosci 3 Suppl:1218-1223 (検索要約で確認) | homes.cs.washington.edu PDF (読めず) | could not open | 選択的注意を統計・情報の観点から論じる (検索要約) | なし | カルマンフィルタへの言及は未確認。使用不可 |
| 固定特徴空間のカルマン型批判 (T7) | 未発見 | 検索のみ | 未確認 | 関連: 切替えカルマン (上記 2014)、Piray & Daw の volatile Kalman filter (検索要約のみ) | なし | 批判論文は見つからず |
| Gershman & Niv 2010/2012 潜在構造学習 | 未検証 | doi 10.1016/j.conb.2010.02.008 はリダイレクト先未取得 | could not open | なし | なし | 本文未確認 |

結論: 全文を読めたのは Gershman et al. 2013 のみ。依頼文の「Science」と「2017 文脈依存消去」は誤りで、採用前に修正が必要。
