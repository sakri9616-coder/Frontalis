# verify-T3: T3 文献の検証結果

検証日: 2026-10-17 / 検証者: literature verifier (Haiku 5.5)
方法: doi.org 経由で原典サイトを WebFetch。出版社サイトの多くは 403 で本文・抄録に届かず。書誌は Crossref API、抄録は Semantic Scholar / Crossref の抄録項目で補った。全文を読めた項目は 0 件。

検証レベル集計: full text read 0件 / abstract only 4件 (#10, #11 部分, B, C) / could not open 5件 (#9, #12, #13, A, D)

## 既存項目 (#9-#13)

### #9 Moors, Ellsworth, Scherer, Frijda 2013
- 書誌: 確認。Emotion Review 5(2):119-124, doi:10.1177/1754073912468165 (Crossref)。
- 開いたURL: https://journals.sagepub.com/doi/10.1177/1754073914534477 (403)。書誌のみ https://api.crossref.org/works/10.1177/1754073912468165
- 検証レベル: could not open (本文・抄録とも未取得)
- 要点: appraisal 理論の現状と今後を論じる論評。評価次元を事前指定するか経験から得るかの議論の参照点として使える見込み。
- 確認箇所: なし (書誌のみ)
- 注意: 元ファイルの「冒頭部分のみ確認」は裏付けられない。本文の確認を取り消す。URL のDOIは別番号で、対象論文は未確認。DOI形式のURLに差し替えるべき。

### #10 Wilson & Gilbert 2005
- 書誌: 確認。Current Directions in Psychological Science 14(3):131-134, doi:10.1111/j.0963-7214.2005.00355.x (Crossref)。CDPS が正しい。
- 開いたURL: 元PDF (lifesci.ucla.edu) はメタデータのみ取得。抄録は https://api.crossref.org/works/10.1111/j.0963-7214.2005.00355.x から確認。
- 検証レベル: abstract only
- 抄録引用: "People fail to anticipate how quickly they will cope psychologically." (11 words)
- 要点: 予測される感情の強さと持続が過大に見積もられる。信念 (予測) と実際の経験 (価値) のずれを示す根拠として使える。
- 確認箇所: 抄録 (節・頁は未確認)
- 注意: impact bias と focalism の記述は未確認。

### #11 Gross 1998
- 書誌: 確認。Review of General Psychology 2(3):271-299, doi:10.1037/1089-2680.2.3.271 (Crossref)。
- 開いたURL: https://doi.org/10.1037/1089-2680.2.3.271 (sagepub へ転送、403)。抄録は Crossref 経由。
- 検証レベル: abstract only (部分)
- 抄録引用: "The field of emotion regulation promises new insights into age-old questions" (11 words)
- 要点: 感情調整を統合的に整理したレビュー。再評価 (cognitive change) を調整方略の一つとして位置づける。
- 確認箇所: 抄録冒頭のみ (節・頁は未確認)
- 注意: 「5つの過程」の内容は抄録で確認できず未確認。元ファイルの PubMed URL は未取得。

### #12 Dayan & Niv 2008
- 書誌: 確認。Current Opinion in Neurobiology 18(2):185-196, doi:10.1016/j.conb.2008.08.003 (Crossref)。
- 開いたURL: https://doi.org/10.1016/j.conb.2008.08.003 (linkinghub.elsevier.com へ転送、403)。Crossref / Semantic Scholar とも抄録なし。
- 検証レベル: could not open
- 要点: 未確認。元ファイルの「価値と予測を分けて扱う枠組み」は本文未読のため未確認。
- 注意: 元ファイルのURL (princeton) は未確認。

### #13 De Houwer, Barnes-Holmes, Moors 2013
- 書誌: 確認。"What is learning? On the nature and merits of a functional definition of learning", Psychonomic Bulletin & Review 20(4):631-642, doi:10.3758/s13423-013-0386-3 (Crossref)。
- 開いたURL: https://doi.org/10.3758/s13423-013-0386-3 (link.springer.com 経由、ログイン画面へ転送で停止)。抄録なし。
- 検証レベル: could not open
- 要点: 学習の機能的定義を論じる論文。題名から、評価的条件づけのレビューではないことは確認できた。
- 注意: 元ファイルの「ontogenetic adaptation」の説明は未確認。評価的条件づけの出典としては使えない。

## 追加項目 (元ファイルに無い)

### A. Hofmann, De Houwer, Perugini, Baeyens, Crombez 2010 (評価的条件づけのメタ分析)
- 書誌: 確認。Psychological Bulletin 136(3):390-421, doi:10.1037/a0018916 (Crossref)。
- 開いたURL: https://doi.org/10.1037/a0018916 (doi.apa.org, psycnet.apa.org へ転送、本文表示なし)。Crossref / Semantic Scholar とも抄録なし。
- 検証レベル: could not open
- 要点: 刺激の対呈示による好みの変化 (評価的条件づけ) のメタ分析。学習された評価の典拠として最も直接的な候補。
- 注意: 検索結果の「認識度・自己報告で効果が強い」は二次資料の記述で、原典では未確認。

### B. Scherer 2009 (component process model)
- 書誌: 確認。Cognition and Emotion 23(7):1307-1351, doi:10.1080/02699930902928969 (Crossref)。単著。
- 開いたURL: https://doi.org/10.1080/02699930902928969 (tandfonline へ転送、403)。抄録は Semantic Scholar 経由。
- 検証レベル: abstract only (S2 経由。出版社の抄録とは照合していない)
- 要点: 情動を主観的評価に基づく創発的で動的な過程と捉え、評価チェックの順序と反応系の同期を強調する。
- 注意: 評価を経験で更新する点は抄録に明示されていない。

### C. Lichtenstein & Slovic 2006 (構成的選好)
- 書誌: 確認。The Construction of Preference (編著), Cambridge University Press 2006, doi:10.1017/CBO9780511618031。1971年の原論文は未確認。
- 開いたURL: https://www.cambridge.org/core/product/identifier/9780511618031/type/book
- 検証レベル: abstract only (出版社の書籍説明文のみ)
- 抄録引用: "One of the main themes that has emerged from behavioral decision research during the past" (15 words)
- 要点: 選好は、尋ね方や提示方法によって提示の時点で構成されることが多いという主張。選好を固定の評価器とみなす前提への批判材料。
- 注意: 章単位の検証は未実施。書籍説明文に基づく。

### D. Gross 2015 (感情調整の現状)
- 書誌: 確認。Psychological Inquiry 26(1):1-26, doi:10.1080/1047840X.2014.940781 (Crossref)。
- 開いたURL: https://doi.org/10.1080/1047840X.2014.940781 (tandfonline へ転送、403)。Crossref / Semantic Scholar とも抄録なし。
- 検証レベル: could not open
- 要点: 検索要約では、拡張過程モデルで調整を「価値付け」の一種として扱うとある。原典未読のため未確認。
- 注意: 検索要約の内容は二次情報。

## 元ファイルへの訂正案
1. #9: 「冒頭部分のみ確認」を「書誌のみ、本文未取得」に修正。URL をDOI形式に。
2. #10, #11: 検証レベルを secondary source only から abstract only に。
3. #13: 内容記述 (ontogenetic adaptation) は未確認と明記。
4. 「情動的条件づけ専用の出典は未発見」は、A (Hofmann 2010) を候補として追加できる。ただし原典は未読。
5. 検証レベル集計の「全文を読んだ項目は 0 件」は本検証でも変わらない。
