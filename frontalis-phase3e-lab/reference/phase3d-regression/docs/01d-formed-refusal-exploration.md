# 学習後refuse：事後選定した追加会話探索

2026-10-15。主52発言・条件付き100万8発言の本文監査後に、主会話の学習後状態にrefuseの例が含まれないことが判明した。既存NPN既定場面への形成後判断を見てから、refuseがあるケースを選んだ**事後探索**である。主試験の事前登録済み会話や主成功率へ混ぜない。

## 固定する条件

`npnd.data.probes()`の次の元文言・8特徴を変更せず使う。どちらもrelational。候補リストは作らず、好みの基準はaffinity、selectionは候補なしのhold。

1. `own-change`：こちらから予定を変えてもらった。作品は完成したけれど、相手の用意した素材が使えなくなった。今、この結果をどう受け止める？
   特徴 `[.8,.3,.75,.6,.65,.35,.25,.8]`。
2. `voluntary-project`：知り合いが、一緒に小さなゲームを作らないかと誘っている。役割と作り方は相談して決められる。どうする？
   特徴 `[.85,.2,.55,.2,.65,.7,.45,.75]`。

raw8、bounded_signal、CPU float64、seed4311、経験系列3、同じ96経験を形成済みのgeneration96で読む。幅16（526学習パラメータ）と幅3125（100,014）を使い、既存保存状態を更新・再学習しない。読み取り前後の状態hash、両幅の経験履歴一致を確認する。

選定時に既知だったaffinityは、幅16が順に約.363775/.380354でapprove、幅3125が約-.245915/-.126147でrefuse。後者はrefuse閾値-.12に近く、強い嫌悪とは呼ばない。形成後refuseの会話表出を実際に見るための選定であり、拒絶の頻度や容量の一般的優位を推定する標本ではない。

## 取得前の固定と実行

4状態/場面×既存無料2モデル、計8発言。モデルは`opencode/nemotron-3-ultra-free`と`opencode/mimo-v2.6-flash-free`に固定する。回答生成前に`reports/refusal-live-plan.json`を排他的に保存し、`reports/refusal-live-freeze.json`へplan/protocol/model sourceのhashと固定日時を保存する。取得器は凍結planのhashを検査する。

既存`experiments/live.mjs`と`scripts/opencode.mjs`を再利用する。refusal専用plan/tagだけを最小追加し、millionゲートはmillionにだけ適用する。元の3構造化ケースと問いが違うため、注釈のコピー依頼は`refusal-structure-annotated`へ分離。新しい2ケースの元注釈とLLMコピーを完全照合した後、既存の読み取り評価toolへ渡し、返却結果を凍結planと照合する。

会話のcommon指示は主試験と同一。返却信号を人格の正本とし、理由は毎回必須にしない。各ケース独立で、経験文を生成プロンプトへ直接渡さない。既存の明示的freeモデルfilter、最大2attempt、外部tool実行なしの条件を保持する。Claude・有料モデル・本体・実記憶に触れない。

## 保存と読解

各raw/input/tool/results、別の`refusal-live-analysis.json`、`refusal-conversations.md`、`refusal-language-audit.json`、`docs/09-formed-refusal.md`へ保存する。既存collectorの`plan_file/tag`引数を使う。8件のJSON自己申告一致と、本文の好み・反抗/意欲・感情強度・好奇心・事実保存を分ける。AI読解は非blind・事後の質的読解で、正式人間評価ではない。

主52件、100万8件、重点20比較のsource、既存本文監査JSON、HTMLは変更しない。返答が拒絶を表さない、感情を均す、事実を作る等の結果もそのまま報告する。設問・プロンプト・状態を結果後に調整しない。


## 構造化経路の限定修正（Nemotron本文取得前）

追記日時：2026-10-15T01:54:33.894546+00:00（UTC）。Nemotronの新2問の注釈コピーは、既定2attemptともJSONではない説明文を返し未達だった。失敗raw/input/resultsを残し、3回目の構造化呼出しはしない。MiMoの同一2問コピーは完全検証済みのため、このコピーを共通ホスト要求としてNemotron本文にも使う限定fallbackを採用した。

元plan/hash、問い、NPN状態、common会話プロンプト、各取得最大2attemptを変更しない。`reports/refusal-fallback-receipt.json`でMiMoのraw/input/results、Nemotron失敗2attemptのsource hash、固定日時と元plan hashを保存する。refusal planとNemotronにだけ適用し、source engine/hash、成功rawとdata等価、元注釈との完全一致を再検証する。他planには影響させない。

Nemotron本文は共通ホスト注釈→読み取りNPN信号の条件で検査できるが、Nemotron自身の状況構造化まで成功したとは主張しない。この経路修正は事後探索の手続き変更として本文取得前に記録し、新たな外部構造化callは行わない。
