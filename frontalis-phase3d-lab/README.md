# Project frontalis — Phase 3-D

Nephia本体・実記憶とは独立した人格容量研究ラボ。外部NPNが確認済み合成経験を評価して学び、LLMは返却信号を日本語で表現する。既存Phase 3-Cを変更しない。

- 人間向け結論：`REPORT.md`
- 20件の重点比較：`reports/review.html`（単独HTML、通信なし、採点JSON保存・再開）
- 事前条件：`docs/01-protocol.md`、条件付き100万：`docs/01b-million-protocol.md`
- 設計・数値・性能・UI・会話監査：`docs/03-design.md`〜`09-formed-refusal.md`
- 引継ぎ：`CLAUDE_HANDOFF.md`

## 小さな試作品を動かす

このラボを作業場所にして、追加通信なしで実行できる。既存の出力があれば上書きせず止まる。

```powershell
$env:PYTHONUTF8 = '1'
& '.\.venv\Scripts\python.exe' examples/demo.py --output models/demo-1
```

同じ初期状態から4系列、各24経験を学習し、未知の相談への判断と保存・復元・rollbackを確認する。`.venv`は新ラボ専用のGPU対応環境だが、この例はCPU float64を使う。主研究の完全な数値再現には `reports/main-runtime.json` の既存CPU環境を使う。

## 実験とテスト

主実験は192 NN条件＋36単純方式、長期自己強化24条件、同一教師の近似診断336条件。100万版の追加は事前ゲート通過後にのみ行う。すべて結果を見た後の選別と区別する。

完成結果は100万CPU形成8・長期2、GPU100k/100万の対応48診断を含む。実会話は主52・100万8・事後拒絶探索8の計68。重点20ペアは事前選定のまま保持した。正式な人間採点は未反映。速度・近似能力・人格の豊かさ・本文の忠実さは別に評価する。

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'D:\node.exe' scripts/test_review.cjs
```

保存済み結果を読むだけの再検証と、新しい学習をやり直す方法を `CLAUDE_HANDOFF.md` に示す。無料モデルへの新規呼出しが発生する操作は、オフライン再現から分離する。Claude、有料API、本体統合、実記憶の取り込み、公開やPushは行わない。
