# Project frontalis — Phase 3-E

経験の意味・人格容量・長期可塑性の独立研究ラボ。自然文→意味抽出→NPN評価→自己学習→次の評価→言語化を実行できる。旧Phase、本体、実記憶、設定には書き込まない。

## 動かす

```powershell
./start-workbench.ps1
```

`http://127.0.0.1:8903/` を開く。既存の研究用Pythonを使用し、環境のANTHROPIC_API_KEYをサーバーのメモリだけで読む。画面では容量5段階、事実8／意味44、初期／形成済み、標準／探索方式を選べる。経験の解釈を確認し、NPN自身の評価から学習して前後を比較する。保存世代の復元とZIP出力もできる。画面操作は主実験と分けた探索用。

## 通信なしのデモと読み取り専用tool

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$labPython = '..\frontalis-phase3d-lab\.venv\Scripts\python.exe'
& $labPython -B examples/offline_demo.py --output examples/demo-new-1
Get-Content -Raw examples/tool-request.json | & $labPython -B tools/evaluate.py
& $labPython -B -m unittest discover -s tests -v
```

既存の出力世代は上書きしない。保存済みの4個体デモは `examples/demo-output/`、全研究用人格は `models/`、画面上の個体は `playground/states/` にある。

## 結果を読む

- [成果報告](REPORT.md)：主形成240、長期60、近似144、追加6条件。長期可塑性の主条件は未達。追加方式の成功と限界を別記。
- [46組の前後比較](reports/review.html)：実会話92発言。採点を保存・再開できる。正式な人間評価は未実施。
- [Claude Codeへの引き継ぎ](CLAUDE_HANDOFF.md)：再現手順、API、保存形式、未解決点。
- [事前計画](docs/01-protocol.md)、[ユーザー承認のHaiku追加群](docs/01a-haiku-fallback.md)、[事後追加実験](docs/01b-plasticity-followup.md)。

人格正本はNPNの数値パラメータ。LLMの感情自己申告は学習教師にしない。JSONの判断一致と本文の忠実性を区別する。旧成果6,618ファイルの保全と334研究状態の復元を最終検証する。詳細な証拠は `reports/final-verification.json`。
