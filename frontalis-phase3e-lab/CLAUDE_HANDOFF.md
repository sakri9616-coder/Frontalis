# frontalis Phase 3-E — Claude Code引き継ぎ

## 成果物と境界

まずREPORT.mdを読む。実行用画面・NPNコード・保存世代・主240形成/60長期/144近似・92発言・46比較が実在する。実記憶、本体、旧ラボ、設定へ書かない。本体統合、公開、pushは今回の範囲外。ANTHROPIC_API_KEYの値を表示・ファイル化・引数化しない。Haiku 5.5はユーザーが再開時に明示承認した別の補完群であり、元の無料モデルの結果を改名していない。

## 使う

新ラボを作業場所にする。既存Pythonを読み取り利用し、旧環境へ依存を追加していない。

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$labPython = 'C:\Users\sakur\AppData\Local\asist-shisa\frontalis-phase3d-lab\.venv\Scripts\python.exe'
& .\start-workbench.ps1
# 同じポートで既に実行中なら http://127.0.0.1:8903/ を開く。
# 起動時の既存キーをメモリだけで使う。User環境変数しかない場合はstart-workbench.ps1が取得する。
& $labPython -B examples/offline_demo.py --output examples/demo-new-1
& $labPython -B -m unittest discover -s tests -v
& $labPython -B scripts/freeze.py verify
```

画面で作る個体はplayground/states/に独立保存される。既存モデルは読み取り複製。世代復元は古い保存をloadし、上書きしない。現在の世代と経験をZIP出力できる。画面の操作は事後の探索で、主研究へ混ぜない。意味の修正は事実だけを扱い、不確かな出来事や外部感情教師を学ばせない。標準方式と結果8軸分離の探索方式を選べる。

## API

- npne.meaning.validate/encode：事実の意味schemaと8/44要素の符号化。cueは事後情報を隠す。
- npne.core.Personality：CPU/GPU float64、evaluate/evaluate_batch/update/save/load。人格正本は重み。updateの教師は旧NPNの結果評価で、LLMの感情自己申告ではない。
- npne.learning.prepare_experience：fullが主方式。facts8-neutral-contextは事後6条件の方式。通常のcueは保持してobservedの文脈だけを中立化する。
- tools/evaluate.py：登録済みモデルID、meaning、options最大4。stdin JSON。学習、任意path、実行権限は公開しない。
- playground/server.py：127.0.0.1限定。Haikuで解釈と表出、NPNが評価と自己学習。stale hash/重複/未確認を拒否する。言語化ではパラメータを更新しない。

## 数値再現

完全な再学習は `scripts/reproduction_copy.py frontalis-phase3e-repro-1` で隣の新ディレクトリを作り、そこを作業場所にする。凍結入力とコードだけをコピーし、reports/models/playground/statesと、既存の会話計画・応答を持ち込まない。既存ディレクトリは上書きしない。追加探索を再現する場合は主集約後にplasticity_followup.py freezeを実行して、そのrunの主結果に対する追加計画を凍結する。

```powershell
& $labPython -B experiments/run.py main --device cuda
& $labPython -B experiments/run.py stress --device cuda
& $labPython -B experiments/run.py diagnostic --device cuda
& $labPython -B experiments/run.py performance
```

主環境はTorch2.7.0+cu128、NumPy2.5.4、float64、1thread、RTX3050。全条件でCPU/GPUを混ぜていない。キャッシュはpreregistrationSha256と結びつく。既存集約がある完了runでは再実行せず別コピーを作る。中断runは同じ装置・版・条件でcacheから継続できる。実行時はCUBLAS_WORKSPACE_CONFIGをコードで設定する。

LLMの原応答はreports/live。無料2モデルの再照合はFRONTALIS_VERIFY_ONLY=1を設定してexperiments/live.mjsを使う。Haikuはexperiments/haiku.mjsでextract/dialogue/pipelineを分け、検証のみなら同じverify-only設定を使う。APIキーはログにしない。新規取得をするなら同じ結果ファイルを消して再使用せず、別のrunへ保存する。

## 残る研究課題

主semantic44の長期可塑性は未達。結果評価の文脈分離は追加探索であり主成功へ遡及しない。遠い同分枝の判断が大きく変わる問題、抽出の小差が自己学習へ波及する問題、本文の感情追加、限定された意味schemaを残す。変更を試す場合は新条件と成功基準を先に固定する。人格を穏やか・道徳的に戻す教師は追加しない。

人間評価は未実施。reports/review.htmlだけを先に見て採点する。review-key.jsonとREPORT.mdを読んだ後は完全blindではない。ブラウザー動作試験の点数を正式採点に混ぜない。機械的JSON一致、AIの事後読解、人間採点を区別する。保全と最終証拠はreports/final-verification.json、全成果物のhashはartifact-sha256.json。
