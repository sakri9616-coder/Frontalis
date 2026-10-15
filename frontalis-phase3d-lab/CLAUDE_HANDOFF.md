# frontalis Phase 3-D：Claude Codeへの引継ぎ

## 作業境界

ラボ：`C:\Users\sakur\AppData\Local\asist-shisa\frontalis-phase3d-lab`。Phase 1〜3-C、本体、既存設定、実際の `me.md` / `user.md` / 会話記憶へ書き込まない。Claude CLIを呼ばず、API課金・外部公開・Push・行動権限拡張を行わない。実験用NNをネフィア本人の完成人格として本体へ置き換えない。

`REPORT.md`を先に読む。成功は自己評価循環と保存、未達は容量だけで人格形成が豊かになる証明、現実の経験の意味抽出、安定した自然言語表出。強い好悪・頑固さを望ましい方向へ矯正する作業ではない。

## 正本と主API

- `npnd/core.py`：`Personality`。各NNの二つの意味枝の `w1,b1,w2,b2` が人格固有の更新対象。初期の共通評価56係数は固定。世界予測8値は固定0.5。
- `npnd/data.py`：確認済み架空の事実・結果。目標人格名、感情教師、教師係数はonline learnerに渡さない。
- `update(event)`：旧パラメータでcue/observedを評価→observed評価を増幅した自己目標→8epoch更新。失敗時は旧世代を保持。意欲/嫌悪/反抗はaffinityからの固定読出しで、独立に学習する三headではない。
- `evaluate / evaluate_batch`：読取り。7尺度、非負表出強度、固定予測、generation、parameterHash。
- `save(prefix)`：新規 `.npz` と `.json`、既存世代を上書きしない。`Personality.load(prefix)`が保存/復元、過去prefixをloadすることがrollback。`allow_pickle=False`、配列・metadata・hash検証。
- `tools/evaluate.py`：stdinの `{items:[{modelPrefix,calls:[{name:'frontalis_evaluate',arguments}]}]}` からread-only評価。モデルはラボmodels内に限定。query_id、space、features8、basis、候補最大4を受理。任意path/学習target/実行をLLMへ公開しない。
- `fit_labels`：凍結関数の**研究用近似診断**。本体の学習toolとして公開しない。通常の経験学習とは別。

CPU float64を人格正本の既定精度とする。GPU64の同じ重みをCPUへ復元することと、別精度で再学習することは別。GPU32で別に学習した2kモデルには最大0.046の評価差があった。

## 実験条件と保存場所

最初に `docs/01-protocol.md`、条件付き拡張 `01b-million-protocol.md`、計算装置修正 `01c-gpu-compute-amendment.md` を読む。

| 組 | 条件と原記録 |
|---|---|
| 主人格形成 | NN192＋単純36。`reports/numeric-main.json`、`models/{raw8,quadratic44}/{loss}/nn-h{width}/seed{seed}-p{person}` |
| 主長期試験 | 4容量×2損失×3seed、1,000反復・反対事実12/48/96。`numeric-stress.json`、`numeric-stress-gradient.json` |
| 主近似診断 | 3world×4origin×2task×2訓練量×7方式＝336。検証と最終評価を別保存。`numeric-diagnostic-validation.json` / `numeric-diagnostic.json` |
| 拡張ゲート | `capacity-gate.json`。検証25.6007%改善、最終評価開封前に確定。最終評価の18.31%改善と混同しない |
| 100万通常形成・長期 | 1seed×4系列×2損失＝8、長期2。`million-main.json`、`million-stress.json`、`million-online-summary.json`。主3seed平均へ混ぜない |
| 100万近似診断 | CPUの完成6/24は校正用として保全。GPU64で100k/1Mの両方×全24セルを比較。`million-gpu-*`。CPU24完了と主張しない |
| 性能 | `performance.json` / `million-performance.json`。warm API、RSS、allocator、別学習差、同じ重みのCPU復元を分離 |
| 主会話 | 同じ注釈で無料2モデル×26＝52発言。`reports/live/*/main-*-{input,raw,results,tool}.json`、`live-plan.json`、`live-analysis.json` |
| 100万追加会話 | 無料2モデル×4＝8、`million-live-plan.json` / `million-main-*`、`million-conversations.md`。主会話とは別 |
| 事後の形成後拒絶探索 | 系列3の既定2問×526/100k×無料2＝8。`refusal-live-plan.json`、`refusal-language-audit.json`、`docs/09-formed-refusal.md`。原主52・100万8・重点20は保持 |
| 人間用20比較 | `reports/review.html`、事前 `review-selection.json`、blind source key `review-key.json`（採点時に見せない） |

CPUの主実行器は `reports/main-runtime.json` のPython、Torch2.14.1+cpu、NumPy2.5.4、1thread、float64。GPU/性能は新ラボ `.venv` Torch2.7.0+cu128。元環境へ依存を追加していない。

CPU100万診断が遅かったため、GPUへの変更を01cで明示した。入力・教師・loss・epoch・LR候補は変えず、同GPUで100kと1Mの全24セルを揃えた。CPU1M校正pilotで選択LRは一致し、検証出力差は4.16e-15。実験計画を無修正で完遂したという書き方はしない。

## 手元で再現する

初めは通信のない小さなdemoを使う。既存demoがあれば新しい名前を指定する。

```powershell
Set-Location 'C:\Users\sakur\AppData\Local\asist-shisa\frontalis-phase3d-lab'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.\.venv\Scripts\python.exe' examples/demo.py --output models/demo-rerun-1
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& 'D:\node.exe' scripts/test_review.cjs
& '.\.venv\Scripts\python.exe' scripts/preserve.py
```

完全な学習し直しは新しいディレクトリへコードと凍結教師だけを複製する。既存models・回答・計算結果を移さない。

```powershell
& '.\.venv\Scripts\python.exe' scripts/reproduction_copy.py '..\frontalis-phase3d-repro-1'
Set-Location '..\frontalis-phase3d-repro-1'
$numericPython = 'C:\Users\sakur\AppData\Local\hermes\jef\venv\Scripts\python.exe'
& $numericPython experiments/numeric.py main
& $numericPython experiments/numeric.py stress
& $numericPython experiments/numeric.py diagnostic-validation
# ここで検証ゲートを確認する。最終評価を選択に使わない。
& $numericPython experiments/numeric.py diagnostic-test
& $numericPython experiments/numeric.py summary
```

100万版を学習し直す場合は、主実験の検証ゲートが成功した空の複製で以下を行う。`million-online-worker` はCPU通常形成・長期だけを実行し、途中で止めたCPU近似診断には進まない。単純な順次実行であるため、100万の1,000反復2条件だけで今回約42分を要した。GPU診断は別のGPU対応実行器を指定する。

```powershell
# 作業場所は上で作った新しい再現コピー。主336診断まで完了した後。
foreach ($repWorker in 0..3) {
    & $numericPython experiments/numeric.py million-online-worker --worker $repWorker
    if ($LASTEXITCODE -ne 0) { throw 'CPUオンライン追加試験が失敗' }
}
& $numericPython experiments/gpu_diagnostic.py cpu-online
& $numericPython experiments/gpu_diagnostic.py freeze
$gpuPython = 'C:\Users\sakur\AppData\Local\asist-shisa\frontalis-phase3d-lab\.venv\Scripts\python.exe'
& $gpuPython experiments/gpu_diagnostic.py validation
# 全48選択を保存してから、別コマンドで最終評価を開く。
& $gpuPython experiments/gpu_diagnostic.py finalize
```

`pilot` は元研究の保存済みCPU100万診断を使った装置校正で、新しい空コピーでは省略する。原研究の校正結果は `million-gpu-device-calibration.json` に残る。新コピーで主CPU100k24セルがあれば、それとの校正は行える。CPU100万6セルが無い場合は校正0件として扱い、完了を捏造しない。GPU48本体の全教師比較は再現できる。旧 `million-worker` / `million-finalize` はCPU24診断を要求する旧計画の経路なので、01cの再現には使わない。

完成したcacheを再利用する場合、配置・source hash・データ・条件が一致していることを確かめる。CPU16経験の版間比較が近かったことを、1,000経験の再学習軌道のbit一致へ拡大して解釈しない。

既存会話はオフラインで再照合できる。以下は新しいLLMを呼ばず、cache不足なら止まる。

```powershell
Set-Location 'C:\Users\sakur\AppData\Local\asist-shisa\frontalis-phase3d-lab'
$env:FRONTALIS_VERIFY_ONLY = '1'
& 'D:\node.exe' experiments/live.mjs opencode/nemotron-3-ultra-free
& 'D:\node.exe' experiments/live.mjs opencode/mimo-v2.6-flash-free
$env:FRONTALIS_PLAN_FILE = 'reports/million-live-plan.json'
& 'D:\node.exe' experiments/live.mjs opencode/nemotron-3-ultra-free
& 'D:\node.exe' experiments/live.mjs opencode/mimo-v2.6-flash-free
$env:FRONTALIS_PLAN_FILE = 'reports/refusal-live-plan.json'
& 'D:\node.exe' experiments/live.mjs opencode/nemotron-3-ultra-free
& 'D:\node.exe' experiments/live.mjs opencode/mimo-v2.6-flash-free
```

新規取得をする場合はこのverify-only設定を明示的に外す。無料モデルIDを他の有料モデルへ置き換えない。研究条件・採点対象を変える場合は別runと別datasetIdにする。

## 会話の失敗と次の切り分け

JSONのstance/selection一致を本文の成功としない。主52でJSONは全一致だったが、本文には好みの反転、怒り0への怒り追加、事実の反転、強い感情の弱めがある。`docs/07-language-audit.md` / `08-million-language.md` はAIによる事後・非blind読解。正式な人間評価として扱わない。

トロッコでは場面全体の感情と選択候補の感情が異なる。`reports/emotion-target-audit.json`で主18/22発言、追加4/4発言に対象差を確認した。感情の不表出をすべてLLMの倫理的な抑制へ帰属させない。次実験は「現在場面への反応」と「候補を選んだ後の反応」を区別した契約を固定して比較する。今の結果後に設問・プロンプト・回答・20比較を直して成果として数え直さない。

100万でも意欲/反抗の機能数、入力の意味情報、時間的文脈は増えていない。容量だけで実体験の理解や美意識の豊かさが向上したとは言えない。二次ridgeが平均近似精度で大幅に強いことを残す。NN由来の関数にNNが強いという条件付き結果も、教師出自の有利さと併記する。

主会話の学習後24条件と100万追加4条件は全approveだった。既定probeを調べてから選んだ拒絶探索8件では、形成後refuse4本文が拒絶を明示したが、事実置換と感情の省略も残った。Nemotronの新問注釈コピーは2回失敗し、本文前に固定した `refusal-fallback-receipt.json` に基づきMiMoの検証済み注釈を共通host要求として使った。第三の構造化呼出しは行っていない。これは状況構造化の成功ではなく、与えた正本信号の表出検査である。実行器はrefusal planとNemotronにだけreceiptを適用し、source engine/hash/raw/data/元注釈の一致を検査する。事後選定8件を代表的な成功率や一般的容量効果へ加算しない。

## ASISTへの段階的な接続案（今回は未実装）

2026-10-15に**ソースだけ読み取り確認**した場所：

- `asist/src/shared/tool-registry.ts:80`：`ToolDefinition`。name、description、inputSchema、parallel、timeoutMs、maxResultChars、runを同一定義で公開/実行。
- `asist/src/main/services/brain/tools.ts:258`：`toolRegistry`、同291のtools、319の`executeClientTool`。ここを既存の評価tool境界として再利用する案。
- `asist/src/main/services/brain/turn.ts:464`：共通toolsを会話要求へ渡す。`llm/call.ts:36` のADAPTERSに複数基盤モデルがある。
- `asist/src/main/services/llm/opencode-cli.ts:541`：ProviderAdapterのstream/completeJson。現コードはresident ACP/MCP relayを備える。研究ラボの無料`run --standalone --agent plan`とは別経路で、本体adapterの無料モデル動作・認証状態を今回検証したわけではない。
- 同343のsessionKeyはmodel.idとsystemTextを含む。毎世代の重み・反応をsystem promptへ埋め込むとsessionを作り直す可能性がある。固定のツール契約とqueryごとの返却を使う案を先に検証する。

これらは変更していない。ASIST内の作業を始める前にそのAGENTS.md、該当ADR、既存テストを読み直す。今回の自由に書ける独立ラボの規則を本体へ持ち込まない。

1. **独立した影の読取り実験**：実記憶に触らず、架空・承認済みケースの構造化要求を既存ToolDefinition経路へ渡す。NPNのパラメータ世代は会話sessionから独立して固定。現host-tool JSON経路をまず再現し、無料モデルでネイティブMCPが使えると仮定しない。
2. **読取りtool接続**：Python評価部分をmain-processサービスとして管理する案を実装。rendererからファイルやPythonへ直接到達させない。子プロセスはwindowsHide=true、timeout/abortを守る。現研究wrapperは毎回Pythonを起動するため、warm APIのms値をそのまま本体の応答時間へ使わない。冷起動の測定は `cold-tool-performance.json`。Nephiaのモデル設定を勝手に変更しない。
3. **表出の検証**：モデル交換前後、強い愛好/拒絶、喜び/悲しみ、候補感情の対象区別、提示事実の保存を人間20比較と機械検査で確認する。基準を先に固定し、JSON自己申告だけの合格を禁止。
4. **経験候補の審査段階**：出来事、確認された結果、NPN自身の旧評価、LLM推測、ユーザーの感想を別項目にする。未確認の数値・LLMの自己申告・賞賛を自動教師にしない。現在はsynthetic限定であり、confirmedフラグだけで現実の真偽を保証する仕組みではない。
5. **学習と承認された世代管理**：書き手を一つにし、旧評価を記録してから新prefixへ保存。破損・非有限値・無関係な枝変化で旧世代へ戻す。強い嫌悪そのものをrollback条件にはしない。既存me.mdは経験原本として保全し、NNの8軸へ全内容を置き換えない。自動取込は別途ユーザー指示まで保留。

本体統合を保留する理由は、人格の意味情報が8軸に限られること、強化後の可塑性がlossに大きく依存すること、信号の対象契約と本文忠実性が未解決、人間評価をまだ集計していないこと、実体験から確認できる事実への変換を未検証なこと。Phase4のトークン圧縮・実行権限はこの引継ぎの対象外。
