"""Concise reports and standard plots, derived from completed immutable results."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def write(p,s):
    with (ROOT/p).open('x',encoding='utf8') as f:f.write(s)
n=read('reports/numerical-summary.json');sem=read('reports/semantics.json');perf=read('reports/performance.json');follow=read('reports/plasticity-followup.json');dialog=read('reports/dialogue-analysis.json')
cap=n['capacity'];plastic=n['plasticity'];colors={'raw8':'#8c6e53','semantic44':'#217765'}
fig,axes=plt.subplots(1,4,figsize=(16,3.7),layout='constrained')
for rep in ['raw8','semantic44']:
    rows=[r for r in cap if r['representation']==rep];x=[r['parameterCount'] for r in rows]
    axes[0].plot(x,[r['diagnosticMSE'] for r in rows],'o-',label=rep,color=colors[rep]);axes[1].plot(x,[r['meanEffectiveRank'] for r in rows],'o-',label=rep,color=colors[rep])
    cpu=[r for r in perf['conditions'] if r['representation']==rep and r['device']=='cpu'];axes[3].plot([r['parameterCount'] for r in cpu],[r['evaluateMedianMs'] for r in cpu],'o-',label=rep,color=colors[rep])
for i in [0,1,3]:axes[i].set_xscale('log');axes[i].set_xlabel('Trainable parameters');axes[i].grid(alpha=.2);axes[i].legend(fontsize=8)
axes[0].set_yscale('log');axes[0].set_title('Frozen task test MSE');axes[0].set_ylabel('MSE (lower is better)')
axes[1].set_title('Formed appraisal complexity');axes[1].set_ylabel('Covariance effective rank')
labels=[];values=[]
for rep in ['raw8','semantic44']:
    for method in ['signal','logit']:labels.append(rep+'\n'+method);values.append(float(np.mean([r[method+'Change'] for r in plastic if r['representation']==rep])))
axes[2].bar(range(4),values,color=['#c4a98e','#8c6e53','#84b6a6','#217765']);axes[2].set_xticks(range(4),labels,fontsize=8);axes[2].set_title('After 1,000 reinforcements');axes[2].set_ylabel('Cue change after 96 contrary facts');axes[2].set_ylim(0,2.15)
axes[3].set_title('Warm CPU array evaluation');axes[3].set_ylabel('1 sample, median ms')
for ext in ['png','svg']:fig.savefig(ROOT/'reports'/f'capacity-overview.{ext}',dpi=180)
plt.close(fig)

rich=[r for r in cap if r['representation']=='semantic44'];improvement=1-rich[3]['diagnosticMSE']/rich[0]['diagnosticMSE'];million=1-rich[4]['diagnosticMSE']/rich[3]['diagnosticMSE']
lines=['# frontalis Phase 3-E — 実装・実測結果','',
'**自然文の意味を受け取り、既存NPNが評価して自己学習し、次の経験への判断を変える経路を実装・実動確認した。意味を増やすと文脈による判断差が生じた。一方、豊かな入力の長期可塑性と、本文の忠実性には未達が残る。**','',
'## すぐ使う','',
'- 実行用画面：`http://127.0.0.1:8903/`。容量5段階、意味44／事実8、初期／形成済み、経験の解釈・自己学習・言語化・保存復元・ZIP出力。起動は `./start-workbench.ps1`。',
'- 通信なしのデモ：`examples/offline_demo.py`。保存済み4個体は `examples/demo-output/`。',
'- 人間用46比較：`reports/review.html`。実会話92発言。正式な人間採点は未実施。動作確認の採点は未採点へ戻した。',
'- 実装は `npne/`、読み取り専用toolは `tools/evaluate.py`、再現手順は `CLAUDE_HANDOFF.md`。本体・実記憶・既存設定は変更していない。','',
'## 完了した比較','',
'主形成240個体×48経験、学習off30対照、長期60個体×1,096経験、凍結関数近似144条件を完了。全主形成で有限・保存復元一致、別分枝保持、中立経験不変を確認。主形成の平均履歴間差は %.3f。初期56係数と評価軸・自己強化則は設計上の仮定である。'%n['gates']['meanHistoryDifferentiation'],'',
'| 入力 | 容量 | 未知関数MSE | 文脈感度 | 有効ランク |',
'|---|---:|---:|---:|---:|']
for r in cap:lines.append(f"| {r['representation']} | {r['parameterCount']:,} | {r['diagnosticMSE']:.6f} | {r['meanContextSensitivity']:.3f} | {r['meanEffectiveRank']:.3f} |")
lines+=['',f'意味44の534→10万でこの固定課題のMSEは{improvement:.1%}改善、10万→100万は{million:.1%}改善。意味44の線形ridgeは0.032462、二次ridgeは0.036417、検索は0.187777。10万NNは線形とほぼ同等。教師は44要素を使う合成関数なので、8軸側には設計上の情報欠落がある。容量の一般的優越や人格の良さの証明ではない。',
'','![容量と長期可塑性の比較](reports/capacity-overview.png)','',
'## 自然文からの学習','',
'32文は8話型×4表現。研究者作成の架空注釈との比較であり、32独立状況や人間の唯一の解釈ではない。','',
'| 抽出モデル | schema成功 | カテゴリ一致 | 8軸MAE | 事前基準 |','|---|---:|---:|---:|---|']
for r in sem['engines']:lines.append(f"| {r['engine']} | {r['schemaSuccess']:.1%} | {r['categoricalAccuracy']:.1%} | {r['facts8MAE']:.4f} | {'達成' if r['predeclaredSemanticSuccess'] else '未達'} |")
lines+=['','Haikuの追加はユーザー承認後の別群。OpenCodeのタイムアウト・tool使用・欠測を置き換えて成功扱いしていない。Haikuは15経験を全件学習できたが、正解意味で学んだ個体との未知入力判断差MAEは534版0.380、10万0.318、100万0.251。抽出精度の高さだけで、学習後の判断まで同一になるとは言えない。未学習の意味はheldout8件中3話型に限られ、他は未知の言い換え。','',
'## 長期可塑性：主実験は未達、追加で経路を切り分けた','',
f"主条件の『反対経験後>.05かつlogit法がsignal法より大きい』は{n['gates']['plasticityCriterionFraction']:.1%}で、事前80%基準は未達。raw8は15/15、意味44は1/15。raw8の大きな変化も、遠い同分枝への大きな波及を伴い、局所的な価値更新の成功とは言えない。",
'',f"事後に計画・凍結した結果評価の文脈分離は{sum(r['criterion'] for r in follow['conditions'])}/6で暫定基準を{'通過' if follow['success'] else '未達'}。cueの意味は保ち、結果評価の末尾36要素だけを中立化した。望ましい感情の教師は追加していない。関係性・意図を結果評価から除く制約があり、遠い判断への変化も残る。2容量・3seedだけの追加探索で、主実験の結果へ混ぜない。画面で探索方式として選択できる。",
'','## 言語と実装の限界','',
'主会話72/72、自然文接続20/24を取得。取得92件のJSON stance・候補選択は一致したが、MiMoには怒り信号0でも「怒りはある」と表す例、Nemotronには喜び信号0でも「嬉しい」と表す例がある。実行用画面の追加試験にも感情の付加があった。主会話の候補選択は24条件ともBで、候補分化や広い倫理判断の実証にはなっていない。本文の忠実性・自然さ・事実保持は正式な人間評価を待つ。詳細は `reports/failure-cases.json`。',
'','意味schemaは最大2出来事・固定行為分類・限定した関係性で、任意概念や長い経験列の一般的な意味理解は未実証。別分枝保持は構造による保証で、同分枝の無関係な価値保持は未解決。本体統合は別判断とする。','',
'## 実行コスト・検証','',
'| 意味44 | CPU評価1件ms | CPU更新ms | GPU更新ms | 重みbytes |','|---:|---:|---:|---:|---:|']
for c in range(5):
    a=next(r for r in perf['conditions'] if r['representation']=='semantic44' and r['capacityIndex']==c and r['device']=='cpu');b=next(r for r in perf['conditions'] if r['representation']=='semantic44' and r['capacityIndex']==c and r['device']=='cuda')
    lines.append(f"| {a['parameterCount']:,} | {a['evaluateMedianMs']:.3f} | {a['updateMedianMs']:.2f} | {b['updateMedianMs']:.2f} | {a['weightBytes']:,} |")
lines+=['','読み込み済み配列APIのwarm値で、tool起動・LLM待機は含まない。RSS/privateとCUDA allocatorの全記録は `reports/performance.json`。同GPU重みのCPU復元は全10条件でhash一致、最大出力差 %.3g。4回の別学習校正から1000回の軌道一致は主張しない。'%max(r['sameWeightMaxError'] for r in perf['calibration']),
'','新実装34テスト、独立コピーで旧94回帰テストが成功。事前凍結18ファイルの変更0、旧成果6,618ファイルの変更・欠落0。実画面で2経験の学習、言語化、前後世代の復元を確認し、人間評価画面の自動保存・JSON保存読込も確認した。再現用数値、重み、原応答、失敗、hash台帳を保存した。']
write('REPORT.md','\n'.join(lines)+'\n')

handoff='''# frontalis Phase 3-E — Claude Code引き継ぎ

## 成果物と境界

まずREPORT.mdを読む。実行用画面・NPNコード・保存世代・主240形成/60長期/144近似・92発言・46比較が実在する。実記憶、本体、旧ラボ、設定へ書かない。本体統合、公開、pushは今回の範囲外。ANTHROPIC_API_KEYの値を表示・ファイル化・引数化しない。Haiku 5.5はユーザーが再開時に明示承認した別の補完群であり、元の無料モデルの結果を改名していない。

## 使う

新ラボを作業場所にする。既存Pythonを読み取り利用し、旧環境へ依存を追加していない。

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$labPython = 'C:\\Users\\sakur\\AppData\\Local\\asist-shisa\\frontalis-phase3d-lab\\.venv\\Scripts\\python.exe'
& .\\start-workbench.ps1
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
'''
write('CLAUDE_HANDOFF.md',handoff)
print('Report, handoff, and comparison plots saved.')
