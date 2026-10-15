"""Human-facing artifacts derived from immutable primary and additional results."""
import html,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def fmt(x):return f'{x:.4f}'
def main():
    n=read('reports/numerical-analysis.json');e=read('reports/extended-analysis.json');s=read('reports/semantic-analysis.json');c=read('reports/conversation-analysis.json')
    perf=read('reports/performance-repeat.json');x=np.asarray([r['parameters'] for r in n['capacities']]);caps=['~500','~2k','~10k','~100k','~1M']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    a=ax[0,0];p0=[np.mean([r['pastDifferenceLong'] for r in n['historyGroups'] if r['capacity']==cap and r['profile']==0]) for cap in range(5)]
    dual=[np.mean([r['pastDifferenceLong'] for r in e['dual']['groups'] if r['capacity']==cap]) for cap in range(5)]
    a.plot(x,p0,'o-',label='Observed (primary P0)');a.plot(x,dual,'o-',label='Dual view (post hoc P0)');a.axhline(.01,color='gray',ls=':',label='Fixed criterion');a.set_xscale('log');a.set_title('Same present, different past (3 seeds)');a.set_ylabel('Long-term appraisal RMS difference');a.legend(fontsize=8)
    a=ax[0,1];m=read('reports/mechanisms.json')['rows']
    for mode,color in [('idle','#298b72'),('rumination','#dd8844')]:
      for field,style in [('short','-'),('mood','--')]:
        rs=[r for r in m if r['mode']==mode];v=np.mean([[p[field] for p in r['timeline']] for r in rs],axis=0);a.plot(np.arange(1,33),v,style,color=color,label=mode+' / '+field)
    a.set_title('Affect persistence and reactivation');a.set_xlabel('Information updates');a.set_ylabel('Seven-channel norm');a.legend(fontsize=8)
    a=ax[0,2];modes=n['decay'];a.bar([0,1,2],[v['retention'] for v in modes],color=['#5c88a4','#9bb7b9','#2b9777']);a.set_xticks([0,1,2],[v['mode'] for v in modes],rotation=10);a.set_ylim(0,1.1);a.set_title('Retention after 64 blank updates');a.set_ylabel('Distance from initial / formed distance')
    a=ax[1,0]
    for kind in ['npn','linear','quadratic','retrieval']:
        rs=[r for r in n['diagnostic'] if r['kind']==kind]
        if kind=='npn':a.plot(x,[v['testMSE'] for v in rs],'o-',label='RBF NPN')
        else:a.axhline(rs[0]['testMSE'],ls='--',color={'linear':'#499a73','quadratic':'#c79440','retrieval':'#a878aa'}[kind],label=kind)
    a.set_xscale('log');a.set_title('Fixed-teacher held-out diagnostic');a.set_ylabel('MSE (lower for this diagnostic only)');a.legend(fontsize=8)
    a=ax[1,1]
    for device in ['cpu','cuda']:
        r=[r for r in perf['rows'] if r['device']==device];a.plot(x,[v['updateMs'] for v in r],'o-',label=device)
    a.set_xscale('log');a.set_title('Update cost including parameter hashing');a.set_ylabel('Milliseconds');a.legend()
    a=ax[1,2];initial=n['capacities'][0]['parameters'];future=read('reports/future-choice.json')['metrics'];first=[r for r in future if r['capacity']==0];a.bar(np.arange(4),[r['candidateValues'][0]-r['candidateValues'][1] for r in first],color=['#98a4ae','#2b9777','#cf7a50','#2b9777']);a.axhline(0,color='gray',lw=1);a.set_xticks(range(4),['Initial\n(exact tie)','Cooperation','Misbelief','Corrected'],rotation=8);a.set_title('Unknown outcome choice (post hoc, one seed)');a.set_ylabel('Friend value minus new-person value')
    for a in [ax[0,0],ax[1,0],ax[1,1]]:a.set_xticks(x,caps);a.set_xlabel('Reserved trainable parameters')
    fig.suptitle('Project frontalis 3-F — primary failures and additional findings',fontsize=15)
    fig.savefig(ROOT/'reports/results.png',dpi=170);fig.savefig(ROOT/'reports/results.svg');plt.close(fig)
    tests=(ROOT/'reports/tests-complete.txt').read_text(encoding='utf8');assert 'Ran 33 tests' in tests and '\nOK' in tests
    lines=['# Project frontalis — Phase 3-F 成果報告','',
      '時間・一時感情・気分・長期人格・経験記録を分けた研究実装と、5容量の再現実験を作成した。主条件では時間分離と学び直しが成立した一方、同じ現在への長期的な過去差は事前基準を満たさなかった。独立した二視点方式で改善したが、容量に伴う人格の複雑化は今回の系列では実証できていない。',
      '', '![数値結果](reports/results.png)','',
      '## 使える成果物','',
      '- [ワークベンチ起動](start-workbench.ps1) → http://127.0.0.1:8905/ 。経験の反復、時間経過、反すう、訂正、自然文の型付き抽出、確認後の学習、言語化、人格ZIP保存。主方式と二視点の追加方式を明示して選べる。',
      '- [NPN実装](npnf/core.py)、[二視点方式](npnf/dual_view.py)、[独立評価ツール](tools/evaluate.py)、[二視点ツール](tools/evaluate_dual.py)、[API不要デモ](examples/offline_demo.py)。',
      '- [主数値結果](reports/numerical-analysis.json)、[追加結果](reports/extended-analysis.json)、[16組の対話型比較](reports/dashboard.html)、[主会話72件](reports/conversation-analysis.json)、[追加会話80件](reports/additional-conversations.json)、[実ツール往復10件](reports/actual-tool-calls.json)。',
      '- [主会話の匿名評価44対](reports/review.html) / [追加会話40対](reports/additional-review.html)。正式な人間採点は未実施。機械の点数を人間評価へ転記していない。',
      '- [原方式の6段階会話](reports/workbench-conversation.json)、[二視点の6段階会話](reports/workbench-conversation-dual.json)、[保存した二視点人格](reports/workbench-live-dual-personality.zip)、[Claude Code引継ぎ](CLAUDE_HANDOFF.md)。',
      '', '## 主条件と独立した追加条件','',
      '[事前計画](PREREGISTRATION.md)と18ソースを最初の本数値実験前に固定。seedは6311/6312/6313。4つの数値初期傾向を用い、人物名やその説明を言語化LLMに与えない。言語入力の実験IDには状態名が含まれたため、状態名からの推測の影響を完全には除けていない。',
      '', '| 主条件の判定 | 結果 |','|---|---|',
      f"| 16更新後の短期感情 < 直後の1% | {n['criteria']['shortPassed']}/240 |",
      f"| 同時点の気分 < 直後の50% | {n['criteria']['moodPassed']}/240 |",
      f"| 長期の形成効果を50%以上保持 | {n['criteria']['retentionPassed']}/240 |",
      f"| 同じ現在への長期的な過去差 >0.01 | {n['criteria']['historyPassed']}/60群、失敗 |",
      f"| 反復後64経験で変化>0.01かつ無関係/関連<0.5 | {n['criteria']['plasticityPassed']}/60 |",
      '| 学習なし / 一時感情なし対照 | 重み不変 / 一時状態ゼロ |','',
      f"16更新後の短期ノルム比は平均 {n['capacities'][0]['shortRatio16']:.3g}、気分 {n['capacities'][0]['moodRatio16']:.3f}、長期保持 {n['capacities'][0]['longTermRetention16']:.3f}。これは独自更新則の作動確認であり、人間の実時間への適合を示すものではない。",'',
      '主時系列は240の過去系列から現在4を分岐し960枝、逆順も960枝。減衰探索135検証条件と45テスト系列、反復ストレス60、反すう等75、対照30、固定教師診断24手法条件。逆順比較には直近の出来事と空白位置の効果も含む。独立な人間被験者数のように数えない。',
      '', '## 二視点方式と未来の選択（事後探索）','',
      f"手掛かりへの学習効果は大きかったが、結果・因果も含む観測とは入力空間の距離が離れていた。cueとobservedのlogitを固定で半分ずつ用いる別構造では、P0・3seed・5容量の {e['dual']['passed']}/{e['dual']['total']}群で過去差基準を超え、平均差は {e['dual']['meanPastDifference']:.4f}。60系列の学習重みは元構造の対照と一致した。改善は同じ学習パラメータを読み出す構造の効果で、追加の人格容量の効果ではない。新しい評価を一時感情へ反映し、識別した構造付き保存・復元も実装した。",'',
      '結果が未知で相手だけが異なる候補では、この固定seedの全5容量で、協力後は友人、誤解後は未知の相手、訂正後は友人が選ばれた。初期の友人選択は同値時の先頭規則であり好意の証拠ではない。誤解時の候補差は約0.0048と小さい。反応の望ましさを採点せず、この探索を主条件の成功へ置換しない。',
      '', '## 容量と計算効率','', '| 予約した学習パラメータ | 主系列の使用数 | 未知固定教師MSE | CPU更新ms | GPU更新ms |','|---:|---:|---:|---:|---:|']
    for cap in range(5):
        d=next(r for r in n['diagnostic'] if r['kind']=='npn' and r['capacity']==cap);cpu=next(r for r in perf['rows'] if r['device']=='cpu' and r['capacity']==cap);gpu=next(r for r in perf['rows'] if r['device']=='cuda' and r['capacity']==cap)
        lines.append(f"| {n['capacities'][cap]['parameters']:,} | {n['capacities'][cap]['activeParametersMean']:.0f} | {d['testMSE']:.4f} | {cpu['updateMs']:.2f} | {gpu['updateMs']:.2f} |")
    lines += ['', '主系列は実質2中心しか使わず、5容量で判断が一致した。固定教師診断でも10万/100万は128中心・6,528パラメータの使用に留まり、高次元未知入力への転移が弱い。線形ridgeのMSEは0.1592、二次0.1966、検索0.3779で、今回のRBFは線形を上回らない。大容量の複雑化を実証したとは言えず、軽量版が一般に優れているという結論にもならない。',
      '', '固定初期写像56、占有ビット、定着トレース、感情状態、経験記録を別計上。主測定と独立コピーでCPU/GPU性能を測った。値はローカルで併走処理の影響を含む参考値で、LLM待ちを除く。RSS/privateはプロセス全体、CUDA割当/予約はPyTorch allocator。CPU/GPUの同一重み・4更新の比較は最大誤差約2.8e-17。詳細は[測定値](reports/performance-repeat.json)。',
      '', '## 意味・ツール・言語を分けた結果','', '| 経路 | スキーマ利用可能 | 8軸MAE（欠測も含む） |','|---|---:|---:|']
    for row in s['rows']:lines.append(f"| {row['engine']} 主条件 | {row['valid']}/16 | {row['facts8MAE']:.4f} |")
    for row in e['losslessFormatDiagnosis'][:2]:lines.append(f"| {row['engine']} 軸辞書の無損失変換後 | {row['valid']}/16 | {row['facts8MAE']:.4f} |")
    row=e['strictSchema'];lines.append(f"| Haiku JSON Schema追加 | {row['valid']}/16 | {row['facts8MAE']:.4f} |")
    lines += ['', '主条件の型失敗には契約の配列指定不足が含まれた。無損失変換は値を補わず、未知を0とする等の意味誤差は残す。JSON Schema条件は否定2/2、事実と解釈の分離2/2、非欠測のカテゴリ精度0.8922。単独著者の合成goldで、二重注釈済みの人間の正解ではない。ローカルQwenは短いreadyに応答したが、本条件ではHTTP400のterminated/engine protocol fetch failedと不完全出力があり、生成上限追加条件でも学習可能な抽出を得られなかった。意味能力の優劣と断定しない。',
      '', '全正解16件とLLM経由の比較に加え、欠測時を両側で空白にした[利用可能数を揃えた比較](reports/availability-matched.json)で、輸送欠測と意味誤差を分離。利用可能0は意味MAEをNAとした。',
      '', f"実際のHaiku→NPN→Haiku往復は、10/10で要求を正しくコピーし、保存モデルと返却信号が一致、最終の態度・選択も10/10一致。ここにはgold意味を直接与え、自然文理解とは独立して測った。元のツール境界24ケースも検証。",'',
      '発言は主会話72件、未来選択40件、二視点40件、実ツール10件、2方式の各6段階会話を保存。追加会話では選択一致40/40ずつでも、態度一致は未来Haiku16/20・DeepSeek15/20、二視点Haiku15/20・DeepSeek9/20だった。JSONや選択の一致だけで本文の忠実性を保証しない。',
      '', '主72件の匿名LLM補助点検では感情付加1件、省略7件、反転0件を検出した。これはHaikuによる事後点検で、自己評価を含み、人間評価ではない。自然さ・事実保持も独立した機械尺度を保存。[失敗事例](reports/failure-cases.json)と未採点の人間用資料を用意した。',
      '', '## 文献・保全・限界','',
      '[心理学文献と設計仮説](LITERATURE.md)に持続、慣性、反すう、消去、調整、多言語の原著出典を記載。減衰率0.45/0.9や定着更新則は独自仮説。文脈差は観測したが、人間の消去・自然回復、感情体験、普遍的7感情を再現したとは主張しない。',
      '', 'ASISTの日記は雛形・検索単位・保存/curationソースのみ調査。実日記、実記憶、設定、ネフィア本体への統合は行わない。3-Eの固定2,252ファイルと従来6,618ファイルを終了時に照合する。詳細は[最終検証](reports/final-verification.json)。',
      '', '新規33テスト、既存3-E34テスト、3-D94テストを実行。新規コピーで5容量の学習・16更新・訂正・保存復元を再現（全数値系列の独立再実行ではない）。数値発散と保存不整合を検査し、個性の極端さ自体は不良としない。極端化の十分な再現、広い意味転移、大容量の複雑化、正式な人間採点は残る課題。',
      '', '## 再実行','',
      '元の成果フォルダへ上書きしない。`scripts/reproduction_copy.py <新しい名前>`で隣接する空の再現コピーを作り、そこで`experiments/run.py all`を実行する。固定入力と事前計画はコピー済みでprepare/freezeの再実行は不要。追加実験はreference内の各freezeを参照。API呼出しは再現に必須ではなく、保存済みraw応答と要求ハッシュから集計できる。詳しいコマンドと注意点は引継ぎへ。','']
    lines += ['基礎の8軸写像は固定で、経験はRBF残差を変える。基礎価値全体が自由に変わる実証ではない。時間リンクは記録・参照検査に使うが、NPNがその参照先を注意機構で読む構造ではない。人物個別の信頼も扱っていない。継承した44次元では同じ事実軸ならinstitutionとstrangerの符号化が一致する場合があり、容量を増しても失われた区別は復元できない。局所性の合格は固定した無関係プローブと音響領域の範囲であり、広い類似場面への保持を保証しない。DeepSeekは形式変換後の8軸誤差が小さい一方、事実と解釈の分離は0/2だった。','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    (ROOT/'CLAUDE_HANDOFF.md').write_text('''# Phase 3-F handoff

## Start here

This is an isolated research lab. Read REPORT.md, PREREGISTRATION.md, reports/final-verification.json, then the relevant code. Preserve Phase 3-E, all earlier labs, ASIST, Nephia, real memory, and settings. No body integration or Git publishing has been performed. The user authorized continued API use for this Phase 3-F work; credentials are environment-only. Prefer artifacts and runnable validation over additional prose.

## Implemented

- npnf/core.py: manually learned RBF centers and normalized LMS heads, two domains, numerical seven-axis appraisal, information-update clock, transient affect, mood, coefficient forgetting, consolidation trace, append-only episodes with facts versus interpretation and correction/recall links. No LLM sentiment is a training target. Long-term inference does not consult episodes.
- npnf/dual_view.py: separately identified post hoc architecture with equal cue/observed logit weighting. New affect uses that appraisal. Underlying learned weights match the observed-only control; architecture is included in the personality hash. A .view.json is REQUIRED for dual snapshots.
- npnf/transport.py: measured JSON schema and exact eight-axis lossless object-to-array conversion. Never fill missing facts. tools/evaluate.py and tools/evaluate_dual.py are JSON stdin/stdout read-only boundaries with model-ID allowlists.
- playground/server.py + index.html, local port 8905: editable interpretation, explicit confirmation, time/recall/correction, initial tendencies, both architectures, language, immutable snapshots, ZIP exports, old-session restoration, CSRF, save-before-RAM commit. start-workbench.ps1 starts it using the existing runtime.

## Evidence and limitations

Primary sources were frozen before numerical runs (reference/preregistration.json). Every added condition has a distinct reference/*-freeze.json and explanation. Do not change frozen files to improve results.
Primary history criterion FAILED 0/60, while temporal separation passed 240/240 and plasticity/locality passed 60/60. Dual-view P0 additional criterion passed 15/15; do not merge this with the primary failure. Main history uses only 102 active parameters at every reserved capacity, so high-capacity complexity is NOT demonstrated. Fixed-teacher RBF generalization is worse than linear ridge. Structural comparison with 3-E changes both architecture and learning, so it is not an isolated architecture effect; its sourceHash refers to the imported 3-E preregistration, and structure-freeze.json binds the added code.
Original extraction had array/object ambiguity. Retain strict primary failures, lossless diagnosis, and schema-enforced results separately. Qwen engine errors are not proof of poor semantic capability. Actual NPN tool calls used gold meaning, so they test tool execution separately. The actual-tool verbalizer received structured meaning, not the original narrative; do not interpret missing narrative details as facts it was given.
Primary 72 utterances and additional 80 were generated, plus 10 real tool-mediated utterances and two six-stage workbench conversations. LLM audit is NOT formal human evaluation. Human forms have no human results. Experimental IDs contained state names; this may cue the verbalizer. Use opaque IDs in a future cleaner replication. Initial future choice is an exact tie resolved by input order. Some later margins are small. Do not claim robust human-like forgiveness or universal emotional semantics.

## Runtime and reproduction

Existing Python: ../frontalis-phase3d-lab/.venv/Scripts/python.exe, torch 2.7.0+cu128, numpy 2.5.4. Use -B and PYTHONUTF8=1; do not install into or modify the preserved environment. Node D:/node.exe. OpenCode executable is C:/Users/sakur/AppData/Local/Programs/@opencodedesktop/resources/opencode-cli.exe; exact free ID opencode/deepseek-v4.1-flash-free. LM Studio URL http://127.0.0.1:1234. Never print keys or dump environment variables.

From this lab:
```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$researchPython='..\\frontalis-phase3d-lab\\.venv\\Scripts\\python.exe'
& $researchPython -B -m unittest discover -s tests -v
& $researchPython -B scripts/reproduction_copy.py another-run
```
In the fresh adjacent frontalis-phase3f-repro-another-run directory, run experiments/run.py all and optionally experiments/performance.py. Do NOT rerun prepare.py or freeze.py over copied immutable inputs. API scripts perform paid/free live calls if their cache is absent; offline tests/demo do not. Main numeric conditions are resumable by verified cache; many additional experiments require a fresh copy because snapshots are immutable. structure.py requires the preserved sibling Phase 3-E lab.
repro-smoke verifies five capacities offline and repeated device measurements. The final fresh copy verifies current source/tests and offline restoration. Neither is a second complete execution of every numeric trajectory.

## Next research, if requested

Test richer and held-out semantic contexts that use the available capacity; compare locality against transfer without relabeling preference as correctness. Study explicit unknown/missing masks and cue/observed geometry. Replicate dual-view on unseen seeds and initial tendencies. Resolve annotation ambiguity with independent human annotators, then conduct the prepared human evaluation. Keep the current primary failures as the canonical result. Decide Nephia integration separately with the user.
''',encoding='utf8')
    # Self-contained interactive matrix, with primary versus additional data explicitly separated.
    data=json.dumps(dict(primary=read('reports/timelines.json')['rows'],dual=read('reports/dual-view.json')['rows']),ensure_ascii=False).replace('</',r'<\/')
    dashboard='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 3-F 数値結果</title><style>body{font:15px system-ui;background:#101922;color:#e4edf3;margin:auto;max-width:1200px;padding:26px;line-height:1.7}a{color:#8edec8}select{font:inherit;background:#203241;color:inherit;padding:8px;margin:4px}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:15px;border:1px solid #415565;text-align:center}img{width:100%;background:white;border-radius:8px}p{color:#b5c6d1}#note{color:#8edec8}</style><h1>同じ現在を、違う過去から受け止める</h1><p>4つの過去×4つの現在。色は選択した数値軸の符号と強さで、反応の良し悪しを示しません。主条件と事後探索を分けています。</p><a href="/">操作ワークベンチ</a> · <a href="/review">主会話の人間評価</a> · <a href="/review-additional">追加会話の人間評価</a><div><select id="view" aria-label="比較する評価構造"><option value="primary">主条件：観測のみ</option><option value="dual">追加探索：二視点</option></select><select id="cap" aria-label="容量"></select><select id="seed" aria-label="乱数seed"></select><select id="profile" aria-label="初期数値傾向"></select><select id="axis" aria-label="数値軸"></select><select id="when" aria-label="学習時点"><option value="before">現在の経験を学ぶ前</option><option value="after">現在を8回学んだ後</option></select><select id="signal" aria-label="長期と結合"><option value="longTerm">長期判断</option><option value="combined">一時感情・気分も含めた結合</option></select></div><p id="note"></p><div id="matrix"></div><h2>全体の結果</h2><img src="/figure" alt="数値結果の6パネル"><p>主条件：過去差0/60群、時間分離240/240、可塑性と波及抑制60/60。追加二視点：P0の過去差15/15。主系列の使用パラメータは全容量102で、高容量による複雑化は未実証。</p><script>const DATA=__DATA__;const $=id=>document.getElementById(id),past=['協力','裏切り','思い込み','自分の失敗'],now=['謝罪と修復','再び裏切り','責任なしと判明','不明なまま再訪'];$('cap').innerHTML=['約500','約2千','約1万','約10万','約100万'].map((x,i)=>`<option value="${i}">${x}</option>`).join('');$('seed').innerHTML=[6311,6312,6313].map(x=>`<option>${x}</option>`).join('');$('profile').innerHTML=[0,1,2,3].map(x=>`<option value="${x}">P${x}</option>`).join('');$('axis').innerHTML=['好意/接近','好奇心','価値判断','美的関心','怒り方向','喜び方向','悲しみ方向'].map((x,i)=>`<option value="${i}">${x}</option>`).join('');function draw(){let dual=$('view').value==='dual';$('profile').disabled=dual;let rs=DATA[$('view').value].filter(r=>r.capacity===+$('cap').value&&r.seed===+$('seed').value&&(dual||r.profile===+$('profile').value));$('note').textContent=dual?'事後探索。P0のみ。元の学習重みを変えず、cueとobservedを等分して評価。':'事前固定した主比較。高容量を確保しても、この系列では使用される中心が同じ。';$('matrix').innerHTML='<table><tr><th>過去 → 現在</th>'+now.map(x=>'<th>'+x+'</th>').join('')+'</tr>'+past.map((p,i)=>'<tr><th>'+p+'</th>'+now.map((_,j)=>{let r=rs.find(x=>x.past===i),v=r.branches[j][$('when').value][$('signal').value][+$('axis').value],color=v>=0?'57,169,170':'222,139,88';return `<td style="background:rgba(${color},${.12+.68*Math.abs(v)})">${v.toFixed(4)}</td>`}).join('')+'</tr>').join('')+'</table>'}document.querySelectorAll('select').forEach(x=>x.onchange=draw);draw();</script></html>'''
    dashboard=dashboard.replace('src="/figure"','src="results.png"')
    (ROOT/'reports/dashboard.html').write_text(dashboard.replace('__DATA__',data),encoding='utf8')
    failures=[]
    for output in c['outputs']:
        actual=output['response'];expected=output['case']['toolResult']
        if actual['stance']!=expected['scene']['stance']:
            failures.append(dict(type='Metadata stance mismatch',engine=output['engine'],mode=output['mode'],id=actual['id'],expected=expected['scene']['stance'],actual=actual['stance'],reply=actual['reply']))
    failures.extend(dict(type='Post hoc machine prose flag, not human adjudication',**r) for r in e['audit']['flags'])
    failures += [dict(type='Primary history criterion failed',passed=0,total=60),dict(type='Capacity underused',mainActive=102,maxReserved=999906),dict(type='RBF heldout worse than linear',linearMSE=.15922417831132982),dict(type='Cue and observed geometry differ',effect='Primary long-term history effect attenuated'),dict(type='Extraction shape ambiguity',format='Named axis object versus eight-element array'),dict(type='Qwen operational failures',error='HTTP400 terminated / Engine protocol predict request failed: fetch failed'),dict(type='Single authored gold',effect='Annotation ambiguity not independently resolved'),dict(type='Experimental state names in IDs',effect='Possible verbalizer cueing'),dict(type='Human evaluation pending')]
    (ROOT/'reports/failure-cases.json').write_text(json.dumps(dict(cases=failures),ensure_ascii=False,indent=2),encoding='utf8')
    print('Published plot, interactive matrix, report, failures, and handoff.')

if __name__=='__main__':main()
