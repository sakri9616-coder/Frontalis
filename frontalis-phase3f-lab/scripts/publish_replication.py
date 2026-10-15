"""Derived publication of the separately frozen additional study."""
import hashlib,itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.replication_longrun import verify

def main():
    r=json.loads((ROOT/'reports/replication-longrun.json').read_text(encoding='utf8'))
    if r['sourceHash']!=verify()[0]:raise ValueError('Additional report source hash mismatch.')
    s=r['summary'];groups=r['replication'];long=r['longrun']
    spreads=[]
    for seed,profile,mode,repeats in itertools.product(r['seeds'],[0,2],['uniform','consolidation'],[32,1024]):
        values=[p['cueAfter'] for row in long if (row['seed'],row['profile'],row['decay'])==(seed,profile,mode) for p in row['checkpoints'] if p['repeats']==repeats]
        if len(values)!=5:raise ValueError('Incomplete capacity comparison.')
        spreads.append(max(float(np.sqrt(np.mean((np.array(a)-b)**2))) for a,b in itertools.combinations(values,2)))
    descriptive=dict(inputSha256=hashlib.sha256((ROOT/'reports/replication-longrun.json').read_bytes()).hexdigest(),
        maxCapacitySpread=max(spreads),activeParameterCounts=sorted({p['activeParameters'] for row in long for p in row['checkpoints']}),
        ranges=[dict(mode=q['mode'],repeats=q['repeats'],retentionMin=min(p['retentionAfter1024'] for row in long if row['decay']==q['mode'] for p in row['checkpoints'] if p['repeats']==q['repeats']),retentionMax=max(p['retentionAfter1024'] for row in long if row['decay']==q['mode'] for p in row['checkpoints'] if p['repeats']==q['repeats'])) for q in s['longrun']])
    target_drift=[]
    for cap in range(5):
        path=ROOT/f'models/replication-longrun/c{cap}-reacquired.json'
        records=json.loads(path.read_text(encoding='utf8'))['metadata']['episodes']
        repetitions=[x for x in records if x['episode']['id'].startswith('past-0-') and int(x['episode']['id'].split('-')[-1])<1024]
        if len(repetitions)!=1024:raise ValueError('Incomplete saved formation trajectory.')
        a,b=repetitions[0],repetitions[-1]
        drift=lambda key:float(np.sqrt(np.mean((np.asarray(a[key])-b[key])**2)))
        target_drift.append(dict(capacity=cap,seed=7901,profile=0,targetLogitDrift=drift('targetLogit'),outcomeAppraisalDrift=drift('outcomeAppraisalBefore'),recordSourceSha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    descriptive['posthocTargetDrift']=dict(type='Post hoc descriptive mechanism inspection; one seed and one initial tendency, shared across capacities',rows=target_drift)
    (ROOT/'reports/replication-analysis.json').write_text(json.dumps(descriptive,ensure_ascii=False,indent=2),encoding='utf8')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for phase,color,style in [('before','#157d8a','-'),('after','#d4791c','--')]:
        for j,seed in enumerate(r['seeds']):
            vals=[np.mean([g['metrics'][phase+'-dual50']['meanPastDifference'] for g in groups if g['seed']==seed and g['profile']==p]) for p in range(4)]
            axs[0,0].plot(range(4),vals,marker=['o','s','^'][j],ls=style,color=color,alpha=.65,label=f'dual {phase}, seed {seed}')
    vals=[np.mean([g['metrics']['before-observed']['meanPastDifference'] for g in groups if g['profile']==p]) for p in range(4)]
    axs[0,0].plot(range(4),vals,color='#77828e',label='observed before (mean seeds)')
    axs[0,0].axhline(.01,color='black',ls='--',lw=1);axs[0,0].set(xticks=range(4),xlabel='Initial numerical tendency',ylabel='Past-dependent RMS at same present',title='History difference before / after 8 current updates')
    axs[0,0].legend(fontsize=7,ncol=2)
    for j,mode in enumerate(['uniform','consolidation']):
        values=[];changes=[]
        for repeats in [32,1024]:
            points=[p for row in long if row['decay']==mode for p in row['checkpoints'] if p['repeats']==repeats]
            values.append([p['retentionAfter1024'] for p in points]);changes.append([p['learningSpecificChange256'] for p in points])
        positions=np.array([0,1])+(j-.5)*.24
        for ax,data in [(axs[0,1],values),(axs[1,0],changes)]:
            av=[np.mean(a) for a in data];lo=[np.min(a) for a in data];hi=[np.max(a) for a in data]
            ax.errorbar(positions,av,yerr=[np.array(av)-lo,np.array(hi)-av],fmt='o-',capsize=4,label=mode)
    axs[0,1].set(xticks=[0,1],xticklabels=['32','1024'],xlabel='Repeated successful-help experiences',ylabel='Learned residual retention ratio',title='After 1024 idle updates');axs[0,1].legend()
    axs[1,0].axhline(.01,color='black',ls='--',lw=1)
    axs[1,0].set(xticks=[0,1],xticklabels=['32','1024'],xlabel='Formation repetitions',ylabel='RMS vs 256-update decay-only control',title='Effect of 256 contrary-outcome experiences');axs[1,0].legend()
    labels=[];vals=[]
    for q in s['longrun']:
        labels.append(q['mode']+'\n'+str(q['repeats']));vals.append(q['meanFormedSaturation'])
    axs[1,1].bar(labels,vals,color=['#77828e','#77828e','#157d8a','#157d8a']);axs[1,1].set(ylabel='Fraction of 7 cue signals with |value| > .95',ylim=(0,1),title='Extremity is reported separately')
    fig.suptitle('Phase 3-F additional study — descriptive means and min/max ranges\nSame capacities can overlap; ranges are not confidence intervals',fontsize=12)
    fig.savefig(ROOT/'reports/replication-longrun.png',dpi=170);fig.savefig(ROOT/'reports/replication-longrun.svg');plt.close(fig)
    rows=[]
    for q in s['longrun']:
        rows.append(f"| {q['mode']} | {q['repeats']} | {q['meanRetention']:.5f} | {q['meanLearningSpecificChange']:.5f} | {q['learningSpecificPassed']}/{q['total']} | {q['localityPassed']}/{q['total']} | {q['meanFormedSaturation']:.1%} |")
    before=s['replication']['before-dual50'];observed=s['replication']['before-observed']
    byprofile=[]
    for p in range(4):
        gs=[g for g in groups if g['profile']==p];byprofile.append(f"| {p} | {sum(g['metrics']['before-dual50']['passed'] for g in gs)}/{len(gs)} | {np.mean([g['metrics']['before-dual50']['meanPastDifference'] for g in gs]):.5f} |")
    text=f'''# Phase 3-F 追加研究結果

ユーザーから追加研究の依頼を受け、主結果を確認した後に[計画](reference/replication-longrun.md)と実行コードを固定した。未使用seed7901/7902/7903で実行。API呼出しは追加していない。[全数値](reports/replication-longrun.json)、[実装](experiments/replication_longrun.py)、[図](reports/replication-longrun.png)を保存した。

## 未使用seed・初期人格への再検証

同じ学習パラメータの読み出しを比較。二視点方式の過去差 >.01 は **{before['passed']}/{before['total']}群**、平均RMS **{before['mean']:.5f}**。観測視点だけでは **{observed['passed']}/{observed['total']}群**、平均 **{observed['mean']:.5f}**。二視点の事前基準（80%以上）: **{s['replicationCriterionPassed']}**。

現在経験を8回学習した後は、二視点の過去差 >.01 は **{s['replication']['after-dual50']['passed']}/60群**、平均RMS **{s['replication']['after-dual50']['mean']:.5f}** まで縮まった。新経験への追従が強く、履歴由来の差が急速に揃う。人格の持続性が十分とは扱わない。これは事前に記録対象とした学習後の副次結果で、学習前を主対象にした成立判定と分けて報告する。

| 初期傾向 | 二視点の成立群 | 過去差平均RMS |
|---|---:|---:|
{chr(10).join(byprofile)}

60群は5容量×4初期傾向×3seed。基礎乱数を共有する容量・傾向を独立な60標本とは扱わない。新しい自然文や人物識別への一般化ではなく、既存の合成4×4条件での再検証。短期感情は同一の状態に固定して長期信号だけを比較した。

## 長期反復と学び直し

善意の援助と成功を反復した後、同じ善意の援助が失敗する新経験256回を与えた。新経験のない空白256更新を対照とし、減衰だけでは説明できない長期信号の変化を測定した。自己評価目標を維持し、反応の方向に正解を置いていない。

| 減衰 | 成功反復 | 空白1024後の保持比 | 減衰対照との差RMS | 対照差>.01 | 固定無関連への波及比<.5 | 形成時の飽和率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

保持比は初期状態からの学習効果を分母にする。1,024回反復後の学び直し成立（80%以上）: **{s['longrunCriterion']}**。反復32/1024は同じ系列のチェックポイントで独立標本ではない。数値有限性と以前の先頭記録の保持は全120チェックポイントで検査し、5容量の追加人格を別フォルダへ保存・厳密復元した。

## 事後の機構点検

保存済みのseed7901・初期傾向0・定着度依存の記録を点検すると、成功経験の1回目から1,024回目までの自己学習目標の変化はRMS **{target_drift[0]['targetLogitDrift']:.7f}**、結果に対する自己評価の変化は **{target_drift[0]['outcomeAppraisalDrift']:.7f}** だった。5容量で同じ値だが、別の5標本とは扱わない。これは事後の限定点検で、全seedでの事前検定ではない。評価の帰還が大きく増幅される循環は、この保存例では示せていない。結果評価の文脈距離と固定の初期写像が関係するという説明は設計上の仮説であり、原因を分ける介入実験はまだ行っていない。

## この結果の範囲

容量間の学び直し後信号の最大RMS差は **{descriptive['maxCapacitySpread']:.3g}**、使用中の学習パラメータ数は **{descriptive['activeParameterCounts']}**。この反復は少数の同じ手掛かりを使うため大容量の複雑化を測る条件ではない。極端な人格の形成、広い場面への局所性、人間の忘却や感情体験への対応は別の問い。係数が減衰しても中心と初期写像は残る。二視点探索の再検証が成立しても、元の主条件での0/60という不成立は変更しない。正式な人間評価84対は引き続き未採点。

再実行は新しい再現コピーで `experiments/replication_longrun.py all`。キャッシュのsourceHashと計画を照合して再開し、既存の不変スナップショットを上書きしない。
'''
    (ROOT/'ADDITIONAL_RESULTS.md').write_text(text,encoding='utf8')
    print(json.dumps(s,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
