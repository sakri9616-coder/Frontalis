"""Render descriptive figures from completed evidence; do not train or select models."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'reports'/name).read_text(encoding='utf8'))
font_path=Path('C:/Windows/Fonts/meiryo.ttc')
if font_path.is_file():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})

main=read('numeric-summary.json'); gpu=read('million-gpu-summary.json')
widths=[16,64,312,3125,31250];counts=[526,2062,9998,100014,1000014]
approx=[]
for width in widths:
    rows=gpu['frozenApproximation'] if width==31250 else main['frozenApproximation']
    selected=[x for x in rows if x.get('kind','nn')=='nn' and x['width']==width and x['trainCount']==192]
    assert len(selected)==2 and all(x['completeCells']==12 for x in selected)
    approx.append(float(np.mean([x['testMSEMean'] for x in selected])))
baseline={kind:float(np.mean([x['testMSEMean'] for x in main['frozenApproximation'] if x['kind']==kind and x['trainCount']==192])) for kind in ('linear','quadratic','memory')}
stress=read('numeric-stress.json')['conditions']+read('million-stress.json')['conditions']
plasticity={loss:[next(x for x in stress if x['configuration']['seed']==4311 and x['configuration']['width']==w and x['configuration']['loss']==loss)['contraryCheckpoints']['96']['changeSince1000']['meanAbsoluteChange'] for w in widths] for loss in ('bounded_signal','bounded_logit')}
perf=read('performance.json')['models']+read('million-performance.json')['models']
latency={device:[next(x for x in perf if x['kind']=='nn' and x['width']==w and x['device']==device and x['dtype']=='float64')['singleInference']['medianMs'] for w in widths] for device in ('cpu','cuda')}

fig,axs=plt.subplots(1,3,figsize=(16.3,5.1),layout='constrained')
axs[0].plot(counts,approx,'o-',color='#2454a6',label='NPN')
for kind,label,color in [('linear','線形','#a45e31'),('quadratic','二次回帰','#19805a'),('memory','経験検索','#747b86')]:
    axs[0].axhline(baseline[kind],color=color,linestyle='--',label=label)
axs[0].set(xscale='log',yscale='log',title='未知入力の近似誤差',ylabel='MSE（小さいほどよい）',xlabel='学習パラメータ数')
axs[0].legend(fontsize=9,loc='lower left')
for loss,label,color in [('bounded_signal','信号の誤差で更新','#bf5e4f'),('bounded_logit','出力前の値の誤差で更新','#226f8c')]:
    axs[1].plot(counts,plasticity[loss],'o-',color=color,label=label)
axs[1].set(xscale='log',yscale='log',title='自己強化後、新しい事実で変われるか',ylabel='新経験96件後の判断変化量',xlabel='学習パラメータ数')
axs[1].legend(fontsize=9,loc='lower left')
for device,label,color in [('cpu','CPU','#2454a6'),('cuda','GPU','#a8742b')]:
    axs[2].plot(counts,latency[device],'o-',color=color,label=label)
axs[2].set(xscale='log',title='読み込み済みNPNの1件評価',ylabel='中央値 ms（小さいほど速い）',xlabel='学習パラメータ数')
axs[2].legend(fontsize=9,loc='upper left')
for ax in axs:
    ax.grid(True,alpha=.19)
    ax.set_xticks(counts,labels=['526','2千','1万','10万','100万'])
fig.savefig(ROOT/'reports/capacity-overview.png',dpi=150)
fig.savefig(ROOT/'reports/capacity-overview.svg')
plt.close(fig)
value={'derivedFromSavedResults':True,'approximationTrainCount':192,'approximation':approx,'baseline':baseline,'stressSeed':4311,'stressNewExperienceCount':96,'stress':plasticity,'warmMilliseconds':latency,'counts':counts,
       'limits':['Approximation is a frozen-function diagnostic, not a personality quality score.','Million approximation uses GPU64, with matched 100k calibration.','Plasticity shows one matched seed, not a confidence interval.','Warm API excludes new process startup; ordinary concurrent desktop workload.']}
(ROOT/'reports/capacity-overview-data.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(value,ensure_ascii=False))
