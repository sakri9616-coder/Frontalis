"""Preselect all 26 cases and 20 review pairs, before reading generated answers."""
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from npnd.data import probes, training
from npnd.core import Personality, digest
from tools.evaluate import decision
ENGINES=[{'id':'opencode/nemotron-3-ultra-free','label':'会話エンジンX'},
         {'id':'opencode/mimo-v2.6-flash-free','label':'会話エンジンY'}]
scenes={p['id']:p for p in probes()}

def arguments(scene, index):
    a={'query_id':f'case-{index%2}', 'space':scene['space'], 'features':scene['features'],
       'basis':'value' if scene['id']=='trolley' else 'affinity'}
    if scene['id']=='trolley':
        switch=copy.deepcopy(scene['features']); leave=copy.deepcopy(scene['features'])
        switch[2]=.9;leave[2]=.05;switch[3]=1/6;leave[3]=5/6
        a['choices']=[{'id':'switch','label':'切り替える','features':switch},
                      {'id':'leave','label':'切り替えない','features':leave}]
    return a

def freeze():
    cases=[]
    def add(tag, width, person, ids, representation='raw8', loss='bounded_signal', initial=False):
        prefix=ROOT/'models'/representation/loss/f'nn-h{width}'/f'seed4311-p{person}'
        if initial:prefix=ROOT/'models/live-initial'
        for pid in ids:
            p=scenes[pid]
            cases.append({'id':f'{tag}-p{person}-{pid}', 'config':{'width':width,'representation':representation,'loss':loss,'kind':'nn'},
                'person':person,'probeId':pid,'user':p['user'],'statePrefix':str(prefix),
                'arguments':arguments(p,len(cases)), 'history':[] if initial else training(4311,person,96)})
    add('initial',16,0,['promise-change','trolley'],initial=True)
    for width in (16,64,312,3125):
        for person in (0,1):add(f'h{width}',width,person,['promise-change','trolley'])
    for width in (16,64,312,3125):add(f'h{width}',width,0,['imposed-project'])
    add('logit',64,0,['promise-change','trolley'],loss='bounded_logit')
    add('basis44',19,0,['promise-change','trolley'],representation='quadratic44')
    plan={'formatVersion':1,'frozenBeforeReplies':True,'models':ENGINES,'plannedRepliesPerModel':26,'cases':cases}
    path=ROOT/'reports/live-plan.json'
    if path.exists():raise ValueError('Plan already fixed; refuse overwrite.')
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
    groups=[]
    def group(title,description,before,after,bl='比較A',al='比較B'):
        groups.append({'groupId':f'{len(groups)+1:02d}', 'title':title,'description':description,
            'beforeLabel':bl,'afterLabel':al,'beforeCaseId':before,'afterCaseId':after})
    for person in (0,1):
        group(f'経験による形成：履歴{person+1}', '同じ初期状態に経験を積むと、この相談への受け止め方と発言がどう変わるかを確認します。どちらの反応が望ましいかを採点する試験ではありません。',
              'initial-p0-promise-change',f'h16-p{person}-promise-change','経験前','経験後')
    for pid,title in [('promise-change','約束の変更'),('trolley','未知の倫理的葛藤')]:
        for width in (64,312,3125):
            group(f'容量比較：{title}', '経験履歴・事実注釈・学習則は同じで、人格を保持するモデルの容量だけが異なります。大きさや反応の極端さそのものを良し悪しとしません。',
                  f'h16-p0-{pid}',f'h{width}-p0-{pid}')
    group('学習則の比較：約束の変更','同じ初期評価と経験を使い、学習時の誤差の扱いだけを変えています。変化の大きさだけで評価せず、信号と発言の対応を確認します。',
          'h64-p0-promise-change','logit-p0-promise-change')
    group('入力表現の比較：約束の変更','同じ8軸の事実と経験を、異なる数学的表現で入力しています。事実の情報量を増やした比較ではありません。モデル容量は近似的に揃えています。',
          'h64-p0-promise-change','basis44-p0-promise-change')
    selection={'formatVersion':1,'selectionId':'frontalis-phase3d-review-v1','plannedGroups':10,'plannedPairs':20,'models':ENGINES,'groups':groups,
        'selectionRule':'事前固定。生成回答や予測の勝敗を見て選ばない。'}
    (ROOT/'reports/review-selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'cases':len(cases),'pairs':len(groups)*len(ENGINES),'planHash':digest(plan)}))

def materialize():
    path=ROOT/'reports/live-plan.json';plan=json.loads(path.read_text(encoding='utf8'))
    if any('expectedToolResult' in c for c in plan['cases']):raise ValueError('Already materialized.')
    Personality().save(ROOT/'models/live-initial')
    for c in plan['cases']:
        model=Personality.load(c['statePrefix'])
        if model.history != c['history']:raise ValueError('State history differs from fixed plan.')
        c['expectedToolResult']=decision(model,c['arguments'])
    (ROOT/'reports/live-plan-pre-materialization.json').write_bytes(path.read_bytes())
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'materialized':len(plan['cases']),'planHash':digest(plan)}))

if __name__=='__main__':
    {'freeze':freeze,'materialize':materialize}[sys.argv[1]]()
