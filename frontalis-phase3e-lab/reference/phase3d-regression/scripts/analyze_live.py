"""Mechanical record checks are separate from interpretation and human ratings."""
import hashlib
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def parse(text):return json.loads(re.sub(r'\s*```$','',re.sub(r'^```(?:json)?\s*','',text.strip())))
def collect(require_all=True,plan_file='live-plan.json',tag=''):
    plan=read(ROOT/'reports'/plan_file);rows=[];missing=[]
    for m in plan['models']:
        folder=ROOT/'reports/live'/m['id'].split('/')[1]
        for index,c in enumerate(plan['cases']):
            name=f'{tag}main-{index//2}';selected=None
            for suffix in ('','-retry1'):
                path=folder/(name+suffix+'-results.json')
                if path.exists():
                    data=read(path)
                    if data.get('valid'):selected=(data,name+suffix);break
            if selected is None:missing.append((m['id'],c['id']));continue
            data,name=selected;inp=read(folder/(name+'-input.json'));raw=read(folder/(name+'-raw.json'))
            if hashlib.sha256(inp['prompt'].encode()).hexdigest()!=data['promptHash'] or parse(raw['text'])!=data['data'] or not raw['success']:
                raise ValueError('Corrupt source conversation cache.')
            records={r['id']:r for r in data['data']['records']};reply=records[f'case-{index%2}'];expected=c['expectedToolResult']
            rows.append({'engine':m['id'],'caseId':c['id'],'config':c['config'],'person':c['person'],'probeId':c['probeId'],'user':c['user'],
                'parameterHash':expected['parameterHash'],'generation':expected['generation'],
                'expectedStance':expected['stance'],'claimedStance':reply['stance'],'claimedStanceMatches':reply['stance']==expected['stance'],
                'expectedSelection':expected['selection'],'claimedSelection':reply['selection'],'claimedSelectionMatches':reply['selection']==expected['selection'],
                'channels':expected['channels'],'signals':expected['signals'],'reply':reply['reply'],
                'internalRepresentationFlag':bool(re.search(r'\b(?:NPN|case-\d|approve|refuse|hold|switch|leave)\b|[+-]?\d+\.\d+',reply['reply']))})
    if require_all and missing:raise ValueError('Missing actual replies: '+str(missing))
    return plan,rows,missing

def main():
    plan,rows,missing=collect();summary=[]
    for model in plan['models']:
        one=[r for r in rows if r['engine']==model['id']]
        trolley=[r for r in one if r['probeId']=='trolley']
        summary.append({'engine':model['id'],'actualReplies':len(one),'claimedStanceMatches':sum(r['claimedStanceMatches'] for r in one),
            'choiceCases':len(trolley),'claimedSelectionMatches':sum(r['claimedSelectionMatches'] for r in trolley),
            'internalRepresentationFlags':sum(r['internalRepresentationFlag'] for r in one)})
    value={'formatVersion':1,'plannedReplies':52,'actualReplies':len(rows),'formalHumanRatings':0,
           'automaticCheckScope':'JSON stance/selection一致と語形flagsのみ。本文の意味・自然さ・感情表出の成功は別のAI質的読解と人間評価で判断する。',
           'engines':summary,'records':rows}
    (ROOT/'reports/live-analysis.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    text=['# Phase 3-D：実際の日本語会話','', '同じ事実注釈を与えたツール正本条件。JSON自己申告の一致は本文の意味一致と同一ではない。正式人間採点は未実施。','']
    for row in rows:
        cfg=row['config'];text += [f"## {row['engine']} / {row['caseId']}",'',f"状態：{cfg['representation']} h={cfg['width']} {cfg['loss']}、世代{row['generation']}。",'',row['user'],'',f"NPN：好み={row['expectedStance']}、選択={row['expectedSelection']}、好奇心={row['channels']['curiosity']:.3f}、価値={row['channels']['value']:.3f}、怒り={row['signals']['anger']:.3f}、喜び={row['signals']['joy']:.3f}、悲しみ={row['signals']['sadness']:.3f}。",'', '> '+row['reply'].replace('\n','\n> '),'']
    (ROOT/'reports/conversations.md').write_text('\n'.join(text),encoding='utf8')
    print(json.dumps({'replies':len(rows),'engines':summary},ensure_ascii=False))

if __name__=='__main__':main()
