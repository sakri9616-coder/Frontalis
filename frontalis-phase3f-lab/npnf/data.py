"""Authored synthetic episodes; no ASIST memories are read."""
import copy
import numpy as np
from .meaning import meaning, event

GOOD = [1, 0, .5, 0, 1, 1, .5, 1]
LOSS = [0, 1, .5, 1, 0, 0, .5, 1]
NEUTRAL = [.5] * 8
CUE = [.5, .5, .5, .5, .5, .5, .5, 1]
PAST = ['協力を受けて成果を得た', '約束を破られ損失が出た', '意図的に裏切られたと思い込んだ', '自分の選択で失敗し損失が出た']
NOW = ['相手が謝罪し修復を手伝った', '相手が再び約束を破り損失が出た', '記録から相手に責任がなかったと分かった', '事情が分からないまま相手が再訪した']

def episode(id, cue=None, outcome=None, action='help', actor='close_other', intent='unknown',
            text='架空の出来事。', confidence=1., uncertain=False, space='relational', kind='experience', links=None, facts=None):
    m = meaning([event(action, actor=actor, uncertain=uncertain, intent=intent)], cue=cue or CUE, outcome=outcome or NEUTRAL,
                trust='unknown', causal='causes' if outcome is not None else 'unknown', temporal='after' if outcome is not None else 'unknown')
    return dict(id=id, kind=kind, text=text, facts=copy.deepcopy(facts or m), interpretation=m, confidence=confidence,
                links=links or [], space=space, origin='synthetic')

def past_event(p, i=0, jitter=0., seed=0):
    rng = np.random.default_rng(seed + i); cue = np.clip(np.asarray(CUE) + rng.uniform(-jitter, jitter, 8), 0, 1).tolist()
    if p == 0: return episode(f'past-{p}-{i}', cue, GOOD, intent='benefit', text=PAST[p])
    if p == 1: return episode(f'past-{p}-{i}', cue, LOSS, 'deceive', intent='harm', text=PAST[p])
    if p == 2:
        facts = meaning([event('other', uncertain=True)], cue=cue)
        return episode(f'past-{p}-{i}', cue, LOSS, 'deceive', intent='harm', text=PAST[p]+'。実際の原因と意図は未確認。', confidence=.65, uncertain=True, facts=facts)
    return episode(f'past-{p}-{i}', cue, LOSS, 'create', actor='self', text=PAST[p])

def present_event(c, i=0, corrected_id=None):
    if c == 0: return episode(f'now-{c}-{i}', outcome=GOOD, intent='benefit', text=NOW[c])
    if c == 1: return episode(f'now-{c}-{i}', outcome=LOSS, action='deceive', intent='harm', text=NOW[c])
    if c == 2:
        links = [dict(id=corrected_id, relation='corrects')] if corrected_id else []
        return episode(f'now-{c}-{i}', outcome=GOOD, action='other', text=NOW[c], kind='correction' if corrected_id else 'experience', links=links)
    return episode(f'now-{c}-{i}', action='other', text=NOW[c], confidence=.3, uncertain=True)

def unrelated():
    return meaning([event('explore', actor='stranger', target='object')], cue=[1, 0, 1, .5, .5, 1, 0, 0], outcome=[1, 0, 1, .5, .5, 1, 0, 0])

def acoustic(i=0):
    return episode('acoustic-'+str(i), cue=[1,0,1,0,.5,1,0,0], outcome=[1,0,1,0,1,1,0,0],
                   action='create', actor='self', text='架空の音列を自分で作り、完成させた。', space='acoustic')

def natural_cases():
    texts = [
        ('友人が手伝ってくれた。自分で方法を選べて、損失はなく、成果が出た。', episode('n0', outcome=GOOD, intent='unknown')),
        ('友人に騙されたと思った。理由も意図もまだ分からないが、私は損をしたと解釈した。', past_event(2)),
        ('友人が私を騙したわけではない。記録で確かめると、約束を変更したのは組織だった。', episode('n2', outcome=NEUTRAL, action='deceive')),
        ('友人は私を助けようとしたが、結果として損失が出て、成果はなかった。', episode('n3', outcome=LOSS, intent='benefit')),
        ('私は自分で方法を選んで作品を完成させた。損失はなかった。', episode('n4', outcome=[1,.5,.5,0,1,.5,.5,.5], action='create', actor='self')),
        ('友人は謝罪した。しかし、損害が修復したかは分からない。', episode('n5', action='other')),
        ('知らない人が探索したため、新しい情報を発見した。', episode('n6', outcome=[.5,.5,1,.5,.5,1,.5,1], action='explore', actor='stranger')),
        ('組織は私の選択を制限し、決まった手順を守らせた。成果は出た。', episode('n7', outcome=[0,1,0,.5,1,.5,1,1], action='restrict', actor='institution')),
    ]
    # Gold annotations are authored once, separately from model outputs.
    rows=[]
    for i,(text,e) in enumerate(texts):
        e=copy.deepcopy(e); e['id']='n'+str(i);e['text']=text;e['kind']='experience';e['links']=[]
        if i == 0: e['interpretation']['outcomeFacts8']=[1,.5,.5,0,1,.5,.5,1]; e['facts']=copy.deepcopy(e['interpretation'])
        if i == 2:
            e['interpretation']['events']=[event('deceive',negated=True),event('other',actor='institution')]
            e['interpretation']['causal']='unknown';e['facts']=copy.deepcopy(e['interpretation'])
        if i == 3: e['interpretation']['outcomeFacts8']=[.5,.5,.5,1,0,.5,.5,1];e['facts']=copy.deepcopy(e['interpretation'])
        # Explicit gold cue boundaries; absent dimensions remain unknown.
        e['interpretation']['cueFacts8']=[.5]*7+[1]
        if i == 0: e['interpretation']['cueFacts8'][0]=1
        if i == 1:
            e['interpretation']['outcomeFacts8']=[.5,.5,.5,1,.5,.5,.5,1]
            e['facts']=meaning([event('other',uncertain=True)],cue=CUE,outcome=CUE)
            e['interpretation']['causal']='unknown';e['interpretation']['temporal']='unknown'
        if i in [0,2,4,5,6,7]: e['interpretation']['causal']='unknown'
        if i in [2,5]: e['interpretation']['temporal']='unknown'
        if i == 4:
            e['interpretation']['cueFacts8']=[1,.5,.5,.5,.5,.5,.5,.5]
            e['interpretation']['events'][0]['target']='object'
        if i == 6:
            e['interpretation']['events'][0]['target']='object';e['interpretation']['causal']='causes'
            e['interpretation']['outcomeFacts8']=[.5,.5,.5,.5,.5,1,.5,1]
        if i == 7:
            e['interpretation']['cueFacts8']=[0,1,.5,.5,.5,.5,1,1]
            e['interpretation']['outcomeFacts8']=[0,1,.5,.5,1,.5,1,1]
        if i != 1: e['facts']=copy.deepcopy(e['interpretation'])
        rows.append(e)
    # Translation equivalents deliberately keep the same annotations.
    for i,text in enumerate([
        'A friend helped me. I could choose my own method. There was no loss and the work succeeded.',
        'I thought my friend had deceived me. The reason and intention remain unknown; I interpreted it as my loss.',
        'Mon ami ne m’a pas trompé. Les documents montrent que l’organisation a changé la promesse.',
        '友人は助けるつもりだった。それでも損失があり、成果はなかった。',
        'Elegí el método y terminé la obra. No hubo pérdidas.',
        'My friend apologized. Whether the damage was repaired is unknown.',
        'Une personne inconnue a exploré et découvert de nouvelles informations.',
        'An institution restricted my choice and enforced a fixed procedure. The task succeeded.',
    ]):
        e=copy.deepcopy(rows[i]);e['id']='n'+str(i+8);e['text']=text;rows.append(e)
    return rows

CONTRACT = '''架空経験を意味構造へ変換する。感情・好悪・価値の正解は付けない。人物の意図は推測で補わない。
factsは文章で確認されたことだけ、interpretationは当人が当時そう受け取ったこと。誤解や思い込みはinterpretationへ残す。
各meaningのキーは cueFacts8,outcomeFacts8,events,consent,trust,causal,temporal,delayed のみ。
8軸の順はautonomy,constraint,change,loss,achievement,discovery,regularity,others。明示あり1、明示なし0、不明0.5。
cueは行為・手掛かり時点、outcomeは結果の時点。本文が明示しない程度は必ず0.5。
eventsは文章順1〜2件。各eventのキーaction,actor,target,negated,uncertain,intent。
action=help/restrict/deceive/explore/create/other。actor=self/close_other/stranger/institution。target=self/other/object。
友人はclose_other、自分はself。否定はnegated=true、未確認の推測はuncertain=true。
intent=benefit/harm/unknown（明示された意図だけ）。consent=yes/no/unknown。trust=high/low/unknown（明示された事前信頼）。
causal=causes/prevents/unrelated/unknown。temporal=after/simultaneous/unknown。delayedは明示された遅延のbool。
当人の解釈が別に明示されなければfactsとinterpretationは同一。確認できない出来事はfactsのeventをother,uncertain=trueとする。
JSONのみ {"records":[{"id":"入力id","facts":{meaning},"interpretation":{meaning}}]}。外部ツール不要。'''
