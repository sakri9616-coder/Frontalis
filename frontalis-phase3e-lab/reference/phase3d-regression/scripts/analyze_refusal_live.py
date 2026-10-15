"""Collect the separate post hoc refusal exploration using the existing reader."""
import hashlib
import json
from analyze_live import ROOT, collect

plan, rows, missing = collect(plan_file='refusal-live-plan.json', tag='refusal-')
freeze = json.loads((ROOT / 'reports/refusal-live-freeze.json').read_text(encoding='utf8'))
if hashlib.sha256((ROOT / 'reports/refusal-live-plan.json').read_bytes()).hexdigest() != freeze['planFileSha256']:
    raise ValueError('Frozen refusal plan changed.')
protected_after = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                   for rel in freeze['protectedSourceHashesBefore']}
if protected_after != freeze['protectedSourceHashesBefore']:
    raise ValueError('Protected primary/million/review source changed.')
model_after = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in freeze['modelSourceHashes']}
if model_after != freeze['modelSourceHashes']:
    raise ValueError('Frozen trained model source changed.')
value = {'formatVersion': 1, 'actualAdditionalReplies': len(rows), 'plannedAdditionalReplies': 8,
         'postHocExploration': True, 'formalHumanRatings': 0, 'primaryReviewUnchanged': True,
         'scope': '事後に選定した形成後refuse探索。主52/100万8/重点20と別資料。JSON一致は本文成功と別。',
         'planFileSha256': freeze['planFileSha256'], 'protectedSourcesUnchanged': True,
         'modelSourcesUnchanged': True, 'records': rows}
(ROOT / 'reports/refusal-live-analysis.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
text = ['# 学習後refuse：事後探索の実会話', '',
        '既定2問×形成済み2規模×無料2モデルの8発言。事後のrefuse候補選定であり、主試験成功率・重点20評価・容量一般効果へ混ぜない。', '']
for row in rows:
    cfg = row['config']
    text += [f"## {row['engine']} / {row['caseId']}", '',
             f"raw8 / {cfg['loss']} / h={cfg['width']} / seed4311 / person3 / generation96", '', row['user'], '',
             f"NPN：好み={row['expectedStance']}、選択={row['expectedSelection']}、好奇心={row['channels']['curiosity']:.6f}、怒り={row['signals']['anger']:.6f}、喜び={row['signals']['joy']:.6f}、悲しみ={row['signals']['sadness']:.6f}。", '',
             '> ' + row['reply'].replace('\n', '\n> '), '']
(ROOT / 'reports/refusal-conversations.md').write_text('\n'.join(text), encoding='utf8')
print(json.dumps({'additionalReplies': len(rows), 'claimedStanceMatches': sum(r['claimedStanceMatches'] for r in rows),
                  'claimedSelectionMatches': sum(r['claimedSelectionMatches'] for r in rows),
                  'protectedSourcesUnchanged': True, 'modelSourcesUnchanged': True}))
