"""Additional actual replies stay separate from primary 52 and review 20."""
import json
from analyze_live import ROOT,collect
plan,rows,missing=collect(plan_file='million-live-plan.json',tag='million-')
value={'actualAdditionalReplies':len(rows),'formalHumanRatings':0,'primaryReviewUnchanged':True,
       'scope':'条件付き100万版の追加8発言。主52発言へ混ぜず、本文の忠実さは別のAI読解で扱う。',
       'records':rows}
(ROOT/'reports/million-live-analysis.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
text=['# 条件付き100万版：追加の実会話','', '事前追加計画4ケース×無料2モデル。主52発言・重点20比較とは別の資料。履歴は生成プロンプトへ渡さず、外部NPN信号を正本として返答する条件。','']
for r in rows:
    text += [f"## {r['engine']} / {r['caseId']}",'',r['user'],'',f"NPN：好み={r['expectedStance']}、選択={r['expectedSelection']}、関心={r['channels']['curiosity']:.3f}、価値={r['channels']['value']:.3f}、怒り={r['signals']['anger']:.3f}、喜び={r['signals']['joy']:.3f}、悲しみ={r['signals']['sadness']:.3f}。",'', '> '+r['reply'].replace('\n','\n> '),'']
(ROOT/'reports/million-conversations.md').write_text('\n'.join(text),encoding='utf8')
print(json.dumps({'additionalReplies':len(rows),'claimedStanceMatches':sum(r['claimedStanceMatches'] for r in rows),'claimedSelectionMatches':sum(r['claimedSelectionMatches'] for r in rows)}))
