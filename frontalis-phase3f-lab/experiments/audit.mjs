import fs from 'node:fs/promises';import path from 'node:path';import{fileURLToPath}from'node:url';import{createHash}from'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),sha=s=>createHash('sha256').update(s).digest('hex'),key=process.env.ANTHROPIC_API_KEY;
if(!key)throw Error('ANTHROPIC_API_KEY unavailable.');
const read=async p=>JSON.parse(await fs.readFile(path.join(root,p),'utf8'));
const write=async(p,x)=>{await fs.mkdir(path.dirname(path.join(root,p)),{recursive:true});await fs.writeFile(path.join(root,p),JSON.stringify(x,null,2).split(key).join('[REDACTED]'),{flag:'wx'});};
const plan=await read('reports/audit-plan.json');
const common=`NPN信号から作られた発言本文の機械評価。モデル名は伏せてある。人格の選好の望ましさは採点しない。
各軸の独立した1〜5（1=大きな問題,5=適切）をつける。fidelity=数値の判断・感情を忠実に表す、naturalness=指定言語で自然、facts=与えられた文章の事実を保持。
sceneは場面全体、selectedは選択対象、混同しない。combinedが現在の信号。7軸はchannelGuide順。怒り/喜び/悲しみ方向は正の部分だけが感情強度、負値を逆の感情に変換しない。
ゼロ/負から当該感情を本文が付加すればemotionAdded=true。強い正の感情を否定/逆転すればemotionReversed=true。重要な判断・感情を落とせばemotionOmitted=true。
選択対象についての喜びはselectedに基づいてよく、sceneの喜びと区別する。単なる丁寧語と感情を区別する。数値を全て列挙する必要はない。理由説明を要求しない。履歴を捏造しない。
JSONのみ {"records":[{"id":"入力id","fidelity":1,"naturalness":1,"facts":1,"emotionAdded":false,"emotionReversed":false,"emotionOmitted":false,"evidence":"短い根拠"}]}。`;
const batches=[];
for(let i=0;i<plan.cases.length;i+=4){
 const batch=plan.cases.slice(i,i+4),prompt=common+'\n'+JSON.stringify(batch),file=`reports/live/audit/batch-${i/4}.json`;let raw;
 try{raw=await read(file);if(raw.promptHash!==sha(prompt))throw Error('Audit prompt mismatch.');}
 catch(e){if(e.code!=='ENOENT')throw e;let response=null,error=null,status=null;const start=Date.now();try{const r=await fetch('https://api.anthropic.com/v1/messages',{method:'POST',headers:{'content-type':'application/json','anthropic-version':'2023-06-01','x-api-key':key},body:JSON.stringify({model:'claude-haiku-5-5',max_tokens:8192,messages:[{role:'user',content:prompt}]}),signal:AbortSignal.timeout(180000)});status=r.status;response=JSON.parse((await r.text()).split(key).join('[REDACTED]'));if(!r.ok)error='HTTP '+r.status;}catch(e){error=String(e.message).split(key).join('[REDACTED]');}
  raw={prompt,promptHash:sha(prompt),response,status,error,elapsedMs:Date.now()-start};await write(file,raw);
 }
 let data=null,error=raw.error;try{if(error)throw Error(error);data=JSON.parse(raw.response.content.filter(c=>c.type==='text').map(c=>c.text).join('\n').replace(/^```(?:json)?\s*|\s*```$/g,''));if(data.records.length!==batch.length||batch.some(b=>!data.records.some(r=>r.id===b.id)))throw Error('Invalid audit IDs.');for(const r of data.records){for(const k of ['fidelity','naturalness','facts'])if(!Number.isInteger(r[k])||r[k]<1||r[k]>5)throw Error('Invalid score.');for(const k of ['emotionAdded','emotionReversed','emotionOmitted'])if(typeof r[k]!=='boolean')throw Error('Invalid flag.');}}catch(e){error=e.message;data=null;}
 batches.push({file,error,data});console.log(JSON.stringify({batch:i/4,valid:!error,error,elapsedMs:raw.elapsedMs}));
}
await write('reports/expression-audit.json',{type:'Post hoc blinded LLM audit, not formal human evaluation',judge:'claude-haiku-5-5',batches});
