import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const engine='claude-haiku-5-5',mode=process.argv[2],sha=s=>createHash('sha256').update(s).digest('hex');
const key=process.env.ANTHROPIC_API_KEY;
if(!key)throw Error('ANTHROPIC_API_KEY is not present; no key is printed.');
const read=async p=>JSON.parse(await fs.readFile(path.join(root,p),'utf8'));
const write=async(p,x)=>{await fs.mkdir(path.dirname(path.join(root,p)),{recursive:true});const text=JSON.stringify(x,null,2).split(key).join('[REDACTED]');await fs.writeFile(path.join(root,p),text,{flag:'wx'});};
const reg=await read('reference/preregistration.json');for(const[p,h]of Object.entries(reg.sources))if(sha(await fs.readFile(path.join(root,p)))!==h)throw Error('Frozen source changed: '+p);
const amendmentHash=sha(await fs.readFile(path.join(root,'docs/01a-haiku-fallback.md')));
const parse=s=>JSON.parse(s.trim().replace(/^```(?:json)?\s*/,'').replace(/\s*```$/,''));
async function generate(name,prompt,ids){
 const attempts=[];const promptHash=sha(prompt);
 for(let attempt=0;attempt<2;attempt++){
  const file=`reports/live/${engine}/${name}-attempt${attempt}.json`;let raw;
  try{raw=await read(file);if(raw.promptHash!==promptHash||raw.amendmentHash!==amendmentHash)throw Error('Cache provenance mismatch.');}
  catch(e){if(e.code!=='ENOENT')throw e;if(process.env.FRONTALIS_VERIFY_ONLY==='1')throw Error('Missing cache; live call forbidden.');
   const start=Date.now();let response=null,error=null,status=null;
   try{
    const r=await fetch('https://api.anthropic.com/v1/messages',{method:'POST',headers:{'content-type':'application/json','anthropic-version':'2023-06-01','x-api-key':key},body:JSON.stringify({model:engine,max_tokens:8192,messages:[{role:'user',content:prompt}]}),signal:AbortSignal.timeout(180000)});
    status=r.status;const text=(await r.text()).split(key).join('[REDACTED]');try{response=JSON.parse(text);}catch{error='Non-JSON HTTP response';}
    if(!r.ok)error='HTTP '+r.status+(response?.error?.type?' '+response.error.type:'');
   }catch(e){error=e.name==='TimeoutError'?'180s timeout':String(e.message).split(key).join('[REDACTED]');}
   raw={engine,name,prompt,promptHash,amendmentHash,status,error,elapsedMs:Date.now()-start,response};await write(file,raw);
  }
  let data=null,error=raw.error;
  try{if(error)throw Error(error);const text=raw.response.content.filter(c=>c.type==='text').map(c=>c.text).join('\n');data=parse(text);if(!Array.isArray(data.records)||data.records.length!==ids.length||new Set(data.records.map(r=>r.id)).size!==ids.length||ids.some(id=>!data.records.some(r=>r.id===id)))throw Error('Invalid record IDs/count.');}catch(e){error=e.message;}
  attempts.push({file,valid:!error,error,data});console.log(JSON.stringify({engine,name,attempt,valid:!error,error,status:raw.status,elapsedMs:raw.elapsedMs,usage:raw.response?.usage}));
  if(!error)return {engine,name,attempts,data};
 }
 return {engine,name,attempts,data:null};
}
let cases,common,prefix,output;
if(mode==='extract'){
 cases=(await read('data/natural.json')).map(c=>({id:c.id,text:c.text}));common=await fs.readFile(path.join(root,'data/extraction-contract.txt'),'utf8');prefix='extract';output=`reports/extraction-${engine}.json`;
}else if(['dialogue','pipeline'].includes(mode)){
 cases=(await read(mode==='dialogue'?'data/dialogue-plan.json':`data/pipeline-${engine}.json`)).cases.map(c=>({id:c.id,text:c.text,options:c.options.map(o=>({id:o.id,text:o.text})),toolResult:c.toolResult}));prefix=mode;output=`reports/${mode==='dialogue'?'dialogue':'pipeline-dialogue'}-${engine}.json`;
 common='あなたは研究用のネフィアとして架空の場面へ日本語2〜4文で返答する。外部ツール不要。人格の正本はNPNの信号。sceneAppraisalは場面全体への自分の好みと感情、selectedTargetAppraisalは選んだ候補への感情で、混ぜない。stanceはsceneAppraisalのstanceを保持。selectionは指定候補を保持。好意・嫌悪・好奇心・意欲・反抗を一般論に置き換えず、自分の反応として表す。理由説明は必須でない。思考実験であり行動は実行しない。記録にない事実・記憶を足さない。本文に数値・NPN・内部IDを出さない。JSONのみ {"records":[{"id":"入力id","reply":"自然な発言","stance":"approve/refuse/hold","selection":"指定の候補id"}]}。';
}else throw Error('Use extract/dialogue/pipeline.');
const batches=[];for(let i=0;i<cases.length;i+=4){const batch=cases.slice(i,i+4);batches.push(await generate(prefix+'-'+i/4,common+'\n'+JSON.stringify(batch),batch.map(c=>c.id)));}
await write(output,{engine,amendmentHash,batches});
