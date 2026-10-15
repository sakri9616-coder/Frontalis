import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const [engine,mode]=process.argv.slice(2);
const models={haiku:'claude-haiku-5-5',qwen:'qwen3-4b-instruct-2507',deepseek:'opencode/deepseek-v4.1-flash-free'};
if(!models[engine])throw Error('Unknown engine.');
const key=process.env.ANTHROPIC_API_KEY;
if(engine==='haiku'&&!key)throw Error('ANTHROPIC_API_KEY unavailable.');
const sha=s=>createHash('sha256').update(s).digest('hex');
const read=async p=>JSON.parse(await fs.readFile(path.join(root,p),'utf8'));
const redact=s=>key?s.split(key).join('[REDACTED]'):s;
const write=async(p,x)=>{await fs.mkdir(path.dirname(path.join(root,p)),{recursive:true});await fs.writeFile(path.join(root,p),redact(JSON.stringify(x,null,2)),{flag:'wx'});};
const reg=await read('reference/preregistration.json');
for(const[p,h]of Object.entries(reg.sources))if(sha(await fs.readFile(path.join(root,p)))!==h)throw Error('Frozen source changed: '+p);
const parse=s=>JSON.parse(s.trim().replace(/^```(?:json)?\s*/,'').replace(/\s*```$/,''));
const expression=`架空の場面に自然文で2〜4文の返答をする。指定languageで発言する。人格・感情の唯一の正本はtoolResult。
scene=場面全体、selected=選んだ対象。混同しない。combinedが現在の反応、shortTermは一時感情、moodは残る気分、longTermは持続的傾向。
channelGuideは7数値軸の読み方であり、人物像の説明はない。感情方向の強度は正の部分で、負値を逆感情と解釈しない。
scene.stanceとselectionを保持。経験文にある事実だけ使う。過去の記憶を推測で足さない。喜び・怒り・悲しみをゼロ/負の信号から付加しない。
好意、嫌悪、好奇心、意欲、抵抗は信号に従う。選択対象への感情と場面全体の感情が違えばその区別を自然に示す。理由の説明は必須でない。
本文に内部のID、NPN、数値を出さない。JSONのみ {"records":[{"id":"入力id","reply":"発言","stance":"approve/refuse/hold","selection":"指定id"}]}。外部ツール不要。`;
async function opencode(prompt){
 const env={...process.env};for(const k of Object.keys(env))if(/API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET/i.test(k))delete env[k];
 return await new Promise(resolve=>{
  const child=spawn('C:\\Users\\sakur\\AppData\\Local\\Programs\\@opencodedesktop\\resources\\opencode-cli.exe',['run','--standalone','--format','json','--agent','plan','-m',models.deepseek],{cwd:root,env,windowsHide:true,stdio:['pipe','pipe','pipe']});
  let stdout='',stderr='',timed=false;const timeout=setTimeout(()=>{timed=true;child.kill();},180000);
  child.stdout.on('data',b=>stdout+=b);child.stderr.on('data',b=>stderr+=b);child.on('error',e=>{clearTimeout(timeout);resolve({error:e.message});});
  child.on('close',code=>{clearTimeout(timeout);let lines=[];try{lines=stdout.trim().split(/\r?\n/).filter(Boolean).map(s=>JSON.parse(s));}catch{}
   const tools=lines.filter(l=>l.type==='tool_use'),steps=lines.filter(l=>l.type==='step_finish');
   const cost=steps.reduce((a,s)=>a+(s.part?.cost||0),0);
   resolve({code,stdout,stderr,timedOut:timed,tools:tools.length,cost,text:lines.filter(l=>l.type==='text').map(l=>l.part?.text||'').join('\n'),error:timed?'180s timeout':code!==0?'CLI exit '+code:tools.length?'Unexpected tool execution':cost!==0?'Unexpected nonzero cost':null});});
  child.stdin.end(prompt);
 });
}
async function generate(name,prompt,ids){
 const attempts=[];
 for(let attempt=0;attempt<2;attempt++){
  const file=`reports/live/${engine}/${name}-attempt${attempt}.json`;let raw;
  try{raw=await read(file);if(raw.promptHash!==sha(prompt))throw Error('Cache prompt mismatch.');}
  catch(e){if(e.code!=='ENOENT')throw e;if(process.env.FRONTALIS_VERIFY_ONLY==='1')throw Error('Missing cache; live forbidden.');
   const start=Date.now();let response=null,error=null,status=null,text='';
   try{
    if(engine==='deepseek'){response=await opencode(prompt);error=response.error;text=response.text||'';}
    else{
     const url=engine==='haiku'?'https://api.anthropic.com/v1/messages':'http://127.0.0.1:1234/v1/chat/completions';
     const headers=engine==='haiku'?{'content-type':'application/json','anthropic-version':'2023-06-01','x-api-key':key}:{'content-type':'application/json'};
     const body=engine==='haiku'?{model:models[engine],max_tokens:8192,messages:[{role:'user',content:prompt}]}:{model:models[engine],max_tokens:6000,temperature:0,messages:[{role:'user',content:prompt}]};
     const r=await fetch(url,{method:'POST',headers,body:JSON.stringify(body),signal:AbortSignal.timeout(180000)});status=r.status;
     response=JSON.parse(redact(await r.text()));if(!r.ok)error='HTTP '+r.status+' '+(response?.error?.type||'');
     text=engine==='haiku'?(response.content||[]).filter(c=>c.type==='text').map(c=>c.text).join('\n'):response.choices?.[0]?.message?.content||'';
    }
   }catch(e){error=e.name==='TimeoutError'?'180s timeout':redact(String(e.message));}
   raw={engine,model:models[engine],name,prompt,promptHash:sha(prompt),status,error,elapsedMs:Date.now()-start,response,text};await write(file,raw);
  }
  let data=null,error=raw.error;
  try{if(error)throw Error(error);data=parse(raw.text);if(!Array.isArray(data.records)||data.records.length!==ids.length||new Set(data.records.map(r=>r.id)).size!==ids.length||ids.some(id=>!data.records.some(r=>r.id===id)))throw Error('Invalid JSON record IDs/count.');}catch(e){error=e.message;}
  attempts.push({file,error,valid:!error,data});console.log(JSON.stringify({engine,name,attempt,valid:!error,error,status:raw.status,elapsedMs:raw.elapsedMs,usage:raw.response?.usage}));
  if(!error)return {name,attempts,data};
 }
 return {name,attempts,data:null};
}
let cases,common;
if(mode==='extract'){cases=(await read('data/natural.json')).map(r=>({id:r.id,text:r.text}));common=await fs.readFile(path.join(root,'data/extraction-contract.txt'),'utf8');}
else if(['dialogue','pipeline','future','dual'].includes(mode)){cases=(await read(`data/${mode==='dialogue'?'dialogue-plan':`${mode}-${engine}`}.json`)).cases.map(c=>({id:c.id,text:c.text,options:c.options,language:c.language,toolResult:c.toolResult}));common=expression;}
else throw Error('Use extract/dialogue/pipeline.');
const batches=[];for(let i=0;i<cases.length;i+=2){const batch=cases.slice(i,i+2);batches.push(await generate(mode+'-'+i/2,common+'\n'+JSON.stringify(batch),batch.map(c=>c.id)));}
await write(`reports/${mode}-${engine}.json`,{engine,model:models[engine],batches});
