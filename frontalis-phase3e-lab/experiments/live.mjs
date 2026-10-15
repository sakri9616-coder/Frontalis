import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const cli='C:/Users/sakur/AppData/Local/Programs/@opencodedesktop/resources/opencode-cli.exe';
const engines=['opencode/mimo-v2.6-flash-free','opencode/nemotron-3-ultra-free'];
const sha=s=>createHash('sha256').update(s).digest('hex');
const read=p=>fs.readFile(path.join(root,p),'utf8').then(JSON.parse);
const write=async(p,x)=>{await fs.mkdir(path.dirname(path.join(root,p)),{recursive:true});await fs.writeFile(path.join(root,p),JSON.stringify(x,null,2),{flag:'wx'});};
const regText=await fs.readFile(path.join(root,'reference/preregistration.json'),'utf8');const reg=JSON.parse(regText);
for(const [p,h] of Object.entries(reg.sources))if(sha(await fs.readFile(path.join(root,p)))!==h)throw Error('Frozen source changed: '+p);
async function call(args,prompt='',timeout=180000){
 await fs.mkdir(path.join(root,'sandbox'),{recursive:true});
 return new Promise((resolve,reject)=>{
  const env=Object.fromEntries(Object.entries(process.env).filter(([k])=>!/API_?KEY|SECRET|TOKEN/i.test(k)));
  const start=Date.now();let stdout='',stderr='',timedOut=false;
  const p=spawn(cli,args,{cwd:path.join(root,'sandbox'),env,windowsHide:true,stdio:['pipe','pipe','pipe']});
  const timer=setTimeout(()=>{timedOut=true;p.kill();},timeout);
  p.stdout.setEncoding('utf8');p.stderr.setEncoding('utf8');p.stdout.on('data',s=>stdout+=s);p.stderr.on('data',s=>stderr+=s);p.on('error',reject);
  p.on('close',code=>{clearTimeout(timer);const events=[];for(const line of stdout.split(/\r?\n/)){try{events.push(JSON.parse(line));}catch{}}
   const texts=events.filter(e=>e.type==='text').map(e=>e.part?.text??'');const toolCalls=events.filter(e=>e.type==='tool_use').length;
   const costs=events.filter(e=>e.type==='step_finish').map(e=>e.part?.cost??null);
   resolve({code,elapsedMs:Date.now()-start,timedOut,stdout,stderr,text:texts.join('\n'),toolCalls,costs,success:code===0&&texts.length>0&&toolCalls===0&&costs.every(c=>c===null||c===0)});
  });p.stdin.on('error',()=>{});p.stdin.end(prompt);
 });
}
function parse(text){return JSON.parse(text.trim().replace(/^```(?:json)?\s*/,'').replace(/\s*```$/,''));}
async function generation(engine,name,prompt,ids){
 const folder=`reports/live/${engine.split('/')[1]}/${name}`;const promptHash=sha(prompt);
 const attempts=[];
 for(let i=0;i<2;i++){
  const file=folder+`-attempt${i}.json`;let raw;
  try{raw=await read(file);if(raw.promptHash!==promptHash||raw.preregistrationSha256!==sha(regText))throw Error('Cache mismatch.');}
  catch(e){if(e.code!=='ENOENT')throw e;
   if(process.env.FRONTALIS_VERIFY_ONLY==='1')throw Error('Missing cache; live call forbidden.');
   const result=await call(['run','--standalone','--format','json','--agent','plan','-m',engine],prompt);
   raw={engine,name,prompt,promptHash,preregistrationSha256:sha(regText),...result};await write(file,raw);
  }
  let data=null,error=null;
  try{if(!raw.success)throw Error(`CLI failure: code=${raw.code}, tools=${raw.toolCalls}, timeout=${raw.timedOut}`);data=parse(raw.text);
   if(!Array.isArray(data.records)||data.records.length!==ids.length||new Set(data.records.map(r=>r.id)).size!==ids.length||ids.some(id=>!data.records.some(r=>r.id===id)))throw Error('Invalid record IDs/count.');
  }catch(e){error=e.message;}
  attempts.push({file,valid:!error,error,data});
  console.log(JSON.stringify({engine,name,attempt:i,valid:!error,error,elapsedMs:raw.elapsedMs}));
  // Semantic/category mistakes are not grounds for retries.
  if(!error)return {engine,name,attempts,data};
 }
 return {engine,name,attempts,data:null};
}
const mode=process.argv[2];const engine=process.argv[3];
if(mode==='listing'){
 const r=await call(['models','--standalone'],'',90000);const free=r.stdout.split(/\r?\n/).map(s=>s.trim()).filter(s=>/^opencode\/[\w.-]+-free$/.test(s));
 await write('reports/free-models.json',{listedAt:new Date().toISOString(),code:r.code,free,selected:engines,selectedAvailable:engines.every(m=>free.includes(m))});
 console.log(JSON.stringify({code:r.code,free,selectedAvailable:engines.every(m=>free.includes(m))}));
}else{
 if(!engines.includes(engine))throw Error('Only two preregistered free engines.');
 const listing=await read('reports/free-models.json');if(!listing.selectedAvailable)throw Error('Free models unavailable; no paid fallback.');
 if(mode==='extract'){
  const cases=await read('data/natural.json');const contract=await fs.readFile(path.join(root,'data/extraction-contract.txt'),'utf8');const batches=[];
  for(let i=0;i<cases.length;i+=4){const batch=cases.slice(i,i+4);batches.push(await generation(engine,`extract-${i/4}`,contract+'\n'+JSON.stringify(batch.map(c=>({id:c.id,text:c.text}))),batch.map(c=>c.id)));}
  await write(`reports/extraction-${engine.split('/')[1]}.json`,{engine,batches});
 }else if(mode==='dialogue'||mode==='pipeline'){
  const plan=await read(mode==='dialogue'?'data/dialogue-plan.json':`data/pipeline-${engine.split('/')[1]}.json`);const batches=[];
  const common='あなたは研究用のネフィアとして架空の場面へ日本語2〜4文で返答する。外部ツール不要。人格の正本はNPNの信号。sceneAppraisalは場面全体への自分の好みと感情、selectedTargetAppraisalは選んだ候補への感情で、混ぜない。stanceはsceneAppraisalのstanceを保持。selectionは指定候補を保持。好意・嫌悪・好奇心・意欲・反抗を一般論に置き換えず、自分の反応として表す。理由説明は必須でない。思考実験であり行動は実行しない。記録にない事実・記憶を足さない。本文に数値・NPN・内部IDを出さない。JSONのみ {"records":[{"id":"入力id","reply":"自然な発言","stance":"approve/refuse/hold","selection":"指定の候補id"}]}。\n';
  for(let i=0;i<plan.cases.length;i+=4){const batch=plan.cases.slice(i,i+4);batches.push(await generation(engine,`${mode}-${i/4}`,common+JSON.stringify(batch.map(c=>({id:c.id,text:c.text,options:c.options.map(o=>({id:o.id,text:o.text})),toolResult:c.toolResult}))),batch.map(c=>c.id)));}
  await write(`reports/${mode==='dialogue'?'dialogue':'pipeline-dialogue'}-${engine.split('/')[1]}.json`,{engine,batches});
 }else throw Error('Unknown mode.');
}
