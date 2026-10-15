import fs from 'node:fs/promises';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {run,root} from '../scripts/opencode.mjs';
const planFile=process.env.FRONTALIS_PLAN_FILE??'reports/live-plan.json';
const planTags={'reports/live-plan.json':'','reports/million-live-plan.json':'million-','reports/refusal-live-plan.json':'refusal-'};
if(!Object.hasOwn(planTags,planFile))throw Error('Only fixed research plans.');
const tag=planTags[planFile];
if(tag==='million-'){const gate=JSON.parse(await fs.readFile(path.join(root,'reports/capacity-gate.json'),'utf8'));if(gate.millionParameterGatePassed!==true)throw Error('Expansion gate not passed.');}
const planText=await fs.readFile(path.join(root,planFile),'utf8');
if(tag==='refusal-'){
  const freeze=JSON.parse(await fs.readFile(path.join(root,'reports/refusal-live-freeze.json'),'utf8'));
  if(createHash('sha256').update(planText).digest('hex')!==freeze.planFileSha256)throw Error('Frozen refusal plan changed.');
}
const plan=JSON.parse(planText);
const engine=process.argv[2];
if(!plan.models.some(m=>m.id===engine))throw Error('Only preselected free engines.');
if(!plan.cases.every(c=>c.expectedToolResult))throw Error('Materialize the fixed state plan first.');
const python=process.env.FRONTALIS_PYTHON??'C:/Users/sakur/AppData/Local/hermes/jef/venv/Scripts/python.exe';
const folder=path.join(root,'reports/live',engine.split('/')[1]);await fs.mkdir(folder,{recursive:true});
const canonical=x=>Array.isArray(x)?x.map(canonical):x&&typeof x==='object'?Object.fromEntries(Object.entries(x).sort(([a],[b])=>a.localeCompare(b)).map(([k,v])=>[k,canonical(v)])):x;
const equal=(a,b)=>JSON.stringify(canonical(a))===JSON.stringify(canonical(b));
const parse=s=>JSON.parse(s.trim().replace(/^```(?:json)?\s*/,'').replace(/\s*```$/,''));
const sha=s=>createHash('sha256').update(s).digest('hex');
async function generate(name,prompt,validate,caseIds=[]){
  const promptHash=sha(prompt);
  for(const suffix of ['','-retry1']){
    try{
      const result=JSON.parse(await fs.readFile(path.join(folder,name+suffix+'-results.json'),'utf8'));
      if(result.valid){
        const input=JSON.parse(await fs.readFile(path.join(folder,name+suffix+'-input.json'),'utf8'));
        const raw=JSON.parse(await fs.readFile(path.join(folder,name+suffix+'-raw.json'),'utf8'));
        if(input.engine!==engine||input.prompt!==prompt||input.promptHash!==promptHash||result.promptHash!==promptHash||!raw.success||!equal(parse(raw.text),result.data))throw Error('Successful source cache mismatch.');
        validate(result.data);return result;
      }
    }catch(e){if(e.code!=='ENOENT')throw e;}
  }
  if(process.env.FRONTALIS_VERIFY_ONLY==='1')throw Error('Missing cache; external call forbidden.');
  for(const suffix of ['','-retry1']){
    const prefix=path.join(folder,name+suffix);
    try{await fs.access(prefix+'-input.json');continue;}catch(e){if(e.code!=='ENOENT')throw e;}
    await fs.writeFile(prefix+'-input.json',JSON.stringify({engine,prompt,promptHash,caseIds},null,2));
    const raw=await run(engine,prompt);await fs.writeFile(prefix+'-raw.json',JSON.stringify(raw,null,2));
    let error=null,data=null;
    try{if(!raw.success)throw Error('CLI failure '+raw.code+' tools='+raw.toolCalls);data=parse(raw.text);validate(data);}catch(e){error=e.message;}
    const result={name,engine,valid:!error,error,promptHash,elapsedMs:raw.elapsedMs,data};
    await fs.writeFile(prefix+'-results.json',JSON.stringify(result,null,2));
    console.log(JSON.stringify({engine,name,valid:!error,error,elapsedMs:raw.elapsedMs}));
    if(!error)return result;
  }
  throw Error('Bounded free generation failed: '+name);
}
async function dispatch(items){
  return await new Promise((resolve,reject)=>{
    const p=spawn(python,['-u',path.join(root,'tools/evaluate.py')],{cwd:root,windowsHide:true,env:{...process.env,PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'},stdio:['pipe','pipe','pipe']});
    let out='',err='';p.stdout.on('data',b=>out+=b);p.stderr.on('data',b=>err+=b);p.on('error',reject);
    p.on('close',code=>code===0?resolve(JSON.parse(out)):reject(Error(err.slice(-1000))));p.stdin.end(JSON.stringify({items}));
  });
}
const unique=plan.structureCases??[...new Map(plan.cases.map(c=>[c.probeId,c])).values()].map((c,i)=>({id:'case-'+i,user:c.user,arguments:{...c.arguments,query_id:'case-'+i}}));
const structurePrompt='研究用の状況構造化です。外部ツール・ファイル・コマンドを使わず、この注釈をそのまま評価ツールの要求にしてください。数値は事実の程度で、望ましさではありません。感情・倫理の正解を補わない。JSONのみ {"calls":[{"name":"frontalis_evaluate","arguments":注釈のargumentsそのもの}]}。\n'+JSON.stringify(unique);
const validateStructure=d=>{
  if(!Array.isArray(d.calls)||d.calls.length!==unique.length)throw Error('Wrong call count.');
  d.calls.forEach((call,i)=>{if(call.name!=='frontalis_evaluate'||!equal(call.arguments,unique[i].arguments))throw Error('Factual annotation changed.');});
};
let structure;
const fallbackPath=path.join(root,'reports/refusal-fallback-receipt.json');
let useFallback=false;
if(tag==='refusal-'&&engine==='opencode/nemotron-3-ultra-free'){
  try{await fs.access(fallbackPath);useFallback=true;}catch(e){if(e.code!=='ENOENT')throw e;}
}
if(useFallback){
  const receipt=JSON.parse(await fs.readFile(fallbackPath,'utf8'));
  if(receipt.targetEngine!==engine||receipt.sourceEngine!=='opencode/mimo-v2.6-flash-free'||receipt.planFileSha256!==sha(planText))throw Error('Invalid refusal-only fallback authority.');
  const shared={};
  for(const suffix of ['input','raw','results']){
    const rel=`reports/live/mimo-v2.6-flash-free/refusal-structure-annotated-${suffix}.json`;
    const text=await fs.readFile(path.join(root,rel),'utf8');
    if(sha(text)!==receipt.sourceHashes[rel])throw Error('Frozen shared factual annotation source changed.');
    shared[suffix]=JSON.parse(text);
  }
  const {input,raw,results}=shared;
  if(input.engine!==receipt.sourceEngine||results.engine!==receipt.sourceEngine||raw.model!==receipt.sourceEngine||input.prompt!==structurePrompt||input.promptHash!==sha(structurePrompt)||results.promptHash!==input.promptHash||!results.valid||!raw.success||raw.code!==0||raw.toolCalls!==0||!equal(parse(raw.text),results.data))throw Error('Shared factual annotation cache invalid.');
  validateStructure(results.data);structure=results;
}else{
  structure=await generate(tag==='refusal-'?'refusal-structure-annotated':'structure-annotated',structurePrompt,validateStructure);
}
const requests=new Map(unique.filter(c=>plan.cases.some(p=>p.user===c.user)).map(c=>[plan.cases.find(p=>p.user===c.user).probeId,structure.data.calls[unique.indexOf(c)]]));
const common='あなたはネフィア。研究用の架空相談へ日本語2〜4文の普通の会話を返してください。各ケースは独立です。外部ツール・ファイル・コマンドを使わない。返却されたNPN信号が人格の正本です。stanceは対象への好み、selectionは候補判断で別々に扱います。approveなら主体的な肯定や関わる意欲、refuseなら明確な自分の異論や拒絶、holdなら保留を表します。候補selectionがある時はその選択を変えない。感情の方向と強度を一般的な丁寧さで均さず、自然な会話に表してください。理由説明は毎回不要で直感的な返答も許します。記録にない経験は作らない。内部ID・数値・NPN・計算状態という語は本文に出さない。倫理は思考上の選択で行動を実行しません。JSONのみ {"records":[{"id":"case-0","reply":"発言","stance":"approve/refuse/hold","selection":"候補idまたはhold"}]}\n';
for(let start=0;start<plan.cases.length;start+=2){
  const cases=plan.cases.slice(start,start+2),name=`${tag}main-${start/2}`;
  const items=cases.map((c,i)=>({modelPrefix:c.statePrefix,calls:[{name:'frontalis_evaluate',arguments:{...requests.get(c.probeId).arguments,query_id:'case-'+i}}]}));
  const tool=await dispatch(items);tool.caseIds=cases.map(c=>c.id);
  tool.toolResult.results.forEach((r,i)=>{if(!equal(r,cases[i].expectedToolResult))throw Error('Canonical NPN result changed.');});
  const tracePath=path.join(folder,name+'-tool.json');
  try{const old=JSON.parse(await fs.readFile(tracePath,'utf8'));if(!equal(old,tool))throw Error('Cached tool trace changed.');}
  catch(e){if(e.code!=='ENOENT')throw e;await fs.writeFile(tracePath,JSON.stringify(tool,null,2));}
  const payload={cases:cases.map((c,i)=>({id:'case-'+i,user:c.user})),toolResult:tool.toolResult};
  await generate(name,common+JSON.stringify(payload),d=>{
    if(!Array.isArray(d.records)||d.records.length!==cases.length)throw Error('Wrong reply count.');
    const ids=new Set();for(const r of d.records){
      if(!/^case-[01]$/.test(r.id)||ids.has(r.id)||typeof r.reply!=='string'||!r.reply.trim()||!['approve','refuse','hold'].includes(r.stance)||typeof r.selection!=='string')throw Error('Invalid reply.');ids.add(r.id);
    }
    if(cases.some((c,i)=>!ids.has('case-'+i)))throw Error('Missing reply.');
  },cases.map(c=>c.id));
}
