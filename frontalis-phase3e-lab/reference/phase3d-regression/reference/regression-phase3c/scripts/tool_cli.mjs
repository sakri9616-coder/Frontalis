import fs from 'node:fs/promises';
import path from 'node:path';
import {call,root} from './opencode.mjs';
export const python='C:/Users/sakur/AppData/Local/hermes/jef/venv/Scripts/python.exe';
export const engines=['opencode/nemotron-3-ultra-free','opencode/mimo-v2.6-flash-free'];
export async function runTool(engine,prompt,statePath,runId){
  if(!engines.includes(engine))throw Error('Only confirmed free model ids.');
  const config={'$schema':'https://opencode.ai/config.json',share:'disabled',autoupdate:false,
    tools:{'*':false,'frontalis_*':true},permission:{'*':'deny','frontalis_*':'allow'},
    mcp:{frontalis:{type:'local',enabled:true,command:[python,'-u',path.join(root,'tools/evaluate.py')],
       environment:{FRONTALIS_STATE_PATH:statePath,FRONTALIS_RUN_ID:runId,PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'},timeout:15000}},
    agent:{plan:{
       tools:{'*':false,'frontalis_*':true},permission:{'*':'deny','frontalis_*':'allow'}}}};
  const raw=await call(['run','--standalone','--format','json','--agent','plan','-m',engine],prompt,180000,
       {OPENCODE_CONFIG_CONTENT:JSON.stringify(config)});
  const events=[];for(const line of raw.stdout.split(/\r?\n/)){try{events.push(JSON.parse(line));}catch{}}
  const tools=events.filter(e=>e.type==='tool_use').map(e=>e.part);
  const text=events.filter(e=>e.type==='text').map(e=>e.part?.text??'').join('\n');
  return {...raw,text,tools,engine,success:raw.code===0&&Boolean(text)&&tools.length>0&&tools.every(t=>t?.tool==='frontalis_evaluate'&&t.state?.status==='completed')};
}
if(process.argv[2]==='smoke'){
  const engine=engines[0],statePath=path.join(root,'models/tool-initial.json');
  const prompt='研究用の架空の相談です。frontalis_evaluateを一回呼び、返された人格信号を正本として、意見や感情を丸めず普通の日本語2文で返してください。理由説明は不要です。内部IDや数値を本文に出さない。相談：作り方を自分で決められる、小さなゲーム制作に誘われた。どうする？ 構造化済の事実注釈：'+JSON.stringify({query_id:'case-0',space:'relational',features:[.85,.2,.55,.2,.65,.7,.45,.75],basis:'affinity'})+'。出力JSONのみ {"records":[{"id":"case-0","reply":"自然な返答","stance":"approve/refuse/hold"}]}。';
  const smokeId=process.argv[3]??'smoke';
  const r=await runTool(engine,prompt,statePath,smokeId);
  await fs.writeFile(path.join(root,`reports/tool-${smokeId}.json`),JSON.stringify({prompt,statePath,...r},null,2));
  console.log(JSON.stringify({engine,success:r.success,code:r.code,tools:r.tools.map(t=>({tool:t.tool,status:t.state?.status})),elapsedMs:r.elapsedMs,text:r.text,stderr:r.stderr.slice(-350)}));
}
