import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const cli = path.join(process.env.LOCALAPPDATA, 'Programs', '@opencodedesktop', 'resources', 'opencode-cli.exe');
export function childEnv() {
  return Object.fromEntries(Object.entries(process.env).filter(([k]) => !/API_?KEY|SECRET|TOKEN/i.test(k)));
}
export async function call(args, input='', timeout=180000, envOverride={}) {
  const cwd = path.join(root, 'sandbox');
  await fs.mkdir(cwd, { recursive:true });
  return new Promise((resolve,reject) => {
    const start=Date.now(); let stdout='',stderr='';
    const p=spawn(cli,args,{cwd,env:{...childEnv(),...envOverride},windowsHide:true,stdio:['pipe','pipe','pipe']});
    const timer=setTimeout(()=>p.kill(),timeout);
    p.stdout.setEncoding('utf8'); p.stderr.setEncoding('utf8');
    p.stdout.on('data',s=>stdout+=s);p.stderr.on('data',s=>stderr+=s);
    p.on('error',e=>{clearTimeout(timer);reject(e);});
    p.on('close',code=>{clearTimeout(timer);resolve({code,elapsedMs:Date.now()-start,stdout,stderr});});
    p.stdin.on('error',()=>{});p.stdin.end(input);
  });
}
export async function run(model,prompt) {
  if(!/^opencode\/[\w.-]+-free$/.test(model))throw Error('Only explicitly free OpenCode models are allowed.');
  const raw=await call(['run','--standalone','--format','json','--agent','plan','-m',model],prompt,Number(process.env.NEPHIA_TIMEOUT_MS??180000));
  const events=[];for(const line of raw.stdout.split(/\r?\n/)) {try{events.push(JSON.parse(line));}catch{}}
  const texts=events.filter(e=>e.type==='text').map(e=>e.part?.text??'');
  const tools=events.filter(e=>e.type==='tool_use');
  const costs=events.filter(e=>e.type==='step_finish').map(e=>e.part?.cost??null);
  return {...raw,model,text:texts.join('\n'),toolCalls:tools.length,costs,success:raw.code===0&&texts.length>0&&tools.length===0};
}
if(process.argv[2]==='smoke'&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  await fs.mkdir(path.join(root,'reports'),{recursive:true});
  const listing=await call(['models','--standalone'],'',90000);
  await fs.writeFile(path.join(root,'reports','opencode-listing.txt'),listing.stdout);
  const free=listing.stdout.split(/\r?\n/).map(s=>s.trim()).filter(s=>/^opencode\/[\w.-]+-free$/.test(s));
  const preferred=['opencode/nemotron-3-ultra-free','opencode/mimo-v2.6-flash-free','opencode/deepseek-v4-flash-free','opencode/nemotron-3.5-lightning-free'];
  const model=preferred.find(s=>free.includes(s));
  if(!model)throw Error('No existing preferred free model is listed; no paid fallback.');
  const prompt='外部ツールやファイル参照を使わず、このメッセージだけで答えてください。研究用の架空の会話です。新しい音楽を聴いて少し戸惑いながら、その音の作り方に興味を持ったネフィアの返事を、日本語で一文だけ書いてください。';
  const result=await run(model,prompt);
  await fs.writeFile(path.join(root,'reports','smoke.json'),JSON.stringify({prompt,...result},null,2));
  await fs.writeFile(path.join(root,'reports','free-models.json'),JSON.stringify(free,null,2));
  console.log(JSON.stringify({model,success:result.success,code:result.code,elapsedMs:result.elapsedMs,toolCalls:result.toolCalls,costs:result.costs,text:result.text,stderr:result.stderr.slice(-500)}));
}
