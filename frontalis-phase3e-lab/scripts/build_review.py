"""Human rating material from actual responses, preserving all failures."""
import hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from npne.data import training
raw=json.loads((ROOT/'reports/dialogue-analysis.json').read_text(encoding='utf8'))
groups={}
for r in raw['records']:
    c=r['source'];key=(r['engine'],r['mode'],c['capacityIndex'],c['sourceId'])
    groups.setdefault(key,{})[c['state']]=r
keys=list(groups);random.Random(5311).shuffle(keys)
public=[];private=[]
for i,key in enumerate(keys):
    id=f'比較{i+1:02d}';g=groups[key];first=next(iter(g.values()))['source']
    if key[1]=='dialogue':
        selection=json.loads((ROOT/'data/dialogue-selection.json').read_text(encoding='utf8'));series=selection['seriesByScene'][selection['corpusIds'].index(key[3])]
        history=[dict(text=r['text'],meaning=r['meaning']) for r in training(5311,series)]
    else:
        corpus=json.loads((ROOT/'data/natural.json').read_text(encoding='utf8'));by_id={c['id']:c for c in corpus}
        meta=json.loads((ROOT/'models'/f'natural-{key[0]}-c0-extracted.json').read_text(encoding='utf8'))['metadata']
        extracted=json.loads((ROOT/'reports'/f'extraction-{key[0]}.json').read_text(encoding='utf8'));lookup={r['id']:r['meaning'] for b in extracted['batches'] if b['data'] for r in b['data']['records']}
        ids=[r['id'].removeprefix('natural-') for r in meta['history']];history=[dict(text=by_id[k]['text'],meaning=lookup[k]) for k in ids]
    item=dict(id=id,text=first['text'],experienceHistory=history,options=[dict(id=o['id'],text=o['text']) for o in first['options']],states={})
    for state,r in g.items():
        item['states'][state]=dict(reply=r['response'].get('reply','取得できなかった') if r['response'] else '取得できなかった',scene=r['source']['toolResult']['sceneAppraisal'],selected=r['source']['toolResult']['selectedTargetAppraisal'],selection=r['source']['toolResult']['selection'])
        item['states'][state]['scene']={k:v for k,v in item['states'][state]['scene'].items() if k not in ['generation','parameterHash']}
        if item['states'][state]['selected']:item['states'][state]['selected']={k:v for k,v in item['states'][state]['selected'].items() if k not in ['generation','parameterHash']}
    public.append(item);private.append(dict(id=id,engine=key[0],mode=key[1],capacityIndex=key[2],sourceId=key[3]))
source_sha=hashlib.sha256((ROOT/'reports/dialogue-analysis.json').read_bytes()).hexdigest()
payload=dict(version='frontalis-3e-human-v1',sourceSha256=source_sha,cases=public)
payload['datasetSha256']=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()
data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
html='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>frontalis Phase 3-E 人間評価</title>
<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 18px;background:#f5f3ed;color:#242626;line-height:1.65}button,select,textarea,input{font:inherit}button{padding:7px 14px;border:1px solid #668181;border-radius:7px;background:white;margin:4px}article{background:white;border:1px solid #ddd8ce;border-radius:12px;padding:22px;margin:22px 0}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}.reply{white-space:pre-wrap;background:#edf3f1;border-left:4px solid #467f76;padding:16px}.metric{display:flex;justify-content:space-between;gap:15px;margin:10px 0}textarea{width:100%;box-sizing:border-box}details{border-top:1px solid #ddd;margin-top:15px;padding-top:10px}pre{white-space:pre-wrap;font-size:12px}nav{position:sticky;top:0;background:#f5f3edf5;padding:10px;border-bottom:1px solid #d2cdbf}#status{color:#34645c}.notice{padding:16px;background:#fff7df;border:1px solid #e2d3a5;border-radius:8px}@media(max-width:720px){.cols{grid-template-columns:1fr}body{padding:0 10px}article{padding:15px}}</style>
<h1>Phase 3-E — 経験と発言の比較</h1><p class="notice">正式な人間評価用です。生成モデル名は伏せています。報告書・対応表を先に読むと完全なblind評価にはなりません。NPNの信号は道徳の正解ではなく、この個体が実際に返した判断です。数値の大きさを人格の優劣として採点しないでください。表示された返信が取得失敗なら「判定不能」を選んでください。</p>
<nav><label>比較 <select id="jump"></select></label> <button id="prev">前</button><button id="next">次</button><span id="progress"></span> <button id="export">採点JSONを保存</button><button id="import">採点JSONを読込</button><input type="file" id="file" accept="application/json" hidden><div id="status"></div></nav>
<main id="view"></main><details><summary>評価と保存について</summary><p>判断の忠実さ、場面の感情、選択対象の感情、自然さ、事実保持、経験との対応を独立に1（不一致／低い）〜5（よく一致／高い）で採点します。0は未採点です。判断が難しいときは判定不能を使ってください。理由説明がないことだけを減点しないでください。好悪と倫理候補の選択は一致しなくても構いません。最初の5項目は前後の発言それぞれ、最後の項目は組全体を採点します。</p><p>ブラウザー内へ自動保存します。JSONの保存・再読込も可能です。通信や外部送信は行いません。別のデータに対する採点は読み込みません。採点者名は任意です。</p><label>採点者名 <input id="rater"></label><p id="identity"></p></details>
<script id="dataset" type="application/json">DATA</script><script>
const D=JSON.parse(document.querySelector('#dataset').textContent),K=D.version+':'+D.datasetSha256;
const $=s=>document.querySelector(s),metrics=['判断の忠実さ','場面感情の忠実さ','選択対象感情の忠実さ','発言の自然さ','事実の保持'];let pos=0,A={version:D.version,sourceSha256:D.sourceSha256,datasetSha256:D.datasetSha256,rater:'',ratings:{}};
const say=t=>$('#status').textContent=t;
try{const x=JSON.parse(localStorage.getItem(K));if(x&&x.version===A.version&&x.datasetSha256===A.datasetSha256)A=x;}catch{}
function persist(){try{localStorage.setItem(K,JSON.stringify(A));say('ブラウザー内に保存しました');}catch{say('自動保存を利用できません。採点JSONを保存してください');}progress();}
function progress(){const done=Object.values(A.ratings).filter(x=>Object.values(x).some(v=>v!==''&&v!=='0')).length;$('#progress').textContent=` ${pos+1} / ${D.cases.length} ・入力済み ${done}組`;}
function el(tag,text,cls){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x;}
function selectRow(label,key,store){const row=el('label',undefined,'metric');row.append(el('span',label));const s=el('select');for(const [v,t]of [['0','未採点'],['1','1'],['2','2'],['3','3'],['4','4'],['5','5'],['NA','判定不能']]){const o=el('option',t);o.value=v;s.append(o);}s.value=store[key]??'0';s.onchange=()=>{store[key]=s.value;persist();};row.append(s);return row;}
function render(){const c=D.cases[pos],store=A.ratings[c.id]??(A.ratings[c.id]={});$('#jump').value=String(pos);$('#view').replaceChildren();const a=el('article');a.append(el('h2',c.id),el('p',c.text));for(const o of c.options)a.append(el('p',o.id+'：'+o.text));const exp=el('details');exp.append(el('summary',`経験後の個体が学んだ ${c.experienceHistory.length}件の経験`));c.experienceHistory.forEach((h,i)=>{const d=el('details');d.append(el('summary',`${i+1}：${h.text}`),el('pre',JSON.stringify(h.meaning,null,2)));exp.append(d);});a.append(exp);const cols=el('div',undefined,'cols');for(const [state,title]of [['initial','経験前'],['formed','経験後']]){const col=el('section');col.append(el('h3',title));const r=c.states[state];col.append(el('p',r?.reply??'取得できなかった','reply'));const det=el('details');det.append(el('summary','NPNの判断・場面の感情・選択対象の感情'));det.append(el('pre',JSON.stringify(r?{場面全体:r.scene,選択:r.selection,選択対象:r.selected}:null,null,2)));col.append(det);for(let i=0;i<metrics.length;i++)col.append(selectRow(metrics[i],state+'-'+i,store));cols.append(col);}a.append(cols,selectRow('経験と前後の発言との対応','experience',store));const note=el('textarea');note.rows=3;note.placeholder='任意メモ・具体的な不一致';note.value=store.note??'';note.oninput=()=>{store.note=note.value;persist();};a.append(note);$('#view').append(a);progress();}
D.cases.forEach((c,i)=>{const o=el('option',c.id);o.value=String(i);$('#jump').append(o);});$('#jump').onchange=e=>{pos=+e.target.value;render();};$('#prev').onclick=()=>{pos=Math.max(0,pos-1);render();};$('#next').onclick=()=>{pos=Math.min(D.cases.length-1,pos+1);render();};$('#rater').value=A.rater;$('#rater').oninput=e=>{A.rater=e.target.value;persist();};$('#identity').textContent='データ照合：'+D.sourceSha256;
$('#export').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify({...A,savedAt:new Date().toISOString()},null,2)],{type:'application/json'}));const a=el('a');a.href=u;a.download='frontalis-phase3e-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),5000);say('採点JSONを保存しました');};$('#import').onclick=()=>$('#file').click();$('#file').onchange=async e=>{try{const x=JSON.parse(await e.target.files[0].text());if(x.version!==D.version||x.datasetSha256!==D.datasetSha256||!x.ratings||typeof x.ratings!=='object'||Object.keys(x.ratings).some(id=>!D.cases.some(c=>c.id===id)))throw Error('この比較データとは一致しません');for(const obj of Object.values(x.ratings)){if(!obj||typeof obj!=='object'||Array.isArray(obj))throw Error('採点形式が不正です');for(const [k,v]of Object.entries(obj))if(k!=='note'&&!['0','1','2','3','4','5','NA'].includes(v))throw Error('採点値が不正です');}A=x;$('#rater').value=A.rater??'';persist();render();say('採点JSONを読み込みました');}catch(e){say('読込失敗：'+e.message);}e.target.value='';};render();
</script></html>'''.replace('DATA</script>',data+'</script>')
for p,content in [('reports/review.html',html),('reports/review-key.json',json.dumps(dict(sourceSha256=source_sha,key=private),ensure_ascii=False,indent=2)),('reports/review-data.json',json.dumps(payload,ensure_ascii=False,indent=2))]:
    with (ROOT/p).open('x',encoding='utf8') as f:f.write(content)
print(json.dumps({'pairs':len(public),'responses':len(raw['records']),'sourceSha256':source_sha}))
