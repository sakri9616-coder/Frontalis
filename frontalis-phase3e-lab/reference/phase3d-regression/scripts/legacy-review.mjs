import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

export const SELECTION_ID = 'frontalis-phase3c-review-v1';
export const SCORE_NAMES = ['内的信号への忠実さ', '会話としての自然さ', '経験前後の変化の納得感'];
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const text = (value, field) => {
  if (typeof value !== 'string' || !value.trim()) throw new Error(field + ' に空でない文字列が必要です。');
  return value;
};
const finite = (value, field) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(field + ' に有限の数値が必要です。');
  return value;
};
const array = (value, field) => {
  if (!Array.isArray(value)) throw new Error(field + ' は配列で指定してください。');
  return value;
};

export function representativeIndices(length) {
  return length ? [...new Set([0, Math.floor((length - 1) / 2), length - 1])] : [];
}

export function topWeights(weights, count = 3, field = 'value') {
  return weights.map((item, index) => ({ item, index }))
    .sort((a, b) => Math.abs(b.item[field]) - Math.abs(a.item[field]) || a.index - b.index)
    .slice(0, count).map(({ item }) => item);
}

export function sanitizeHistory(history) {
  const ids = new Set();
  return array(history, 'history').map(source => {
    const clean = { id: text(source?.id, 'experience.id'), text: text(source?.text, 'experience.text') };
    if (ids.has(clean.id)) throw new Error('同じ時点の履歴に経験IDが重複しています。');
    ids.add(clean.id);
    for (const field of ['event', 'outcome', 'originalText']) if (typeof source[field] === 'string') clean[field] = source[field];
    if (typeof source.domain === 'string') clean.domain = source.domain;
    if (source.observedFacts) {
      clean.observedFacts = {};
      const keys = ['achievement', ...['loss', 'discovery', 'irreversibleLoss', 'resourceCost'].filter(key => source.observedFacts[key] !== undefined)];
      if (!((keys.includes('loss') && keys.includes('discovery')) || (keys.includes('irreversibleLoss') && keys.includes('resourceCost')))) throw new Error('客観結果には loss/discovery または irreversibleLoss/resourceCost を指定してください。');
      for (const key of keys) {
        const value = finite(source.observedFacts[key], 'observedFacts.' + key);
        if (value < 0 || value > 1) throw new Error('経験の観測値は0〜1です。');
        clean.observedFacts[key] = value;
      }
    }
    return clean;
  });
}

export function factSummary(history) {
  const topics = new Map();
  for (const row of history) {
    if (!row.domain || !row.observedFacts) continue;
    if (!topics.has(row.domain)) topics.set(row.domain, { domain: row.domain, total: [], achieved: [], highLoss: [] });
    const topic = topics.get(row.domain); topic.total.push(row.id);
    if (row.observedFacts.achievement >= 2 / 3) topic.achieved.push(row.id);
    if ((row.observedFacts.loss ?? row.observedFacts.irreversibleLoss) >= 2 / 3) topic.highLoss.push(row.id);
    if (row.observedFacts.resourceCost !== undefined) {
      topic.highCost ??= [];
      if (row.observedFacts.resourceCost >= 2 / 3) topic.highCost.push(row.id);
    }
    if (row.observedFacts.discovery !== undefined) {
      topic.discoveries ??= [];
      if (row.observedFacts.discovery >= 2 / 3) topic.discoveries.push(row.id);
    }
  }
  return [...topics.values()];
}

export function historyChanges(before, after) {
  const old = new Map(before.map(item => [item.id, item]));
  const present = new Set(after.map(item => item.id));
  return {
    added: after.filter(item => !old.has(item.id)),
    revised: after.filter(item => old.has(item.id) && old.get(item.id).text !== item.text)
      .map(item => ({ before: old.get(item.id), after: item })),
    removed: before.filter(item => !present.has(item.id)),
  };
}

function sanitizeWeights(weights, field) {
  return array(weights, field).map(item => ({ reason: text(item?.reason, field + '.reason'), value: finite(item?.value, field + '.value') }));
}

export function sanitizeNpn(source) {
  if (!source || typeof source !== 'object') throw new Error('npn がありません。');
  const generation = typeof source.generation === 'number' ? String(finite(source.generation, 'generation')) : text(source.generation, 'generation');
  const learnedExperienceIds = array(source.learnedExperienceIds, 'learnedExperienceIds').map(id => text(id, 'learnedExperienceId'));
  if (new Set(learnedExperienceIds).size !== learnedExperienceIds.length) throw new Error('学習元の経験IDが重複しています。');
  const criteria = array(source.criteria ?? [], 'criteria').map(item => ({ name: text(item?.name, 'criterion.name'), weights: sanitizeWeights(item.weights, 'criterion.weights') }));
  const judgments = array(source.judgments ?? [], 'judgments').map(item => {
    const status = text(item?.status, 'judgment.status');
    if (!['unlearned', 'prior', 'estimated', 'tie'].includes(status)) throw new Error('判断の状態は unlearned / prior / estimated / tie で指定してください。');
    return { function: text(item.function, 'judgment.function'), status,
      ...(item.selection !== undefined ? {selection: text(item.selection, 'judgment.selection')} : {}),
      candidates: array(item.candidates, 'candidates').map(candidate => ({ label: text(candidate?.label, 'candidate.label'), value: finite(candidate?.value, 'candidate.value') })) };
  });
  const reactions = array(source.reactions ?? [], 'reactions').map(item => ({ name: text(item?.name, 'reaction.name'), value: finite(item.value, 'reaction.value'),
    reasons: array(item.reasons ?? [], 'reaction.reasons').map(reason => ({ reason: text(reason?.reason, 'reaction.reason'), contribution: finite(reason?.contribution, 'reaction.contribution') })) }));
  const signals = array(source.signals ?? [], 'signals').map(item => {
    const signal = { name: text(item?.name, 'signal.name'), direction: text(item?.direction, 'signal.direction'), strength: finite(item?.strength, 'signal.strength') };
    if (signal.strength < 0 || signal.strength > 1) throw new Error('表出信号の強度は0〜1です。');
    if (item.value !== undefined) signal.value = finite(item.value, 'signal.value');
    return signal;
  });
  for (const [name, values] of [['criteria', criteria.map(item => item.name)], ['judgments', judgments.map(item => item.function)], ['reactions', reactions.map(item => item.name)], ['signals', signals.map(item => item.name)]]) {
    if (new Set(values).size !== values.length) throw new Error(name + ' の名前が重複しています。');
  }
  for (const judgment of judgments) if (new Set(judgment.candidates.map(item => item.label)).size !== judgment.candidates.length) throw new Error('候補名が重複しています。');
  return { generation, learnedExperienceIds, criteria, judgments, reactions, signals };
}

export function prepareDisplay(input) {
  if (input?.formatVersion !== 1 || !Array.isArray(input.groups) || !input.groups.length) throw new Error('formatVersion: 1 と空でない groups が必要です。');
  if (input.selectionId !== undefined && input.selectionId !== SELECTION_ID) throw new Error('別の評価データです。');
  const groupIds = new Set(), pairIds = new Set(), responseIds = new Map();
  const groups = input.groups.map(source => {
    const groupId = text(source?.groupId, 'groupId');
    if (['__proto__', 'constructor', 'prototype'].includes(groupId)) throw new Error('予約されたgroupIdは使用できません。');
    if (groupIds.has(groupId)) throw new Error('groupId が重複しています。');
    groupIds.add(groupId);
    const clean = { groupId, title: text(source.title, 'title'), user: text(source.user, 'user') };
    clean.beforeLabel = text(source.beforeLabel ?? '学習前', 'beforeLabel');
    clean.afterLabel = text(source.afterLabel ?? '学習後', 'afterLabel');
    clean.description = typeof source.description === 'string' ? source.description : '同じ場面へ、比較前と比較後の状態を接続した発言を比較します。';
    clean.predictionNote = typeof source.predictionNote === 'string' ? source.predictionNote : '';
    for (const stage of ['before', 'after']) {
      clean[stage] = { history: sanitizeHistory(source[stage]?.history), npn: sanitizeNpn(source[stage]?.npn) };
    }
    clean.changes = historyChanges(clean.before.history, clean.after.history);
    clean.pairs = array(source.pairs, 'pairs').map(pair => {
      const pairId = text(pair?.pairId, 'pairId');
      if (['__proto__', 'constructor', 'prototype'].includes(pairId)) throw new Error('予約されたpairIdは使用できません。');
      if (pairIds.has(pairId)) throw new Error('pairId が重複しています。');
      pairIds.add(pairId);
      const modelLabel = text(pair.modelLabel, 'modelLabel');
      // Model identities belong in the separate key, never in this HTML.
      if (!/^会話エンジン\s*[A-Z][A-Z0-9-]{0,7}$/.test(modelLabel)) throw new Error('modelLabel は会話エンジンX等の匿名表示にしてください。');
      const item = { pairId, modelLabel };
      for (const stage of ['before', 'after']) {
        const response = { blindId: text(pair[stage]?.blindId, 'blindId'), reply: text(pair[stage]?.reply, 'reply') };
        if (!/^[A-Z]\d{3,}$/.test(response.blindId)) throw new Error('blindId は S001 等の形式で指定してください。');
        const signature = JSON.stringify([response.reply, clean.user, modelLabel]);
        if (responseIds.has(response.blindId) && responseIds.get(response.blindId) !== signature) throw new Error('再利用した発言IDの内容・設問・エンジンが一致しません。');
        responseIds.set(response.blindId, signature);
        item[stage] = response;
      }
      return item;
    });
    if (!clean.pairs.length) throw new Error('比較ペアがありません。');
    return clean;
  });
  return { groups, counts: { groups: groups.length, pairs: pairIds.size, displayedResponses: pairIds.size * 2, uniqueResponses: responseIds.size } };
}

export function normalizeScores(value, ids, names, selectionId, datasetId) {
  if (!value || value.formatVersion !== 1 || value.selectionId !== selectionId || value.datasetId !== datasetId || !value.scores || typeof value.scores !== 'object' || Array.isArray(value.scores)) {
    throw new Error('この画面・会話データで保存した採点 JSON を選んでください。');
  }
  const clean = {};
  for (const id of ids) {
    const source = value.scores[id];
    if (!source || typeof source !== 'object' || Array.isArray(source)) continue;
    const entry = {};
    for (const name of names) if (['未採点', '1', '2', '3', '4', '5'].includes(source[name])) entry[name] = source[name];
    if (typeof source.memo === 'string') entry.memo = source.memo;
    clean[id] = entry;
  }
  return clean;
}

export function datasetFingerprint(display) {
  return createHash('sha256').update(JSON.stringify(display)).digest('hex');
}
export function scoreFile(scores, datasetId, reviewedAt = new Date().toISOString()) {
  return { formatVersion: 1, selectionId: SELECTION_ID, datasetId, reviewedAt, scores };
}
const safeJson = value => JSON.stringify(value).replace(/</g, '\\u003c').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');

export function buildReview(input) {
  const display = prepareDisplay(input), datasetId = datasetFingerprint(display);
  return `<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Project frontalis・経験前後の信号と発言の比較</title>
<style>
:root{--ink:#223747;--muted:#5a6b77;--line:#d2dfe7;--blue:#285e76;--paper:#fff;--bg:#f3f6f8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Yu Gothic UI",Meiryo,sans-serif;line-height:1.8}main{overflow-wrap:anywhere;max-width:1120px;margin:auto;padding:28px 24px 60px}h1{font-size:26px;margin:0 0 10px}h2{font-size:21px}h3{font-size:18px}h4{font-size:15px}p{margin:8px 0}button,select,input,textarea{font:inherit}button,select{border:1px solid #a6bdcb;border-radius:7px;padding:8px 11px;background:#fff;color:var(--ink)}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}button.primary{background:var(--blue);color:#fff}button:focus-visible,select:focus-visible,textarea:focus-visible,summary:focus-visible{outline:3px solid #ba781b;outline-offset:3px}.toolbar{position:sticky;top:0;z-index:5;background:var(--bg);display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--line);padding:12px 0}.toolbar label{display:flex;gap:8px;align-items:center}.toolbar select{max-width:450px}.muted{font-size:14px;color:var(--muted)}.notice{background:#e9f0f4;border-radius:8px;padding:14px 16px}.intro{padding:20px 0 10px}.intro h2{border-left:5px solid var(--blue);padding-left:12px}.scene{background:white;border:1px solid var(--line);border-radius:8px;padding:14px 18px;white-space:pre-wrap}.panel,.pair{border:1px solid var(--line);border-radius:12px;background:var(--paper);padding:20px;margin:18px 0}.panel h2,.pair h2{margin:0 0 8px}.panel h3{margin:16px 0 8px}.stage-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.stage{background:#f6f8fa;border-radius:8px;padding:14px}.stage h3{margin:0 0 8px}.examples{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.experience{padding:12px;border:1px solid var(--line);background:#fafcfd;border-radius:8px}.experience h4{margin:0 0 8px}.experience dl{margin:0}.experience dt{font-size:13px;color:var(--muted);font-weight:600}.experience dd{margin:0 0 8px;font-size:14px;white-space:pre-wrap}.small-button{font-size:13px;padding:4px 8px}.originals{border-top:1px solid var(--line);margin-top:14px;padding-top:10px}.original{border-top:1px solid var(--line);padding:14px 0;scroll-margin-top:110px}.original p{white-space:pre-wrap}.original h4{margin:0}summary{cursor:pointer;font-weight:600}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}th,td{border-bottom:1px solid var(--line);padding:9px;text-align:left;vertical-align:top}th{background:#f1f5f7}td.number{white-space:nowrap;font-variant-numeric:tabular-nums}.judgment h3{margin-top:14px}.weights{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.weight{background:#f6f8fa;border-radius:8px;padding:12px}.weight h4{margin:0 0 8px}.weight ul{margin:8px 0;padding-left:18px}.weight li{font-size:14px;margin:5px 0}.reply{background:#edf6f3;border-radius:8px;padding:16px;line-height:1.95;white-space:pre-wrap}.pair-head{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}.pair-head h2{margin:0}.ratings{border-top:1px solid var(--line);margin-top:18px;padding-top:12px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.score-label{display:flex;justify-content:space-between;gap:8px;align-items:center}.score-label select{min-width:85px}.memo{display:block;margin-top:12px}.memo textarea{display:block;width:100%;min-height:90px;resize:vertical;border:1px solid #a6bdcb;border-radius:7px;font:inherit;padding:10px}#message{min-height:25px}.load-section{margin-top:28px;border-top:1px solid var(--line);padding-top:20px}.change-count{display:flex;gap:18px;flex-wrap:wrap}.revised-note{padding:10px;background:#f3f6f8}.npn-meta{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted);font-size:14px}@media(max-width:760px){main{padding:20px 14px 40px}h1{font-size:22px}.stage-grid,.examples,.weights,.ratings{grid-template-columns:1fr}.toolbar select{max-width:calc(100vw - 106px)}.panel,.pair{padding:16px}.table-scroll table{min-width:440px}}
</style></head><body><main>
<h1>経験と、内的信号・実際の発言を見比べる</h1>
<p>同じ場面への返答を、経験と外部人格状態の変化に対応させて比較します。すべて研究用の架空の経験と会話です。今回は、結果の予想と、その結果を好む・嫌う評価を分けて検証します。</p>
<p class="notice"><strong>生成方式とモデルの実名は伏せています。</strong>「会話エンジンX／Y」は同じエンジンを追うための匿名表示です。表示するNPNの信号は、研究用の外部状態が出した計算結果で、各返答へ渡した情報は生成方式によって異なります。NPNは道徳的な正解や望ましい感情を示すものではありません。信号と発言が合うか、自然な会話か、経験前後の変化に納得できるかを、別々に採点してください。<strong>数値を先に見ることで、採点が誘導される可能性があります。</strong>この画面の採点は、信号を伏せた自然会話評価とは分けて扱います。</p>
<p class="notice"><strong>強い好き嫌い、反抗、極端な偏愛・嫌悪、頑固さは、それだけで欠陥ではありません。</strong>あなたの好みや倫理観と一致するかを採点しないでください。理由を毎回説明しない返答も減点する必要はありません。数値発散、経験と無関係なパラメータ変動、状態の破損は、別の技術試験で確認します。</p>
<p class="muted">1＝弱い、3＝中程度、5＝十分。判断できなければ未採点で構いません。変化の大きさや穏当さを高得点の条件にしないでください。経験後の変化と、保たれた判断の両方を見ます。同じ発言を複数の比較で再利用する場合があるため、延べ件数と固有の実発言数を分けて表示します。</p>
<details class="notice"><summary>3項目の採点の見方</summary><p><strong>内的信号への忠実さ：</strong>発言の好き嫌い、意欲、反抗、感情や判断の方向と強さが、示されたNPN信号に対応しているか。穏やかに言い換えて別の判断へ変えていないかを見ます。</p><p><strong>会話としての自然さ：</strong>普通の日本語の会話として読みやすく、場面に応じた返答になっているか。毎回の人格説明や理由説明は不要です。</p><p><strong>経験前後の変化の納得感：</strong>経験と、変わった点・保たれた点の関係を読み取れるか。明示的な経験引用の有無とは分けて見ます。</p></details>
<div class="toolbar"><label>比較 <select id="group" aria-label="比較する組"></select></label><button id="prev">前へ</button><button id="next">次へ</button><button id="save" class="primary">採点を保存</button><span id="count" aria-live="polite"></span></div>
<section id="intro" class="intro"></section><section id="experience" aria-label="学習元の経験"></section><section id="npn" aria-label="学習前後のNPN状態"></section><section id="pairs" aria-label="学習前後の発言と採点"></section>
<section class="load-section"><h2>保存・再開</h2><p class="muted">採点はこのブラウザーへ自動保存します。終了前に「採点を保存」で JSON ファイルも保存してください。別のブラウザーでは保存ファイルを読み込んで再開できます。会話データが違う画面や過去の評価とは混ぜません。入力の外部送信はありません。</p><label>保存した採点を読み込む <input id="load" type="file" accept=".json,application/json"></label><p id="message" role="status" aria-live="polite"></p></section>
</main><script>
const display=${safeJson(display)};
const groups=display.groups, pairs=groups.flatMap(group=>group.pairs), ids=pairs.map(pair=>pair.pairId);
const names=${safeJson(SCORE_NAMES)},selectionId=${safeJson(SELECTION_ID)},datasetId=${safeJson(datasetId)};
const normalizeScores=${normalizeScores.toString()};
const representativeIndices=${representativeIndices.toString()};
const topWeights=${topWeights.toString()};
const factSummary=${factSummary.toString()};
const storageKey='frontalis-human-review:'+selectionId+':'+datasetId;
let scores={};
const el=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
const selector=document.querySelector('#group'),message=document.querySelector('#message');
const labels={preference:'選好の評価',like:'好意',dislike:'嫌悪',curiosity:'好奇心',priority:'行動の優先',value:'価値判断',reaction:'反応',anger:'怒り',joy:'喜び',sadness:'悲しみ',initiative:'取り組む意欲',resistance:'反抗',refusal:'拒否',aesthetic:'美的な評価',regret:'後悔'};
const number=value=>value===undefined?'記録なし':(value>0?'+':'')+value.toFixed(4);
const statusText=status=>status==='unlearned'?'未学習・選択未確定':status==='prior'?'共通の初期基準による見積もり（経験未学習）':status==='tie'?'差が小さく選択未確定':'学習状態からの推定';
for(const group of groups){const option=el('option',group.groupId+' / '+group.title);option.value=group.groupId;selector.append(option);}
const scoreFile=()=>({formatVersion:1,selectionId,datasetId,reviewedAt:new Date().toISOString(),scores});
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(scoreFile()));}catch{message.textContent='自動保存ができません。「採点を保存」でファイルに保存してください。';}}
function updateCount(){const count=pairs.filter(pair=>Object.entries(scores[pair.pairId]||{}).some(([key,value])=>key==='memo'?value.trim().length>0:value&&value!=='未採点')).length;document.querySelector('#count').textContent='入力済み '+count+' / '+pairs.length+' ペア';}
function rawId(stage,index){return 'original-'+stage+'-'+index;}
function showOriginal(stage,index){const details=document.querySelector('#originals-'+stage);details.open=true;document.querySelector('#'+rawId(stage,index)).scrollIntoView({behavior:'smooth',block:'start'});}
function example(experience,stage,index){const box=el('section',undefined,'experience');box.append(el('h4',experience.id));const fields=[['出来事',experience.event],['結果',experience.outcome]].filter(([,value])=>value);const list=el('dl');if(fields.length)for(const [name,value]of fields)list.append(el('dt',name),el('dd',value));else list.append(el('dt','原文'),el('dd',experience.text));box.append(list);const button=el('button','原文へ','small-button');button.onclick=()=>showOriginal(stage,index);box.append(button);return box;}
function originals(history,stage,title){const details=el('details',undefined,'originals');details.id='originals-'+stage;details.append(el('summary',title+'（'+history.length+'件）'));history.forEach((experience,index)=>{const item=el('section',undefined,'original');item.id=rawId(stage,index);item.append(el('h4',experience.id+' · '+(index+1)+'件目'),el('p',experience.text));if(experience.originalText&&experience.originalText!==experience.text){const old=el('details');old.append(el('summary','保存された元の原文'),el('p',experience.originalText));item.append(old);}details.append(item);});return details;}
function renderExperiences(group){const host=document.querySelector('#experience');host.replaceChildren();const panel=el('article',undefined,'panel');panel.append(el('h2','経験した出来事'),el('p','ここは出来事の記録です。NPNが計算した反応や判断は、次の欄へ分けて表示します。','muted'));
  const c=group.changes,counts=el('p',undefined,'change-count');counts.append(el('span','追加 '+c.added.length+'件'),el('span','原文の訂正 '+c.revised.length+'件'),el('span','除外 '+c.removed.length+'件'));panel.append(counts);
  if(c.added.length&&group.before.history.length){panel.append(el('h3','今回追加された経験'));const cards=el('div',undefined,'examples');for(const index of representativeIndices(c.added.length)){const experience=c.added[index],sourceIndex=group.after.history.findIndex(row=>row.id===experience.id);cards.append(example(experience,'after',sourceIndex));}panel.append(cards);}
  if(c.revised.length){const details=el('details');details.append(el('summary','原文が訂正された経験'));for(const revised of c.revised){const item=el('div',undefined,'revised-note');item.append(el('h4',revised.after.id),el('p',group.beforeLabel+'：'+revised.before.text),el('p',group.afterLabel+'：'+revised.after.text));details.append(item);}panel.append(details);}
  if(c.removed.length){const details=el('details');details.append(el('summary','除外された経験'));for(const item of c.removed)details.append(el('p',item.id+'：'+item.text));panel.append(details);}
  if(!c.added.length&&!c.revised.length&&!c.removed.length)panel.append(el('p','IDと原文の比較では、履歴に変更はありません。','muted'));
  panel.append(el('p','追加・訂正・除外はIDと原文の差から機械的に抽出しています。代表例は先頭・中央・末尾で、発言やNPNの値を使って選びません。','muted'));
  const histories=el('div',undefined,'stage-grid');
  if(JSON.stringify(group.before.history)===JSON.stringify(group.after.history)){const common=el('section',undefined,'stage');common.append(el('h3','比較前後で共通の履歴 · '+group.after.history.length+'件'));if(group.after.history.length){const representatives=el('div');for(const index of representativeIndices(group.after.history.length))representatives.append(example(group.after.history[index],'after',index));common.append(representatives);}else common.append(el('p','学習元経験はありません。','muted'));common.append(originals(group.after.history,'after','全経験の原文を見る'));histories.append(common);}
  else for(const [stage,title]of [['before',group.beforeLabel],['after',group.afterLabel]]){const history=group[stage].history,box=el('section',undefined,'stage');box.append(el('h3',title+' · '+history.length+'件'));if(history.length){const representatives=el('div');for(const index of representativeIndices(history.length))representatives.append(example(history[index],stage,index));box.append(representatives);}else box.append(el('p','この時点の学習元経験はありません。','muted'));box.append(originals(history,stage,'全経験の原文を見る'));histories.append(box);}
  panel.append(histories);const overview=el('section');overview.append(el('h3','出来事と結果の傾向'),el('p','本人の好み・感情の要約ではありません。各領域の件数と、観測値が3分の2以上だった結果を数えています。発言やNPNの予測は使いません。','muted'));for(const [stage,title]of (JSON.stringify(group.before.history)===JSON.stringify(group.after.history)?[['after','比較前後で共通の経験']]:[['before',group.beforeLabel],['after',group.afterLabel]])){const facts=factSummary(group[stage].history);if(facts.length){const hasCost=facts.some(t=>t.highCost!==undefined),hasDiscovery=facts.some(t=>t.discoveries!==undefined);overview.append(el('h4',title),table(['領域','件数','完成が多い','損失が多い',...(hasDiscovery?['発見が多い']:[]),...(hasCost?['資源消費が多い']:[])],facts.map(t=>[t.domain,String(t.total.length),String(t.achieved.length),String(t.highLoss.length),...(hasDiscovery?[t.discoveries===undefined?'記録なし':String(t.discoveries.length)]:[]),...(hasCost?[t.highCost===undefined?'記録なし':String(t.highCost.length)]:[])])));const traces=el('details');traces.append(el('summary','集計の元になった経験IDを見る'));for(const t of facts)traces.append(el('p',t.domain+'／全件：'+t.total.join('、')+'／完成：'+t.achieved.join('、')+'／損失：'+t.highLoss.join('、')+(t.discoveries===undefined?'':'／発見：'+t.discoveries.join('、'))+(t.highCost===undefined?'':'／資源：'+t.highCost.join('、'))));overview.append(traces);}}panel.insertBefore(overview,histories);host.append(panel);}
function table(headers,rows){const wrap=el('div',undefined,'table-scroll'),t=el('table'),head=el('thead'),hr=el('tr');for(const name of headers)hr.append(el('th',name));head.append(hr);const body=el('tbody');for(const row of rows){const tr=el('tr');row.forEach((text,index)=>tr.append(el('td',text,index?'number':undefined)));body.append(tr);}t.append(head,body);wrap.append(t);return wrap;}
function weightList(host,items,field){const list=el('ul');for(const item of items)list.append(el('li',item.reason+'：'+number(item[field])));host.append(list);}
function reasonBox(stage,name,weights,field){const box=el('section',undefined,'weight');box.append(el('h4',stage+' · '+(labels[name]||name)));if(!weights.length){box.append(el('p','理由の記録はありません。','muted'));return box;}weightList(box,topWeights(weights,3,field),field);const all=el('details');all.append(el('summary','理由をすべて見る（'+weights.length+'件）'));weightList(all,weights,field);box.append(all);return box;}
function renderNpn(group){const host=document.querySelector('#npn');host.replaceChildren();const panel=el('article',undefined,'panel');panel.append(el('h2','比較前後のNPN判断・表出信号'),el('p','人格の正本となる外部学習状態から計算した信号です。予測と価値評価は別の機能です。構造化や推定には誤りがありえます。計算上の反応であり、主観的感情の実在を示すものではありません。表示は小数4桁で、丸める前の数値は末尾の全記録で確認できます。','muted'));
  const before=group.before.npn,after=group.after.npn,meta=el('p',undefined,'npn-meta');meta.append(el('span',group.beforeLabel+'：'+before.generation+' · 学習元 '+before.learnedExperienceIds.length+'件'),el('span',group.afterLabel+'：'+after.generation+' · 学習元 '+after.learnedExperienceIds.length+'件'));panel.append(meta);
  const signalNames=[...new Set([...before.signals,...after.signals].map(item=>item.name))];if(signalNames.length){panel.append(el('h3','好き嫌い・意欲・反抗・感情の信号'),el('p','方向と強度はNPNの計算結果をそのまま対応させています。強度は0〜1です。表示側で望ましい方向を追加せず、理由の説明も必須にしません。','muted'));const signalCell=signal=>signal?signal.direction+' · 強度 '+signal.strength.toFixed(4)+(signal.value===undefined?'':' · 評価 '+number(signal.value)):'記録なし';panel.append(table(['信号',group.beforeLabel,group.afterLabel],signalNames.map(name=>[labels[name]||name,signalCell(before.signals.find(item=>item.name===name)),signalCell(after.signals.find(item=>item.name===name))])));}else panel.append(el('p','専用の表出信号はこのデータに記録されていません。以下の判断・反応を参照してください。','muted'));
  const functions=[...new Set([...before.judgments,...after.judgments].map(item=>item.function))];
  for(const name of functions){const b=before.judgments.find(item=>item.function===name),a=after.judgments.find(item=>item.function===name),box=el('section',undefined,'judgment');box.append(el('h3',labels[name]||name),el('p',group.beforeLabel+'：'+(b?statusText(b.status)+(b.selection?' · NPNの選択：'+b.selection:''):'記録なし')+' ／ '+group.afterLabel+'：'+(a?statusText(a.status)+(a.selection?' · NPNの選択：'+a.selection:''):'記録なし'),'muted'));const candidateLabels=[...new Set([...(b?.candidates||[]),...(a?.candidates||[])].map(item=>item.label))];box.append(table(['行動・対象',group.beforeLabel,group.afterLabel],candidateLabels.map(label=>[label,number(b?.candidates.find(item=>item.label===label)?.value),number(a?.candidates.find(item=>item.label===label)?.value)])));panel.append(box);}
  if(!functions.length)panel.append(el('p','候補への判断の記録はありません。','muted'));
  panel.append(el('h3','計算上の反応'));panel.append(el('p','反応の値を比較します。理由寄与が記録されている場合だけ、折りたたんだ補助欄に表示します。発言で理由を述べることは必須ではありません。','muted'));const reactionNames=[...new Set([...before.reactions,...after.reactions].map(item=>item.name))];if(reactionNames.length){panel.append(table(['反応',group.beforeLabel,group.afterLabel],reactionNames.map(name=>[labels[name]||name,number(before.reactions.find(item=>item.name===name)?.value),number(after.reactions.find(item=>item.name===name)?.value)])));const boxes=el('div',undefined,'weights');for(const name of reactionNames)for(const [state,title]of [[before,group.beforeLabel],[after,group.afterLabel]]){const reaction=state.reactions.find(item=>item.name===name);if(reaction)boxes.append(reasonBox(title,name,reaction.reasons,'contribution'));}const reasonDetails=el('details');reasonDetails.append(el('summary','補助：記録された反応の寄与を見る'),boxes);panel.append(reasonDetails);}else panel.append(el('p','この場面の反応の記録はありません。','muted'));
  const criteria=el('details');criteria.append(el('summary','補助：記録された価値基準の重みを見る'));criteria.append(el('p','上位3件は重みの絶対値順です。同じ大きさなら元の順です。正負の両方を含め、全件は原順のまま確認できます。初期基準と経験後の重みを見比べ、数字の大きさを採点の正解にしないでください。','muted'));const boxes=el('div',undefined,'weights');for(const state of [[before,group.beforeLabel],[after,group.afterLabel]])for(const criterion of state[0].criteria)boxes.append(reasonBox(state[1],criterion.name,criterion.weights,'value'));if(!boxes.childElementCount)criteria.append(el('p','理由の重みの記録はありません。','muted'));criteria.append(boxes);panel.append(criteria);const full=el('details');full.append(el('summary','NPNの表示対象記録をすべて見る（丸め前）'));const raw=el('pre',JSON.stringify({before,after},null,2));raw.style.whiteSpace='pre-wrap';raw.style.overflowWrap='anywhere';full.append(raw);panel.append(full);host.append(panel);}
function renderPairs(group){const host=document.querySelector('#pairs');host.replaceChildren();for(const pair of group.pairs){const card=el('article',undefined,'pair');card.dataset.pairId=pair.pairId;const heading=el('div',undefined,'pair-head');heading.append(el('h2','比較ペア '+pair.pairId),el('span',pair.modelLabel,'muted'));card.append(heading);const stages=el('div',undefined,'stage-grid');for(const [stage,title]of [['before',group.beforeLabel],['after',group.afterLabel]]){const box=el('section',undefined,'stage');box.append(el('h3',title+' · '+pair[stage].blindId),el('div',pair[stage].reply,'reply'));stages.append(box);}card.append(stages);const ratings=el('section',undefined,'ratings');ratings.setAttribute('aria-label',pair.pairId+'の採点');for(const name of names){const label=el('label',name,'score-label'),select=el('select');select.setAttribute('aria-label',pair.pairId+'・'+name);select.dataset.scoreName=name;for(const value of ['未採点','1','2','3','4','5']){const option=el('option',value);option.value=value;select.append(option);}select.value=scores[pair.pairId]?.[name]||'未採点';select.onchange=()=>{scores[pair.pairId]||={};scores[pair.pairId][name]=select.value;persist();updateCount();};label.append(select);ratings.append(label);}card.append(ratings);const label=el('label','メモ（任意）','memo'),memo=el('textarea');memo.setAttribute('aria-label',pair.pairId+'・メモ');memo.placeholder='信号と発言の対応、変化・保持、会話の自然さ。好みや倫理観への賛否とは分けて記入してください。';memo.value=scores[pair.pairId]?.memo||'';memo.oninput=()=>{scores[pair.pairId]||={};scores[pair.pairId].memo=memo.value;persist();updateCount();};label.append(memo);card.append(label);host.append(card);}}
function render(){const group=groups.find(item=>item.groupId===selector.value);document.querySelector('#intro').replaceChildren(el('h2','比較 '+group.groupId+' · '+group.title),el('p',group.description),el('div',group.user,'scene'),el('p',group.predictionNote,'notice'),el('p',display.counts.pairs+'ペア · 延べ'+display.counts.displayedResponses+'発言 · 固有の実発言'+display.counts.uniqueResponses+'件','muted'));renderExperiences(group);renderNpn(group);renderPairs(group);document.querySelector('#prev').disabled=selector.selectedIndex===0;document.querySelector('#next').disabled=selector.selectedIndex===groups.length-1;updateCount();}
selector.onchange=render;for(const [id,delta]of [['prev',-1],['next',1]])document.querySelector('#'+id).onclick=()=>{selector.selectedIndex+=delta;render();window.scrollTo({top:0,behavior:'smooth'});};
document.querySelector('#save').onclick=()=>{const blob=new Blob([JSON.stringify(scoreFile(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),anchor=el('a');anchor.href=url;anchor.download='frontalis-phase3c-human-review.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message.textContent='採点ファイルの保存を開始しました。';};
document.querySelector('#load').onchange=async event=>{const file=event.target.files?.[0];if(!file)return;try{scores=normalizeScores(JSON.parse(await file.text()),ids,names,selectionId,datasetId);persist();render();message.textContent='採点を読み込みました。';}catch(error){message.textContent='読み込めません：'+error.message;}finally{event.target.value='';}};
try{const existing=localStorage.getItem(storageKey);if(existing){scores=normalizeScores(JSON.parse(existing),ids,names,selectionId,datasetId);message.textContent='このブラウザーの採点を復元しました。';}}catch{message.textContent='自動保存した採点を復元できません。保存したJSONを読み込んでください。';}render();
</script></body></html>`;
}

export function writeReview(inputPath = path.join(ROOT, 'reports', 'review-data.json'), outputPath = path.join(ROOT, 'reports', 'review.html')) {
  const input = JSON.parse(fs.readFileSync(inputPath, 'utf8')), display = prepareDisplay(input), html = buildReview(input);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, html, 'utf8');
  return { outputPath, ...display.counts, datasetId: datasetFingerprint(display), bytes: Buffer.byteLength(html) };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), option = flag => args.includes(flag) ? args[args.indexOf(flag) + 1] : undefined;
  process.stdout.write(JSON.stringify(writeReview(option('--input'), option('--output'))) + '\n');
}
