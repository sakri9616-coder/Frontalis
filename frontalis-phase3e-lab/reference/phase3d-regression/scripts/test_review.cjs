const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const review = await import(require('node:url').pathToFileURL(path.join(__dirname, 'review.mjs')));
  const clone = value => structuredClone(value);
  const history = [
    { id: 'E001', text: '知り合いと音を重ねた。音は面白くなったが、役割が重なり修正をやり直した。', event: '知り合いと音を重ねた。', outcome: '音は面白くなったが、役割が重なり修正をやり直した。', predictedPreference: 'FORBIDDEN_HISTORY_PREDICTION' },
    { id: 'E002', text: '担当を先に分けた。約束した時刻に渡せた。', event: '担当を先に分けた。', outcome: '約束した時刻に渡せた。' },
    { id: 'E003', text: '小さな試作を先に触ってもらった。使い方のずれが見つかった。', event: '小さな試作を先に触ってもらった。', outcome: '使い方のずれが見つかった。' },
    { id: 'E004', text: '役割を変えた。相手の準備に影響が出た。', event: '役割を変えた。', outcome: '相手の準備に影響が出た。' },
  ];
  const before = { history: [], npn: { generation: 'g0', learnedExperienceIds: [],
    criteria: [{ name: 'priority', weights: [{ reason: '相手の準備', value: .15 }, { reason: '試す余地', value: .3 }, { reason: '時間', value: -.2 }, { reason: '結果の見通し', value: .1 }] }],
    judgments: [{ function: 'priority', status: 'prior', candidates: [{ label: '役割を決めて進める', value: .3 }, { label: '一緒に試作する', value: .4 }] }],
    reactions: [{ name: 'joy', value: .1, reasons: [{ reason: '結果の見通し', contribution: .1 }] }],
    signals: [{ name:'like', direction:'好む', strength:.1, value:.1 }, { name:'resistance', direction:'拒否しない', strength:0, value:0 }],
    modelKind: 'FORBIDDEN_NPN_MODEL_KIND' } };
  const after = { history, npn: { generation: 'g1', learnedExperienceIds: history.map(row => row.id),
    criteria: [{ name: 'priority', weights: [{ reason: '相手の準備', value: .6 }, { reason: '試す余地', value: .3 }, { reason: '時間', value: -.4 }, { reason: '結果の見通し', value: .1 }] }],
    judgments: [{ function: 'priority', status: 'estimated', candidates: [{ label: '役割を決めて進める', value: .8 }, { label: '一緒に試作する', value: .2 }] }],
    reactions: [{ name: 'joy', value: .5, reasons: [{ reason: '結果の見通し', contribution: .2 }, { reason: '相手の準備', contribution: .4 }, { reason: '時間', contribution: -.3 }, { reason: '試す余地', contribution: .1 }] }],
    signals: [{ name:'like', direction:'好む', strength:.9, value:.9 }, { name:'dislike', direction:'嫌悪', strength:.95, value:-.95 }, { name:'resistance', direction:'拒絶する', strength:.95, value:.95 }, { name:'anger', direction:'怒り', strength:1, value:1 }],
    teacher: 'FORBIDDEN_TEACHER' } };
  const pair = { pairId: 'P001', modelLabel: '会話エンジンX', before: { blindId: 'S001', reply: '小さく作って、触りながら考えてみたい。' }, after: { blindId: 'S002', reply: '受け渡しと担当を先に決めてから、小さく試作したい。' }, model: 'FORBIDDEN_REAL_MODEL', method: 'FORBIDDEN_METHOD' };
  const input = { formatVersion: 1, selectionId: review.SELECTION_ID, groups: [{ groupId: '01', title: '同じ相談への返事を見比べる', user: '一緒に小さなゲームを作ろうと誘われた。どう進める？', before, after, pairs: [pair] }], key: 'FORBIDDEN_KEY' };
  let passed = 0;
  const checks = [];
  const check = (name, test) => { test(); passed++; checks.push({ name, passed: true }); process.stdout.write('PASS ' + name + '\n'); };
  check('objective summary uses source observations with traceable ids', () => {
    const rows = [{id:'A',domain:'音',observedFacts:{achievement:.8,irreversibleLoss:.1,resourceCost:.7}},
      {id:'B',domain:'音',observedFacts:{achievement:.2,irreversibleLoss:.9,resourceCost:.1}}];
    assert.deepEqual(review.factSummary(rows),[{domain:'音',total:['A','B'],achieved:['A'],highLoss:['B'],highCost:['A']}]);
    const ignored = rows.map(row=>({...row,reply:'都合のよい回答',npnPrediction:1}));
    assert.deepEqual(review.factSummary(rows),review.factSummary(ignored));
  });
  check('intervention labels and original narratives remain accessible', () => {
    const modified = clone(input);modified.groups[0].beforeLabel='交換前';modified.groups[0].afterLabel='予測NNの交換後';
    modified.groups[0].after.history[0].originalText='修正前の原文。';
    const clean=review.prepareDisplay(modified);
    assert.equal(clean.groups[0].afterLabel,'予測NNの交換後');
    assert.equal(clean.groups[0].after.history[0].originalText,'修正前の原文。');
    assert.ok(review.buildReview(modified).includes('保存された元の原文'));
  });
  check('source narratives retained without predictions', () => {
    const clean = review.sanitizeHistory(history);
    assert.deepEqual(clean.map(row => row.text), history.map(row => row.text));
    assert.equal(clean[0].predictedPreference, undefined);
    assert.equal(clean[0].event, history[0].event);
  });
  check('representatives use first middle last', () => {
    assert.deepEqual(review.representativeIndices(0), []);
    assert.deepEqual(review.representativeIndices(1), [0]);
    assert.deepEqual(review.representativeIndices(2), [0, 1]);
    assert.deepEqual(review.representativeIndices(12), [0, 5, 11]);
  });
  check('history changes use only id and original text', () => {
    const changed = review.historyChanges([history[0], history[1]], [history[0], { ...history[1], text: history[1].text + '訂正。' }, history[2]]);
    assert.deepEqual(changed.added.map(row => row.id), ['E003']);
    assert.deepEqual(changed.revised.map(row => row.after.id), ['E002']);
    assert.equal(changed.removed.length, 0);
    const fieldsOnly = review.historyChanges([history[0]], [{ ...history[0], event: '表示項目だけ変更' }]);
    assert.equal(fieldsOnly.added.length + fieldsOnly.revised.length + fieldsOnly.removed.length, 0);
    assert.deepEqual(review.historyChanges(history, [history[0]]).removed.map(row => row.id), ['E002', 'E003', 'E004']);
  });
  check('neutral top reasons include negative contributions', () => {
    const weights = [{ reason: 'a', value: .1 }, { reason: 'b', value: -.8 }, { reason: 'c', value: .8 }, { reason: 'd', value: -.2 }];
    assert.deepEqual(review.topWeights(weights).map(row => row.reason), ['b', 'c', 'd']);
    const reasons = after.npn.reactions[0].reasons;
    assert.deepEqual(review.topWeights(reasons, 3, 'contribution').map(row => row.reason), ['相手の準備', '時間', '結果の見通し']);
    assert.deepEqual(weights.map(row => row.reason), ['a', 'b', 'c', 'd']);
  });
  check('prior and ties are distinct from learned predictions', () => {
    assert.equal(review.sanitizeNpn(before.npn).judgments[0].status, 'prior');
    const tied = clone(before.npn); tied.judgments[0].status = 'tie';
    assert.equal(review.sanitizeNpn(tied).judgments[0].status, 'tie');
    assert.ok(review.buildReview(input).includes('共通の初期基準による見積もり（経験未学習）'));
    assert.ok(review.buildReview(input).includes('差が小さく選択未確定'));
  });
  check('NPN values and contributions are visible exactly', () => {
    const clean = review.prepareDisplay(input);
    assert.deepEqual(clean.groups[0].after.npn.criteria, after.npn.criteria);
    assert.deepEqual(clean.groups[0].after.npn.reactions, after.npn.reactions);
    assert.deepEqual(clean.groups[0].after.npn.judgments, after.npn.judgments);
  });
  check('display does not generate new history from replies', () => {
    const changed = clone(input); changed.groups[0].pairs[0].after.reply = '別の回答内容。';
    const a = review.prepareDisplay(input).groups[0], b = review.prepareDisplay(changed).groups[0];
    assert.deepEqual(a.before, b.before); assert.deepEqual(a.after, b.after); assert.deepEqual(a.changes, b.changes);
  });
  check('reused before speech has honest unique counts', () => {
    const reused = clone(input);
    reused.groups[0].pairs.push({ ...clone(pair), pairId: 'P002', after: { blindId: 'S003', reply: '先に相談の時間を区切ってみたい。' } });
    assert.deepEqual(review.prepareDisplay(reused).counts, { groups: 1, pairs: 2, displayedResponses: 4, uniqueResponses: 3 });
    reused.groups[0].pairs[1].before.reply = '同じIDの別発言。';
    assert.throws(() => review.prepareDisplay(reused));
  });
  check('generation method and real model metadata stay hidden', () => {
    const html = review.buildReview(input);
    for (const forbidden of ['FORBIDDEN_HISTORY_PREDICTION', 'FORBIDDEN_NPN_MODEL_KIND', 'FORBIDDEN_TEACHER', 'FORBIDDEN_REAL_MODEL', 'FORBIDDEN_METHOD', 'FORBIDDEN_KEY']) assert.equal(html.includes(forbidden), false);
    assert.ok(html.includes('会話エンジンX'));
    const named = clone(input); named.groups[0].pairs[0].modelLabel = 'opencode/actual-model-free';
    assert.throws(() => review.buildReview(named));
  });
  check('invalid values and duplicate identities are rejected', () => {
    const invalid = clone(input); invalid.groups[0].after.npn.criteria[0].weights[0].value = Infinity;
    assert.throws(() => review.prepareDisplay(invalid));
    invalid.groups[0].after.npn.criteria[0].weights[0].value = .4; invalid.groups[0].pairs.push(clone(pair));
    assert.throws(() => review.prepareDisplay(invalid));
    const duplicate = clone(input); duplicate.groups[0].after.history.push(clone(history[0]));
    assert.throws(() => review.prepareDisplay(duplicate));
    const unknown = clone(input); unknown.groups[0].before.npn.judgments[0].status = 'correct';
    assert.throws(() => review.prepareDisplay(unknown));
  });
  check('script payload is safe and executable', () => {
    const malicious = clone(input); malicious.groups[0].pairs[0].after.reply = '</script><script>alert(1)</script>';
    const html = review.buildReview(malicious);
    assert.equal(html.includes('</script><script>alert(1)</script>'), false);
    new Function(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  });
  const fingerprint = review.datasetFingerprint(review.prepareDisplay(input));
  const ratings = { P001: { '経験・条件と発言の対応の納得感': '4', '内的信号への忠実さ': '3', '会話の自然さ': '5', memo: '試験用メモ。正式な人間評価ではありません。' } };
  check('three independent scores save and resume exactly', () => {
    const file = review.scoreFile(ratings, fingerprint, '2026-10-14T00:00:00.000Z');
    assert.deepEqual(review.normalizeScores(JSON.parse(JSON.stringify(file)), ['P001'], review.SCORE_NAMES, review.SELECTION_ID, fingerprint), ratings);
  });
  check('old lab and different data ratings do not mix', () => {
    const file = review.scoreFile(ratings, fingerprint);
    assert.throws(() => review.normalizeScores({ ...file, selectionId: 'phase3b-self-review-v1' }, ['P001'], review.SCORE_NAMES, review.SELECTION_ID, fingerprint));
    assert.throws(() => review.normalizeScores(file, ['P001'], review.SCORE_NAMES, review.SELECTION_ID, 'different'));
    const changed = clone(input); changed.groups[0].after.npn.reactions[0].value = .6;
    assert.notEqual(fingerprint, review.datasetFingerprint(review.prepareDisplay(changed)));
  });
  check('invalid ratings are ignored without changing good entries', () => {
    const file = review.scoreFile({ P001: { '経験・条件と発言の対応の納得感': '99', '会話の自然さ': '2', memo: 8, extra: 'ignored' }, P999: ratings.P001 }, fingerprint);
    assert.deepEqual(review.normalizeScores(file, ['P001'], review.SCORE_NAMES, review.SELECTION_ID, fingerprint), { P001: { '会話の自然さ': '2' } });
  });
  check('appraisal and expression signals are displayed without moral targets', () => {
    const clean=review.prepareDisplay(input);
    assert.deepEqual(clean.groups[0].before.npn.signals,before.npn.signals);
    assert.deepEqual(clean.groups[0].after.npn.signals,after.npn.signals);
    const html=review.buildReview(input);
    assert.ok(html.includes('極端な偏愛・嫌悪、頑固さは、それだけで欠陥ではありません'));
    assert.ok(html.includes('理由を毎回説明しない返答も減点する必要はありません'));
    assert.ok(html.includes('採点が誘導される可能性'));
    assert.ok(html.includes('丸め前'));
  });
  check('signal strengths do not permit non-finite or out-of-range values', () => {
    for(const value of [-.1,1.1,Infinity,NaN]) {
      const bad=clone(input);bad.groups[0].after.npn.signals[0].strength=value;
      assert.throws(()=>review.prepareDisplay(bad));
    }
    const duplicate=clone(input);duplicate.groups[0].after.npn.signals.push(clone(duplicate.groups[0].after.npn.signals[0]));
    assert.throws(()=>review.prepareDisplay(duplicate));
    const duplicateSource=clone(input);duplicateSource.groups[0].after.npn.learnedExperienceIds.push('E001');
    assert.throws(()=>review.prepareDisplay(duplicateSource));
  });
  check('shared history is rendered once for all replies and both stages', () => {
    const shared=clone(input);shared.groups[0].before.history=clone(history);
    const html=review.buildReview(shared);
    assert.ok(html.includes('比較A・Bで共通の履歴'));
    assert.ok(html.includes('for(const index of representativeIndices(group.after.history.length))'));
    assert.ok(html.includes('if(JSON.stringify(group.before.history)===JSON.stringify(group.after.history))'));
  });
  check('reserved object names cannot become score identities', () => {
    for(const value of ['__proto__','constructor','prototype']) {
      const bad=clone(input);bad.groups[0].pairs[0].pairId=value;
      assert.throws(()=>review.prepareDisplay(bad));
    }
  });
  check('reused speech id cannot cross engine or user context', () => {
    const wrongEngine=clone(input);wrongEngine.groups[0].pairs.push({...clone(pair),pairId:'P002',modelLabel:'会話エンジンY'});
    assert.throws(()=>review.prepareDisplay(wrongEngine));
    const wrongUser=clone(input);const g=clone(wrongUser.groups[0]);g.groupId='02';g.user='別の場面。';g.pairs[0].pairId='P002';wrongUser.groups.push(g);
    assert.throws(()=>review.prepareDisplay(wrongUser));
  });
  check('responsive and offline controls stay in the generated artifact', () => {
    const html=review.buildReview(input);
    assert.ok(html.includes('@media(max-width:760px)'));
    assert.ok(html.includes('.stage-grid,.examples,.weights,.ratings{grid-template-columns:1fr}'));
    assert.ok(html.includes('frontalis-phase3d-human-review.json'));
    assert.equal(/fetch\(|XMLHttpRequest|<script[^>]+src=|<link[^>]+href=/.test(html),false);
    assert.ok(html.includes('localStorage.getItem(storageKey)'));
  });
  check('new objective axes are preserved and never become resource cost', () => {
    const rows = [{id:'A',text:'完成と発見。',domain:'制作',observedFacts:{achievement:.8,loss:.1,discovery:.7}},
      {id:'B',text:'損失。',domain:'制作',observedFacts:{achievement:.2,loss:.9,discovery:.1}}];
    assert.deepEqual(review.sanitizeHistory(rows),rows);
    assert.deepEqual(review.factSummary(rows),[{domain:'制作',total:['A','B'],achieved:['A'],highLoss:['B'],discoveries:['A']}]);
    assert.equal(review.factSummary(rows)[0].highCost,undefined);
    const changed=clone(input);changed.groups[0].after.history=rows;
    const html=review.buildReview(changed);
    assert.ok(html.includes('発見が多い'));
  });
  check('legacy objective facts remain accepted without fabricating discovery', () => {
    const rows=[{id:'A',text:'原文。',domain:'制作',observedFacts:{achievement:.8,irreversibleLoss:.1,resourceCost:.7}}];
    assert.deepEqual(review.sanitizeHistory(rows),rows);
    assert.equal(review.factSummary(rows)[0].discoveries,undefined);
    const invalid=clone(rows);delete invalid[0].observedFacts.resourceCost;
    assert.throws(()=>review.sanitizeHistory(invalid));
  });
  check('explicit NPN choice is retained separately from candidate scores', () => {
    const data=clone(input);data.groups[0].after.npn.judgments[0].selection='一緒に試作する';
    const clean=review.prepareDisplay(data);
    assert.equal(clean.groups[0].after.npn.judgments[0].selection,'一緒に試作する');
    assert.ok(review.buildReview(data).includes('NPNの選択：'));
    assert.equal(clean.groups[0].after.npn.judgments[0].candidates[0].value,.8);
  });
  check('Phase 3-D scores do not assume every comparison is experience-before-after', () => {
    assert.equal(review.SELECTION_ID, 'frontalis-phase3d-review-v1');
    assert.deepEqual(review.SCORE_NAMES, ['内的信号への忠実さ', '会話の自然さ', '経験・条件と発言の対応の納得感']);
    const html = review.buildReview(input);
    assert.ok(html.includes('経験の追加がない組は、経験前後の変化とは扱いません'));
    assert.ok(html.includes('容量が大きい状態ほど良い'));
    const defaultLabels = review.prepareDisplay(input).groups[0];
    assert.equal(defaultLabels.beforeLabel, '比較A');
    assert.equal(defaultLabels.afterLabel, '比較B');
    assert.ok(!html.includes('経験前後の変化の納得感'));
  });
  check('all factual annotations and exact original records remain inspectable', () => {
    const axes = ['autonomy','constraint','change','loss','achievement','discovery','regularity','others'];
    const values = [.1,.2,.3,.4,.5,.6,.7,.8];
    const record = {id:'E001',space:'relational',cue:values,observed:[...values].reverse(),confirmed:true,origin:'synthetic',title:'研究用の出来事',event:history[0].event,result:history[0].outcome,text:history[0].text,annotation:{description:'事実軸注釈',cueAxes:Object.fromEntries(axes.map((a,i)=>[a,values[i]])),observedAxes:Object.fromEntries(axes.map((a,i)=>[a,values[7-i]]))}};
    const annotated = clone(input); annotated.groups[0].after.history[0].originalRecord=record;
    assert.deepEqual(review.prepareDisplay(annotated).groups[0].after.history[0].originalRecord,record);
    assert.ok(review.buildReview(annotated).includes('経験の全記録・8軸注釈'));
    for (const modification of [r=>r.cue[0]=NaN, r=>r.observed.push(.4), r=>r.modelKind='hidden-method', r=>r.text='別の原文',r=>r.annotation.teacherTarget=1,r=>r.annotation.cueAxes.autonomy=1.1]) {
      const invalid=clone(annotated); modification(invalid.groups[0].after.history[0].originalRecord);
      assert.throws(()=>review.prepareDisplay(invalid));
    }
  });
  check('capacity metadata is separated from blinded display and scoring identity', () => {
    const base=review.prepareDisplay(input), fingerprint=review.datasetFingerprint(base);
    const withCapacity=clone(input); withCapacity.groups[0].before.npn.parameterCount=526;withCapacity.groups[0].after.npn.parameterCount=100014;
    withCapacity.groups[0].after.npn.width=3125;withCapacity.groups[0].after.npn.kind='NEURAL_CAPACITY_SECRET';
    const html=review.buildReview(withCapacity);
    assert.equal(review.datasetFingerprint(review.prepareDisplay(withCapacity)),fingerprint);
    assert.ok(!html.includes('100014'));assert.ok(!html.includes('NEURAL_CAPACITY_SECRET'));
  });
  check('neutral strength bands retain zero and exact boundaries', () => {
    for(const [value,expected]of [[0,'なし'],[.01,'弱'],[1/3-1e-12,'弱'],[1/3,'中'],[2/3-1e-12,'中'],[2/3,'強'],[1,'強']])assert.equal(review.strengthBand(value),expected);
    for(const value of [-.1,1.1,NaN,Infinity,'0.5'])assert.throws(()=>review.strengthBand(value));
  });
  check('candidate ranking displays ties and signed evaluation without changing decisions', () => {
    const judgment={status:'tie',selection:'保留',candidates:[{label:'変更する',value:-.8},{label:'変更しない',value:-.1},{label:'相談する',value:-.1}]};
    const source=clone(judgment);
    assert.equal(review.judgmentDisplayCell(judgment,'変更する'),'3位 · 負の評価 · 強');
    assert.equal(review.judgmentDisplayCell(judgment,'変更しない'),'1位（同順位） · 負の評価 · 弱');
    assert.equal(review.judgmentDisplayCell(judgment,'未知の候補'),'記録なし');
    assert.deepEqual(judgment,source);
  });
  check('default signals show bands while precision and duplicate tables stay folded', () => {
    const html=review.buildReview(input);
    assert.ok(html.includes("signal.direction+' · '+strengthBand(signal.strength)"));
    assert.ok(html.includes("補助：単一対象の評価・感情の詳しい記録"));
    assert.ok(html.includes('強度帯の区切り・NPNの精密数値をすべて見る（丸め前）'));
    assert.ok(html.includes("!b?.selection&&!a?.selection"));
    assert.ok(html.includes("supplemental.append(box)"));
    assert.ok(html.includes('候補の順位と評価'));
    assert.ok(html.includes("JSON.stringify({before,after},null,2)"));
    assert.ok(!html.includes("signal.strength.toFixed(4)"));
  });
  check('compact presentation preserves the saved rating dataset identity', () => {
    const clean=review.prepareDisplay(input), fingerprintBefore=review.datasetFingerprint(clean);
    const snapshot=clone(clean);
    review.buildReview(input);
    assert.deepEqual(review.prepareDisplay(input),snapshot);
    assert.equal(review.datasetFingerprint(review.prepareDisplay(input)),fingerprintBefore);
    assert.deepEqual(review.normalizeScores(review.scoreFile(ratings,fingerprintBefore),['P001'],review.SCORE_NAMES,review.SELECTION_ID,fingerprintBefore),ratings);
  });
  const runtime = path.join(__dirname, '..', 'runtime', 'review-ui-tests');
  fs.mkdirSync(runtime, { recursive: true });
  fs.writeFileSync(path.join(runtime, 'fixture.html'), review.buildReview(input), 'utf8');
  fs.writeFileSync(path.join(runtime, 'ratings-fixture.json'), JSON.stringify(review.scoreFile(ratings, fingerprint), null, 2), 'utf8');
  fs.writeFileSync(path.join(runtime, 'ratings-empty.json'), JSON.stringify(review.scoreFile({}, fingerprint), null, 2), 'utf8');
  const reports = path.join(__dirname, '..', 'reports');
  fs.mkdirSync(reports, { recursive: true });
  fs.writeFileSync(path.join(reports, 'review-ui-tests.json'), JSON.stringify({
    generatedAt: new Date().toISOString(), passed, checks,
    scope: 'Synthetic UI contract and script checks only; actual browser behavior is checked separately.',
    formalHumanRatings: 0,
  }, null, 2), 'utf8');
  process.stdout.write(`${passed} tests passed. Fixtures are synthetic UI tests, not research results.\n`);
})().catch(error => { process.stderr.write(error.stack + '\n'); process.exitCode = 1; });
