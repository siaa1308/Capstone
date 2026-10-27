const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const fixture = (bank, id) => JSON.parse(fs.readFileSync(path.join(root,
  'outputs/updated/adaptive_federated/seed_42/local', bank, 'reports', `validation_${id}`, 'explanation.json'), 'utf8'));
function harness(fetch) {
  const elements = new Map();
  function element(selector) {
    if (!elements.has(selector)) elements.set(selector, {
      value: selector === '#bank' ? 'Wells_Fargo' : selector === '#view' ? 'flagged' : '',
      innerHTML: '', textContent: '', dataset: {}, disabled: false,
      addEventListener() {}, querySelectorAll() { return []; },
    });
    return elements.get(selector);
  }
  const context = vm.createContext({document: {querySelector: element, querySelectorAll: () => []},
    URLSearchParams, location: {search: ''}, history: {replaceState() {}}, fetch,
    setTimeout, clearTimeout});
  const source = fs.readFileSync(path.join(root, 'static/dashboard.js'), 'utf8').replace(/\ninit\(\);\s*$/, '\n');
  vm.runInContext(source + '\nthis.api={render,validateExplanation,loadRows,loadExplanation,graphEdgeDetails,readableTechnicalDetails,observedPatterns};', context);
  return {api: context.api, element};
}

test('every cached bank/transaction renders feature directions from its own SHAP', () => {
  const h = harness();
  let count = 0;
  for (const bank of ['JPMorgan_Chase', 'Wells_Fargo', 'Key_Bank']) {
    const folder = path.join(root, 'outputs/updated/adaptive_federated/seed_42/local', bank, 'reports');
    for (const name of fs.readdirSync(folder)) {
      const reportPath = path.join(folder, name, 'explanation.json');
      if (!fs.existsSync(reportPath)) continue;
      const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
      h.api.validateExplanation(report, bank, report.transaction_id);
      h.api.render(report);
      const html = h.element('#detail').innerHTML;
      assert.ok(html.includes(`<h2>Transaction ${report.transaction_id}</h2>`));
      assert.ok(!html.includes('The main factor or its direction changed'));
      assert.ok(!html.includes('Reference stability'));
      for (const feature of report.feature_explanation.features) {
        const direction = Math.abs(feature.shap_logit) < 1e-8 ? 'No clear measured effect'
          : feature.shap_logit > 0 ? 'Pushed the score higher' : 'Pushed the score lower';
        const label = feature.feature === 'transfer_terms' ? 'Amount and currencies' : feature.label;
        assert.ok(html.includes(`<strong>${label}</strong><span class="chip ${Math.abs(feature.shap_logit)<1e-8?'':feature.shap_logit>0?'alert':'known'}">${direction}`));
      }
      assert.ok(!html.includes('undefined'));
      count++;
    }
  }
  assert.ok(count >= 6);
});

test('empty eligible history renders an accurate message and no history table', () => {
  const h = harness();
  h.api.render(fixture('Wells_Fargo', '2954161'));
  const html = h.element('#detail').innerHTML;
  assert.ok(html.includes('No earlier transfers involving either account were available in the model’s eligible history'));
  assert.ok(!html.includes('no verified score test'));
  assert.ok(!html.includes('class="recorded-history"'));
  assert.ok(!html.includes('class="flow-table"'));
  assert.ok(html.includes('Observed amount / currencies:'));
});

test('untested existing history is not described as absent', () => {
  const report = fixture('JPMorgan_Chase', '180245');
  report.transaction_story.model_link_effects = [];
  const h = harness(); h.api.render(report);
  const html = h.element('#detail').innerHTML;
  assert.ok(html.includes('Earlier transfers are available, but no verified score test is available'));
  assert.ok(html.includes('class="recorded-history"'));
  assert.ok(!html.includes('No earlier transfers involving either account were available'));
});

test('mixed identities, scores, feature values, decisions and stale graph effects are rejected', () => {
  const h = harness();
  const mutations = [
    r => r.transaction_id = 'wrong',
    r => r.transaction_story.selected_transfer.transaction_id = 'wrong',
    r => r.feature_explanation.target_probability = .7,
    r => r.feature_explanation.features[0].value = 'Wire',
    r => r.feature_explanation.features[1].value = 'amount: 0 / currency: USD',
    r => r.prediction.class = 'not_fraud',
    r => r.transaction_story.model_link_effects[0].original_model_score = .8,
  ];
  for (const mutate of mutations) {
    const report = fixture('JPMorgan_Chase', '180245'); mutate(report);
    assert.throws(() => h.api.validateExplanation(report, 'JPMorgan_Chase', '180245'), /consistency check failed/);
  }
});

function deferredFetch() {
  const requests = [];
  return {requests, fetch: url => new Promise(resolve => requests.push({url, resolve: data => resolve({ok:true,json:async()=>data})}))};
}
test('out-of-order catalog responses cannot overwrite the current bank threshold', async () => {
  const deferred = deferredFetch(), h = harness(deferred.fetch);
  const first = h.api.loadRows();
  h.element('#bank').value = 'Key_Bank';
  const second = h.api.loadRows();
  const catalog = (bank, threshold) => ({bank,threshold,total:0,counts:{known_fraud:0,missed_fraud:0},transactions:[]});
  deferred.requests[1].resolve(catalog('Key_Bank', .3759)); await second;
  deferred.requests[0].resolve(catalog('Wells_Fargo', .1869)); await first;
  assert.ok(h.element('#threshold').innerHTML.includes('0.3759'));
  assert.ok(!h.element('#threshold').innerHTML.includes('0.1869'));
});

test('out-of-order explanation responses cannot overwrite the selected transaction', async () => {
  const deferred = deferredFetch(), h = harness(deferred.fetch);
  h.element('#bank').value = 'JPMorgan_Chase';
  const first = h.api.loadExplanation('180245'), second = h.api.loadExplanation('2580817');
  deferred.requests[1].resolve(fixture('JPMorgan_Chase','2580817')); await second;
  deferred.requests[0].resolve(fixture('JPMorgan_Chase','180245')); await first;
  assert.ok(h.element('#detail').innerHTML.includes('<h2>Transaction 2580817</h2>'));
});


test('selected transfer cards across every bank show model score without historical-test wording', () => {
  const h = harness();
  for (const [bank,id] of [['Wells_Fargo','2954161'],['JPMorgan_Chase','180245'],['Key_Bank','2974317']]) {
    const report = fixture(bank,id);
    const html = h.api.graphEdgeDetails(report.transaction_story.selected_transfer, undefined, report.prediction);
    assert.ok(html.includes(`Model score: ${report.prediction.probability.toFixed(4)}`));
    assert.ok(!html.includes('Connection only:'));
    assert.ok(!html.includes('With this earlier transfer'));
    assert.ok(!html.includes('Without its history update'));
  }
  const report = fixture('JPMorgan_Chase','180245');
  const earlier = report.transaction_story.transfers.find(event => event.transaction_id === '180238');
  const effect = report.transaction_story.model_link_effects.find(effect => effect.transaction_id === '180238');
  const measured = h.api.graphEdgeDetails(earlier,effect,report.prediction);
  assert.ok(measured.includes('Helped lower the score'));
  assert.ok(measured.includes('Without its history update: 0.2785'));
  const context = h.api.graphEdgeDetails(earlier,undefined,report.prediction);
  assert.ok(context.includes('Selected transaction’s model score: 0.0862'));
  assert.ok(!context.includes('score effect has not been tested'));
});


test('technical threshold is bank-specific and a different bank threshold is rejected', () => {
  const h = harness();
  for (const [bank,id,threshold] of [['JPMorgan_Chase','180245','0.01780519'],['Wells_Fargo','2954161','0.18689416'],['Key_Bank','2974317','0.37591952']]) {
    const report=fixture(bank,id);
    h.api.validateExplanation(report,bank,id);
    h.api.render(report);
    assert.ok(h.element('#detail').innerHTML.includes(`Saved configured threshold for <strong>${bank.replaceAll('_',' ')}</strong> (validation-selected): <strong>${threshold}</strong>`));
  }
  const wrong=fixture('Wells_Fargo','2954161');
  wrong.prediction.threshold=wrong.xai_provenance.banks.JPMorgan_Chase.threshold;
  assert.throws(()=>h.api.validateExplanation(wrong,'Wells_Fargo','2954161'),/threshold differs from the selected bank/);
});


test('plain-language technical details adapt to selected score, directions, history and available checks', () => {
  const h=harness();
  const wells=fixture('Wells_Fargo','2954161');
  const empty=h.api.readableTechnicalDetails(wells);
  assert.ok(empty.includes('0.20514992'));
  assert.ok(empty.includes('-0.0349'));
  assert.ok(empty.includes('Contributed toward a lower output'));
  assert.ok(empty.includes('there was no earlier transfer to remove and test'));
  assert.ok(empty.includes('Not recorded for this transaction. This does not mean the check passed.'));
  assert.ok(!empty.includes('0.53210872'));
  const key=h.api.readableTechnicalDetails(fixture('Key_Bank','2974317'));
  assert.ok(key.includes('+2.7856'));
  assert.ok(key.includes('1 earlier transfer has a verified score test'));
  assert.ok(!key.includes('The leading factor changed.'));
  const untested=h.api.readableTechnicalDetails(fixture('JPMorgan_Chase','2580817'));
  assert.ok(untested.includes('Earlier transfers are recorded, but no verified score-effect test is available'));
  const legitimate=h.api.readableTechnicalDetails(fixture('Wells_Fargo','2965480'));
  assert.ok(legitimate.includes('Why was this transaction not flagged?'));
  assert.ok(legitimate.includes('below'));
});


test('observed pattern labels count distinct peers and require a chronological cycle', () => {
 const h=harness();
 const event=(source,target,day,kind='earlier')=>({source,target,timestamp:`2025-07-${String(day).padStart(2,'0')}T00:00:00`,kind});
 const selected=event('A','B',20,'selected');
 const labels=earlier=>Array.from(h.api.observedPatterns({selected_transfer:selected,transfers:[...earlier,selected]}),p=>p.label);
 assert.deepEqual(labels([]),[]);
 assert.deepEqual(labels([event('A','B',1),event('A','B',2)]),[]);
 assert.ok(labels([event('A','C',1),event('A','D',2)]).includes('Fan-out'));
 assert.ok(labels([event('C','B',1),event('D','B',2)]).includes('Fan-in'));
 assert.ok(labels([event('B','C',1),event('C','A',2)]).includes('Cycle'));
 assert.ok(!labels([event('B','C',2),event('C','A',1)]).includes('Cycle'));
 assert.ok(!labels([event('B','A',21)]).includes('Cycle'));
});
