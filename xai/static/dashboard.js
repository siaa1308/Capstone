'use strict';
const $=selector=>document.querySelector(selector);
const bank=$('#bank'),view=$('#view'),query=$('#query'),body=$('#rows');
let currentPage=0,total=0,selected=null,requestSequence=0,catalogSequence=0,debounce;
const pageSize=75;
const esc=value=>String(value??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const score=value=>Number(value).toFixed(4);
const preciseScore=value=>Number(value).toFixed(8);
const amount=event=>event.amount==null?'Amount unavailable':`${Number(event.amount).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})} ${esc(event.currency||'')}`;
const time=value=>esc(String(value||'').replace('T',' '));
async function json(url){const r=await fetch(url);const data=await r.json();if(!r.ok)throw Error(data.error||'The server could not complete this request.');return data}

async function loadRows(){
 const sequence=++catalogSequence;
 body.innerHTML='<tr><td colspan="4">Loading transactions…</td></tr>';
 $('#threshold').textContent='Loading configured review threshold…';
 $('#count').textContent='';
 try{
  const params=new URLSearchParams({bank:bank.value,view:view.value,q:query.value,page:String(currentPage),page_size:String(pageSize)});
  const d=await json('/api/catalog?'+params);if(sequence!==catalogSequence)return;total=d.total;
  if(d.bank!==bank.value)throw Error('Transaction catalog does not match the selected bank.');
  $('#threshold').innerHTML=`Configured review threshold: <strong>${score(d.threshold)}</strong>. Selected using validation data; scores at or above it are flagged.`;
  $('#count').textContent=`${total.toLocaleString()} matching transactions · ${d.counts.known_fraud} known fraud · ${d.counts.missed_fraud} missed fraud`;
  body.innerHTML=d.transactions.length?d.transactions.map(x=>`<tr class="case ${selected===x.transaction_id?'selected':''}" data-id="${esc(x.transaction_id)}"><td>${esc(x.transaction_id)}</td><td>${score(x.probability)}</td><td><span class="chip ${x.label_value?'known':''}">${x.label_value?'Fraud':'Not fraud'}</span></td><td><span class="chip ${x.flagged?'alert':''}">${x.flagged?'Flagged':'Not flagged'}</span></td></tr>`).join(''):'<tr><td colspan="4">No transactions match this view.</td></tr>';
  const pages=Math.max(1,Math.ceil(total/pageSize));$('#page').textContent=`${currentPage+1} / ${pages}`;
  $('#prev').disabled=currentPage===0;$('#next').disabled=currentPage+1>=pages;
  body.querySelectorAll('.case').forEach(row=>row.addEventListener('click',()=>loadExplanation(row.dataset.id)));
 }catch(error){if(sequence===catalogSequence)body.innerHTML=`<tr><td colspan="4" class="error">${esc(error.message)}</td></tr>`}
}
async function loadExplanation(id){
 selected=id;const sequence=++requestSequence;
 body.querySelectorAll('.case').forEach(row=>row.classList.toggle('selected',row.dataset.id===id));
 $('#detail').innerHTML=`<p class="loading">Preparing the transfer map and checking the model evidence for ${esc(id)}. The first visit can take a few minutes; verified results are saved for later visits.</p>`;
 try{const params=new URLSearchParams({bank:bank.value,transaction_id:id});const x=await json('/api/explanation?'+params);if(sequence===requestSequence){validateExplanation(x,bank.value,id);history.replaceState(null,'','?'+params);render(x)}}
 catch(error){if(sequence===requestSequence)$('#detail').innerHTML=`<h2>Explanation unavailable</h2><p class="error">${esc(error.message)}</p><p class="muted">The transaction is still listed. A verified model explanation could not be loaded for this request.</p>`}
}

function validateExplanation(x,expectedBank,expectedId){
 const fail=message=>{throw Error('Explanation consistency check failed: '+message)};
 if(x.bank!==expectedBank||String(x.transaction_id)!==String(expectedId))fail('requested transaction identity differs');
 const story=x.transaction_story,event=story?.selected_transfer,prediction=x.prediction,feature=x.feature_explanation;
 if(!event||String(event.transaction_id)!==String(expectedId))fail('graph explains a different transaction');
 if(!prediction||![prediction.probability,prediction.threshold].every(value=>typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=1))fail('invalid score or threshold');
 const savedThreshold=x.xai_provenance?.banks?.[expectedBank]?.threshold;
 if(!Number.isFinite(savedThreshold)||Math.abs(prediction.threshold-savedThreshold)>1e-12)fail('threshold differs from the selected bank’s saved threshold');
 if(!['fraud','not_fraud'].includes(prediction.class)||(prediction.probability>=prediction.threshold)!==(prediction.class==='fraud'))fail('decision differs from score and threshold');
 if(!feature||!Number.isFinite(feature.target_probability)||Math.abs(feature.target_probability-prediction.probability)>1e-6)fail('SHAP targets a different score');
 if(!feature.features?.length||feature.features.some(item=>!Number.isFinite(item.shap_logit)))fail('invalid feature contributions');
 const payment=feature.features.find(item=>item.feature==='payment_method');
 if(payment&&String(payment.value).replace(/^payment_format:\s*/,'')!==String(event.payment_method))fail('payment feature differs from selected transfer');
 const terms=feature.features.find(item=>item.feature==='transfer_terms');
 if(terms){const fields=transferFields(terms);if(!Number.isFinite(Number(fields.amount))||!Number.isFinite(Number(event.amount))||Math.abs(Number(fields.amount)-Number(event.amount))>1e-8||fields.currency!==event.currency)fail('amount/currency feature differs from selected transfer')}
 const selectedEvents=(story.transfers||[]).filter(item=>item.kind==='selected');
 if(selectedEvents.length!==1||String(selectedEvents[0].transaction_id)!==String(expectedId))fail('selected graph edge differs');
 for(const key of ['source_account_id','target_account_id','timestamp','amount','currency']){
  if(String(selectedEvents[0][key])!==String(event[key]))fail('selected graph edge details differ');
 }
 if((story.model_link_effects||[]).length&&!Number(story.earlier_related_transfers_considered)&&!(story.transfers||[]).some(item=>item.kind==='earlier'))fail('history effects have no eligible history');
 for(const effect of story.model_link_effects||[]){
  if(!Number.isFinite(effect.original_model_score)||Math.abs(effect.original_model_score-prediction.probability)>1e-6)fail('history effect targets a different score');
  if(!Number.isFinite(effect.model_score_without_event)||!Number.isFinite(effect.score_support)||Math.abs(effect.score_support-(effect.original_model_score-effect.model_score_without_event))>1e-6)fail('history effect arithmetic differs');
 }
}

function observedPatterns(story){
 const key=(event,side)=>String(event[side+'_account_id']??event[side]);
 const selected=story.selected_transfer;
 const source=key(selected,'source'),target=key(selected,'target');
 const when=event=>Date.parse(String(event.timestamp).replace(' ','T'));
 const selectedTime=when(selected);
 const earlier=(story.transfers||[]).filter(event=>event.kind==='earlier'&&when(event)<selectedTime);
 const events=[...earlier,selected],patterns=[];
 const senders=new Set(events.filter(event=>key(event,'target')===target&&key(event,'source')!==target).map(event=>key(event,'source')));
 const recipients=new Set(events.filter(event=>key(event,'source')===source&&key(event,'target')!==source).map(event=>key(event,'target')));
 if(senders.size>=3)patterns.push({label:'Fan-in',description:`Account ${target} received transfers from ${senders.size} distinct accounts.`});
 if(recipients.size>=3)patterns.push({label:'Fan-out',description:`Account ${source} sent transfers to ${recipients.size} distinct accounts.`});
 // A chronological earlier path B -> ... -> A closes with the selected A -> B.
 function returnPath(node,lastTime,visited){
  for(const event of earlier){
   if(key(event,'source')!==node||when(event)<=lastTime)continue;
   const next=key(event,'target');
   if(next===source)return true;
   if(!visited.has(next)&&returnPath(next,when(event),new Set([...visited,next])))return true;
  }
  return false;
 }
 if(source!==target&&returnPath(target,-Infinity,new Set([target])))patterns.push({label:'Cycle',description:'Earlier transfers form a time-ordered return path that closes with the selected transfer.'});
 return patterns;
}
function patternLabels(story){
 const patterns=observedPatterns(story);
 return `<div class="observed-patterns"><span class="muted">Observed transaction patterns:</span> ${patterns.length?patterns.map(pattern=>`<span class="pattern-chip" title="${esc(pattern.description)}">${pattern.label}</span>`).join(' '):'<span class="muted">No fan-in, fan-out or cycle found in the displayed transfers.</span>'}<small>Based only on the transfers shown, including the selected transfer. Fan-in/out requires at least 3 distinct counterpart accounts. These labels describe connections, not confirmed fraud or measured score effects.</small></div>`;
}
function flowDiagram(story){
 const measured=new Map((story.model_link_effects||[]).map(effect=>[String(effect.transaction_id),effect]));
 const positions=new Map(),others=[[360,55],[360,305],[215,65],[505,295],[505,65],[215,295]];let connectedIndex=0;
 for(const account of story.accounts){positions.set(account.alias,account.alias==='A'?[125,180]:account.alias==='B'?[595,180]:others[connectedIndex++]||[360,180])}
 const pairs=new Map();
 const paths=story.transfers.map(event=>{
  const [ax,ay]=positions.get(event.source_alias),[bx,by]=positions.get(event.target_alias),current=event.kind==='selected';
  const key=[event.source_alias,event.target_alias].sort().join('-'),index=pairs.get(key)||0;pairs.set(key,index+1);
  const distance=Math.hypot(bx-ax,by-ay)||1,dx=(bx-ax)/distance,dy=(by-ay)/distance;
  const sx=ax+39*dx,sy=ay+39*dy,tx=bx-43*dx,ty=by-43*dy;
  const effect=measured.get(String(event.transaction_id));
  const evidenceClass=current?'selected-transfer':!effect?'observed-only':effect.score_support>1e-6?'tested-up':effect.score_support< -1e-6?'tested-down':'tested-neutral';
  const evidenceText=current?'Transaction being explained':effect?`Tested: score ${score(effect.original_model_score)} → ${score(effect.model_score_without_event)} without this transfer; absolute change ${score(Math.abs(effect.score_support))}`:'Recorded earlier transfer';
  const offset=current?-22:30+index*22,cx=(ax+bx)/2-dy*offset,cy=(ay+by)/2+dx*offset;
  const path=event.source_alias===event.target_alias?`M ${ax-22} ${ay-24} C ${ax-90} ${ay-120}, ${ax+90} ${ay-120}, ${ax+22} ${ay-24}`:`M ${sx} ${sy} Q ${cx} ${cy} ${tx} ${ty}`;
  return `<g class="graph-edge" data-edge="${esc(event.transaction_id)}" tabindex="0" role="button" aria-label="Inspect transaction ${esc(event.transaction_id)}"><path d="${path}" class="flow-hit"/><path d="${path}" class="flow-line ${evidenceClass}" marker-end="url(#${current?'arrow-current':'arrow-earlier'})"><title>Transaction ${esc(event.transaction_id)}: account ${esc(event.source_account_id)} → account ${esc(event.target_account_id)}; ${amount(event)}; ${time(event.timestamp)}; ${esc(evidenceText)}</title></path><text class="flow-transaction ${current?'current':''}" x="${cx}" y="${cy-10}" text-anchor="middle">TX ${esc(event.transaction_id)}${effect?' [tested]':''}</text>${current?`<text class="flow-amount" x="${cx}" y="${cy+10}" text-anchor="middle">${amount(event)}</text>`:''}</g>`;
 }).join('');
 const nodes=story.accounts.map(account=>{const [x,y]=positions.get(account.alias),id=String(account.account_id??account.node_id);return `<g class="graph-account" data-account="${esc(id)}" tabindex="0" role="button" aria-label="Inspect account ${esc(id)}"><circle class="flow-node ${['sender','receiver','sender and receiver'].includes(account.role)?'current':''}" cx="${x}" cy="${y}" r="36"/><text class="flow-label" style="font-size:${id.length>7?11:14}px" x="${x}" y="${y+5}" text-anchor="middle">${esc(id)}</text><text class="flow-role" x="${x}" y="${y+55}" text-anchor="middle">${esc(account.role)}</text><title>Dataset account ID ${esc(id)}; model node ${esc(account.node_id)}</title></g>`}).join('');
 return `<figure class="flow-card"><svg viewBox="0 0 720 365" role="group" aria-label="Observed transfers labeled with actual transaction and dataset account IDs"><defs><marker id="arrow-current" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#f3c477"/></marker><marker id="arrow-earlier" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8aaed1"/></marker></defs>${paths}${nodes}</svg></figure>`;
}

function transferFields(feature){return Object.fromEntries(String(feature.value).split(' / ').map(part=>{const i=part.indexOf(': ');return [part.slice(0,i),part.slice(i+2)]}))}
function observedFeature(feature){
 if(feature.feature!=='transfer_terms')return esc(feature.feature==='payment_method'?String(feature.value).replace(/^payment_format:\s*/,''):feature.value);
 const fields=transferFields(feature);
 const sent=amount({amount:fields.amount,currency:fields.currency});
 const received=amount({amount:fields.Amount_Received,currency:fields.Receiving_Currency});
 return sent===received?sent:`Sent ${sent}; received ${received}`;
}
function historyEffectCard(effect,story,prediction){
 const event=story.transfers.find(event=>String(event.transaction_id)===String(effect.transaction_id));
 const support=Number(effect.score_support),neutral=Math.abs(support)<=1e-6;
 const date=event?String(event.timestamp).slice(0,10):'';
 const direction=neutral?'The tested change was too small to count as a clear effect.':support>0?'This earlier transfer helped push the score higher.':'This earlier transfer helped keep the score lower.';
 const delta=Math.abs(support),displayDelta=delta>0&&delta<.0001?'less than 0.0001':score(delta);
 const outcome=effect.model_score_without_event>=prediction.threshold;
 return `<div class="history-effect"><p class="muted">Earlier transfer <strong>TX ${esc(effect.transaction_id)}</strong>${date?' · '+esc(date):''}</p><div class="score-comparison"><div><span>With this earlier transfer</span><strong>${score(effect.original_model_score)}</strong></div><span class="comparison-arrow" aria-hidden="true">→</span><div><span>Without its history update</span><strong>${score(effect.model_score_without_event)}</strong></div></div><p><strong>${neutral?'No clear score change':`Score ${support>0?'decreased':'increased'} by ${displayDelta} in the test.`}</strong> ${direction}</p></div>`;
}
function render(x){
 const story=x.transaction_story;
 if(!story){$('#detail').innerHTML='<p class="error">The transfer map is unavailable. Restart the updated dashboard server.</p>';return}
 const event=story.selected_transfer,flagged=x.prediction.class==='fraud',known=x.evaluation_label==='fraud';
 const features=[...(x.feature_explanation?.features||[])].sort((a,b)=>Math.abs(b.shap_logit)-Math.abs(a.shap_logit));
 const earlier=(story.transfers||[]).filter(transfer=>transfer.kind==='earlier');
 const hasHistory=Number(story.earlier_related_transfers_considered||0)>0||earlier.length>0;
 const noHistory='No earlier transfers involving either account were available in the model’s eligible history before this prediction.';
 const effects=[...(story.model_link_effects||[])].sort((a,b)=>Math.abs(b.score_support)-Math.abs(a.score_support));
 const strongest=features.filter(f=>flagged?f.shap_logit>1e-8:f.shap_logit< -1e-8).sort((a,b)=>flagged?b.shap_logit-a.shap_logit:a.shap_logit-b.shap_logit)[0];
 const featureReason=strongest?`<strong>${esc(strongest.label||strongest.feature)}${strongest.feature==='payment_method'?' ('+observedFeature(strongest)+')':''}</strong> had the largest measured feature contribution toward a ${flagged?'higher':'lower'} score in this explanation.`:'The feature check did not find a clear contribution toward this decision.';
 const meaning=flagged&&!known?'This alert is a false positive against that label.':!flagged&&known?'The model missed this transaction against that label.':`The model result agrees with the dataset label (${known?'Fraud':'Not fraud'}).`;
 const maxContribution=Math.max(1e-8,...features.map(f=>Math.abs(f.shap_logit)));
 const featureCards=features.map((feature,index)=>{
  const value=Number(feature.shap_logit),neutral=Math.abs(value)<1e-8;
  const direction=neutral?'No clear measured effect':value>0?'Pushed the score higher':'Pushed the score lower';
  return `<li class="factor-card"><div class="factor-heading"><strong>${esc(feature.feature==='transfer_terms'?'Amount and currencies':feature.label||feature.feature)}</strong><span class="chip ${neutral?'':value>0?'alert':'known'}">${direction}</span></div><p class="factor-value"><span class="observed-label">${feature.feature==='transfer_terms'?'Observed amount / currencies':'Observed value'}:</span> ${observedFeature(feature)}</p><div class="factor-track" aria-hidden="true"><span class="factor-bar ${value<0?'lower':''}" style="width:${neutral?0:Math.max(1,100*Math.abs(value)/maxContribution)}%"></span></div><small>${!neutral&&index===0?'Largest feature effect in this comparison. ':''}${feature.feature==='transfer_terms'?'Amount and currencies were checked together.':''}</small></li>`;
 }).join('');
 const faithfulness=x.evaluation_checks?.faithfulness;
 const faithfulnessSummary=faithfulness&&!faithfulness.groups.every(group=>group.direction_agrees)?'<p class="plain-warning">Changing one feature group alone did not always match its average contribution. Details are available below.</p>':'';
 const historySummary=!hasHistory?noHistory:!effects.length?'Earlier transfers are available, but no verified score test is available for them. The connections provide context only.':effects.some(effect=>effect.score_support>1e-6)?'At least one tested earlier transfer helped push the score higher.':'The tested earlier transfers did not help push the score higher.';
 const historyCards=effects.length?historyEffectCard(effects[0],story,x.prediction)+(effects.length>1?`<details class="more-history"><summary>See ${effects.length-1} more tested transfer${effects.length===2?'':'s'}</summary>${effects.slice(1).map(effect=>historyEffectCard(effect,story,x.prediction)).join('')}</details>`:''):`<p>${historySummary}</p>`;
 const transferRows=earlier.map(transfer=>`<tr><td>TX ${esc(transfer.transaction_id)}<br>Account ${esc(transfer.source_account_id)} → ${esc(transfer.target_account_id)}<br><small>${transfer.kind==='selected'?'Selected transaction':'Earlier transaction'}</small></td><td>${amount(transfer)}<br>${esc(transfer.payment_method)}</td><td>${time(transfer.timestamp)}</td></tr>`).join('');
 $('#detail').innerHTML=`
 <p class="eyebrow bank">${esc(x.bank.replaceAll('_',' '))}</p>
 <h2>Transaction ${esc(x.transaction_id)}</h2>
 <p class="intro">Account <strong>${esc(event.source_account_id)}</strong> sent <strong>${amount(event)}</strong> to account <strong>${esc(event.target_account_id)}</strong> using <strong>${esc(event.payment_method)}</strong>.</p>
 <p class="muted">${time(x.timestamp)} · Score from one saved global model</p>
 <div class="facts"><div class="fact"><span>MODEL RESULT</span><strong>${flagged?'Flagged for review':'Not flagged'}</strong></div><div class="fact"><span>MODEL SCORE / CONFIGURED THRESHOLD</span><strong>${score(x.prediction.probability)} / ${score(x.prediction.threshold)}</strong></div><div class="fact"><span>TRANSACTION AMOUNT</span><strong>${amount(event)}</strong></div></div>
 <section class="section" aria-labelledby="decision-question"><h3 id="decision-question">Why did the model ${flagged?'flag':'not flag'} this transaction?</h3><div class="reason"><p><strong>The model ${flagged?'flagged':'did not flag'} this transaction because its score (${score(x.prediction.probability)}) ${flagged?'met or exceeded':'was below'} the configured review threshold (${score(x.prediction.threshold)}).</strong></p><p>${featureReason}</p><p>${historySummary}</p></div><p class="muted">This explains the model’s decision. It does not prove fraud, and the score is not a confirmed probability of fraud.</p></section>
 <section class="section" aria-labelledby="feature-question"><h3 id="feature-question">Main factors influencing the score</h3><p class="muted">These details were compared with examples from the training data, keeping earlier account activity the same.</p><ol class="factor-list">${featureCards}</ol><p class="muted">Bar lengths compare the size of these feature contributions. They are not percentages of the score.</p>${faithfulnessSummary}</section>
 <section class="section" aria-labelledby="history-question"><h3 id="history-question">Effect of earlier transfers</h3>${historyCards}</section>
 <section class="section" aria-labelledby="graph-question"><h3 id="graph-question">Explore the account connections</h3>${patternLabels(story)}<p>${hasHistory?esc(story.observation):noHistory}</p><div class="graph-layout">${flowDiagram(story)}<aside id="graph-inspector" class="graph-inspector" aria-live="polite"><h4>Select an account or transfer</h4><p>View its details and any tested effect on the score.</p></aside></div>${earlier.length?`<details class="recorded-history"><summary>View earlier transaction history</summary><table class="flow-table"><thead><tr><th>Earlier transaction / accounts</th><th>Observed amount / method</th><th>Time</th></tr></thead><tbody>${transferRows}</tbody></table></details>`:hasHistory?'<p class="history-empty">Earlier history exists, but no earlier transfers fit the current graph display.</p>':''}</section>
 ${readableTechnicalDetails(x)}`;
 attachGraphInteractions(story,x.prediction);
}
function readableTechnicalDetails(x){
 const feature=x.feature_explanation,story=x.transaction_story;
 const features=[...feature.features].sort((a,b)=>Math.abs(b.shap_logit)-Math.abs(a.shap_logit));
 const effects=story.model_link_effects||[];
 const hasHistory=Number(story.earlier_related_transfers_considered||0)>0||story.transfers.some(event=>event.kind==='earlier');
 const flagged=x.prediction.class==='fraud',faithfulness=x.evaluation_checks?.faithfulness;
 const label=f=>f.feature==='transfer_terms'?'Amount and currencies':f.label||f.feature;
 const signed=value=>(Number(value)>=0?'+':'')+Number(value).toFixed(4);
 const rows=features.map(f=>`<tr><td>${esc(label(f))}<br><small>Observed: ${observedFeature(f)}</small></td><td>${signed(f.shap_logit)}</td><td>${Math.abs(f.shap_logit)<1e-8?'No clear measured effect':f.shap_logit>0?'Contributed toward a higher output':'Contributed toward a lower output'}</td></tr>`).join('');
 const equation=[`Comparison starting point: ${Number(feature.base_logit).toFixed(4)}`,...features.map(f=>`${label(f)}: ${signed(f.shap_logit)}`),`Target internal output: ${Number(feature.target_logit).toFixed(4)}`,`Converted model score: ${preciseScore(x.prediction.probability)}`].join('\n');
 const historyText=effects.length?`${effects.length} earlier transfer${effects.length===1?' has':'s have'} a verified score test. Each test leaves out one transfer’s history update and recalculates this transaction’s score.`:hasHistory?'Earlier transfers are recorded, but no verified score-effect test is available for this transaction. Connections alone do not show that they influenced the prediction.':'No eligible earlier transfers involving either account were available, so there was no earlier transfer to remove and test.';
 const effectTable=effects.length?`<table class="technical-table"><thead><tr><th>Earlier transfer</th><th>Original score</th><th>Without its update</th></tr></thead><tbody>${effects.map(effect=>`<tr><td>TX ${esc(effect.transaction_id)}</td><td>${preciseScore(effect.original_model_score)}</td><td>${preciseScore(effect.model_score_without_event)}</td></tr>`).join('')}</tbody></table>`:'';
 const faithText=!faithfulness?'Not recorded for this transaction. This does not mean the check passed.':faithfulness.groups.every(group=>group.direction_agrees)?'The contribution directions agreed with the check that changes one group at a time.':'At least one contribution direction differed when that group was changed alone. The checks answer different questions.';
 const verification=x.verification||{};
 const numericalText=verification.passed===true?`Passed: the saved score, direct model score and explanation score agree within the allowed difference of ${verification.tolerance??0.000001}.`:'A passed numerical verification is not recorded for this transaction.';
 const alternatives=x.counterfactual_explanation?.counterfactuals||[];
 const whatIf=alternatives.length?`<ul>${alternatives.map(item=>`<li>Changing the amount to ${Number(item.changed_features?.amount??item.changed_features?.Amount_Received).toLocaleString()} gave a model score of ${preciseScore(item.counterfactual_probability)}.</li>`).join('')}</ul><p>These amounts can be unrealistic. This tests model sensitivity; it is not a recommendation. A displayed zero may be a rounded value.</p>`:'No alternative amount result is recorded for this transaction.';
 const raw={prediction:x.prediction,feature_explanation:feature,verification,model:x.model,evaluation_checks:x.evaluation_checks,graph_effects:effects,history_scope:story.history_scope,provenance:x.xai_provenance};
 return `<details class="technical-details"><summary>Technical methods and verification</summary>
 <h4>1. Which model made the prediction?</h4><p>One saved adaptive federated model: training seed <strong>${esc(x.model.seed)}</strong>, selected round <strong>${esc(x.model.best_round)}</strong>. No three-model ensemble. It reuses its learned weights; it is not training again.</p>
 <h4>2. Why was this transaction ${flagged?'flagged':'not flagged'}?</h4><p>Its score was <strong>${preciseScore(x.prediction.probability)}</strong>, ${flagged?'at or above':'below'} the configured threshold.</p><p>Saved configured threshold for <strong>${esc(x.bank.replaceAll('_',' '))}</strong> (validation-selected): <strong>${preciseScore(x.prediction.threshold)}</strong>.</p><p>The threshold was chosen using validation data and saved with the model’s results. The dashboard reuses it. Decisions use exact values; the main view rounds scores to four decimals.</p>
 <h4>3. How were the transaction details explained?</h4><p>We used <strong>${esc(feature.background_size)} training transactions as comparison examples</strong>. We temporarily replaced groups of details and asked the same model to calculate outputs, keeping earlier account memory unchanged. This method is called <strong>grouped SHAP</strong>.</p><p>These are comparison examples, not additional models or necessarily earlier transfers involving these accounts. Amounts and currencies move together from the same example.</p><table class="technical-table"><thead><tr><th>Feature group and observed value</th><th>Contribution</th><th>Meaning</th></tr></thead><tbody>${rows}</tbody></table>
 <h4>4. What do the contribution numbers mean?</h4><p>They use <strong>logit</strong>, the model’s internal scale before conversion to the displayed score. Positive values push the output higher; negative values push it lower. They are not percentages.</p><pre class="score-equation">${esc(equation)}</pre><p>The starting point comes from the comparison examples with account memory held fixed; it is not another transaction’s score. Values above are rounded. Changing one group alone can give a different effect because SHAP averages across feature combinations.</p>
 <h4>5. Did earlier transfers affect this score?</h4><p>${historyText}</p>${effectTable}
 <h4>6. How was the explanation checked?</h4><ul class="plain-checks"><li><strong>Numerical agreement — do the model and explanation reproduce the same score?</strong><br>${numericalText}</li><li><strong>Feature-change check — do targeted changes have the expected measured effect?</strong><br>${faithText}</li></ul><p>These are separate checks. Passing one does not prove that the others passed or that the transaction was fraudulent.</p>
 <h4>7. What are the other checks and IDs?</h4><p><strong>Model fingerprint (SHA-256):</strong> identifies the exact saved model file. <strong>Account → model node:</strong> links a dataset account ID to the model’s internal account number.</p><details><summary>Optional amount-change test</summary>${whatIf}</details><details><summary>Account lookup and model fingerprint</summary><p>Fingerprint: <code>${esc(x.model.checkpoint_sha256)}</code></p><p>${story.accounts.map(account=>`Account ${esc(account.account_id)} → internal node ${esc(account.node_id)}`).join('<br>')}</p></details><details><summary>Exact values and audit record</summary><p>This record includes full-precision contributions, the comparison transaction IDs, test results and shared metadata for all banks. This transaction uses only the ${esc(x.bank.replaceAll('_',' '))} threshold shown above.</p><pre>${esc(JSON.stringify(raw,null,2))}</pre></details>
 <p class="technical-conclusion"><strong>What this tells you:</strong> the model ${flagged?'flagged':'did not flag'} this transaction because its score ${flagged?'met or exceeded':'was below'} its configured threshold. The explanation describes the model’s behaviour; it does not establish actual fraud.</p></details>`;
}
function graphEdgeDetails(event,effect,prediction){
 const tested=event.kind==='earlier'&&effect;
 const status=tested?(effect.score_support>1e-6?'Helped raise the score':effect.score_support< -1e-6?'Helped lower the score':'Tested change was too small to measure clearly'):'';
 const scoreLabel=event.kind==='selected'?'Model score':'Selected transaction’s model score';
 return `<h4>Transaction ${esc(event.transaction_id)}</h4><p>Account ${esc(event.source_account_id)} → ${esc(event.target_account_id)}</p><p>${time(event.timestamp)}<br>${amount(event)}<br>${esc(event.payment_method)}</p>${status?`<p><strong>${status}</strong></p>`:''}<p>${scoreLabel}: ${score(prediction.probability)}</p>${tested?`<p>Without its history update: ${score(effect.model_score_without_event)}</p><p class="muted">Earlier account memory is retained. Full replay scope and exact scores are in Technical methods and verification.</p>`:''}`;
}
function attachGraphInteractions(story,prediction){
 const panel=$('#graph-inspector');
 const effects=new Map((story.model_link_effects||[]).map(effect=>[String(effect.transaction_id),effect]));
 const earlier=story.transfers.filter(event=>event.kind==='earlier');
 const verified=earlier.filter(event=>effects.has(String(event.transaction_id))).length;
 function inspectEdge(id){
  const event=story.transfers.find(event=>String(event.transaction_id)===String(id));
  if(!event)return;
  const effect=effects.get(String(id));
  panel.innerHTML=graphEdgeDetails(event,effect,prediction);
 }
 function inspectAccount(id){
  const account=story.accounts.find(account=>String(account.account_id)===String(id));
  const related=story.transfers.filter(event=>String(event.source_account_id)===String(id)||String(event.target_account_id)===String(id));
  panel.innerHTML=`<h4>Account ${esc(id)}</h4><p>${esc(account?.role)} · ${related.length} displayed transfers</p><p>Connections describe recorded activity; they do not identify account ownership, payment purpose, or fraud.</p>${related.map(event=>`<button class="inspect-transfer" data-inspect-edge="${esc(event.transaction_id)}">Inspect TX ${esc(event.transaction_id)}</button>`).join('')}`;
  panel.querySelectorAll('[data-inspect-edge]').forEach(button=>button.addEventListener('click',()=>inspectEdge(button.dataset.inspectEdge)));
 }
 document.querySelectorAll('[data-edge],[data-account]').forEach(element=>{
  const activate=()=>element.dataset.edge?inspectEdge(element.dataset.edge):inspectAccount(element.dataset.account);
  element.addEventListener('click',activate);
  element.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();activate()}});
 });
}
function reset(){currentPage=0;loadRows()}
bank.addEventListener('change',()=>{requestSequence++;selected=null;query.value='';history.replaceState(null,'','?bank='+encodeURIComponent(bank.value));$('#detail').innerHTML='<div class="empty"><h2>Select a transaction from this bank</h2></div>';reset()});view.addEventListener('change',reset);
query.addEventListener('input',()=>{clearTimeout(debounce);debounce=setTimeout(reset,220)});
$('#prev').addEventListener('click',()=>{if(currentPage){currentPage--;loadRows()}});
$('#next').addEventListener('click',()=>{if((currentPage+1)*pageSize<total){currentPage++;loadRows()}});
async function init(){try{const data=await json('/api/meta');bank.innerHTML=data.banks.map(value=>`<option value="${esc(value)}">${esc(value.replaceAll('_',' '))}</option>`).join('');const params=new URLSearchParams(location.search);if(params.has('bank')&&!data.banks.includes(params.get('bank')))throw Error('Unknown bank in transaction link.');if(data.banks.includes(params.get('bank')))bank.value=params.get('bank');const id=params.get('transaction_id');if(id){query.value=id}await loadRows();if(id)await loadExplanation(id)}catch(error){$('#detail').innerHTML=`<p class="error">${esc(error.message)}</p>`}}
init();
