/*
 * AMCShield Adversarial Attacks page.
 * No attack metrics are hardcoded. All experiment results and threat
 * configuration are supplied by FastAPI.
 */
const API_BASE_URL="http://localhost:8000/api";
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
const num=x=>Number.isFinite(Number(x))?Number(x):null;
const fmt=x=>num(x)===null?"—":new Intl.NumberFormat("en-IN",{maximumFractionDigits:4}).format(num(x));
const pct=x=>num(x)===null?"—":`${num(x).toFixed(2)}%`;

async function api(path,options={}){
  const r=await fetch(API_BASE_URL+path,{...options,headers:{"Content-Type":"application/json",Accept:"application/json",...(options.headers||{})}});
  if(!r.ok){let d="";try{const x=await r.json();d=x.detail||x.message||""}catch(_){}throw new Error(d||`HTTP ${r.status}`)}
  return r.json();
}
function setAPI(ok,msg){$("apiStatus").classList.toggle("online",ok);$("apiStatus").classList.toggle("offline",!ok);$("apiText").textContent=ok?"API ONLINE":"API OFFLINE";$("apiDetail").textContent=msg}
function fillSelect(id,rows,placeholder,key="name"){const s=$(id);s.innerHTML=`<option value="">${placeholder}</option>`;(rows||[]).forEach(x=>{const v=typeof x==="object"?(x[key]??x.value??x.name):x;if(v===undefined)return;const o=document.createElement("option");o.value=v;o.textContent=typeof x==="object"?(x.label??v):v;s.appendChild(o)})}

function renderOverview(d){
  const x=d.attack||d;
  $("activeAttack").textContent=x.name||x.attack||x.method||"—";$("activeAttackDetail").textContent=x.target_model||x.model||"Target model";
  $("attackStatus").textContent=(x.status||"—").toUpperCase();$("attackStatusDetail").textContent=x.message||"Backend status";
  $("meanASR").textContent=pct(x.attack_success_rate??x.asr??x.mean_asr);$("meanASRDetail").textContent=x.asr_scope||"Backend metric";
  $("attackAccuracy").textContent=pct(x.attack_accuracy??x.accuracy);$("attackAccuracyDetail").textContent=x.clean_accuracy!==undefined?`Clean: ${pct(x.clean_accuracy)}`:"Backend metric";
  $("epsilon").textContent=x.epsilon!==undefined?fmt(x.epsilon):"—";$("epsilonDetail").textContent=x.norm||"L∞ constraint";
  $("queryBudget").textContent=x.query_budget!==undefined?fmt(x.query_budget):"—";$("queryBudgetDetail").textContent=x.query_budget!==undefined?"Maximum queries":"Not applicable";
  $("engineStatus").textContent=x.status||"Backend";$("threatText").textContent=(x.status||"READY").toUpperCase();$("threatDetail").textContent=x.message||"Attack backend";
  const p=Math.max(0,Math.min(100,num(x.progress_percent)??0));$("progressPct").textContent=x.progress_percent!==undefined?pct(p):"—";$("progressFill").style.width=`${p}%`;
  $("progressLabel").textContent=x.phase||x.status||"No active attack";$("eta").textContent=x.eta||"ETA —";
  $("processed").textContent=fmt(x.processed);$("remaining").textContent=fmt(x.remaining);$("currentSnr").textContent=x.current_snr!==undefined?`${x.current_snr} dB`:"—";$("queries").textContent=fmt(x.queries);
}
function renderConfig(d){
  const x=d.config||d;
  fillSelect("targetModel",x.models||x.target_models||[],"Select target model", "name");
  fillSelect("attackMethod",x.attacks||x.methods||[],"Select attack method","name");
  fillSelect("snrInput",x.snr_levels||x.snr||[],"All SNR");
  if(x.epsilon!==undefined)$("epsilonInput").value=x.epsilon;
  if(x.steps!==undefined)$("stepsInput").value=x.steps;
  if(x.samples_per_snr!==undefined)$("sampleInput").value=x.samples_per_snr;
  $("configBadge").textContent=x.source||"Backend";
}
function renderTable(d){
  const rows=Array.isArray(d)?d:(d.results||d.attacks||d.data||[]);const box=$("attackTable");box.innerHTML="";
  if(!rows.length){box.innerHTML='<tr><td colspan="7"><div class="empty">No attack results returned by backend.</div></td></tr>';return}
  rows.forEach(x=>{
    const asr=num(x.attack_success_rate??x.asr);const e=document.createElement("tr");e.className="row-button";
    e.innerHTML=`<td>${esc(x.attack??x.method??x.name)}</td><td>${pct(x.clean_accuracy)}</td><td>${pct(x.attack_accuracy??x.accuracy??x.cw_accuracy)}</td><td class="${asr!==null&&asr>=50?'asr-high':'asr-low'}">${pct(asr)}</td><td>${fmt(x.mean_linf??x.linf)}</td><td>${fmt(x.mean_l2??x.l2)}</td><td><span class="status-pill ${asr!==null&&asr>=50?'warn':''}">${esc(x.status??"EVALUATED")}</span></td>`;
    e.addEventListener("click",()=>renderDetail(x));box.appendChild(e);
  });
}
function renderSNR(d){
  const rows=Array.isArray(d)?d:(d.data||d.results||d.snr||[]);const box=$("snrChart");box.innerHTML="";
  const a=rows.map(x=>({snr:num(x.snr??x.level),asr:num(x.attack_success_rate??x.asr??x.value)})).filter(x=>x.snr!==null&&x.asr!==null).sort((a,b)=>a.snr-b.snr);
  if(!a.length){box.innerHTML='<div class="empty">No SNR attack profile returned by backend.</div>';return}
  const chart=document.createElement("div");chart.className="snr-bars";a.forEach(x=>{const c=document.createElement("div");c.className="snr-col";const b=document.createElement("div");b.className="snr-bar";b.style.height=`${Math.max(2,Math.min(100,x.asr))}%`;b.title=`${x.snr} dB · ${x.asr.toFixed(2)}% ASR`;const l=document.createElement("span");l.className="snr-label";l.textContent=x.snr;c.append(b,l);chart.appendChild(c)});box.appendChild(chart);
}
function renderMethods(d){
  const rows=Array.isArray(d)?d:(d.methods||d.attacks||d.data||[]);const box=$("methodList");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="empty">No attack method metadata returned by backend.</div>';return}
  rows.forEach(x=>{const e=document.createElement("div");e.className="method";e.innerHTML=`<span class="method-icon">${esc((x.short_name||x.name||"A").slice(0,2))}</span><div><strong>${esc(x.name||x.attack)}</strong><small>${esc(x.description||x.objective||"Backend attack method")}</small></div><span class="method-tag">${esc(x.type||"ATTACK")}</span>`;box.appendChild(e)})
}
function renderConstraints(d){
  const x=d.constraints||d;const entries=[["Norm",x.norm],["ε",x.epsilon],["Steps",x.steps],["α / Step size",x.alpha??x.step_size],["Queries",x.query_budget],["Confidence / κ",x.kappa],["Optimizer",x.optimizer],["Input clipping",x.input_clipping]].filter(a=>a[1]!==undefined&&a[1]!==null);
  const box=$("constraintGrid");box.innerHTML="";if(!entries.length){box.innerHTML='<div class="empty">No constraint metadata returned by backend.</div>';return}
  entries.forEach(([k,v])=>{const e=document.createElement("div");e.className="constraint";e.innerHTML=`<span>${esc(k)}</span><strong>${esc(typeof v==="object"?JSON.stringify(v):v)}</strong>`;box.appendChild(e)})
}
function renderDetail(x){
  $("detailBadge").textContent=x.attack||x.method||"Selected";
  const entries=Object.entries(x).filter(([k])=>!["attack","method","name"].includes(k)).slice(0,16);const box=$("detailGrid");box.innerHTML="";
  entries.forEach(([k,v])=>{const e=document.createElement("div");e.className="detail";e.innerHTML=`<span>${esc(k.replaceAll("_"," "))}</span><strong>${esc(typeof v==="object"?JSON.stringify(v):v)}</strong>`;box.appendChild(e)})
}
function renderLog(d){
  const rows=Array.isArray(d)?d:(d.logs||d.data||[]);const box=$("attackLog");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="log-empty">No attack telemetry received.</div>';return}
  rows.slice(-30).forEach(x=>{const e=document.createElement("div");e.style.padding="3px 0";e.textContent=`[${x.time??x.timestamp??"—"}] ${x.message??x.text??""}`;box.appendChild(e)});box.scrollTop=box.scrollHeight;
}

async function load(){
  $("refresh").disabled=true;let ok=0;
  const endpoints=[["/attacks/overview",renderOverview],["/attacks/config",renderConfig],["/attacks/results",renderTable],["/attacks/snr-profile",renderSNR],["/attacks/methods",renderMethods],["/attacks/constraints",renderConstraints],["/attacks/logs",renderLog]];
  for(const [path,fn] of endpoints){try{fn(await api(path));ok++}catch(_){}}
  setAPI(ok>0,ok?`${ok}/${endpoints.length} attack endpoints responding`:"Attack endpoints unavailable");
  $("updated").textContent=ok?`Updated ${new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})}`:"Not loaded";
  $("refresh").disabled=false;
}
async function runAttack(){
  const body={target_model:$("targetModel").value,attack:$("attackMethod").value,epsilon:num($("epsilonInput").value),steps:num($("stepsInput").value),snr:$("snrInput").value===""?null:num($("snrInput").value),samples_per_snr:num($("sampleInput").value)};
  $("commandStatus").textContent="Attack request sent…";
  try{const d=await api("/attacks/run",{method:"POST",body:JSON.stringify(body)});$("commandStatus").textContent=d.message||"Attack request accepted.";load()}catch(e){$("commandStatus").textContent=`Attack unavailable: ${e.message}`}
}
async function stopAttack(){
  try{const d=await api("/attacks/stop",{method:"POST"});$("commandStatus").textContent=d.message||"Stop request accepted.";load()}catch(e){$("commandStatus").textContent=`Stop unavailable: ${e.message}`}
}
$("refresh").addEventListener("click",load);$("launchAttack").addEventListener("click",runAttack);$("cancelAttack").addEventListener("click",stopAttack);
$("menu").addEventListener("click",()=>{$("sidebar").classList.add("open");$("overlay").classList.add("show")});$("overlay").addEventListener("click",()=>{$("sidebar").classList.remove("open");$("overlay").classList.remove("show")});
$("logout").addEventListener("click",()=>window.location.href="../login/index.html");load();
