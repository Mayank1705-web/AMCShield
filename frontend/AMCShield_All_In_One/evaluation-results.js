/*
 * AMCShield Evaluation Results page.
 * All evaluation metrics are fetched from FastAPI.
 * The frontend contains no hardcoded experiment results.
 */
const API_BASE_URL="http://localhost:8000/api";
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
const n=x=>Number.isFinite(Number(x))?Number(x):null;
const fmt=x=>n(x)===null?"—":new Intl.NumberFormat("en-IN",{maximumFractionDigits:4}).format(n(x));
const pct=x=>n(x)===null?"—":`${n(x).toFixed(2)}%`;
let allRows=[];let filteredRows=[];

async function api(path,options={}){
  const r=await fetch(API_BASE_URL+path,{...options,headers:{Accept:"application/json",...(options.headers||{})}});
  if(!r.ok){let d="";try{const x=await r.json();d=x.detail||x.message||""}catch(_){}throw new Error(d||`HTTP ${r.status}`)}
  return r.json();
}
function setAPI(ok,msg){$("apiStatus").classList.toggle("online",ok);$("apiStatus").classList.toggle("offline",!ok);$("apiText").textContent=ok?"API ONLINE":"API OFFLINE";$("apiDetail").textContent=msg}
function fillSelect(id,values,placeholder){
  const s=$(id);s.innerHTML=`<option value="">${placeholder}</option>`;(values||[]).forEach(v=>{const value=typeof v==="object"?(v.value??v.name??v.attack??v.snr):v;if(value===undefined)return;const o=document.createElement("option");o.value=value;o.textContent=typeof v==="object"?(v.label??value):value;s.appendChild(o)})
}
function renderOverview(d){
  const x=d.summary||d;
  $("robustClean").textContent=pct(x.robust_clean_accuracy);$("robustCleanDetail").textContent=x.robust_samples!==undefined?`${fmt(x.robust_samples)} samples`:"Backend metric";
  $("baselineClean").textContent=pct(x.baseline_clean_accuracy);$("baselineCleanDetail").textContent=x.baseline_samples!==undefined?`${fmt(x.baseline_samples)} samples`:"Backend metric";
  $("robustASR").textContent=pct(x.mean_robust_asr);$("robustASRDetail").textContent=x.attack_scope||"All attacks";
  $("baselineASR").textContent=pct(x.mean_baseline_asr);$("baselineASRDetail").textContent=x.attack_scope||"All attacks";
  $("meanGap").textContent=x.mean_gap!==undefined?`${Number(x.mean_gap).toFixed(2)} pp`:"—";$("meanGapDetail").textContent=x.gap_definition||"Baseline ASR − Robust ASR";
  $("snrCount").textContent=fmt(x.snr_levels);$("snrRange").textContent=x.snr_min!==undefined?`${x.snr_min} to ${x.snr_max} dB`:"Backend range";
  $("sourceStatus").textContent=x.source||"Backend";$("resultStateText").textContent=(x.status||"READY").toUpperCase();$("resultStateDetail").textContent=x.message||"Evaluation backend";
}
function renderFilters(d){
  const x=d.filters||d;
  fillSelect("modelFilter",x.models||[],"All models");
  fillSelect("attackFilter",x.attacks||[],"All attacks");
  fillSelect("snrFilter",x.snr_levels||x.snr||[],"All SNR");
  $("filterBadge").textContent=x.source||"Backend";
}
function renderComparison(d){
  const rows=Array.isArray(d)?d:(d.data||d.attacks||d.results||[]);
  const box=$("comparisonChart");box.innerHTML="";
  const a=rows.map(x=>({attack:x.attack??x.method??x.name,robust:n(x.robust_asr??x.robust),base:n(x.baseline_asr??x.baseline)})).filter(x=>x.attack&&x.robust!==null&&x.base!==null);
  if(!a.length){box.innerHTML='<div class="empty">No model comparison data returned by backend.</div>';return}
  a.forEach(x=>{
    [["Baseline",x.base,"base"],["Robust",x.robust,"robust"]].forEach(([label,v,cls])=>{
      const row=document.createElement("div");row.className="bar-group";row.innerHTML=`<label>${esc(x.attack)} · ${label}</label><div class="bar-track"><div class="bar-fill ${cls}" style="width:${Math.max(0,Math.min(100,v))}%"></div></div><span class="bar-value">${v.toFixed(2)}%</span>`;box.appendChild(row)
    })
  });
  const leg=document.createElement("div");leg.className="legend";leg.innerHTML='<span><i class="r"></i>Robust</span><span><i class="b"></i>Baseline</span>';box.appendChild(leg);
}
function renderAttackSummary(d){
  const rows=Array.isArray(d)?d:(d.attacks||d.data||d.results||[]);const box=$("attackSummary");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="empty">No attack summary returned by backend.</div>';return}
  rows.forEach(x=>{
    const gap=n(x.gap??x.generalization_gap);const base=n(x.baseline_asr);const robust=n(x.robust_asr);
    const e=document.createElement("div");e.className="attack-card";e.innerHTML=`<div class="attack-card-top"><strong>${esc(x.attack??x.method??x.name)}</strong><span class="gap ${gap!==null&&gap>=0?'gap-positive':'gap-negative'}">${gap===null?'—':`${gap.toFixed(2)} pp`}</span></div><div class="attack-card-bar"><span style="width:${Math.max(0,Math.min(100,Math.abs(gap??0)))}%"></span></div><small style="display:flex;justify-content:space-between;margin-top:6px;color:var(--muted2);font-size:8px"><span>Baseline ${pct(base)}</span><span>Robust ${pct(robust)}</span></small>`;box.appendChild(e)
  })
}
function renderRows(d){
  allRows=Array.isArray(d)?d:(d.rows||d.results||d.data||[]);
  filteredRows=[...allRows];renderTable(filteredRows);
  const attacks=[...new Set(allRows.map(x=>x.attack??x.method).filter(Boolean))];const snrs=[...new Set(allRows.map(x=>x.snr).filter(v=>v!==undefined&&v!==null))].sort((a,b)=>Number(a)-Number(b));fillSelect("attackFilter",attacks,"All attacks");fillSelect("snrFilter",snrs,"All SNR");
}
function renderTable(rows){
  const box=$("resultTable");box.innerHTML="";$("rowCount").textContent=`${rows.length} rows`;
  if(!rows.length){box.innerHTML='<tr><td colspan="8"><div class="empty">No rows match the selected scope.</div></td></tr>';return}
  rows.forEach(x=>{
    const gap=n(x.gap??x.generalization_gap),e=document.createElement("tr");
    e.innerHTML=`<td>${esc(x.attack??x.method??"—")}</td><td>${x.snr!==undefined?esc(x.snr):"—"}</td><td>${pct(x.baseline_asr)}</td><td>${pct(x.robust_asr)}</td><td class="${gap!==null&&gap>=0?'positive':'negative'}">${gap===null?'—':gap.toFixed(4)+" pp"}</td><td>${pct(x.baseline_accuracy??x.baseline_clean_accuracy)}</td><td>${pct(x.robust_accuracy??x.robust_clean_accuracy)}</td><td>${fmt(x.samples??x.n_samples)}</td>`;box.appendChild(e)
  })
}
function svgLine(rows){
  const a=rows.map(x=>({snr:n(x.snr),base:n(x.baseline_accuracy),robust:n(x.robust_accuracy)})).filter(x=>x.snr!==null&&(x.base!==null||x.robust!==null)).sort((a,b)=>a.snr-b.snr);
  if(!a.length)return null;
  const W=900,H=250,p={l:35,r:15,t:20,b:25},vals=a.flatMap(x=>[x.base,x.robust]).filter(v=>v!==null),min=Math.max(0,Math.min(...vals)-5),max=Math.min(100,Math.max(...vals)+5),range=(max-min)||1,lo=a[0].snr,hi=a[a.length-1].snr,er=(hi-lo)||1;
  const X=v=>p.l+(v-lo)/er*(W-p.l-p.r),Y=v=>H-p.b-(v-min)/range*(H-p.t-p.b),path=k=>a.filter(x=>x[k]!==null).map((x,i)=>`${i?'L':'M'}${X(x.snr).toFixed(1)},${Y(x[k]).toFixed(1)}`).join(" ");
  let g="";for(let i=0;i<5;i++){const y=p.t+i*(H-p.t-p.b)/4;g+=`<line class="svg-grid" x1="${p.l}" x2="${W-p.r}" y1="${y}" y2="${y}"/>`}
  return `<div class="chart-legend"><span><i class="gold"></i>Baseline</span><span><i class="cyan"></i>Robust</span></div><svg class="svg-chart" viewBox="0 0 ${W} ${H}">${g}<path class="line-baseline" d="${path("base")}"/><path class="line-robust" d="${path("robust")}"/></svg>`
}
function renderAccuracy(d){const x=svgLine(Array.isArray(d)?d:(d.data||d.results||[]));$("accuracyChart").innerHTML=x||'<div class="empty">No SNR accuracy data returned by backend.</div>'}
function renderGap(d){
  const rows=Array.isArray(d)?d:(d.data||d.results||[]);const a=rows.map(x=>({snr:n(x.snr),gap:n(x.gap??x.generalization_gap)})).filter(x=>x.snr!==null&&x.gap!==null).sort((a,b)=>a.snr-b.snr);const box=$("gapChart");box.innerHTML="";
  if(!a.length){box.innerHTML='<div class="empty">No generalization gap data returned by backend.</div>';return}
  const max=Math.max(1,...a.map(x=>Math.abs(x.gap)));a.forEach(x=>{const c=document.createElement("div");c.className="gap-col";const m=document.createElement("div");m.className="gap-mid";m.innerHTML=`<div class="gap-zero"></div><div class="gap-fill ${x.gap>=0?'pos':'neg'}" style="height:${Math.abs(x.gap)/max*48}%"></div>`;const l=document.createElement("span");l.className="gap-label";l.textContent=x.snr;c.append(m,l);box.appendChild(c)})
}
function renderInsights(d){
  const rows=Array.isArray(d)?d:(d.findings||d.insights||d.data||[]);const box=$("insightGrid");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="empty">No backend findings returned.</div>';return}
  rows.forEach(x=>{const e=document.createElement("div");e.className="insight";e.innerHTML=`<span class="type">${esc(x.type??"FINDING")}</span><strong>${esc(x.title??x.name??"Evaluation finding")}</strong><p>${esc(x.description??x.message??"Backend-provided evaluation note.")}</p>`;box.appendChild(e)})
}
function apply(){
  const attack=$("attackFilter").value,snr=$("snrFilter").value,model=$("modelFilter").value;
  filteredRows=allRows.filter(x=>(!attack||(x.attack??x.method)===attack)&&(!snr||String(x.snr)===String(snr))&&(!model||String(x.model??x.model_name??"")===String(model)));
  renderTable(filteredRows)
}
function reset(){ $("attackFilter").value="";$("snrFilter").value="";$("modelFilter").value="";filteredRows=[...allRows];renderTable(filteredRows)}
function exportCSV(){
  if(!filteredRows.length)return;
  const keys=[...new Set(filteredRows.flatMap(x=>Object.keys(x)))];const lines=[keys.join(",")];filteredRows.forEach(x=>lines.push(keys.map(k=>`"${String(x[k]??"").replaceAll('"','""')}"`).join(",")));
  const blob=new Blob([lines.join("\n")],{type:"text/csv"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="amcshield_evaluation_filtered.csv";a.click();URL.revokeObjectURL(url)
}
async function load(){
  $("refresh").disabled=true;let ok=0;
  const endpoints=[["/evaluation/summary",renderOverview],["/evaluation/filters",renderFilters],["/evaluation/comparison",renderComparison],["/evaluation/attack-summary",renderAttackSummary],["/evaluation/results",renderRows],["/evaluation/accuracy-by-snr",renderAccuracy],["/evaluation/generalization-gap",renderGap],["/evaluation/insights",renderInsights]];
  for(const [path,fn] of endpoints){try{fn(await api(path));ok++}catch(_){}}
  setAPI(ok>0,ok?`${ok}/${endpoints.length} evaluation endpoints responding`:"Evaluation endpoints unavailable");
  $("updated").textContent=ok?`Updated ${new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})}`:"Not loaded";
  $("refresh").disabled=false;
}
$("refresh").addEventListener("click",load);$("applyFilter").addEventListener("click",apply);$("resetFilter").addEventListener("click",reset);$("exportCsv").addEventListener("click",exportCSV);
$("menu").addEventListener("click",()=>{$("sidebar").classList.add("open");$("overlay").classList.add("show")});$("overlay").addEventListener("click",()=>{$("sidebar").classList.remove("open");$("overlay").classList.remove("show")});
$("logout").addEventListener("click",()=>window.location.href="../login/index.html");load();
