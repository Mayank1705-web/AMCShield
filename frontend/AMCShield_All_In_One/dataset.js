/*
 * AMCShield Dataset page.
 * No experiment values are hardcoded here.
 * All metadata, distributions, samples and quality checks come from FastAPI.
 */
const API_BASE_URL = "http://localhost:8000/api";

const $ = id => document.getElementById(id);
const state = { meta:null, classes:null, snr:null, quality:null };

async function api(path, options={}) {
  const r = await fetch(API_BASE_URL + path, {
    ...options,
    headers: {Accept:"application/json", ...(options.headers||{})}
  });
  if (!r.ok) {
    let detail="";
    try { const x=await r.json(); detail=x.detail||x.message||""; } catch(_){}
    throw new Error(detail || `HTTP ${r.status}`);
  }
  return r.json();
}

function setAPI(ok, detail) {
  const box=$("apiStatus");
  box.classList.toggle("online",ok);
  box.classList.toggle("offline",!ok);
  $("apiText").textContent=ok?"API ONLINE":"API OFFLINE";
  $("apiDetail").textContent=detail || (ok?"Dataset backend connected":"Backend unavailable");
}

function fmt(n) {
  if(n===null||n===undefined||!Number.isFinite(Number(n))) return "—";
  return new Intl.NumberFormat("en-IN",{maximumFractionDigits:0}).format(Number(n));
}
function pct(n) {
  if(n===null||n===undefined||!Number.isFinite(Number(n))) return "—";
  return `${Number(n).toFixed(1)}%`;
}

function renderMeta(d) {
  state.meta=d;
  const x=d.dataset||d;
  $("datasetName").textContent=x.name ?? "—";
  $("datasetPath").textContent=x.path ?? x.source ?? "Backend metadata";
  $("totalSamples").textContent=fmt(x.total_samples ?? x.samples);
  $("sampleSplit").textContent=x.test_samples!==undefined ? `Test: ${fmt(x.test_samples)}` : "Dataset total";
  $("classCount").textContent=fmt(x.num_classes ?? x.class_count);
  $("classDetail").textContent=x.classes_path ?? "Class metadata";
  $("signalLength").textContent=fmt(x.signal_length);
  $("inputShape").textContent=x.input_shape ? String(x.input_shape) : "Samples × channels × length";
  $("snrLevels").textContent=fmt(x.snr_levels ?? (Array.isArray(x.snr_values)?x.snr_values.length:null));
  const lo=x.snr_min, hi=x.snr_max;
  $("snrRange").textContent=Number.isFinite(Number(lo))&&Number.isFinite(Number(hi))?`${lo} dB to ${hi} dB`:"Backend SNR range";
  $("channels").textContent=fmt(x.input_channels ?? x.channels);
  $("channelDetail").textContent=x.channel_names ? x.channel_names.join(", ") : "Signal channels";
  $("sourceName").textContent=x.name || "Backend";
  $("datasetStatus").textContent="DATASET READY";
  $("datasetStatusDetail").textContent="Metadata loaded from backend";
  $("formatBadge").textContent=x.format || "Backend";
  const specs=[
    ["Name",x.name],["Source",x.source],["Path",x.path],["Format",x.format],
    ["Classes",x.num_classes ?? x.class_count],["Signal length",x.signal_length],
    ["Input channels",x.input_channels ?? x.channels],["SNR range",x.snr_min!==undefined?`${x.snr_min} to ${x.snr_max} dB`:null],
    ["SNR step",x.snr_step!==undefined?`${x.snr_step} dB`:null],["Train split",x.train_size],["Validation split",x.val_size],["Test split",x.test_size]
  ];
  const grid=$("specGrid"); grid.innerHTML="";
  specs.forEach(([k,v])=>{
    if(v===undefined||v===null) return;
    const el=document.createElement("div"); el.className="spec";
    el.innerHTML=`<span>${escapeHTML(k)}</span><strong>${escapeHTML(String(v))}</strong>`;
    grid.appendChild(el);
  });
}

function renderClassDistribution(data) {
  const rows=Array.isArray(data)?data:(data.data||data.classes||[]);
  const normalized=rows.map(x=>({
    name:x.name??x.class_name??x.label??x.modulation,
    value:Number(x.samples??x.count??x.total??x.value)
  })).filter(x=>x.name!=null&&Number.isFinite(x.value));
  const box=$("classChart"); box.innerHTML="";
  if(!normalized.length){box.innerHTML='<div class="empty">No class distribution returned by backend.</div>';return;}
  const max=Math.max(...normalized.map(x=>x.value));
  normalized.sort((a,b)=>b.value-a.value);
  normalized.forEach(x=>{
    const row=document.createElement("div");row.className="bar-row";
    row.innerHTML=`<span class="bar-label" title="${escapeHTML(String(x.name))}">${escapeHTML(String(x.name))}</span><div class="bar-track"><div class="bar-fill" style="width:${max?x.value/max*100:0}%"></div></div><span class="bar-value">${fmt(x.value)}</span>`;
    box.appendChild(row);
  });
  populateClassSelect(normalized);
}

function renderSNRDistribution(data) {
  const rows=Array.isArray(data)?data:(data.data||data.snr||[]);
  const normalized=rows.map(x=>({
    snr:Number(x.snr??x.level??x.value),
    value:Number(x.samples??x.count??x.total??x.value_count??x.frequency)
  })).filter(x=>Number.isFinite(x.snr)&&Number.isFinite(x.value)).sort((a,b)=>a.snr-b.snr);
  const box=$("snrChart");box.innerHTML="";
  if(!normalized.length){box.innerHTML='<div class="empty">No SNR distribution returned by backend.</div>';return;}
  const max=Math.max(...normalized.map(x=>x.value));
  const chart=document.createElement("div");chart.className="snr-grid";
  normalized.forEach(x=>{
    const col=document.createElement("div");col.className="snr-col";
    const bar=document.createElement("div");bar.className="snr-bar";bar.style.height=`${max?x.value/max*100:0}%`;bar.title=`${x.snr} dB · ${fmt(x.value)} samples`;
    const val=document.createElement("span");val.className="snr-value";val.textContent=fmt(x.value);
    const lab=document.createElement("span");lab.className="snr-label";lab.textContent=x.snr;
    col.append(bar,val,lab);chart.appendChild(col);
  });
  box.appendChild(chart);
  populateSNRSelect(normalized.map(x=>x.snr));
}

function populateClassSelect(rows) {
  const s=$("classSelect");s.innerHTML='<option value="">Select class</option>';
  rows.forEach(x=>{const o=document.createElement("option");o.value=x.name;o.textContent=x.name;s.appendChild(o);});
}
function populateSNRSelect(values) {
  const s=$("snrSelect");s.innerHTML='<option value="">Select SNR</option>';
  values.forEach(x=>{const o=document.createElement("option");o.value=x;o.textContent=`${x} dB`;s.appendChild(o);});
}

function renderSplits(d) {
  const x=d.splits||d;
  const items=[
    ["Train",x.train??x.train_size??x.training],
    ["Validation",x.validation??x.val??x.val_size],
    ["Test",x.test??x.test_size]
  ].filter(x=>x[1]!==undefined&&x[1]!==null);
  const box=$("splitList");box.innerHTML="";
  if(!items.length){box.innerHTML='<div class="empty">No split information returned by backend.</div>';return;}
  items.forEach(([name,value])=>{
    let p=typeof value==="number"?value:(Number(value)||0);
    if(p>1) p=p/100;
    const row=document.createElement("div");row.className="split";
    row.innerHTML=`<label>${escapeHTML(name)}</label><div class="split-track"><div class="split-fill" style="width:${Math.max(0,Math.min(100,p*100))}%"></div></div><span class="split-value">${typeof value==="number"&&value<=1?pct(value):String(value)}</span>`;
    box.appendChild(row);
  });
}

function renderQuality(d) {
  const rows=Array.isArray(d)?d:(d.checks||d.data||[]);
  const box=$("qualityGrid");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="empty">No integrity checks returned by backend.</div>';return;}
  rows.forEach(x=>{
    const ok=x.ok??x.passed??x.status==="passed";
    const card=document.createElement("div");card.className="quality";
    card.innerHTML=`<div class="quality-top"><strong>${escapeHTML(String(x.name||x.check||"Check"))}</strong><span class="${ok?'ok':''}">${ok?'PASSED':String(x.status||'REVIEW')}</span></div><small>${escapeHTML(String(x.detail??x.message??"Backend result"))}</small>`;
    box.appendChild(card);
  });
}

function escapeHTML(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));}

function renderSample(d) {
  const values=Array.isArray(d)?d:(d.samples||d.values||d.waveform||[]);
  const points=values.map(v=>Array.isArray(v)?Number(v[0]):Number(v)).filter(Number.isFinite);
  const box=$("sampleArea");
  if(!points.length){box.innerHTML='<div class="sample-placeholder"><strong>No waveform returned</strong><span>The backend returned no numeric sample for this selection.</span></div>';return;}
  const max=Math.max(...points.map(Math.abs))||1;
  const W=1000,H=210,mid=105;
  const step=W/Math.max(1,points.length-1);
  const dpath=points.map((v,i)=>`${i?'L':'M'} ${(i*step).toFixed(2)} ${(mid-(v/max)*85).toFixed(2)}`).join(" ");
  box.innerHTML=`<svg class="wave" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><path class="grid" d="M0 30H1000M0 105H1000M0 180H1000"></path><path d="${dpath}"></path></svg>`;
}

async function loadSample(){
  const cls=$("classSelect").value,snr=$("snrSelect").value;
  if(!cls||snr===""){alert("Select both a modulation class and SNR level.");return;}
  $("sampleArea").innerHTML='<div class="sample-placeholder"><strong>Loading waveform…</strong><span>Fetching the selected signal from the backend.</span></div>';
  try{
    const q=`?class_name=${encodeURIComponent(cls)}&snr=${encodeURIComponent(snr)}`;
    renderSample(await api("/dataset/sample"+q));
  }catch(e){
    $("sampleArea").innerHTML='<div class="sample-placeholder"><strong>Sample unavailable</strong><span>The backend did not return a waveform for this selection.</span></div>';
  }
}

async function load(){
  $("refresh").disabled=true;
  try{
    const results=await Promise.allSettled([
      api("/dataset/metadata"),
      api("/dataset/class-distribution"),
      api("/dataset/snr-distribution"),
      api("/dataset/splits"),
      api("/dataset/quality")
    ]);
    let any=false;
    if(results[0].status==="fulfilled"){renderMeta(results[0].value);any=true}
    if(results[1].status==="fulfilled"){state.classes=results[1].value;renderClassDistribution(state.classes);any=true}
    if(results[2].status==="fulfilled"){state.snr=results[2].value;renderSNRDistribution(state.snr);any=true}
    if(results[3].status==="fulfilled"){renderSplits(results[3].value);any=true}
    if(results[4].status==="fulfilled"){state.quality=results[4].value;renderQuality(state.quality);any=true}
    setAPI(any,"Dataset endpoints "+(any?"responding":"unavailable"));
    $("lastUpdated").textContent=any?`Loaded ${new Date().toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"})}`:"Not loaded";
  }catch(e){setAPI(false,"Connection error")}
  finally{$("refresh").disabled=false}
}

$("refresh").addEventListener("click",load);
$("loadSample").addEventListener("click",loadSample);
$("menu").addEventListener("click",()=>{$("sidebar").classList.add("open");$("overlay").classList.add("show")});
$("overlay").addEventListener("click",()=>{$("sidebar").classList.remove("open");$("overlay").classList.remove("show")});
$("logout").addEventListener("click",()=>{window.location.href="../login/index.html"});
load();
