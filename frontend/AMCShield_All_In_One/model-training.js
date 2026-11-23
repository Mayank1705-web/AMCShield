/*
 * AMCShield Model Training page.
 * Training values and charts are fetched from FastAPI.
 * No experimental metrics are hardcoded in the UI.
 */
const API_BASE_URL="/api";
const $=id=>document.getElementById(id);

async function api(path, options={}){
  if(!window.AMCShieldSecurity?.fetch) throw new Error("Secure API client is unavailable.");
  const r=await window.AMCShieldSecurity.fetch(API_BASE_URL+path,{
    ...options,
    credentials:"include",
    headers:{Accept:"application/json",...(options.headers||{})}
  });
  if(!r.ok){let d="";try{const x=await r.json();d=x.detail||x.message||""}catch(_){}throw new Error(d||`HTTP ${r.status}`)}
  return r.json();
}
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
const num=(x,d=0)=>Number.isFinite(Number(x))?Number(x):d;
const fmt=x=>Number.isFinite(Number(x))?new Intl.NumberFormat("en-IN",{maximumFractionDigits:2}).format(Number(x)):"—";
const pct=x=>Number.isFinite(Number(x))?`${Number(x).toFixed(2)}%`:"—";

/* CSP-safe dynamic styling: markup carries data-* values, CSSOM applies them. */
function applyDynamicStyles(root){
  const scope=root||document;
  scope.querySelectorAll("[data-w]").forEach(e=>{e.style.width=e.dataset.w});
  scope.querySelectorAll("[data-h]").forEach(e=>{e.style.height=e.dataset.h});
  scope.querySelectorAll("[data-bg]").forEach(e=>{e.style.background=e.dataset.bg});
  scope.querySelectorAll("[data-cols]").forEach(e=>{e.style.setProperty("--cols",e.dataset.cols)});
}

async function applyRoleGate(){
  const buttons=["startRun","resumeRun","stopRun"].map($);
  let isAdmin=false;
  try{
    const r=await fetch("/api/auth/me",{credentials:"include",headers:{Accept:"application/json"}});
    if(r.ok){
      const body=await r.json();
      const user=body.user||{};
      isAdmin=user.role_key==="admin"||String(user.role||"").toLowerCase()==="administrator";
    }
  }catch(_){isAdmin=false}
  if(!isAdmin){
    buttons.forEach(b=>{if(b)b.disabled=true});
    $("commandStatus").textContent="Administrator access required to control training.";
  }
}
function setAPI(ok,msg){
  $("apiStatus").classList.toggle("online",ok);$("apiStatus").classList.toggle("offline",!ok);
  $("apiText").textContent=ok?"API ONLINE":"API OFFLINE";$("apiDetail").textContent=msg;
}
function setMetric(id,value,detail){$(id).textContent=value??"—";if(detail!==undefined)$(id+"Detail") && ($(id+"Detail").textContent=detail)}

function renderOverview(d){
  const x=d.run||d;
  $("activeRun").textContent=x.name||x.run_name||x.id||"—";
  $("activeRunDetail").textContent=x.status||"Training run";
  $("epoch").textContent=x.epoch!==undefined?fmt(x.epoch):"—";
  $("epochDetail").textContent=x.total_epochs!==undefined?`of ${fmt(x.total_epochs)} epochs`:"Current epoch";
  $("trainAcc").textContent=pct(x.train_accuracy);
  $("trainLoss").textContent=`Loss: ${x.train_loss!==undefined?Number(x.train_loss).toFixed(4):"—"}`;
  $("valAcc").textContent=pct(x.val_accuracy??x.validation_accuracy);
  $("valLoss").textContent=`Loss: ${x.val_loss!==undefined?Number(x.val_loss).toFixed(4):"—"}`;
  $("device").textContent=x.device||"—";$("deviceDetail").textContent=x.gpu_name||"Backend hardware";
  $("checkpoint").textContent=x.checkpoint_status||x.checkpoint_name||"—";
  $("checkpointDetail").textContent=x.checkpoint_path||"Checkpoint metadata";
  $("engineStatus").textContent=x.status||"Backend";
  $("runStateText").textContent=(x.status||"READY").toUpperCase();
  $("runStateDetail").textContent=x.message||"Training backend";
  const p=Math.max(0,Math.min(100,num(x.progress_percent)));
  $("progressPct").textContent=x.progress_percent!==undefined?pct(p):"—";
  $("progressFill").style.width=`${p}%`;
  $("progressLabel").textContent=x.phase||x.status||"No active run";
  $("eta").textContent=x.eta||"ETA —";
}

function renderConfig(d){
  const x=d.config||d;const entries=[
    ["Dataset",x.dataset?.name??x.dataset_name],["Batch size",x.batch_size],
    ["Epochs",x.epochs],["Learning rate",x.learning_rate],
    ["Optimizer",x.optimizer],["Loss function",x.loss_function],
    ["Adversarial ratio",x.adv_train_ratio],["FGSM ε",x.epsilon?.fgsm],
    ["PGD ε",x.epsilon?.pgd],["PGD steps",x.pgd_steps??x.steps],
    ["Input shape",x.input_shape],["Classes",x.num_classes]
  ].filter(a=>a[1]!==undefined&&a[1]!==null);
  const box=$("configGrid");box.innerHTML="";
  if(!entries.length){box.innerHTML='<div class="empty">No training configuration returned by backend.</div>';return}
  entries.forEach(([k,v])=>{const e=document.createElement("div");e.className="config";e.innerHTML=`<span>${esc(k)}</span><strong>${esc(typeof v==="object"?JSON.stringify(v):v)}</strong>`;box.appendChild(e)});
  $("configBadge").textContent=x.source||"Backend";
}

function renderProgress(d){
  const rows=Array.isArray(d)?d:(d.phases||d.data||[]);
  const box=$("phaseList");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="empty">No training phases returned by backend.</div>';return}
  rows.forEach(x=>{const p=Math.max(0,Math.min(100,num(x.progress_percent??x.progress)));const e=document.createElement("div");e.className="phase";e.innerHTML=`<label>${esc(x.name??x.phase)}</label><div class="phase-track"><div class="phase-fill" data-w="${p}%"></div></div><span class="phase-value">${p.toFixed(1)}%</span>`;box.appendChild(e)})
}

function chartSVG(rows,type){
  const arr=rows.map(x=>({epoch:num(x.epoch),a:num(x.train_accuracy),v:num(x.val_accuracy),l:num(x.train_loss),vl:num(x.val_loss)})).filter(x=>x.epoch||x.a||x.v||x.l||x.vl);
  if(!arr.length)return null;
  const W=900,H=260,pad={l:40,r:18,t:25,b:30};
  const vals=type==="accuracy"?arr.flatMap(x=>[x.a,x.v]).filter(Number.isFinite):arr.flatMap(x=>[x.l,x.vl]).filter(Number.isFinite);
  const min=Math.min(...vals),max=Math.max(...vals),range=(max-min)||1;
  const ex=Math.min(...arr.map(x=>x.epoch)),ex2=Math.max(...arr.map(x=>x.epoch)),er=(ex2-ex)||1;
  const x=e=>pad.l+(e-ex)/er*(W-pad.l-pad.r), y=v=>H-pad.b-(v-min)/range*(H-pad.t-pad.b);
  const path=k=>arr.filter(z=>Number.isFinite(z[k])).map((z,i)=>`${i?"L":"M"}${x(z.epoch).toFixed(1)},${y(z[k]).toFixed(1)}`).join(" ");
  let lines="";
  for(let i=0;i<5;i++){const gy=pad.t+i*(H-pad.t-pad.b)/4;lines+=`<line class="svg-grid" x1="${pad.l}" x2="${W-pad.r}" y1="${gy}" y2="${gy}"/>`}
  const p1=type==="accuracy"?path("a"):path("l"),p2=type==="accuracy"?path("v"):path("vl");
  const legend=type==="accuracy"?`<div class="legend"><span><i class="cyan"></i>Train</span><span><i class="gold"></i>Validation</span></div>`:`<div class="legend"><span><i class="cyan"></i>Train</span><span><i class="gold"></i>Validation</span></div>`;
  return `${legend}<svg class="svg-chart" viewBox="0 0 ${W} ${H}">${lines}<path class="line-train" d="${p1}"/><path class="line-val" d="${p2}"/></svg>`;
}
function renderHistory(d){
  const rows=Array.isArray(d)?d:(d.history||d.data||[]);
  const a=chartSVG(rows,"accuracy"),l=chartSVG(rows,"loss");
  $("accuracyChart").innerHTML=a||'<div class="empty">No accuracy history returned by backend.</div>';
  $("lossChart").innerHTML=l||'<div class="empty">No loss history returned by backend.</div>';
}

function renderModels(d){
  const rows=Array.isArray(d)?d:(d.models||d.data||[]);const box=$("modelList");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="empty">No model registry data returned by backend.</div>';return}
  rows.forEach(x=>{const e=document.createElement("div");e.className="model-row";e.innerHTML=`<div><strong>${esc(x.name??x.model_name)}</strong><small>${esc(x.architecture??x.description??"Model metadata")}</small></div><span class="model-tag ${x.best?'best':''}">${esc(x.status??(x.best?"BEST":"REGISTERED"))}</span>`;box.appendChild(e)})
}
function renderCheckpoints(d){
  const rows=Array.isArray(d)?d:(d.checkpoints||d.data||[]);const box=$("checkpointList");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="empty">No checkpoint metadata returned by backend.</div>';return}
  rows.forEach(x=>{const e=document.createElement("div");e.className="checkpoint-row";e.innerHTML=`<div><strong>${esc(x.name??x.filename??"Checkpoint")}</strong><small>${esc(x.path??x.created_at??"Artifact metadata")}</small></div><small>${esc(x.status??"AVAILABLE")}</small>`;box.appendChild(e)})
}
function renderLogs(d){
  const rows=Array.isArray(d)?d:(d.logs||d.data||[]);const box=$("logWindow");box.innerHTML="";
  if(!rows.length){box.innerHTML='<div class="log-empty">No telemetry received from backend.</div>';return}
  rows.slice(-250).forEach(x=>{const e=document.createElement("div");e.className="log-line";e.innerHTML=`<span class="time">[${esc(x.time??x.timestamp??"—")}]</span> <span class="${x.level==="error"?"":"info"}">${esc(x.message??x.text??"")}</span>`;box.appendChild(e)});
  box.scrollTop=box.scrollHeight;
}

async function load(){
  $("refresh").disabled=true;
  const endpoints=[
    ["/training/overview",renderOverview],
    ["/training/config",renderConfig],
    ["/training/progress",renderProgress],
    ["/training/history",renderHistory],
    ["/training/models",renderModels],
    ["/training/checkpoints",renderCheckpoints],
    ["/training/logs",renderLogs]
  ];
  let ok=0;
  for(const [path,fn] of endpoints){try{const response=await api(path);fn(response?.data??response);applyDynamicStyles(document);ok++}catch(_){/* each panel reports backend absence */}}
  setAPI(ok>0,ok?`${ok}/${endpoints.length} training endpoints responding`:"Training endpoints unavailable");
  $("updated").textContent=ok?`Updated ${new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})}`:"Not loaded";
  $("refresh").disabled=false;
}

async function command(path,label,includeModel=false){
  $("commandStatus").textContent=`${label} request sent…`;
  try{
    const options={method:"POST"};
    if(includeModel){
      options.headers={"Content-Type":"application/json"};
      options.body=JSON.stringify({model:$("modelSelect").value});
    }
    const response=await api(path,options);
    const result=response?.data??response;
    $("commandStatus").textContent=result?.run?.message||result?.message||`${label} request accepted by backend.`;
    await load();
  }catch(e){
    $("commandStatus").textContent=`${label} unavailable: ${e.message}`;
  }
}
$("startRun").addEventListener("click",()=>command("/training/start","Start training",true));
$("resumeRun").addEventListener("click",()=>command("/training/resume","Resume training",true));
$("stopRun").addEventListener("click",()=>command("/training/stop","Stop training"));
$("refresh").addEventListener("click",load);
applyRoleGate();
$("clearLog").addEventListener("click",()=>$("logWindow").innerHTML='<div class="log-empty">Log view cleared. Backend telemetry remains unchanged.</div>');
$("menu").addEventListener("click",()=>{$("sidebar").classList.add("open");$("overlay").style.display="block"});
$("overlay").addEventListener("click",()=>{$("sidebar").classList.remove("open");$("overlay").style.display="none"});
$("logout").addEventListener("click",async()=>{try{await AMCShieldSecurity.fetch("/api/auth/logout",{method:"POST"});}finally{location.href="/";}});
load();
