const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
const API_BASE_URL="/api";
const $=id=>document.getElementById(id),N=x=>x!==null&&x!==undefined&&x!==""&&Number.isFinite(Number(x))?Number(x):null,pct=x=>N(x)===null?"—":`${N(x).toFixed(2)}%`;
function errorText(body,status){
  const d=body&&(body.detail||body.message);
  if(typeof d==="string"&&d)return d;
  if(Array.isArray(d)&&d.length)return d.map(x=>x&&x.msg?x.msg:String(x)).join("; ");
  return `HTTP ${status}`;
}
async function api(path,options={}){
  const method=String(options.method||"GET").toUpperCase();
  const stateChanging=!["GET","HEAD","OPTIONS"].includes(method);
  if(stateChanging&&!window.AMCShieldSecurity?.fetch)throw new Error("Secure API client is unavailable.");
  const request=stateChanging?window.AMCShieldSecurity.fetch:fetch;
  const r=await request(API_BASE_URL+path,{credentials:"include",...options,headers:{Accept:"application/json",...(options.headers||{})}});
  let body={};
  try{body=await r.json()}catch(_){body={}}
  if(!r.ok||body?.success===false)throw new Error(errorText(body,r.status));
  return body?.data??body;
}

/* CSP-safe dynamic styling: markup carries data-* values, CSSOM applies them. */
function applyDynamicStyles(root){
  const scope=root||document;
  scope.querySelectorAll("[data-w]").forEach(e=>{e.style.width=e.dataset.w});
  scope.querySelectorAll("[data-h]").forEach(e=>{e.style.height=e.dataset.h});
  scope.querySelectorAll("[data-bg]").forEach(e=>{e.style.background=e.dataset.bg});
  scope.querySelectorAll("[data-cols]").forEach(e=>{e.style.setProperty("--cols",e.dataset.cols)});
}
function meanBy(rows,by,key){const g={};rows.forEach(x=>{const k=x[by];const v=N(x[key]);if(k===undefined||k===null||v===null)return;(g[k]=g[k]||[]).push(v)});return Object.entries(g).map(([k,v])=>({[by]:k,[key]:v.reduce((a,b)=>a+b,0)/v.length}))}
function asrBySnr(rows){const g={};rows.forEach(x=>{const k=N(x.snr);if(k===null)return;const e=g[k]=g[k]||{b:[],r:[]};const b=N(x.baseline_asr),r=N(x.robust_asr);if(b!==null)e.b.push(b);if(r!==null)e.r.push(r)});const m=a=>a.length?a.reduce((p,q)=>p+q,0)/a.length:null;return Object.entries(g).map(([k,e])=>({snr:Number(k),baseline_asr:m(e.b),robust_asr:m(e.r)}))}
function sel(id,vals,first){const s=$(id);s.innerHTML=`<option value="">${first}</option>`;(vals||[]).forEach(v=>{const val=typeof v==="object"?(v.value??v.name??v.attack??v.snr):v;if(val===undefined)return;const o=document.createElement("option");o.value=val;o.textContent=typeof v==="object"?(v.label??val):val;s.appendChild(o)})}
function lineChart(rows,mode){const a=rows.map(x=>({x:N(x.snr),b:N(x.baseline_accuracy),r:N(x.robust_accuracy),g:N(x.gap??x.generalization_gap),ba:N(x.baseline_asr),ra:N(x.robust_asr)})).filter(x=>x.x!==null).sort((a,b)=>a.x-b.x);let keys=mode==="accuracy"?["b","r"]:mode==="asr"?["ba","ra"]:["g"];if(!a.length)return null;const W=900,H=280,p={l:35,r:15,t:15,b:25},vals=a.flatMap(x=>keys.map(k=>x[k])).filter(v=>v!==null),lo=mode==="gap"?Math.min(-1,...vals):Math.max(0,Math.min(...vals)-5),hi=mode==="gap"?Math.max(1,...vals):Math.min(100,Math.max(...vals)+5),rg=hi-lo||1,x0=a[0].x,x1=a[a.length-1].x||x0+1,er=x1-x0||1,X=x=>p.l+(x-x0)/er*(W-p.l-p.r),Y=y=>H-p.b-(y-lo)/rg*(H-p.t-p.b);let grid="";for(let i=0;i<5;i++){const y=p.t+i*(H-p.t-p.b)/4;grid+=`<line class="gridline" x1="${p.l}" x2="${W-p.r}" y1="${y}" y2="${y}"/>`}const path=k=>a.filter(z=>z[k]!==null).map((z,i)=>`${i?"L":"M"}${X(z.x).toFixed(1)},${Y(z[k]).toFixed(1)}`).join(" ");let paths="";if(mode==="accuracy")paths=`<path class="line-base" d="${path("b")}"/><path class="line-robust" d="${path("r")}"/>`;else if(mode==="asr")paths=`<path class="line-base" d="${path("ba")}"/><path class="line-robust" d="${path("ra")}"/>`;else paths=`<path class="line-gap" d="${path("g")}"/>`;return `<div class="legend">${mode==="gap"?'<span><i class="green"></i>Gap</span>':'<span><i class="gold"></i>Baseline</span><span><i class="cyan"></i>Robust</span>'}</div><svg class="svg" viewBox="0 0 ${W} ${H}">${grid}${paths}</svg>`}
function bars(rows,key,labelKey="attack"){const a=rows.map(x=>({l:x[labelKey]??x.method??x.name,v:N(x[key])})).filter(x=>x.l&&x.v!==null);if(!a.length)return null;const m=Math.max(1,...a.map(x=>x.v));return `<div class="bars">${a.map(x=>`<div class="bar-col"><div class="bar" data-h="${Math.max(3,x.v/m*190)}px"></div><span>${String(x.l).slice(0,10)}</span></div>`).join("")}</div>`}
function heatmap(rows){const attacks=[...new Set(rows.map(x=>x.attack??x.method).filter(Boolean))],snrs=[...new Set(rows.map(x=>x.snr).filter(v=>v!==undefined&&v!==null))].sort((a,b)=>a-b);if(!attacks.length||!snrs.length)return null;let html=`<div class="heat" data-cols="${snrs.length}"><div class="heat-row"><div class="heat-title">ATTACK / SNR</div>${snrs.map(s=>`<div class="heat-label">${esc(s)}</div>`).join("")}</div>`;attacks.forEach(a=>{html+=`<div class="heat-row"><div class="heat-label">${esc(a)}</div>${snrs.map(s=>{const x=rows.find(r=>(r.attack??r.method)===a&&Number(r.snr)===Number(s));const v=N(x?.robust_asr??x?.asr);const alpha=v===null?.05:Math.max(.08,Math.min(.95,v/100));return `<div class="heat-cell" title="${esc(a)} / ${esc(s)} dB: ${v===null?"—":v.toFixed(2)+"%"}" data-bg="rgba(32,201,216,${alpha})">${v===null?"—":v.toFixed(0)+"%"}</div>`}).join("")}</div>`});return html+"</div>"}
async function load(){
 let ok=0,summary,filters,acc,asr,gap,classes,models,findings;
 try{summary=await api("/visualization/summary");ok++}catch(e){}
 try{filters=await api("/visualization/filters");ok++}catch(e){}
 try{acc=await api("/visualization/accuracy-by-snr");ok++}catch(e){}
 try{asr=await api("/visualization/asr-by-snr");ok++}catch(e){}
 try{gap=await api("/visualization/generalization-gap");ok++}catch(e){}
 try{classes=await api("/visualization/class-accuracy");ok++}catch(e){}
 try{models=await api("/visualization/model-comparison");ok++}catch(e){}
 try{findings=await api("/visualization/insights");ok++}catch(e){}
 const s=summary?.summary||summary||{};$("points").textContent=s.data_points??"—";$("snrs").textContent=s.snr_levels??"—";$("snrRange").textContent=s.snr_min!==undefined?`${s.snr_min} to ${s.snr_max} dB`:"Backend range";$("rasr").textContent=pct(s.mean_robust_asr);$("basr").textContent=pct(s.mean_baseline_asr);$("source").textContent=s.source||"Backend";
 const f=filters?.filters||filters||{};sel("attack",f.attacks||[],"All attacks");sel("model",f.models||[],"All models");sel("snr",f.snr_levels||f.snr||[],"All SNR");
 const ar=Array.isArray(acc)?acc:(acc?.data||acc?.results||[]),sr=Array.isArray(asr)?asr:(asr?.data||asr?.results||[]),gr=Array.isArray(gap)?gap:(gap?.data||gap?.results||[]);
 $("mainChart").innerHTML=lineChart(ar,"accuracy")||'<div class="empty">No accuracy data returned.</div>';
 $("attackChart").innerHTML=bars(meanBy(sr,"attack","robust_asr"),"robust_asr")||'<div class="empty">No ASR data returned.</div>';
 $("gapChart").innerHTML=lineChart(gr,"gap")||'<div class="empty">No gap data returned.</div>';
 $("heatmap").innerHTML=heatmap(sr)||'<div class="empty">No SNR/ASR matrix returned.</div>';
 const cr=Array.isArray(classes)?classes:(classes?.data||classes?.results||[]);$("classChart").innerHTML=bars(cr,"accuracy","class_name")||bars(cr,"accuracy","class")||'<div class="empty">No class accuracy returned.</div>';
 const mr=Array.isArray(models)?models:(models?.data||models?.results||[]);$("modelChart").innerHTML=bars(mr,"accuracy","model")||'<div class="empty">No model comparison returned.</div>';
 applyDynamicStyles(document);
 const fr=Array.isArray(findings)?findings:(findings?.findings||findings?.insights||findings?.data||[]);$("notes").innerHTML=fr.length?`<div class="notes-grid">${fr.map(x=>`<div class="note"><b>${esc(x.type??"FINDING")}</b><strong>${esc(x.title??x.name??"Evaluation finding")}</strong><p>${esc(x.description??x.message??"Backend-provided visualization note.")}</p></div>`).join("")}</div>`:'<div class="empty">No backend findings returned.</div>';
 $("api").classList.toggle("online",ok>0);$("api").classList.toggle("offline",ok===0);$("apiText").textContent=ok?`API ONLINE`:"API OFFLINE";$("apiDetail").textContent=ok?`${ok}/8 visualization endpoints responding`:"Visualization endpoints unavailable";$("liveText").textContent=ok?"VISUALS LIVE":"BACKEND UNAVAILABLE";$("liveDetail").textContent=ok?"Dynamic evaluation data loaded":"Waiting for FastAPI";$("updated").textContent=ok?`Updated ${new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})}`:"Not loaded"
}
$("view").addEventListener("change",async()=>{try{const d=await api("/visualization/"+($("view").value==="accuracy"?"accuracy-by-snr":$("view").value==="asr"?"asr-by-snr":"generalization-gap"));const rows0=Array.isArray(d)?d:(d.data||d.results||[]);const rows=$("view").value==="asr"?asrBySnr(rows0):rows0;$("chartTitle").textContent=$("view").value==="accuracy"?"Accuracy Across SNR":$("view").value==="asr"?"Attack Success Across SNR":"Generalization Gap Across SNR";$("mainChart").innerHTML=lineChart(rows,$("view").value)||'<div class="empty">No data returned.</div>';applyDynamicStyles($("mainChart"))}catch(e){$("mainChart").innerHTML='<div class="empty">Selected visualization is unavailable.</div>'}});
$("refresh").addEventListener("click",load);$("apply").addEventListener("click",load);$("reset").addEventListener("click",()=>{["attack","model","snr"].forEach(id=>$(id).value="");$("view").value="accuracy";load()});
$("menu").addEventListener("click",()=>{$("sidebar").classList.add("open");$("overlay").classList.add("show")});$("overlay").addEventListener("click",()=>{$("sidebar").classList.remove("open");$("overlay").classList.remove("show")});$("logout").addEventListener("click",async()=>{try{await AMCShieldSecurity.fetch("/api/auth/logout",{method:"POST"});}finally{location.href="/";}});load();
