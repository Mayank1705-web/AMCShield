const API_BASE_URL="/api";
const $=id=>document.getElementById(id);
async function api(path){const r=await fetch(API_BASE_URL+path,{headers:{Accept:"application/json"}});if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()}
function text(v){return v===undefined||v===null||v===""?"—":String(v)}
function esc(v){return text(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]))}
function safeUrl(v){
  const raw=text(v);
  try{
    const u=new URL(raw,window.location.origin);
    return ["http:","https:"].includes(u.protocol) ? u.href : "#";
  }catch(_){return "#"}
}
function renderFacts(d){const a=Array.isArray(d)?d:(d.facts||d.data||[]),box=$("facts");box.innerHTML=a.length?a.map(x=>`<div class="fact"><b>${esc(x.label??x.key??"FACT")}</b><strong>${esc(x.value)}</strong></div>`).join(""):'<div class="empty">No project facts returned.</div>'}
function renderModels(d){const a=Array.isArray(d)?d:(d.models||d.data||[]),box=$("modelList");box.innerHTML=a.length?a.map(x=>`<div class="model"><div class="model-top"><strong>${esc(x.name??x.model)}</strong><span>${esc(x.parameters??x.parameter_count??"MODEL")}</span></div><p>${esc(x.description??x.architecture??"Backend model metadata.")}</p></div>`).join(""):'<div class="empty">No model metadata returned.</div>'}
function renderAttacks(d){const a=Array.isArray(d)?d:(d.attacks||d.data||[]),box=$("attackList");box.innerHTML=a.length?a.map(x=>`<div class="attack"><div class="attack-top"><strong>${esc(x.name??x.method)}</strong><span>${esc(x.type??"THREAT")}</span></div><p>${esc(x.description??x.details??"Backend attack metadata.")}</p>${x.epsilon!==undefined?`<span class="tag">ε ${x.epsilon}</span>`:""}</div>`).join(""):'<div class="empty">No attack metadata returned.</div>'}
function renderContext(d){const a=Array.isArray(d)?d:(d.items||d.context||d.data||[]),box=$("context");box.innerHTML=a.length?a.map(x=>`<div class="context"><b>${esc(x.label??x.key??"INFO")}</b><strong>${esc(x.value)}</strong></div>`).join(""):'<div class="empty">No project context returned.</div>'}
function renderLinks(d){const a=Array.isArray(d)?d:(d.links||d.resources||d.data||[]),box=$("links");if(!a.length)return;box.innerHTML=a.map(x=>`<a class="resource" href="${safeUrl(x.url)}" target="_blank" rel="noopener"><b>${text(x.type??"RESOURCE")}</b><strong>${esc(x.title??x.name)}</strong><small>${esc(x.description)}</small><span>↗</span></a>`).join("")}
async function load(){
 let ok=0;
 const calls=[
  ["/project/overview",d=>{const x=d.project||d||{};$("status").textContent=text(x.status);$("classes").textContent=text(x.class_count??x.num_classes);$("snr").textContent=x.snr_min!==undefined?`${x.snr_min} to ${x.snr_max} dB`:"—";$("models").textContent=text(x.model_count);$("source").textContent=text(x.source??"Backend");$("heroDescription").textContent=text(x.description??$("heroDescription").textContent)}],
  ["/project/mission",d=>{const x=d.mission||d||{};const b=$("missionText");b.innerHTML=`${x.mission?`<p>${esc(x.mission)}</p>`:""}${x.objective?`<p>${esc(x.objective)}</p>`:""}${x.scope?`<p>${esc(x.scope)}</p>`:""}${!x.mission&&!x.objective&&!x.scope?b.innerHTML:""}`}],
  ["/project/facts",renderFacts],
  ["/project/models",renderModels],
  ["/project/attacks",renderAttacks],
  ["/project/context",renderContext],
  ["/project/resources",renderLinks]
 ];
 for(const [p,f] of calls){try{f(await api(p));ok++}catch(e){}}
 $("api").classList.toggle("online",ok>0);$("api").classList.toggle("offline",ok===0);
 $("apiText").textContent=ok?"API ONLINE":"API OFFLINE";$("apiDetail").textContent=ok?`${ok}/${calls.length} project endpoints responding`:"Project endpoints unavailable";
 $("liveText").textContent=ok?"PROJECT PROFILE READY":"BACKEND UNAVAILABLE";$("liveDetail").textContent=ok?"Dynamic project metadata loaded":"Waiting for FastAPI";
 $("updated").textContent=ok?`Updated ${new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})}`:"Not loaded"
}
$("refresh").addEventListener("click",load);
$("menu").addEventListener("click",()=>{$("sidebar").classList.add("open");$("overlay").classList.add("show")});
$("overlay").addEventListener("click",()=>{$("sidebar").classList.remove("open");$("overlay").classList.remove("show")});
$("logout").addEventListener("click",async()=>{try{await AMCShieldSecurity.fetch("/api/auth/logout",{method:"POST"});}finally{location.href="/";}});
load();
