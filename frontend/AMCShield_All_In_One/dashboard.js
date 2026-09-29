/*
 * AMCShield Dashboard
 *
 * The dashboard intentionally contains NO hardcoded experimental
 * values. All metrics, chart points, attack results, SNR values
 * and recent activity are expected from the backend API.
 */

const API_BASE_URL = "http://localhost:8000/api";

const state = {
  summary: null,
  performance: null,
  attacks: null,
  gap: null,
  snr: null,
  updates: null,
  loading: false,
  lastUpdated: null
};

const $ = (id) => document.getElementById(id);

const els = {
  sidebar: $("sidebar"),
  overlay: $("mobileOverlay"),
  menu: $("menuButton"),
  refresh: $("refreshButton"),
  refreshIcon: $("refreshIcon"),
  apiStatus: document.querySelector(".api-status"),
  apiStatusText: $("apiStatusText"),
  apiStatusDetail: $("apiStatusDetail"),
  lastUpdated: $("lastUpdated"),
  classes: $("metricClasses"),
  classesSub: $("metricClassesSub"),
  samples: $("metricSamples"),
  samplesSub: $("metricSamplesSub"),
  snr: $("metricSnr"),
  snrSub: $("metricSnrSub"),
  attacks: $("metricAttacks"),
  attacksSub: $("metricAttacksSub"),
  performance: $("performanceChart"),
  performanceLoading: $("performanceLoading"),
  attack: $("attackChart"),
  attackLoading: $("attackLoading"),
  attackSelect: $("attackSelect"),
  heatmap: $("heatmapWrap"),
  comparison: $("comparisonBody"),
  snrSelect: $("snrSelect"),
  snrContent: $("snrContent"),
  updates: $("recentUpdates"),
  toast: $("toast"),
  userName: $("userName"),
  userRole: $("userRole")
};

async function fetchAPI(endpoint, options = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Accept": "application/json",
      ...(options.headers || {})
    }
  });

  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = body.detail || body.message || "";
    } catch (_) {}
    throw new Error(detail || `API request failed (${response.status})`);
  }

  return response.json();
}

function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return new Intl.NumberFormat("en-IN", {
    notation: Math.abs(value) >= 100000 ? "compact" : "standard",
    maximumFractionDigits: 1
  }).format(value);
}

function formatPercent(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return `${value.toFixed(2)}%`;
}

function formatSNR(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return `${value} dB`;
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.classList.remove("show"), 3500);
}

function setAPIStatus(online, detail = "") {
  els.apiStatus.classList.toggle("online", online);
  els.apiStatus.classList.toggle("offline", !online);
  els.apiStatusText.textContent = online ? "API ONLINE" : "API OFFLINE";
  els.apiStatusDetail.textContent = detail || (online ? "Backend connected" : "Backend unavailable");
}

function setLoading(isLoading) {
  state.loading = isLoading;
  els.refresh.classList.toggle("loading", isLoading);
  els.refresh.disabled = isLoading;
}

function setLastUpdated() {
  state.lastUpdated = new Date();
  els.lastUpdated.textContent = state.lastUpdated.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function clearMetricSkeletons() {
  [els.classes, els.samples, els.snr, els.attacks].forEach((el) => {
    el.classList.remove("skeleton");
  });
}

function renderSummary(data) {
  clearMetricSkeletons();

  const dataset = data?.dataset || {};
  const evaluation = data?.evaluation || {};
  const attacks = data?.attacks || data?.attack_methods || [];

  const attackCount = Array.isArray(attacks)
    ? attacks.length
    : Number(evaluation.attack_methods ?? 0);

  els.classes.textContent = formatNumber(dataset.num_classes);
  els.classesSub.textContent = dataset.name || "Dataset";

  els.samples.textContent = formatNumber(
    evaluation.test_samples ?? dataset.test_samples ?? dataset.total_test_samples
  );
  els.samplesSub.textContent = dataset.name || "Test split";

  els.snr.textContent = formatNumber(
    evaluation.snr_levels ?? dataset.snr_levels
  );
  const min = dataset.snr_min;
  const max = dataset.snr_max;
  els.snrSub.textContent =
    Number.isFinite(min) && Number.isFinite(max)
      ? `${formatSNR(min)} to ${formatSNR(max)}`
      : "SNR range from backend";

  els.attacks.textContent = formatNumber(attackCount);
  els.attacksSub.textContent = "Methods in evaluation";
}

function normalizeSeries(payload, key) {
  if (!payload) return [];
  const rows = Array.isArray(payload) ? payload : payload.data;
  if (Array.isArray(rows)) {
    if (!rows.length) return [];
    if (typeof rows[0] === "object") {
      return rows
        .filter(r => Number.isFinite(Number(r.snr)) && Number.isFinite(Number(r[key])))
        .map(r => ({ snr: Number(r.snr), value: Number(r[key]) }));
    }
  }
  const snr = payload.snr || payload.snr_levels;
  const values = payload[key];
  if (Array.isArray(snr) && Array.isArray(values)) {
    return snr
      .map((x, i) => ({ snr: Number(x), value: Number(values[i]) }))
      .filter(r => Number.isFinite(r.snr) && Number.isFinite(r.value));
  }
  return [];
}

function svgEl(name, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function drawLineChart(svg, series, options = {}) {
  svg.replaceChildren();
  const width = svg.clientWidth || 720;
  const height = svg.clientHeight || 290;
  const pad = { left: 44, right: 18, top: 18, bottom: 34 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;

  const all = series.flatMap(s => s.points);
  if (!all.length) return false;

  const xs = all.map(p => p.snr);
  const ys = all.map(p => p.value);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(0, Math.min(...ys));
  const maxY = Math.max(options.maxY ?? 100, Math.max(...ys));
  const xScale = x => pad.left + ((x - minX) / ((maxX - minX) || 1)) * w;
  const yScale = y => pad.top + h - ((y - minY) / ((maxY - minY) || 1)) * h;

  const grid = svgEl("g");
  const axis = svgEl("g");
  const paths = svgEl("g");

  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const y = minY + ((maxY - minY) * i / ticks);
    const yy = yScale(y);
    grid.appendChild(svgEl("line", {
      x1: pad.left, x2: width - pad.right, y1: yy, y2: yy,
      stroke: "rgba(210,225,216,.09)", "stroke-width": "1"
    }));
    const label = svgEl("text", {
      x: pad.left - 8, y: yy + 3, "text-anchor": "end",
      fill: "#77837c", "font-size": "9"
    });
    label.textContent = Number(y).toFixed(0);
    axis.appendChild(label);
  }

  const xTicks = [...new Set(xs)].sort((a,b) => a-b);
  const step = Math.max(1, Math.ceil(xTicks.length / 8));
  xTicks.filter((_, i) => i % step === 0 || i === xTicks.length - 1).forEach(x => {
    const xx = xScale(x);
    const label = svgEl("text", {
      x: xx, y: height - 10, "text-anchor": "middle",
      fill: "#77837c", "font-size": "9"
    });
    label.textContent = x;
    axis.appendChild(label);
  });

  const colors = options.colors || ["#f4c542", "#20c9d8", "#52d273", "#ff7070", "#c58cff"];
  series.forEach((s, idx) => {
    if (!s.points.length) return;
    const d = s.points.map((p, i) =>
      `${i ? "L" : "M"} ${xScale(p.snr).toFixed(2)} ${yScale(p.value).toFixed(2)}`
    ).join(" ");
    paths.appendChild(svgEl("path", {
      d, fill: "none", stroke: colors[idx % colors.length],
      "stroke-width": "2.2", "stroke-linecap": "round", "stroke-linejoin": "round"
    }));

    s.points.forEach(p => {
      const c = svgEl("circle", {
        cx: xScale(p.snr), cy: yScale(p.value), r: "2.7",
        fill: colors[idx % colors.length], stroke: "#07100d", "stroke-width": "1"
      });
      const title = svgEl("title");
      title.textContent = `${s.name} · SNR ${p.snr} dB · ${p.value.toFixed(2)}%`;
      c.appendChild(title);
      paths.appendChild(c);
    });
  });

  const xTitle = svgEl("text", {
    x: pad.left + w / 2, y: height + 1, "text-anchor": "middle",
    fill: "#77837c", "font-size": "9"
  });
  xTitle.textContent = "SNR (dB)";
  axis.appendChild(xTitle);

  svg.append(grid, paths, axis);
  return true;
}

function renderPerformance(data) {
  const baseline = normalizeSeries(data, "baseline");
  const robust = normalizeSeries(data, "robust");

  const hasData = drawLineChart(els.performance, [
    { name: "Baseline", points: baseline },
    { name: "Robust", points: robust }
  ], { maxY: 100, colors: ["#f4c542", "#20c9d8"] });

  els.performance.classList.toggle("has-data", hasData);
  els.performanceLoading.classList.toggle("hidden", hasData);
}

function renderAttackSelect(methods) {
  const previous = els.attackSelect.value;
  els.attackSelect.innerHTML = '<option value="">All attacks</option>';
  methods.forEach(method => {
    const option = document.createElement("option");
    option.value = method;
    option.textContent = method;
    els.attackSelect.appendChild(option);
  });
  if (methods.includes(previous)) els.attackSelect.value = previous;
}

function getAttackSeries(data, selected = "") {
  const rows = Array.isArray(data) ? data : data?.data;
  if (Array.isArray(rows) && rows.length && rows[0].attack) {
    const methods = [...new Set(rows.map(r => r.attack))];
    return methods
      .filter(m => !selected || m === selected)
      .map((method, idx) => ({
        name: method,
        points: rows
          .filter(r => r.attack === method)
          .map(r => ({ snr: Number(r.snr), value: Number(r.baseline_asr ?? r.attack_success_rate ?? r.asr) }))
          .filter(p => Number.isFinite(p.snr) && Number.isFinite(p.value))
      }));
  }

  const methods = data?.methods || data?.attacks || {};
  const entries = Array.isArray(methods)
    ? methods.map(m => [m.name || m.attack, m])
    : Object.entries(methods);

  return entries
    .filter(([name]) => name && (!selected || name === selected))
    .map(([name, payload]) => ({
      name,
      points: normalizeSeries(payload, "baseline_asr").length
        ? normalizeSeries(payload, "baseline_asr")
        : normalizeSeries(payload, "attack_success_rate").length
          ? normalizeSeries(payload, "attack_success_rate")
          : normalizeSeries(payload, "asr")
    }))
    .filter(s => s.points.length);
}

function renderAttackChart(data) {
  const series = getAttackSeries(data, els.attackSelect.value);
  const methods = getAttackSeries(data).map(s => s.name);
  renderAttackSelect(methods);

  const hasData = drawLineChart(els.attack, series, {
    maxY: 100,
    colors: ["#f4c542", "#20c9d8", "#52d273", "#ff7070", "#c58cff"]
  });

  els.attack.classList.toggle("has-data", hasData);
  els.attackLoading.classList.toggle("hidden", hasData);
}

function heatColor(value, min, max) {
  if (!Number.isFinite(value)) return "rgba(255,255,255,.03)";
  const t = Math.max(0, Math.min(1, (value - min) / ((max - min) || 1)));
  if (value >= 0) {
    return `rgba(244,197,66,${0.12 + t * 0.72})`;
  }
  const a = Math.min(1, Math.abs(value - min) / (Math.abs(min) || 1));
  return `rgba(32,201,216,${0.12 + a * 0.62})`;
}

function renderHeatmap(data) {
  els.heatmap.replaceChildren();

  const rows = Array.isArray(data) ? data : data?.data;
  if (!Array.isArray(rows) || !rows.length) {
    const empty = document.createElement("div");
    empty.className = "snr-empty";
    empty.textContent = "No generalization-gap data available from backend.";
    els.heatmap.appendChild(empty);
    return;
  }

  const attacks = [...new Set(rows.map(r => r.attack))];
  const snrs = [...new Set(rows.map(r => Number(r.snr)))].sort((a,b) => a-b);
  const values = rows.map(r => Number(r.gap)).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);

  const grid = document.createElement("div");
  grid.className = "heatmap";
  grid.style.gridTemplateColumns = `110px repeat(${snrs.length}, minmax(25px, 1fr))`;

  grid.appendChild(document.createElement("div"));
  snrs.forEach(snr => {
    const cell = document.createElement("div");
    cell.className = "hm-label";
    cell.style.justifyContent = "center";
    cell.textContent = snr;
    grid.appendChild(cell);
  });

  attacks.forEach(attack => {
    const label = document.createElement("div");
    label.className = "hm-label";
    label.textContent = attack;
    grid.appendChild(label);

    snrs.forEach(snr => {
      const row = rows.find(r => r.attack === attack && Number(r.snr) === snr);
      const value = row ? Number(row.gap) : NaN;
      const cell = document.createElement("div");
      cell.className = "hm-cell";
      cell.style.background = heatColor(value, min, max);
      cell.title = `${attack} · ${snr} dB · ${Number.isFinite(value) ? value.toFixed(2) : "N/A"} pp`;
      cell.textContent = Number.isFinite(value) ? value.toFixed(1) : "—";
      grid.appendChild(cell);
    });
  });

  els.heatmap.appendChild(grid);
}

function renderComparison(data) {
  const rows = Array.isArray(data) ? data : data?.data;
  els.comparison.replaceChildren();

  if (!Array.isArray(rows) || !rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "table-empty";
    td.textContent = "No comparison data available from backend.";
    tr.appendChild(td);
    els.comparison.appendChild(tr);
    return;
  }

  rows.forEach(row => {
    const tr = document.createElement("tr");
    const values = [
      row.attack,
      formatPercent(Number(row.baseline_asr)),
      formatPercent(Number(row.robust_asr)),
      Number.isFinite(Number(row.gap)) ? `${Number(row.gap).toFixed(2)} pp` : "N/A"
    ];

    values.forEach((value, i) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (i === 3 && Number.isFinite(Number(row.gap))) {
        td.className = Number(row.gap) >= 0 ? "gap-positive" : "gap-negative";
      }
      tr.appendChild(td);
    });
    els.comparison.appendChild(tr);
  });
}

function populateSNR(data) {
  const values = Array.isArray(data) ? data : data?.snr || data?.snr_levels || [];
  const snrs = [...new Set(values.map(v => Number(typeof v === "object" ? v.snr : v)).filter(Number.isFinite))].sort((a,b) => a-b);

  els.snrSelect.innerHTML = '<option value="">Select SNR</option>';
  snrs.forEach(snr => {
    const option = document.createElement("option");
    option.value = snr;
    option.textContent = `${snr} dB`;
    els.snrSelect.appendChild(option);
  });
}

function findSNRRow(data, snr) {
  const rows = Array.isArray(data) ? data : data?.data;
  if (Array.isArray(rows)) {
    return rows.find(r => Number(r.snr) === Number(snr)) || null;
  }
  return null;
}

function renderSNRDetail(snr) {
  if (snr === "") {
    els.snrContent.innerHTML = '<div class="snr-empty">Select an SNR level after backend data is loaded.</div>';
    return;
  }

  const row = findSNRRow(state.snr, snr);
  if (!row) {
    els.snrContent.innerHTML = '<div class="snr-empty">No backend data is available for this SNR level.</div>';
    return;
  }

  const metrics = [
    ["Baseline Accuracy", row.baseline_accuracy ?? row.baseline_clean_accuracy],
    ["Robust Accuracy", row.robust_accuracy ?? row.clean_accuracy],
    ["FGSM ASR", row.fgsm_asr],
    ["PGD ASR", row.pgd_asr],
    ["MIM ASR", row.mim_asr],
    ["C&W ASR", row.cw_asr],
    ["Black-box ASR", row.blackbox_asr]
  ];

  els.snrContent.innerHTML = "";
  metrics.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "snr-metric";
    const span = document.createElement("span");
    span.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = formatPercent(Number(value));
    card.append(span, strong);
    els.snrContent.appendChild(card);
  });
}

function renderUpdates(data) {
  const rows = Array.isArray(data) ? data : data?.updates || data?.data;
  els.updates.replaceChildren();

  if (!Array.isArray(rows) || !rows.length) {
    const empty = document.createElement("div");
    empty.className = "table-empty";
    empty.textContent = "No recent updates available from backend.";
    els.updates.appendChild(empty);
    return;
  }

  rows.slice(0, 8).forEach(item => {
    const row = document.createElement("div");
    row.className = "update-item";

    const dot = document.createElement("span");
    dot.className = "update-dot";

    const body = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = item.title || "Untitled update";
    const type = document.createElement("small");
    type.textContent = item.type || "activity";
    body.append(title, type);

    const date = document.createElement("span");
    date.className = "update-date";
    date.textContent = item.date || item.created_at || "—";

    row.append(dot, body, date);
    els.updates.appendChild(row);
  });
}

async function loadDashboard() {
  setLoading(true);

  try {
    const results = await Promise.allSettled([
      fetchAPI("/dashboard/summary"),
      fetchAPI("/dashboard/performance"),
      fetchAPI("/dashboard/attacks"),
      fetchAPI("/dashboard/generalization-gap"),
      fetchAPI("/dashboard/snr"),
      fetchAPI("/dashboard/recent-updates")
    ]);

    const [summary, performance, attacks, gap, snr, updates] = results;

    if (summary.status === "fulfilled") {
      state.summary = summary.value;
      renderSummary(state.summary);
      if (state.summary.user) {
        els.userName.textContent = state.summary.user.name || "User";
        els.userRole.textContent = state.summary.user.role || "AMCShield Console";
      }
    }

    if (performance.status === "fulfilled") {
      state.performance = performance.value;
      renderPerformance(state.performance);
    } else {
      els.performanceLoading.textContent = "Performance data unavailable.";
    }

    if (attacks.status === "fulfilled") {
      state.attacks = attacks.value;
      renderAttackChart(state.attacks);
    } else {
      els.attackLoading.textContent = "Attack data unavailable.";
    }

    if (gap.status === "fulfilled") {
      state.gap = gap.value;
      renderHeatmap(state.gap);
      renderComparison(state.gap);
    } else {
      renderHeatmap(null);
      renderComparison(null);
    }

    if (snr.status === "fulfilled") {
      state.snr = snr.value;
      populateSNR(state.snr);
    }

    if (updates.status === "fulfilled") {
      state.updates = updates.value;
      renderUpdates(state.updates);
    } else {
      renderUpdates(null);
    }

    const anySuccess = results.some(r => r.status === "fulfilled");
    setAPIStatus(anySuccess, anySuccess ? "Backend data loaded" : "No endpoint responded");
    if (!anySuccess) {
      showToast("AMCShield backend is unavailable. No dummy data has been shown.");
    } else {
      setLastUpdated();
    }
  } catch (error) {
    console.error("Dashboard load failed:", error);
    setAPIStatus(false, "Connection error");
    showToast("Unable to load AMCShield backend data.");
  } finally {
    setLoading(false);
  }
}

els.refresh.addEventListener("click", loadDashboard);

els.attackSelect.addEventListener("change", () => {
  if (state.attacks) renderAttackChart(state.attacks);
});

els.snrSelect.addEventListener("change", () => {
  renderSNRDetail(els.snrSelect.value);
});

els.menu.addEventListener("click", () => {
  els.sidebar.classList.add("open");
  els.overlay.classList.add("show");
});

els.overlay.addEventListener("click", () => {
  els.sidebar.classList.remove("open");
  els.overlay.classList.remove("show");
});

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", () => {
    if (window.innerWidth <= 1000) {
      els.sidebar.classList.remove("open");
      els.overlay.classList.remove("show");
    }
  });
});

$("logoutButton").addEventListener("click", async () => {
  try {
    await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
  } finally {
    window.location.href = "/";
  }
});

async function requireAuth() {
  try {
    const response = await fetch("/api/auth/me", {
      headers: { "Accept": "application/json" },
      credentials: "include"
    });
    if (!response.ok) {
      window.location.href = "/";
      return false;
    }
    const body = await response.json();
    const user = body.user || {};
    els.userName.textContent = user.name || user.login || "User";
    els.userRole.textContent = user.role || "AMCShield Console";
    return true;
  } catch (error) {
    console.error("Authentication check failed:", error);
    window.location.href = "/";
    return false;
  }
}

window.addEventListener("resize", () => {
  if (state.performance) renderPerformance(state.performance);
  if (state.attacks) renderAttackChart(state.attacks);
});

requireAuth().then((authenticated) => {
  if (authenticated) loadDashboard();
});
