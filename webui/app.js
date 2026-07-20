/* DoE web UI — talks to doe_server.py, renders with Plotly. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const S = {
  state: null,          // /api/state payload
  model: null,          // /api/model payload
  landscape: null,      // /api/landscape payload (fetched when the tab opens)
  landscapeGen: 0,      // bumped on invalidation, discards in-flight fetches
  landscapeLoading: false,
  landscapeResyncGen: -1, // gen for which the staleness guard already resynced
  tab: "progress",
  respSel: 0,           // response shown in the Model/Landscape tabs
  nSuggest: 1,
  wizardBuilt: false,
};

const COLORS = ["#5ac8fa", "#ffd166", "#ff7a90", "#7bd88f", "#c792ea", "#f2a65a"];
const AXG = {
  gridcolor: "rgba(255,255,255,.07)",
  zerolinecolor: "rgba(255,255,255,.14)",
  linecolor: "rgba(255,255,255,.15)",
};
const PLOTLY_CFG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
};

/* ------------------------------------------------------------- helpers --- */

const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

function fmt(v, digits = 4) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return Number(v).toExponential(2);
  return String(parseFloat(Number(v).toPrecision(digits)));
}
const money = (v) => Number(v).toFixed(2);

function parseNum(s) {
  const v = parseFloat(String(s).trim().replace(",", "."));
  return isFinite(v) ? v : null;
}

async function api(path, opts = {}) {
  const res = await fetch(path, {headers: {"Content-Type": "application/json"}, ...opts});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

let toastTimer;
function toast(msg, isErr = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.className = "show" + (isErr ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = ""; }, 4200);
}

let busyCount = 0;
function busy(msg) {
  busyCount++;
  document.body.classList.add("busy");
  const s = $("#status");
  s.textContent = msg;
  s.classList.remove("hidden");
}
function idle() {
  if (--busyCount <= 0) {
    busyCount = 0;
    document.body.classList.remove("busy");
    $("#status").classList.add("hidden");
  }
}

function ensurePlotly() {
  if (window.Plotly) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    s.onload = resolve;
    s.onerror = () => reject(new Error("Plotly failed to load (no local copy, no CDN)"));
    document.head.appendChild(s);
  });
}

function isPriced() {
  const c = S.state && S.state.costs;
  return !!c && (c.fixed_cost > 0 || c.prices.some((p) => p > 0));
}
const currency = () => (S.state.costs ? S.state.costs.currency : "");

function propCost(x) {
  const c = S.state.costs;
  return x.reduce((sum, v, j) => sum + v * c.prices[j], c.fixed_cost);
}

function baseLayout(extra) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#c7d1e0", family: "ui-sans-serif, system-ui, sans-serif", size: 12},
    margin: {l: 60, r: 20, t: 28, b: 44},
    colorway: COLORS,
    hovermode: "closest",
  }, extra);
}

function plot(el, traces, layout) {
  if (!window.Plotly) return;
  if (!el._fullLayout) el.innerHTML = "";
  Plotly.react(el, traces, layout, PLOTLY_CFG);
}
function ph(el, text) {
  if (window.Plotly && el._fullLayout) Plotly.purge(el);
  el.innerHTML = text ? `<div class="placeholder">${esc(text)}</div>` : "";
}
const range1 = (n) => Array.from({length: n}, (_, i) => i + 1);

/* -------------------------------------------------------------- fetch --- */

async function refreshState() {
  S.state = await api("/api/state");
  render();
}

// The landscape is expensive to compute, so it is cached until the data
// changes. Call this BEFORE refreshState() at every mutation site, so the
// stale payload can never render against the new state.
function invalidateLandscape() {
  S.landscape = null;
  S.landscapeGen++;     // drops any in-flight fetch
}

async function refreshModel() {
  const st = S.state;
  if (!st || !st.configured || st.num_results < 2) {
    S.model = null;
    renderBest();
    renderPlotsFor(S.tab);
    return;
  }
  busy("fitting GP model…");
  try {
    S.model = await api("/api/model");
  } catch (e) {
    S.model = {available: false, reason: e.message};
  } finally {
    idle();
  }
  renderBest();
  renderPlotsFor(S.tab);
}

/* -------------------------------------------------------------- render --- */

function render() {
  const st = S.state;
  $("#setup-view").classList.toggle("hidden", st.configured);
  $("#main-view").classList.toggle("hidden", !st.configured);
  renderChrome();
  if (!st.configured) { buildWizardOnce(); return; }

  $("#tab-btn-costs").classList.toggle("hidden", !isPriced());
  if (S.tab === "costs" && !isPriced()) switchTab("progress");
  renderFitControl();
  const hasPairs = st.factor_names.length >= 2;
  $("#tab-btn-landscape").classList.toggle("hidden", !hasPairs);
  if (S.tab === "landscape" && !hasPairs) switchTab("progress");
  if (S.respSel >= st.response_names.length) S.respSel = 0;

  renderProposals();
  renderManual();
  renderBest();
  renderHistory();
  renderPlotsFor(S.tab);
}

function renderChrome() {
  const st = S.state;
  const btn = $("#session-name");
  btn.textContent = "📁 " + st.session.replace(/^.*\//, "");
  btn.title = (st.description ? st.description + " — " : "") + "manage sessions";
  const show = st.configured && isPriced();
  $("#spend-badge").classList.toggle("hidden", !show);
  if (show) {
    $("#spend-badge").textContent =
      `${st.num_results} runs · spent ${money(st.spend)} ${currency()}`;
  }
}

/* ---- posterior fit choice (GP vs. n-degree polynomial) ---- */

function renderFitControl() {
  const st = S.state;
  const sel = $("#fit-kernel");
  const deg = $("#fit-degree");
  // "|| default": a not-yet-restarted server doesn't send the fields
  const kernel = st.kernel || "rbf";
  // don't clobber a value the user is editing right now
  if (document.activeElement !== sel) sel.value = kernel;
  if (document.activeElement !== deg) deg.value = st.poly_degree || 2;
  deg.classList.toggle("hidden", kernel !== "poly");
}

async function applyFit() {
  const kernel = $("#fit-kernel").value;
  const raw = parseNum($("#fit-degree").value);
  const degree = Math.max(1, Math.min(10, Math.round(raw ?? S.state.poly_degree)));
  busy("refitting…");
  try {
    await api("/api/fit", {method: "POST",
                           body: JSON.stringify({kernel, degree})});
    invalidateLandscape();
    S.model = null;         // the old fit's slices must not render
    await refreshState();
    refreshModel();
    toast(kernel === "poly"
      ? `Fitting a degree-${degree} polynomial`
      : "Fitting the flexible GP (RBF)");
  } catch (e) {
    toast(e.message, true);
    renderFitControl();     // snap the control back to the server's truth
  } finally {
    idle();
  }
}

/* ---- proposals & entry ---- */

function renderProposals() {
  const st = S.state;
  const box = $("#proposals");
  $("#suggest-mode").textContent =
    st.num_results < st.num_init
      ? `Space-filling design — the GP takes over after ${st.num_init} results ` +
        `(${st.num_results} so far).`
      : "GP-guided: expected improvement" +
        (st.cost_aware && isPriced() ? " per cost." : ".");
  box.innerHTML = "";
  if (!st.pending.length) {
    box.innerHTML = `<p class="hint">No open proposals — click “Suggest”.</p>`;
    return;
  }
  st.pending.forEach((x, i) => {
    const row = document.createElement("div");
    row.className = "proposal";
    const chips = x.map((v, j) =>
      `<span class="chip"><b>${esc(st.factor_names[j])}</b>${fmt(v)}</span>`).join("");
    const cost = isPriced()
      ? `<span class="chip cost">~${money(propCost(x))} ${esc(currency())}</span>` : "";
    const inputs = st.response_names.map((nm, t) =>
      `<label class="mini">${esc(nm)}<input type="text" inputmode="decimal"
        class="resp-input" data-t="${t}" placeholder="measured"></label>`).join("");
    row.innerHTML = `<div class="chips">${chips}${cost}</div>
      <div class="entry">${inputs}<button class="primary save">Save result</button></div>`;
    row.querySelector(".save").onclick = () => saveProposal(i, x, row);
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter") saveProposal(i, x, row);
    });
    box.appendChild(row);
  });
}

async function saveProposal(i, x, row) {
  const y = $$(".resp-input", row).map((inp) => parseNum(inp.value));
  if (y.some((v) => v === null)) {
    toast("Enter all measured responses first", true);
    return;
  }
  busy("updating model…");
  try {
    await api("/api/result", {
      method: "POST",
      body: JSON.stringify({x, y, pending_index: i}),
    });
    invalidateLandscape();
    await refreshState();
    toast("Result saved — model updated");
    refreshModel();
  } catch (e) {
    toast(e.message, true);
  } finally {
    idle();
  }
}

function renderManual() {
  const st = S.state;
  const box = $("#manual-body");
  const fIn = st.factor_names.map((nm, j) =>
    `<label class="mini">${esc(nm)}<input class="man-x" inputmode="decimal"
      placeholder="${fmt(st.bounds[j][0])} – ${fmt(st.bounds[j][1])}"></label>`).join("");
  const rIn = st.response_names.map((nm) =>
    `<label class="mini">${esc(nm)}<input class="man-y" inputmode="decimal"></label>`).join("");
  box.innerHTML = `<p class="hint">Concentrations:</p><div class="entry wrap">${fIn}</div>
    <p class="hint">Measured responses:</p>
    <div class="entry wrap">${rIn}<button class="primary" id="manual-add">Add result</button></div>`;
  $("#manual-add").onclick = async () => {
    const x = $$(".man-x", box).map((i) => parseNum(i.value));
    const y = $$(".man-y", box).map((i) => parseNum(i.value));
    if (x.some((v) => v === null) || y.some((v) => v === null)) {
      toast("Fill in all concentrations and responses", true);
      return;
    }
    busy("updating model…");
    try {
      await api("/api/result", {method: "POST", body: JSON.stringify({x, y})});
      invalidateLandscape();
      await refreshState();
      toast("Result saved — model updated");
      refreshModel();
    } catch (e) {
      toast(e.message, true);
    } finally {
      idle();
    }
  };
}

/* ---- best card ---- */

function kv(names, vals, isResp = false, prefix = "") {
  const st = S.state;
  return `<div class="chips">` + names.map((nm, i) =>
    `<span class="chip${isResp ? " resp" : ""}${isResp && i === st.target_task ? " target" : ""}">
      <b>${esc(nm)}</b>${prefix}${fmt(vals[i])}</span>`).join("") + `</div>`;
}

function renderBest() {
  const st = S.state;
  if (!st.configured) return;
  const el = $("#best-body");
  if (!st.best) {
    el.innerHTML = `<p class="hint">No results yet.</p>`;
    return;
  }
  const b = st.best;
  const cur = esc(currency());
  const targetName = esc(st.response_names[st.target_task]);
  let html = `<h4>Best measured <span class="tag">#${b.index + 1}</span></h4>`
    + kv(st.factor_names, b.x) + kv(st.response_names, b.y, true);
  if (isPriced() && b.cost !== undefined) {
    html += `<p class="cost-line">cost ${money(b.cost)} ${cur}`
      + (b.cost_per_yield !== undefined
         ? ` · yield price ${fmt(b.cost_per_yield)} ${cur} / ${targetName}` : "")
      + `</p>`;
  }
  const p = S.model && S.model.available ? S.model.predicted_best : null;
  if (p) {
    html += `<h4>Predicted optimum <span class="tag">model</span></h4>`
      + kv(st.factor_names, p.x) + kv(st.response_names, p.y, true, "~");
    if (isPriced() && p.cost_per_yield !== undefined) {
      html += `<p class="cost-line">predicted yield price ${fmt(p.cost_per_yield)} `
        + `${cur} / ${targetName}</p>`;
    }
  }
  el.innerHTML = html;
}

/* ---- history table ---- */

function renderHistory() {
  const st = S.state;
  const tbl = $("#history");
  $("#history-empty").classList.toggle("hidden", st.num_results > 0);
  if (!st.num_results) { tbl.innerHTML = ""; return; }
  const priced = isPriced();
  const head = `<tr><th>#</th>`
    + st.factor_names.map((n) => `<th>${esc(n)}</th>`).join("")
    + st.response_names.map((n) => `<th class="resp">${esc(n)}</th>`).join("")
    + (priced ? `<th>cost [${esc(currency())}]</th>` : "")
    + `<th></th></tr>`;
  const bestIdx = st.best ? st.best.index : -1;
  const rows = st.results_x.map((x, i) => {
    const cells = x.map((v) => `<td>${fmt(v)}</td>`).join("")
      + st.results_y[i].map((v, k) =>
          `<td class="resp${k === st.target_task ? " target" : ""}">${fmt(v)}</td>`).join("")
      + (priced ? `<td>${money(st.result_costs[i])}</td>` : "");
    return `<tr class="${i === bestIdx ? "best" : ""}">
      <td>${i + 1}${i === bestIdx ? " ★" : ""}</td>${cells}
      <td><button class="del" data-i="${i}" title="delete this result">×</button></td></tr>`;
  }).reverse().join("");
  tbl.innerHTML = head + rows;
  $$(".del", tbl).forEach((btn) => {
    btn.onclick = async () => {
      const i = +btn.dataset.i;
      if (!confirm(`Delete experiment #${i + 1}? The model will refit without it.`)) return;
      busy("deleting…");
      try {
        await api(`/api/result/${i}`, {method: "DELETE"});
        invalidateLandscape();
        await refreshState();
        refreshModel();
      } catch (e) {
        toast(e.message, true);
      } finally {
        idle();
      }
    };
  });
}

/* ----------------------------------------------------------------- plots --- */

function switchTab(tab) {
  S.tab = tab;
  // a cached transport failure (server restart etc.) retries on re-entry
  if (tab === "landscape" && S.landscape && !S.landscape.available &&
      S.landscape.transient) S.landscape = null;
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  $$(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + tab));
  renderPlotsFor(tab);
}

function renderPlotsFor(tab) {
  if (!S.state || !S.state.configured || !window.Plotly) return;
  if (tab === "progress") renderProgress();
  else if (tab === "model") renderModelTab();
  else if (tab === "landscape") renderLandscape();
  else if (tab === "insights") renderInsights();
  else if (tab === "costs") renderCosts();
}

function renderProgress() {
  const st = S.state;
  const el = $("#plot-progress");
  if (!st.num_results) {
    ph(el, "No results yet — suggest experiments to get started.");
    return;
  }
  const n = st.num_results;
  const xs = range1(n);
  const T = st.response_names.length;
  const traces = [];
  const layout = baseLayout({
    showlegend: false,
    height: Math.max(300, 205 * T),
    grid: {rows: T, columns: 1, pattern: "independent"},
    shapes: [],
  });
  st.response_names.forEach((name, t) => {
    const ax = t === 0 ? "" : String(t + 1);
    traces.push({
      x: xs, y: st.results_y.map((r) => r[t]),
      mode: "lines+markers", name,
      line: {color: COLORS[t % COLORS.length], width: 2},
      marker: {size: 6},
      xaxis: "x" + ax, yaxis: "y" + ax,
    });
    if (t === st.target_task) {
      const best = [];
      let b = st.maximize ? -Infinity : Infinity;
      st.results_y.forEach((r) => {
        b = st.maximize ? Math.max(b, r[t]) : Math.min(b, r[t]);
        best.push(b);
      });
      traces.push({
        x: xs, y: best, mode: "lines", name: "best so far",
        line: {color: "#7bd88f", dash: "dot", shape: "hv", width: 1.5},
        xaxis: "x" + ax, yaxis: "y" + ax,
      });
      if (st.best) {
        traces.push({
          x: [st.best.index + 1], y: [st.results_y[st.best.index][t]],
          mode: "markers", name: "best",
          marker: {symbol: "star", size: 13, color: "#7bd88f"},
          xaxis: "x" + ax, yaxis: "y" + ax,
        });
      }
    }
    layout["xaxis" + ax] = {...AXG, dtick: n > 15 ? undefined : 1,
                            title: t === T - 1 ? {text: "experiment #"} : undefined};
    layout["yaxis" + ax] = {...AXG, title: {text: name, font: {size: 11}}};
    // shade the initial space-filling phase
    if (st.num_init && n >= 1) {
      layout.shapes.push({
        type: "rect", xref: "x" + ax, yref: (ax ? "y" + ax : "y") + " domain",
        x0: 0.5, x1: Math.min(st.num_init, n) + 0.5, y0: 0, y1: 1,
        fillcolor: "rgba(255,255,255,.045)", line: {width: 0},
      });
    }
  });
  plot(el, traces, layout);
}

function renderModelTab() {
  const st = S.state;
  renderRespPicker($("#resp-picker"), renderModelTab);
  const el = $("#plot-slices");
  const m = S.model;
  if (!m || !m.available) {
    ph(el, (m && m.reason) || "Add at least 2 results to see the model.");
    return;
  }
  const t = S.respSel;
  const F = st.factor_names.length;
  const cols = F <= 2 ? F : 2;
  const rows = Math.ceil(F / cols);
  const traces = [];
  const layout = baseLayout({
    showlegend: false,
    height: 270 * rows + 40,
    grid: {rows, columns: cols, pattern: "independent"},
    shapes: [],
  });
  m.slices.forEach((sl, j) => {
    const ax = j === 0 ? "" : String(j + 1);
    const xa = "x" + ax, ya = "y" + ax;
    traces.push({x: sl.grid, y: sl.upper[t], mode: "lines",
                 line: {width: 0}, hoverinfo: "skip",
                 xaxis: xa, yaxis: ya});
    traces.push({x: sl.grid, y: sl.lower[t], mode: "lines",
                 line: {width: 0}, fill: "tonexty",
                 fillcolor: "rgba(90,200,250,.16)", hoverinfo: "skip",
                 xaxis: xa, yaxis: ya});
    traces.push({x: sl.grid, y: sl.mean[t], mode: "lines", name: "GP mean",
                 line: {color: "#5ac8fa", width: 2.2}, xaxis: xa, yaxis: ya});
    traces.push({x: st.results_x.map((r) => r[j]),
                 y: st.results_y.map((r) => r[t]),
                 mode: "markers", name: "observed",
                 marker: {color: "#e6ecf5", size: 6, opacity: .85},
                 xaxis: xa, yaxis: ya});
    if (m.predicted_best) {
      traces.push({x: [m.predicted_best.x[j]], y: [m.predicted_best.y[t]],
                   mode: "markers", name: "predicted optimum",
                   marker: {symbol: "diamond", size: 11, color: "#ffd166"},
                   xaxis: xa, yaxis: ya});
    }
    layout.shapes.push({
      type: "line", xref: xa, yref: (ax ? "y" + ax : "y") + " domain",
      x0: m.reference[j], x1: m.reference[j], y0: 0, y1: 1,
      line: {color: "rgba(255,255,255,.3)", dash: "dot", width: 1},
    });
    layout["xaxis" + ax] = {...AXG, title: {text: st.factor_names[j]}};
    layout["yaxis" + ax] = {...AXG};
  });
  plot(el, traces, layout);
}

function renderRespPicker(el, onpick) {
  const st = S.state;
  el.innerHTML = st.response_names.map((nm, t) =>
    `<button class="${t === S.respSel ? "active" : ""}" data-t="${t}">${esc(nm)}</button>`
  ).join("");
  $$("button", el).forEach((b) => {
    b.onclick = () => { S.respSel = +b.dataset.t; onpick(); };
  });
}

/* ---- landscape: corner plot over every factor pair ---- */

async function fetchLandscape() {
  if (S.landscapeLoading) return;
  S.landscapeLoading = true;
  const gen = S.landscapeGen;
  busy("computing landscape…");
  let data;
  try {
    data = await api("/api/landscape");
  } catch (e) {
    data = {available: false, reason: e.message, transient: true};
  } finally {
    S.landscapeLoading = false;
    idle();
  }
  // if the data changed while this was in flight, drop it and refetch
  S.landscape = gen === S.landscapeGen ? data : null;
  if (S.tab === "landscape" && S.state && S.state.configured) renderLandscape();
}

function renderLandscape() {
  const st = S.state;
  renderRespPicker($("#resp-picker-landscape"), renderLandscape);
  const el = $("#plot-landscape");
  let L = S.landscape;
  if (L && L.available && (L.session !== st.session ||
      L.n !== st.num_results ||
      L.kernel !== st.kernel ||
      L.poly_degree !== st.poly_degree ||
      L.reference.length !== st.factor_names.length ||
      L.pairs[0].mean.length !== st.response_names.length)) {
    // The payload belongs to another session or data version -- the server
    // is shared, so a second window can switch it under us. Refetching alone
    // would loop (the server keeps answering for *its* session); resync the
    // state instead and let render() fetch again once both sides agree.
    // At most one automatic resync per generation, in case they never do.
    const resynced = S.landscapeResyncGen === S.landscapeGen;
    invalidateLandscape();
    if (resynced) {
      ph(el, "Out of sync with the server — reload the page.");
      return;
    }
    S.landscapeResyncGen = S.landscapeGen;
    ph(el, "Session changed — reloading…");
    refreshState().then(refreshModel).catch((e) => toast(e.message, true));
    return;
  }
  if (!L) {
    ph(el, "Computing landscape…");
    fetchLandscape();
    return;
  }
  if (!L.available) {
    ph(el, L.reason);
    return;
  }
  const t = S.respSel;
  const F = st.factor_names.length;
  const respName = st.response_names[t];
  const ty = st.results_y.map((r) => r[t]);
  const runs = range1(st.num_results);

  // One color scale shared by the predicted maps and the measured dots, so
  // a dot that clashes with the map around it disagrees with the model.
  let zmin = Infinity, zmax = -Infinity;
  const take = (v) => {
    if (v < zmin) zmin = v;
    if (v > zmax) zmax = v;
  };
  L.pairs.forEach((p) => p.mean[t].forEach((row) => row.forEach(take)));
  ty.forEach(take);

  // shared value range for the diagonal slice cells (only (0,0) shows ticks)
  // -- the model must belong to this session's factors/responses
  const m = S.model && S.model.available &&
    S.model.reference.length === F &&
    S.model.slices[0].mean.length === st.response_names.length
    ? S.model : null;
  let smin = Infinity, smax = -Infinity;
  if (m) {
    m.slices.forEach((sl) => {
      sl.lower[t].forEach((v) => { if (v < smin) smin = v; });
      sl.upper[t].forEach((v) => { if (v > smax) smax = v; });
    });
    ty.forEach((v) => {
      if (v < smin) smin = v;
      if (v > smax) smax = v;
    });
  }

  const traces = [];
  // cells stay roughly square: derive the row height from the pane width
  const cellH = Math.max(110, Math.min(235, (el.clientWidth || 940) / F));
  const layout = baseLayout({
    showlegend: false,
    height: Math.max(440, Math.round(cellH * F) + 70),
    grid: {rows: F, columns: F, pattern: "independent"},
    margin: {l: 62, r: 14, t: 16, b: 46},
  });
  if (!m) {
    // slices come from /api/model -- say why the diagonal is empty
    layout.annotations = [{
      x: 0.5 / F, y: 1 - 0.5 / F, xref: "paper", yref: "paper",
      text: esc((S.model && S.model.reason) || "1D slices load with the model…"),
      showarrow: false, font: {size: 10, color: "#8b94a8"},
    }];
  }
  const suf = (r, c) => (r * F + c ? String(r * F + c + 1) : "");
  const pad = (lo, hi) => {
    const w = (hi - lo) * 0.03;
    return [lo - w, hi + w];
  };
  for (let r = 0; r < F; r++) {
    for (let c = 0; c < F; c++) {
      const ax = suf(r, c);
      layout["xaxis" + ax] = {
        ...AXG, range: pad(...st.bounds[c]), tickfont: {size: 9},
        showticklabels: r === F - 1,
        title: r === F - 1 ? {text: st.factor_names[c], font: {size: 11}} : undefined,
      };
      layout["yaxis" + ax] = r === c
        ? {...AXG, tickfont: {size: 9}, showticklabels: !!m && c === 0,
           range: m && isFinite(smin) && smin < smax ? pad(smin, smax) : [0, 1]}
        : {...AXG, range: pad(...st.bounds[r]), tickfont: {size: 9},
           showticklabels: c === 0,
           title: c === 0 ? {text: st.factor_names[r], font: {size: 11}} : undefined};
    }
  }

  // diagonal: the 1D slices (band + mean + observed)
  if (m) {
    m.slices.forEach((sl, d) => {
      const ax = suf(d, d), xa = "x" + ax, ya = "y" + ax;
      traces.push({x: sl.grid, y: sl.upper[t], mode: "lines", line: {width: 0},
                   hoverinfo: "skip", xaxis: xa, yaxis: ya});
      traces.push({x: sl.grid, y: sl.lower[t], mode: "lines", line: {width: 0},
                   fill: "tonexty", fillcolor: "rgba(90,200,250,.16)",
                   hoverinfo: "skip", xaxis: xa, yaxis: ya});
      traces.push({x: sl.grid, y: sl.mean[t], mode: "lines",
                   line: {color: "#5ac8fa", width: 2},
                   hovertemplate: `${esc(st.factor_names[d])}: %{x:.4g}<br>` +
                     `predicted ${esc(respName)}: %{y:.4g}<extra></extra>`,
                   xaxis: xa, yaxis: ya});
      traces.push({x: st.results_x.map((r) => r[d]), y: ty, mode: "markers",
                   marker: {color: "#e6ecf5", size: 4, opacity: .8},
                   hoverinfo: "skip", xaxis: xa, yaxis: ya});
    });
  }

  L.pairs.forEach((p, idx) => {
    const fi = st.factor_names[p.i], fj = st.factor_names[p.j];
    const xi = st.results_x.map((r) => r[p.i]);
    const xj = st.results_x.map((r) => r[p.j]);

    // lower triangle (row j, col i): predicted response surface
    let ax = suf(p.j, p.i), xa = "x" + ax, ya = "y" + ax;
    traces.push({
      type: "heatmap", x: p.gi, y: p.gj, z: p.mean[t], customdata: p.std[t],
      zmin, zmax, colorscale: "Viridis", zsmooth: "best",
      showscale: idx === 0,
      colorbar: {title: {text: respName}, thickness: 12, outlinewidth: 0},
      hovertemplate: `${esc(fi)}: %{x:.4g}<br>${esc(fj)}: %{y:.4g}<br>` +
        `predicted ${esc(respName)}: %{z:.4g} ± %{customdata:.2g}<extra></extra>`,
      xaxis: xa, yaxis: ya,
    });
    traces.push({x: xi, y: xj, mode: "markers", hoverinfo: "skip",
                 marker: {color: "rgba(255,255,255,.85)", size: 4},
                 xaxis: xa, yaxis: ya});
    if (st.best) {
      traces.push({x: [st.best.x[p.i]], y: [st.best.x[p.j]], mode: "markers",
                   marker: {symbol: "star", size: 12, color: "#7bd88f",
                            line: {color: "#0b101b", width: 1}},
                   hovertemplate: "best measured<extra></extra>",
                   xaxis: xa, yaxis: ya});
    }
    if (L.predicted_best) {
      traces.push({x: [L.predicted_best.x[p.i]], y: [L.predicted_best.x[p.j]],
                   mode: "markers",
                   marker: {symbol: "diamond", size: 10, color: "#ffd166",
                            line: {color: "#0b101b", width: 1}},
                   hovertemplate: "predicted optimum<extra></extra>",
                   xaxis: xa, yaxis: ya});
    }
    if (st.pending.length) {
      traces.push({x: st.pending.map((x) => x[p.i]),
                   y: st.pending.map((x) => x[p.j]), mode: "markers",
                   marker: {symbol: "x-thin", size: 9,
                            line: {color: "#ff7a90", width: 2}},
                   hovertemplate: "open proposal<extra></extra>",
                   xaxis: xa, yaxis: ya});
    }

    // upper triangle (row i, col j): experiments colored by measurement
    ax = suf(p.i, p.j); xa = "x" + ax; ya = "y" + ax;
    traces.push({
      x: xj, y: xi, mode: "markers", customdata: runs,
      marker: {color: ty, cmin: zmin, cmax: zmax, colorscale: "Viridis",
               size: 9, line: {color: "rgba(255,255,255,.55)", width: 1}},
      hovertemplate: `#%{customdata}<br>${esc(fj)}: %{x:.4g}<br>` +
        `${esc(fi)}: %{y:.4g}<br>${esc(respName)}: %{marker.color:.4g}<extra></extra>`,
      xaxis: xa, yaxis: ya,
    });
    if (st.best) {
      traces.push({x: [st.best.x[p.j]], y: [st.best.x[p.i]], mode: "markers",
                   marker: {symbol: "star", size: 13, color: "#7bd88f",
                            line: {color: "#0b101b", width: 1}},
                   hovertemplate: "best measured<extra></extra>",
                   xaxis: xa, yaxis: ya});
    }
    if (st.pending.length) {
      traces.push({x: st.pending.map((x) => x[p.j]),
                   y: st.pending.map((x) => x[p.i]), mode: "markers",
                   marker: {symbol: "x-thin", size: 9,
                            line: {color: "#ff7a90", width: 2}},
                   hovertemplate: "open proposal<extra></extra>",
                   xaxis: xa, yaxis: ya});
    }
  });
  plot(el, traces, layout);
}

function renderInsights() {
  const st = S.state;
  const el1 = $("#plot-importance");
  const el2 = $("#plot-taskcorr");
  const m = S.model;
  if (!m || !m.available) {
    const why = (m && m.reason) || "Add at least 2 results to see model insights.";
    ph(el1, why);
    ph(el2, "");
    return;
  }
  if (!m.importance) {
    ph(el1, "Factor influence is read from the GP's learned lengthscales — " +
            "the polynomial fit has none. Switch the fit to GP (RBF) to see it.");
  } else {
    plot(el1, [{
      x: m.importance, y: st.factor_names, type: "bar", orientation: "h",
      marker: {color: "#5ac8fa"},
      hovertemplate: "%{y}: %{x:.2f}<extra></extra>",
    }], baseLayout({
      height: Math.max(200, 62 * st.factor_names.length + 110),
      xaxis: {...AXG, range: [0, 1.06], title: {text: "relative influence (1/lengthscale)"}},
      yaxis: {...AXG, autorange: "reversed"},
    }));
  }

  const names = st.response_names;
  const corr = m.task_correlation;
  const annotations = [];
  corr.forEach((row, i) => row.forEach((v, j) => annotations.push({
    x: names[j], y: names[i], text: v.toFixed(2), showarrow: false,
    font: {color: Math.abs(v) > 0.5 ? "#fff" : "#223", size: 12},
  })));
  plot(el2, [{
    z: corr, x: names, y: names, type: "heatmap",
    zmin: -1, zmax: 1, colorscale: "RdBu", reversescale: true,
    showscale: true, colorbar: {thickness: 12, outlinewidth: 0},
    hovertemplate: "%{y} × %{x}: %{z:.2f}<extra></extra>",
  }], baseLayout({
    height: Math.max(200, 62 * names.length + 110),
    annotations,
    xaxis: {...AXG},
    yaxis: {...AXG, autorange: "reversed"},
  }));
}

function renderCosts() {
  const st = S.state;
  const el1 = $("#plot-costs");
  const el2 = $("#plot-costyield");
  if (!st.num_results) {
    ph(el1, "No experiments yet.");
    ph(el2, "");
    return;
  }
  const n = st.num_results;
  const xs = range1(n);
  const cum = [];
  st.result_costs.reduce((acc, c, i) => (cum[i] = acc + c, cum[i]), 0);
  const traces = [
    {x: xs, y: st.result_costs, type: "bar", name: "per experiment",
     marker: {color: "rgba(90,200,250,.45)"}},
    {x: xs, y: cum, mode: "lines+markers", name: "cumulative", yaxis: "y2",
     line: {color: "#ffd166"}},
  ];
  const pj = st.projected;
  if (pj && pj["5"] !== undefined) {
    traces.push({
      x: [n, n + 5, n + 10, n + 20],
      y: [cum[n - 1], pj["5"], pj["10"], pj["20"]],
      mode: "lines+markers", name: "projected", yaxis: "y2",
      line: {color: "#ffd166", dash: "dot"},
      marker: {symbol: "diamond-open", size: 8},
    });
  }
  plot(el1, traces, baseLayout({
    height: 330,
    legend: {orientation: "h", y: 1.14, x: 0, bgcolor: "rgba(0,0,0,0)"},
    xaxis: {...AXG, title: {text: "experiment #"}, dtick: n > 15 ? undefined : 1},
    yaxis: {...AXG, title: {text: `cost per run [${currency()}]`}},
    yaxis2: {overlaying: "y", side: "right", gridcolor: "rgba(0,0,0,0)",
             title: {text: `cumulative [${currency()}]`}},
  }));

  const ty = st.results_y.map((r) => r[st.target_task]);
  const t2 = [{
    x: st.result_costs, y: ty, mode: "markers", type: "scatter", name: "",
    marker: {size: 9, color: xs, colorscale: "Viridis", showscale: true,
             colorbar: {title: {text: "run #"}, thickness: 12, outlinewidth: 0}},
    hovertemplate: "experiment %{marker.color}<br>cost %{x:.2f}<br>%{y:.4g}<extra></extra>",
  }];
  if (st.best) {
    t2.push({x: [st.result_costs[st.best.index]], y: [ty[st.best.index]],
             mode: "markers", name: "best",
             marker: {symbol: "star", size: 15, color: "#7bd88f"}});
  }
  plot(el2, t2, baseLayout({
    height: 330, showlegend: false,
    xaxis: {...AXG, title: {text: `experiment cost [${currency()}]`}},
    yaxis: {...AXG, title: {text: st.response_names[st.target_task]}},
  }));
}

/* -------------------------------------------------------- session manager --- */

function openSessions() {
  $("#sessions-modal").classList.remove("hidden");
  refreshSessions();
}
function closeSessions() {
  $("#sessions-modal").classList.add("hidden");
}

async function refreshSessions() {
  try {
    renderSessions(await api("/api/sessions"));
  } catch (e) {
    toast(e.message, true);
  }
}

function fmtWhen(mtime) {
  if (!mtime) return "–";
  const d = new Date(mtime * 1000);
  return d.toLocaleDateString() + " " +
         d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
}

function renderSessions(data) {
  const tbl = $("#sessions-table");
  const list = data.sessions || [];
  $("#sessions-empty").classList.toggle("hidden", list.length > 0);
  if (!list.length) { tbl.innerHTML = ""; return; }
  const head = `<tr><th>session</th><th>description</th><th>results</th>
    <th>last change</th><th></th></tr>`;
  tbl.innerHTML = head + list.map((s, i) => {
    const what = s.configured
      ? `${s.factor_names.join(", ")} → ${s.response_names.join(", ")}`
      : "not configured yet — loading it opens the setup wizard";
    return `<tr class="${s.active ? "best" : ""}">
      <td title="${esc(what)}">${esc(s.name)}${s.active
        ? ` <span class="tag">active</span>` : ""}</td>
      <td class="desc">${esc(s.description) || `<span class="muted">—</span>`}
        <button class="mini-act" data-act="describe" data-i="${i}"
          title="edit description">✎</button></td>
      <td>${s.configured ? s.num_results : "—"}</td>
      <td>${fmtWhen(s.modified)}</td>
      <td class="acts">
        <button class="mini-act" data-act="load" data-i="${i}"
          ${s.active ? "disabled" : ""}>load</button>
        <button class="mini-act" data-act="rename" data-i="${i}">rename</button>
        <button class="mini-act danger" data-act="delete" data-i="${i}"
          ${s.active ? 'disabled title="the active session cannot be deleted"'
                     : 'title="delete the session file"'}>delete</button>
      </td></tr>`;
  }).join("");
  $$("button[data-act]", tbl).forEach((b) => {
    b.onclick = () => sessionAction(b.dataset.act, list[+b.dataset.i]);
  });
}

async function sessionAction(act, s) {
  try {
    if (act === "load") {
      busy("loading session…");
      try {
        await api("/api/sessions/load",
                  {method: "POST", body: JSON.stringify({name: s.name})});
        closeSessions();
        invalidateLandscape();
        S.model = null;     // the old session's model must not render either
        await refreshState();
        refreshModel();
        toast(`Loaded ${s.name}`);
      } finally {
        idle();
      }
    } else if (act === "rename") {
      const nn = prompt(`Rename ${s.name} to:`, s.name.replace(/\.json$/, ""));
      if (nn === null || !nn.trim()) return;
      await api("/api/sessions/rename", {method: "POST",
                body: JSON.stringify({name: s.name, new_name: nn})});
      await refreshSessions();
      if (s.active) await refreshState();
      toast("Session renamed");
    } else if (act === "delete") {
      const runs = s.num_results
        ? `\n\nIts ${s.num_results} recorded experiment(s) are lost permanently.` : "";
      if (!confirm(`Delete ${s.name}?${runs}`)) return;
      await api("/api/sessions/delete",
                {method: "POST", body: JSON.stringify({name: s.name})});
      await refreshSessions();
      toast(`${s.name} deleted`);
    } else if (act === "describe") {
      const d = prompt(`Description for ${s.name}:`, s.description || "");
      if (d === null) return;
      await api("/api/sessions/describe", {method: "POST",
                body: JSON.stringify({name: s.name, description: d})});
      await refreshSessions();
      if (s.active) await refreshState();
      toast("Description saved");
    }
  } catch (e) {
    toast(e.message, true);
  }
}

async function createSessionFile() {
  const name = $("#new-session-name").value.trim();
  const description = $("#new-session-desc").value.trim();
  if (!name) return toast("Give the new session a file name", true);
  busy("creating session…");
  try {
    await api("/api/sessions/create",
              {method: "POST", body: JSON.stringify({name, description})});
    $("#new-session-name").value = "";
    $("#new-session-desc").value = "";
    closeSessions();
    invalidateLandscape();
    S.model = null;
    await refreshState();     // unconfigured -> shows the setup wizard
    refreshModel();
    toast("Session created — define ingredients & responses");
  } catch (e) {
    toast(e.message, true);
  } finally {
    idle();
  }
}

/* ---------------------------------------------------------------- wizard --- */

function addFactorRow(name = "", lo = "", hi = "", price = "0") {
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input class="f-name" placeholder="e.g. Li salt"></td>
    <td class="num"><input class="f-lo" inputmode="decimal" placeholder="0"></td>
    <td class="num"><input class="f-hi" inputmode="decimal" placeholder="1"></td>
    <td class="num"><input class="f-price" inputmode="decimal"></td>
    <td><button class="del" title="remove">×</button></td>`;
  $(".f-name", tr).value = name;
  $(".f-lo", tr).value = lo;
  $(".f-hi", tr).value = hi;
  $(".f-price", tr).value = price;
  $(".del", tr).onclick = () => tr.remove();
  $("#factor-tbody").appendChild(tr);
}

function addResponseRow(name = "") {
  const div = document.createElement("div");
  div.className = "response-row";
  div.innerHTML = `<input class="r-name" placeholder="e.g. capacity">
    <button class="del" title="remove">×</button>`;
  $(".r-name", div).value = name;
  $(".r-name", div).oninput = syncTargetSelect;
  $(".del", div).onclick = () => { div.remove(); syncTargetSelect(); };
  $("#response-rows").appendChild(div);
  syncTargetSelect();
}

function syncTargetSelect() {
  const sel = $("#target-select");
  const prev = sel.value;
  sel.innerHTML = $$(".r-name").map((inp, i) =>
    `<option value="${i}">${esc(inp.value.trim() || `response ${i + 1}`)}</option>`).join("");
  if (prev && +prev < sel.options.length) sel.value = prev;
}

function buildWizardOnce() {
  if (S.wizardBuilt) return;
  S.wizardBuilt = true;
  addFactorRow();
  addFactorRow();
  addResponseRow();
  addResponseRow();
  $("#add-factor").onclick = () => addFactorRow();
  $("#add-response").onclick = () => addResponseRow();
  $("#create-session").onclick = createSession;
}

async function createSession() {
  const factors = $$("#factor-tbody tr").map((tr) => ({
    name: $(".f-name", tr).value.trim(),
    low: parseNum($(".f-lo", tr).value),
    high: parseNum($(".f-hi", tr).value),
    price: parseNum($(".f-price", tr).value) ?? 0,
  }));
  if (!factors.length) return toast("Add at least one ingredient", true);
  for (const f of factors) {
    if (f.low === null || f.high === null || !(f.low < f.high)) {
      return toast(`Ingredient “${f.name || "?"}”: min must be smaller than max`, true);
    }
  }
  const responses = $$(".r-name").map((inp) => inp.value.trim());
  if (!responses.length) return toast("Add at least one response", true);
  const body = {
    factors,
    responses,
    target_task: +($("#target-select").value || 0),
    maximize: $("#direction").value === "max",
    fixed_cost: parseNum($("#fixed-cost").value) ?? 0,
    currency: $("#currency").value.trim() || "USD",
    num_init: Math.round(parseNum($("#num-init").value) ?? 4),
    cost_aware: $("#cost-aware").checked,
  };
  busy("creating session…");
  try {
    await api("/api/setup", {method: "POST", body: JSON.stringify(body)});
    await refreshState();
    toast("Session created — suggest your first experiments!");
  } catch (e) {
    toast(e.message, true);
  } finally {
    idle();
  }
}

/* ------------------------------------------------------------------ init --- */

function wireStaticEvents() {
  $$(".tabs button").forEach((b) => { b.onclick = () => switchTab(b.dataset.tab); });
  $("#fit-kernel").onchange = applyFit;
  $("#fit-degree").onchange = applyFit;
  $("#session-name").onclick = openSessions;
  $("#open-sessions-link").onclick = (e) => { e.preventDefault(); openSessions(); };
  $("#sessions-close").onclick = closeSessions;
  $("#new-session-create").onclick = createSessionFile;
  $("#sessions-modal").addEventListener("click", (e) => {
    if (e.target === e.currentTarget) closeSessions();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSessions();
  });
  $$("#n-seg button").forEach((b) => {
    b.onclick = () => {
      S.nSuggest = +b.dataset.n;
      $$("#n-seg button").forEach((o) => o.classList.toggle("active", o === b));
    };
  });
  $("#btn-suggest").onclick = async () => {
    busy("choosing next experiments…");
    try {
      await api("/api/suggest", {method: "POST", body: JSON.stringify({n: S.nSuggest})});
      await refreshState();
    } catch (e) {
      toast(e.message, true);
    } finally {
      idle();
    }
  };
}

(async function init() {
  wireStaticEvents();
  const hash = location.hash.slice(1);
  if (["progress", "model", "landscape", "insights", "costs"].includes(hash)) switchTab(hash);
  try {
    await refreshState();
  } catch (e) {
    toast("Cannot reach the DoE server: " + e.message, true);
    return;
  }
  if (hash === "sessions") openSessions();
  try {
    await ensurePlotly();
  } catch (e) {
    toast(e.message, true);
  }
  renderPlotsFor(S.tab);
  refreshModel();
})();
