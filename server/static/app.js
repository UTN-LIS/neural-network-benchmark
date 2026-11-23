"use strict";

/* ============================================================
   Estado de la sesión de UI
   ============================================================ */
const state = {
  options: null,
  info: null,
  dataset: "",
  activation: "relu",
  optimizer: "adam",
  lr: 0.001,
  init: "he",
  classWeight: "none",
  seed: 42,
  maxEpochs: 500,
  earlyStop: true,
  includeTest: false,
  liveValidation: null,
  evidence: null,
  evidenceLoaded: false,
};

const runs = [];
let busy = false;
let source = null;
let finished = false;
let curveRows = [];
let activeJobId = null;
let validationTimer = null;
let validationSeq = 0;

/* ============================================================
   Utilidades
   ============================================================ */
const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 3) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
};
const esc = (value) =>
  String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try {
    body = await response.json();
  } catch (_) {
    body = null;
  }
  if (!response.ok) {
    const detail = body && body.detail;
    const message =
      typeof detail === "string" ? detail : detail && detail.message ? detail.message : `HTTP ${response.status}`;
    const error = new Error(message);
    error.body = body;
    error.status = response.status;
    throw error;
  }
  return body;
}

function stat(key, value) {
  return '<div class="stat"><span>' + key + "</span><b>" + value + "</b></div>";
}

function metric(key, value, note, primary) {
  return (
    '<div class="metric' + (primary ? " primary" : "") + '"><div class="k">' + key +
    '</div><div class="v">' + value + '</div><div class="n">' + (note || "&nbsp;") + "</div></div>"
  );
}

/* Textos para mostrar valores técnicos (los códigos de la API no cambian). */
const EXECUTION_STATE_LABELS = {
  COMPLETED: "Completada",
  COMPLETED_WITH_WARNINGS: "Completada con advertencias",
  INCOMPLETE: "Incompleta",
  FAILED: "Fallida",
  VALID: "Válida",
  WARNING: "Advertencia",
  INVALID: "Inválida",
};

const JOB_STATUS_LABELS = {
  queued: "en cola",
  running: "entrenando",
  completed: "completada",
  failed: "fallida",
};

const executionStateLabel = (state) => EXECUTION_STATE_LABELS[state] || state || "—";
const jobStatusLabel = (status) => JOB_STATUS_LABELS[status] || status || "—";

/* Corridas persistidas antes de traducir el dominio guardaron avisos en inglés.
   Esta tabla solo adapta esos metadatos históricos para mostrarlos en español. */
const LEGACY_WARNING_TRANSLATIONS = {
  "Adam with learning rate >= 0.1 may be unstable":
    "Adam con tasa de aprendizaje >= 0.1 puede ser inestable",
  "SGD with learning rate >= 0.1 may diverge":
    "SGD con tasa de aprendizaje >= 0.1 puede divergir",
  "He/Kaiming initialization is not conventionally recommended for sigmoid/tanh activations":
    "La inicialización He/Kaiming no es la recomendación convencional para activaciones sigmoid/tanh",
  "Xavier/Glorot initialization is not conventionally recommended for ReLU activations":
    "La inicialización Xavier/Glorot no es la recomendación convencional para activaciones ReLU",
  "run cancelled by user request": "Corrida cancelada por pedido del usuario",
};

function localizedWarning(message) {
  const text = String(message);
  if (LEGACY_WARNING_TRANSLATIONS[text]) return LEGACY_WARNING_TRANSLATIONS[text];
  const version = /^dataset version unavailable: (.*)$/.exec(text);
  if (version) return "Versión del dataset no disponible: " + version[1];
  const environment = /^environment capture skipped: (.*)$/.exec(text);
  if (environment) return "Captura del entorno omitida: " + environment[1];
  const convergence = /^convergence metric '(.*)' unavailable; running to max_epochs$/.exec(text);
  if (convergence) {
    return (
      "La métrica de convergencia '" + convergence[1] +
      "' no está disponible; se ejecuta hasta max_epochs"
    );
  }
  return text;
}

/* ============================================================
   Inicialización
   ============================================================ */
async function init() {
  try {
    state.options = await api("/api/options");
    $("api-status").textContent = "API conectada";
    $("api-status").className = "tag ok";
  } catch (error) {
    $("api-status").textContent = "sin conexión";
    $("api-status").className = "tag bad";
    $("results").innerHTML = '<div class="alert bad"><span class="t">No se pudo conectar con la API</span>' +
      esc(error.message) + "</div>";
    return;
  }

  try {
    const health = await api("/api/health");
    $("store-root").textContent = health.store_root;
  } catch (_) {
    /* el banner es informativo */
  }

  const datasets = state.options.datasets;
  const select = $("dataset");
  select.innerHTML = "";
  datasets.forEach((dataset) => {
    const option = document.createElement("option");
    option.value = dataset.name;
    option.textContent = dataset.name + (dataset.cached ? " · cacheado" : " · se descargará al seleccionar");
    select.appendChild(option);
  });
  const preferred = datasets.find((item) => item.name === "wine_quality");
  state.dataset = (preferred || datasets[0] || {}).name || "";
  select.value = state.dataset;
  select.addEventListener("change", () => {
    state.dataset = select.value;
    loadDataset();
  });

  wireControls();
  refresh();
  await Promise.all([loadDataset(), loadEvidence()]);
}

function wireControls() {
  $("seed").addEventListener("input", (event) => {
    state.seed = Number(event.target.value || 0);
    renderProtocol();
    scheduleValidate();
  });
  $("max-epochs").addEventListener("input", (event) => {
    state.maxEpochs = Number(event.target.value || 1);
    renderProtocol();
    scheduleValidate();
  });
  $("early-stop").addEventListener("change", (event) => {
    state.earlyStop = event.target.checked;
    renderProtocol();
    scheduleValidate();
  });
  $("include-test").addEventListener("change", (event) => {
    state.includeTest = event.target.checked;
    renderWarnings();
  });
  $("run-btn").addEventListener("click", startRun);
  $("cancel-btn").addEventListener("click", cancelRun);

  document.querySelectorAll("nav button[data-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!button.disabled) selectTab(button.dataset.tab);
    });
  });
}

/* ============================================================
   Controles
   ============================================================ */
function segmented(elementId, options, current, onPick) {
  const element = $(elementId);
  element.innerHTML = "";
  options.forEach((option) => {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = esc(option.label) + (option.note ? "<small>" + esc(option.note) + "</small>" : "");
    button.setAttribute("aria-pressed", String(String(option.value) === String(current)));
    button.addEventListener("click", () => {
      onPick(option.value);
      refresh();
      scheduleValidate();
    });
    element.appendChild(button);
  });
}

function refresh() {
  const options = state.options;
  if (!options) return;
  segmented(
    "seg-activation",
    options.activations.map((value) => ({ value, label: value })),
    state.activation,
    (value) => { state.activation = value; }
  );
  segmented(
    "seg-optimizer",
    options.optimizers.map((value) => ({ value, label: value, note: value === "adam" ? "adaptativo" : "paso único" })),
    state.optimizer,
    (value) => { state.optimizer = value; }
  );
  segmented(
    "seg-lr",
    options.learning_rates.map((value) => ({
      value,
      label: String(value),
      note: value === 0.001 ? "baja" : value === 0.01 ? "media" : "alta",
    })),
    state.lr,
    (value) => { state.lr = Number(value); }
  );
  segmented(
    "seg-init",
    options.initializations.map((value) => ({ value, label: value === "he" ? "He / Kaiming" : "Xavier / Glorot" })),
    state.init,
    (value) => { state.init = value; }
  );
  segmented(
    "seg-cw",
    options.class_weights.map((value) => ({ value, label: value })),
    state.classWeight,
    (value) => { state.classWeight = value; }
  );

  renderDatasetStats();
  renderArchitecture();
  renderProtocol();
  renderEstimate();
  renderWarnings();
  renderCompare();
}

function renderDatasetStats() {
  const info = state.info;
  if (!info) {
    $("dataset-stats").innerHTML = '<div class="loading">Cargando…</div>';
    return;
  }
  $("dataset-stats").innerHTML =
    stat("Fuente", esc(info.source)) +
    stat("Muestras", info.n_samples != null ? Number(info.n_samples).toLocaleString("es-AR") : "—") +
    stat("Features (entrada)", info.n_features != null ? info.n_features : "—") +
    stat("Clases (salida)", info.n_classes != null ? info.n_classes : "—") +
    stat("Desbalance máx/mín", info.imbalance != null ? Number(info.imbalance).toFixed(2) : "—") +
    stat("Cache", info.cached ? "disponible" : "se descargará");
}

function renderProtocol() {
  if (!state.options) return;
  const protocol = state.options.protocol;
  const stopping = protocol.early_stopping;
  $("protocol").innerHTML =
    stat("Partición train/val/test",
      Math.round(protocol.split.train * 100) + " / " + Math.round(protocol.split.val * 100) + " / " +
      Math.round(protocol.split.test * 100) + " (estratificada)") +
    stat("batch_size", protocol.batch_size) +
    stat("Early stopping",
      state.earlyStop
        ? stopping.metric + " · patience " + stopping.patience + " · min_delta " + stopping.min_delta
        : "desactivado") +
    stat("restore_best_weights", "true") +
    stat("seed", state.seed) +
    stat("Pérdida", "cross_entropy (sin ponderar en val/test)") +
    stat("Dispositivo", "auto");
}

function evidenceForDataset() {
  if (!state.evidence) return null;
  const matches = (state.evidence.experiments || []).filter((item) => item.dataset === state.dataset);
  if (!matches.length) return null;
  // La suite completa es el experimento con más corridas del dataset; un smoke
  // de 5 corridas no debe desplazarla.
  return matches.reduce((best, item) => (item.runs > best.runs ? item : best), matches[0]);
}

function currentFamily() {
  const evidence = evidenceForDataset();
  if (!evidence) return null;
  return (evidence.families || []).find(
    (family) => family.optimizer === state.optimizer && Number(family.lr) === Number(state.lr)
  ) || null;
}

function renderEstimate() {
  if (!state.evidenceLoaded) {
    $("estimate").textContent = "Cargando evidencia de las suites…";
    return;
  }
  const family = currentFamily();
  if (family) {
    $("estimate").innerHTML =
      "Costo real medido en la suite para esta familia: <b>" + fmt(family.time_mean, 1) + " s</b> y <b>" +
      fmt(family.epochs_mean, 1) + " épocas</b> de media (n=" + family.n + ").";
  } else {
    $("estimate").textContent = "Sin mediciones previas para esta combinación; el tiempo depende de las épocas ejecutadas.";
  }
}

/* ============================================================
   Advertencias (validación real + evidencia)
   ============================================================ */
function renderWarnings() {
  const items = [];
  const validation = state.liveValidation;

  if (!state.evidenceLoaded) {
    items.push({
      level: "ok",
      title: "Cargando evidencia de las suites…",
      text: "La evidencia empírica se lee de <code>results/exports</code>.",
    });
  }

  if (validation) {
    if (validation.state === "INVALID") {
      items.push({
        level: "bad",
        title: "Configuración inválida: no se puede ejecutar",
        text: (validation.errors || validation.rationale || [])
          .map(localizedWarning)
          .map(esc)
          .join(" · "),
      });
    } else if (validation.state === "WARNING") {
      items.push({
        level: "warn",
        title: "Advertencia de validación",
        text: (validation.warnings || validation.rationale || [])
          .map(localizedWarning)
          .map(esc)
          .join(" · "),
      });
    }
  }

  const evidence = evidenceForDataset();
  const family = currentFamily();
  if (family && family.failure_mode) {
    items.push({
      level: "bad",
      title: "Familia identificada como problemática: " + family.label,
      text:
        esc(family.failure_label) + " Evidencia en <b>" + esc(evidence.experiment_id) + "</b>: F1 macro medio <b>" +
        fmt(family.f1_mean, 4) + "</b>, <b>" + fmt(family.epochs_mean, 1) + "</b> épocas y <b>" +
        fmt(family.time_mean, 1) + " s</b> por corrida (n=" + family.n + ").",
    });
  } else if (family && evidence) {
    items.push({
      level: "ok",
      title: "Evidencia medida para esta familia (" + evidence.experiment_id + ")",
      text:
        "F1 macro medio <b>" + fmt(family.f1_mean, 4) + "</b>, exactitud <b>" + fmt(family.accuracy_mean, 4) +
        "</b>, <b>" + fmt(family.epochs_mean, 1) + "</b> épocas y <b>" + fmt(family.time_mean, 1) +
        " s</b> por corrida (n=" + family.n + ").",
    });
  }

  if (state.info && !state.info.cached) {
    items.push({
      level: "warn",
      title: "Dataset no cacheado",
      text: "La primera carga descarga el dataset desde su fuente. Requiere red.",
    });
  }
  if (state.includeTest) {
    items.push({
      level: "warn",
      title: "Set de prueba habilitado",
      text: "Las métricas de test se devolverán junto con las de validación. No las uses para elegir hiperparámetros.",
    });
  }

  if (!items.length) {
    items.push({
      level: "ok",
      title: "Sin observaciones",
      text: "Configuración convencional y sin familias problemáticas registradas para este dataset.",
    });
  }
  $("warnings").innerHTML = items
    .map((item) => '<div class="alert ' + item.level + '"><span class="t">' + item.title + "</span>" + item.text + "</div>")
    .join("");
}

/* ============================================================
   Validación en vivo contra el dominio real
   ============================================================ */
function buildPayload() {
  return {
    dataset: state.dataset,
    activation: state.activation,
    optimizer: state.optimizer,
    lr: Number(state.lr),
    init: state.init,
    class_weight: state.classWeight,
    seed: Number(state.seed),
    max_epochs: Number(state.maxEpochs),
    early_stop: Boolean(state.earlyStop),
    include_test: Boolean(state.includeTest),
    device: "auto",
  };
}

function scheduleValidate() {
  clearTimeout(validationTimer);
  validationTimer = setTimeout(liveValidate, 250);
}

async function liveValidate() {
  if (!state.dataset) return;
  const sequence = ++validationSeq;
  try {
    const result = await api("/api/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    if (sequence !== validationSeq) return;
    state.liveValidation = result;
  } catch (error) {
    if (sequence !== validationSeq) return;
    state.liveValidation = {
      state: "INVALID",
      rationale: [error.message],
      warnings: [],
      errors: [error.message],
    };
  }
  renderWarnings();
}

/* ============================================================
   Arquitectura (SVG)
   ============================================================ */
function nParams(info) {
  if (!info || info.n_features == null || info.n_classes == null) return null;
  const dims = [info.n_features, ...(info.hidden_layers || [32, 16]), info.n_classes];
  let total = 0;
  for (let i = 0; i < dims.length - 1; i += 1) total += dims[i] * dims[i + 1] + dims[i + 1];
  return total;
}

function renderArchitecture() {
  const info = state.info;
  if (!info || info.n_features == null || info.n_classes == null) {
    $("arch").innerHTML = '<div class="loading">La arquitectura se resuelve al cargar el dataset.</div>';
    $("arch-note").textContent = "";
    return;
  }
  const hidden = info.hidden_layers || [32, 16];
  const sizes = [info.n_features, hidden[0], hidden[1], info.n_classes];
  const names = ["Capa de entrada", "Capa oculta 1", "Capa oculta 2", "Capa de salida"];
  const descriptions = [
    info.n_features + " features estandarizadas",
    hidden[0] + " neuronas · " + state.activation,
    hidden[1] + " neuronas · " + state.activation,
    info.n_classes + " logits · CrossEntropyLoss",
  ];
  const colors = ["#6b7280", "#0b5fa5", "#0b5fa5", "#d97706"];
  const width = 660, height = 250, top = 46, bottom = height - 58, cap = 9;
  const xs = sizes.map((_, index) => 70 + index * ((width - 140) / (sizes.length - 1)));

  let edges = "";
  for (let layer = 0; layer < sizes.length - 1; layer += 1) {
    const from = Math.min(sizes[layer], cap);
    const to = Math.min(sizes[layer + 1], cap);
    for (let a = 0; a < from; a += 1) {
      for (let b = 0; b < to; b += 1) {
        const y1 = top + (from === 1 ? (bottom - top) / 2 : (a * (bottom - top)) / (from - 1));
        const y2 = top + (to === 1 ? (bottom - top) / 2 : (b * (bottom - top)) / (to - 1));
        edges += '<line x1="' + xs[layer] + '" y1="' + y1.toFixed(1) + '" x2="' + xs[layer + 1] +
          '" y2="' + y2.toFixed(1) + '" stroke="#93c5fd" stroke-width="1" opacity="0.30"/>';
      }
    }
  }

  let nodes = "";
  for (let layer = 0; layer < sizes.length; layer += 1) {
    const shown = Math.min(sizes[layer], cap);
    for (let index = 0; index < shown; index += 1) {
      const y = top + (shown === 1 ? (bottom - top) / 2 : (index * (bottom - top)) / (shown - 1));
      nodes += '<circle cx="' + xs[layer] + '" cy="' + y.toFixed(1) + '" r="7" fill="' + colors[layer] +
        '" stroke="#fff" stroke-width="1.6"/>';
    }
    if (sizes[layer] > cap) {
      nodes += '<text x="' + xs[layer] + '" y="' + (bottom + 18) + '" text-anchor="middle" fill="#6b7280">+' +
        (sizes[layer] - cap) + " más</text>";
    }
    nodes += '<text x="' + xs[layer] + '" y="24" text-anchor="middle" fill="#111827" font-weight="600">' + names[layer] + "</text>";
    nodes += '<text x="' + xs[layer] + '" y="' + (height - 12) + '" text-anchor="middle" fill="#6b7280">' +
      esc(descriptions[layer]) + "</text>";
  }

  $("arch").innerHTML =
    '<svg viewBox="0 0 ' + width + " " + height + '" width="100%" role="img" aria-label="Arquitectura de la red">' +
    edges + nodes + "</svg>";
  params = nParams(info);
  $("arch-note").innerHTML =
    "MLP de arquitectura fija <b>[" + hidden.join(", ") + "]</b>. Parámetros entrenables: <b>" +
    (params != null ? params.toLocaleString("es-AR") : "—") + "</b>. Lo único que cambia entre " +
    "configuraciones son las 4 dimensiones libres.";
}

let params = null;

/* ============================================================
   Carga de datos
   ============================================================ */
async function loadDataset() {
  if (!state.dataset) return;
  state.info = null;
  $("dataset-stats").innerHTML = '<div class="loading">Resolviendo dataset…</div>';
  renderArchitecture();
  try {
    state.info = await api("/api/datasets/" + encodeURIComponent(state.dataset));
  } catch (error) {
    state.info = null;
    $("dataset-stats").innerHTML = '<div class="alert bad">' + esc(error.message) + "</div>";
  }
  renderDatasetStats();
  renderArchitecture();
  renderEstimate();
  scheduleValidate();
}

async function loadEvidence() {
  try {
    state.evidence = await api("/api/evidence");
  } catch (_) {
    state.evidence = null;
  } finally {
    state.evidenceLoaded = true;
  }
  renderEstimate();
  renderWarnings();
  renderCompareTab();
}

/* ============================================================
   Corrida individual
   ============================================================ */
async function startRun() {
  if (busy) return;
  busy = true;
  finished = false;
  curveRows = [];
  activeJobId = null;
  $("run-btn").disabled = true;
  $("run-btn").textContent = "Encolando…";
  $("cancel-btn").style.display = "none";
  $("cancel-btn").disabled = false;
  $("cancel-btn").textContent = "■ Cancelar";
  $("run-state").textContent = "encolando";
  $("run-state").className = "tag";
  $("results").innerHTML = '<div class="empty">Esperando al worker…</div>';
  $("meta-box").style.display = "none";
  $("test-box").style.display = "none";
  $("curve").innerHTML = "";
  $("progress").style.width = "0%";
  $("progress-text").textContent = "Encolando…";

  try {
    const created = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    state.liveValidation = created.validation;
    renderWarnings();
    activeJobId = created.job_id;
    $("cancel-btn").style.display = "";
    openStream(created.job_id);
  } catch (error) {
    showRunError(error);
  }
}

async function cancelRun() {
  if (!activeJobId) return;
  const button = $("cancel-btn");
  button.disabled = true;
  button.textContent = "Cancelando…";
  try {
    await api("/api/runs/" + activeJobId + "/cancel", { method: "POST" });
    $("progress-text").textContent =
      "Cancelación solicitada: la corrida se detendrá al terminar la época actual.";
  } catch (error) {
    button.disabled = false;
    button.textContent = "■ Cancelar";
    showRunError(error);
  }
}

function openStream(jobId) {
  if (source) {
    source.close();
    source = null;
  }
  source = new EventSource("/api/runs/" + jobId + "/events");
  $("run-btn").textContent = "Entrenando…";

  source.addEventListener("status", (event) => {
    const data = JSON.parse(event.data);
    $("run-state").textContent = jobStatusLabel(data.status);
    $("run-state").className = "tag";
    $("run-state").title = data.status;
    $("progress-text").textContent =
      data.status === "running" ? "Entrenando…" : jobStatusLabel(data.status);
  });

  source.addEventListener("epoch", (event) => {
    const row = JSON.parse(event.data);
    curveRows.push(row);
    const percent = Math.min(100, Math.round((row.epoch / Math.max(1, Number(state.maxEpochs))) * 100));
    $("progress").style.width = percent + "%";
    $("progress-text").innerHTML = "Época <b>" + row.epoch + "</b> · train_loss " + fmt(row.train_loss) +
      (row.val_loss != null ? " · val_loss " + fmt(row.val_loss) : "") + " · " + percent + "% del tope";
    drawCurve(curveRows, null, false);
  });

  source.addEventListener("done", (event) => {
    const data = JSON.parse(event.data);
    finished = true;
    if (source) {
      source.close();
      source = null;
    }
    finishRun(data.result);
  });

  source.addEventListener("failed", (event) => {
    const data = JSON.parse(event.data);
    finished = true;
    if (source) {
      source.close();
      source = null;
    }
    showRunError(new Error(data.message || "la corrida falló"));
  });

  source.onerror = () => {
    if (finished) return;
    showRunError(new Error("se interrumpió la conexión SSE"));
    if (source) {
      source.close();
      source = null;
    }
  };
}

function finishRun(payload) {
  busy = false;
  activeJobId = null;
  $("run-btn").disabled = false;
  $("run-btn").textContent = "▶ Correr combinación";
  $("cancel-btn").style.display = "none";
  $("cancel-btn").disabled = false;
  $("cancel-btn").textContent = "■ Cancelar";
  const stateName = payload.execution_state || "COMPLETED";
  $("run-state").textContent = executionStateLabel(stateName);
  $("run-state").title = stateName;
  $("run-state").className =
    "tag " + (stateName === "COMPLETED" ? "ok" : stateName === "COMPLETED_WITH_WARNINGS" ? "warn" : "bad");
  $("progress").style.width = "100%";
  $("progress-text").innerHTML = "Corrida finalizada · <b>" + payload.epochs_executed + "</b> épocas · " +
    fmt(payload.training_time, 1) + " s";

  const validation = payload.validation_metrics || {};
  const config = payload.config || {};
  const hidden = config.architecture && config.architecture.hidden_layers ? config.architecture.hidden_layers : [32, 16];
  const warnings = payload.warnings || [];

  drawCurve(curveRows, validation.epoch, true);
  $("res-tag").textContent = "1 semilla · " + state.dataset;
  $("results").innerHTML =
    '<div class="metrics">' +
      metric("F1 macro (val)", fmt(validation.f1_macro, 4), "métrica de decisión", true) +
      metric("Accuracy (val)", fmt(validation.accuracy, 4), "set de validación") +
      metric("Balanced acc.", fmt(validation.balanced_accuracy, 4), "sensible a clases") +
      metric("Precision macro", fmt(validation.precision_macro, 4), "promedio por clase") +
      metric("Recall macro", fmt(validation.recall_macro, 4), "sensibilidad macro") +
      metric("Val loss", fmt(validation.loss, 4), "mejor época") +
      metric("Épocas", payload.epochs_executed, "tope " + state.maxEpochs) +
      metric("Mejor época", validation.epoch != null ? validation.epoch : "—", "restore_best_weights") +
      metric("Tiempo", fmt(payload.training_time, 1) + " s", "CPU") +
      metric("Parámetros", payload.n_params != null ? Number(payload.n_params).toLocaleString("es-AR") : "—",
        "MLP [" + hidden.join(",") + "]") +
    "</div>" +
    '<div class="hint" style="margin-top:10px">config_id <code>' + esc(payload.config_id || "—") +
      "</code> · space_key <code>" + esc(payload.space_key || "—") + "</code> · group_key <code>" +
      esc(payload.group_key || "—") + "</code></div>" +
    (stateName === "INCOMPLETE"
      ? '<div class="alert warn" style="margin-top:12px"><span class="t">Corrida cancelada</span>' +
        "Se conservaron las métricas de la mejor época alcanzada, pero el estado técnico es " +
        "<b>INCOMPLETA</b>: la corrida no se incluye en agregados ni informes.</div>"
      : "") +
    (warnings.length
      ? '<div class="alert warn" style="margin-top:12px"><span class="t">Estado de ejecución: ' +
        esc(executionStateLabel(payload.execution_state)) + "</span>" +
        warnings.map(localizedWarning).map(esc).join(" · ") + "</div>"
      : '<div class="alert ok" style="margin-top:12px"><span class="t">Estado de ejecución: ' +
        esc(executionStateLabel(payload.execution_state)) + "</span>Sin advertencias.</div>");

  $("meta-box").style.display = "";
  $("meta-json").textContent = JSON.stringify(payload.metadata, null, 2);

  if (payload.test_metrics) {
    $("test-box").style.display = "";
    const test = payload.test_metrics;
    $("test-metrics").innerHTML =
      metric("F1 macro (test)", fmt(test.f1_macro, 4), "reservado", false) +
      metric("Accuracy (test)", fmt(test.accuracy, 4), "") +
      metric("Balanced acc.", fmt(test.balanced_accuracy, 4), "") +
      metric("Precision macro", fmt(test.precision_macro, 4), "") +
      metric("Recall macro", fmt(test.recall_macro, 4), "") +
      metric("Loss (test)", fmt(test.loss, 4), "");
    renderConfusion(test.confusion_matrix, state.info ? state.info.n_classes : 0);
  } else {
    $("test-box").style.display = "none";
  }

  runs.unshift({ payload, curve: curveRows.slice() });
  if (runs.length > 3) runs.pop();
  renderCompare();
  if (historyModalOpen()) loadHistory();
}

function showRunError(error) {
  busy = false;
  finished = true;
  activeJobId = null;
  $("run-btn").disabled = false;
  $("run-btn").textContent = "▶ Correr combinación";
  $("cancel-btn").style.display = "none";
  $("cancel-btn").disabled = false;
  $("cancel-btn").textContent = "■ Cancelar";
  $("run-state").textContent = "error";
  $("run-state").className = "tag bad";
  const detail = error.body && error.body.detail;
  let extra = "";
  if (detail && detail.validation) {
    extra = (detail.validation.errors || [])
      .concat(detail.validation.warnings || [])
      .map(localizedWarning)
      .map(esc)
      .join(" · ");
  } else if (detail && typeof detail === "object") {
    extra = esc(JSON.stringify(detail));
  }
  $("results").innerHTML = '<div class="alert bad"><span class="t">No se pudo ejecutar</span>' +
    esc(error.message) + (extra ? "<br>" + extra : "") + "</div>";
}

/* ============================================================
   Curvas
   ============================================================ */
function lossCurveSvg(rows, bestEpoch) {
  const width = 700, height = 250, left = 46, right = 14, top = 16, bottom = 34;
  const maxEpoch = Math.max(rows[rows.length - 1].epoch, 1);
  let maxLoss = 0;
  rows.forEach((row) => {
    maxLoss = Math.max(maxLoss, row.train_loss, row.val_loss != null ? row.val_loss : 0);
  });
  maxLoss = Math.ceil(maxLoss * 10) / 10 || 1;
  const x = (epoch) => left + (epoch / maxEpoch) * (width - left - right);
  const y = (value) => top + (1 - value / maxLoss) * (height - top - bottom);
  const path = (key) =>
    rows
      .filter((row) => row[key] != null)
      .map((row, index) => (index ? "L" : "M") + x(row.epoch).toFixed(1) + " " + y(row[key]).toFixed(1))
      .join(" ");

  let grid = "";
  for (let index = 0; index <= 4; index += 1) {
    const value = (maxLoss * index) / 4;
    grid += '<line x1="' + left + '" y1="' + y(value).toFixed(1) + '" x2="' + (width - right) +
      '" y2="' + y(value).toFixed(1) + '" stroke="#eef0f3"/><text x="' + (left - 7) + '" y="' +
      (y(value) + 3.5).toFixed(1) + '" text-anchor="end" fill="#6b7280">' + value.toFixed(2) + "</text>";
  }
  let xAxis = "";
  for (let index = 0; index <= 5; index += 1) {
    const epoch = Math.round((maxEpoch * index) / 5);
    xAxis += '<text x="' + x(epoch).toFixed(1) + '" y="' + (height - 12) +
      '" text-anchor="middle" fill="#6b7280">' + epoch + "</text>";
  }

  let marker = "";
  if (bestEpoch != null) {
    const row = rows.find((item) => item.epoch === bestEpoch) || rows[rows.length - 1];
    if (row.val_loss != null) {
      marker =
        '<line x1="' + x(bestEpoch).toFixed(1) + '" y1="' + top + '" x2="' + x(bestEpoch).toFixed(1) +
        '" y2="' + (height - bottom) + '" stroke="#d97706" stroke-dasharray="4 3"/>' +
        '<circle cx="' + x(bestEpoch).toFixed(1) + '" cy="' + y(row.val_loss).toFixed(1) +
        '" r="4.5" fill="#d97706"/>' +
        '<text x="' + (x(bestEpoch) + 8).toFixed(1) + '" y="' + (top + 12) + '" fill="#d97706">mejor época ' +
        bestEpoch + "</text>";
    }
  }

  return '<svg viewBox="0 0 ' + width + " " + height + '" width="100%" role="img" aria-label="Curvas de pérdida">' +
    grid + xAxis + marker +
    '<path d="' + path("train_loss") + '" fill="none" stroke="#93c5fd" stroke-width="1.8"/>' +
    '<path d="' + path("val_loss") + '" fill="none" stroke="#0b5fa5" stroke-width="2"/>' +
    '<text x="' + left + '" y="' + (height - 1) + '" fill="#6b7280">época</text>' +
    '<g transform="translate(' + (width - 190) + "," + (top + 2) + ')">' +
      '<rect width="176" height="34" fill="#fff" opacity="0.85" rx="5"/>' +
      '<line x1="10" y1="12" x2="32" y2="12" stroke="#0b5fa5" stroke-width="2"/>' +
      '<text x="38" y="16" fill="#111827">val_loss</text>' +
      '<line x1="98" y1="12" x2="120" y2="12" stroke="#93c5fd" stroke-width="1.8"/>' +
      '<text x="126" y="16" fill="#111827">train_loss</text>' +
    "</g></svg>";
}

function drawCurve(rows, bestEpoch, complete) {
  if (!rows.length) return;
  $("curve").innerHTML = lossCurveSvg(rows, complete ? bestEpoch : null) +
    '<div class="hint">Equivalente visual de <code>epochs.csv</code>: <code>epoch</code>, ' +
    "<code>train_loss</code>, <code>val_loss</code>. La línea ámbar marca la época con menor " +
    "<code>val_loss</code>.</div>";
}

/* ============================================================
   Matriz de confusión
   ============================================================ */
function renderConfusion(matrix, classes) {
  if (!matrix || !matrix.length || !classes) {
    $("cm-wrap").innerHTML = "";
    return;
  }
  let total = 0;
  matrix.forEach((row) => row.forEach((value) => { total += value; }));
  let html = '<div class="hint" style="margin-top:14px">Matriz de confusión del set de prueba.</div>';
  html += '<div class="cm" style="grid-template-columns:34px repeat(' + classes + ",1fr);max-width:" +
    (60 + classes * 52) + 'px">';
  html += "<div></div>";
  for (let column = 0; column < classes; column += 1) {
    html += '<div style="background:none;color:#6b7280">P' + column + "</div>";
  }
  for (let row = 0; row < classes; row += 1) {
    html += '<div style="background:none;color:#6b7280">V' + row + "</div>";
    const rowTotal = matrix[row].reduce((a, b) => a + b, 0) || 1;
    for (let column = 0; column < classes; column += 1) {
      const value = matrix[row][column] / rowTotal;
      const diagonal = row === column;
      const alpha = Math.min(1, 0.12 + value * (diagonal ? 1.5 : 2.2));
      const color = diagonal
        ? "rgba(11,95,165," + alpha.toFixed(2) + ")"
        : "rgba(185,28,28," + (value * 2.4).toFixed(2) + ")";
      html += '<div style="background:' + color + '">' + (value > 0.005 ? (value * 100).toFixed(0) : "") + "</div>";
    }
  }
  html += "</div>";
  $("cm-wrap").innerHTML = html;
}

/* ============================================================
   Comparación rápida (hasta 3 corridas)
   ============================================================ */
function renderCompare() {
  $("cmp-count").textContent = runs.length + " / 3";
  if (!runs.length) {
    $("compare").innerHTML = '<div class="empty">Sin corridas para comparar.</div>';
    return;
  }

  let table =
    "<table><thead><tr><th>Configuración</th><th>Dataset</th><th class='num'>F1 val</th>" +
    "<th class='num'>Acc val</th><th class='num'>Bal. acc</th><th class='num'>Prec</th>" +
    "<th class='num'>Recall</th><th class='num'>Val loss</th><th class='num'>Épocas</th>" +
    "<th class='num'>Tiempo</th><th></th></tr></thead><tbody>";

  const COMPLETE_STATES = ["COMPLETED", "COMPLETED_WITH_WARNINGS"];
  const isComplete = (run) => COMPLETE_STATES.includes(run.payload.execution_state);
  const comparable = runs.filter(isComplete);
  const bestPool = comparable.length ? comparable : runs;
  const incompleteCount = runs.length - comparable.length;
  const best = Math.max(...bestPool.map((run) => Number(run.payload.validation_metrics.f1_macro || 0)));
  let banner = "";
  if (incompleteCount) {
    banner =
      '<div class="alert warn" style="margin-bottom:10px"><span class="t">Comparación con corridas incompletas</span>' +
      incompleteCount + " corrida(s) cancelada(s) o incompleta(s). Se muestran, pero no se usan como referencia " +
      "y no son comparables con las completas.</div>";
  }

  runs.forEach((run, index) => {
    const payload = run.payload;
    const validation = payload.validation_metrics || {};
    const config = payload.config || {};
    const delta = Number(validation.f1_macro || 0) - best;
    const label = [config.activation, config.optimizer, "lr " + config.lr, config.init].filter(Boolean).join(" · ");
    const runComplete = isComplete(run);
    const stateName = payload.execution_state || "COMPLETED";
    const stateClass = stateName === "COMPLETED" ? "ok" : stateName === "COMPLETED_WITH_WARNINGS" ? "warn" : "bad";
    table += "<tr>" +
      "<td>" + esc(label) + "<br><span class='pill'>cw " +
        esc(config.preprocessing ? config.preprocessing.class_weight || "none" : "none") + "</span> " +
        "<span class='pill " + stateClass + "' title='execution_state: " + esc(stateName) + "'>" +
        esc(executionStateLabel(stateName)) + "</span></td>" +
      "<td>" + esc(payload.config && payload.config.dataset ? payload.config.dataset.name : state.dataset) + "</td>" +
      "<td class='num'><b>" + fmt(validation.f1_macro, 4) + "</b><br><span class='pill'>" +
        (runs.length > 1
          ? (runComplete ? (delta === 0 ? "referencia" : (delta > 0 ? "+" : "") + fmt(delta, 4)) : "no comparable")
          : "—") + "</span></td>" +
      "<td class='num'>" + fmt(validation.accuracy, 4) + "</td>" +
      "<td class='num'>" + fmt(validation.balanced_accuracy, 4) + "</td>" +
      "<td class='num'>" + fmt(validation.precision_macro, 4) + "</td>" +
      "<td class='num'>" + fmt(validation.recall_macro, 4) + "</td>" +
      "<td class='num'>" + fmt(validation.loss, 4) + "</td>" +
      "<td class='num'>" + payload.epochs_executed + "</td>" +
      "<td class='num'>" + fmt(payload.training_time, 1) + " s</td>" +
      "<td class='num'><button class='btn ghost remove-run' type='button' data-index='" + index +
      "' title='Quitar esta corrida de la comparación'>Quitar</button></td></tr>";
  });
  table += "</tbody></table>";

  let overlay = "";
  if (runs.length > 1) {
    const width = 700, height = 230, left = 46, right = 14, top = 16, bottom = 34;
    const colors = ["#0b5fa5", "#d97706", "#0f766e"];
    let maxEpoch = 1;
    let maxLoss = 0;
    runs.forEach((run) => run.curve.forEach((row) => {
      maxEpoch = Math.max(maxEpoch, row.epoch);
      maxLoss = Math.max(maxLoss, row.val_loss != null ? row.val_loss : 0);
    }));
    maxLoss = Math.ceil(maxLoss * 10) / 10 || 1;
    const x = (epoch) => left + (epoch / maxEpoch) * (width - left - right);
    const y = (value) => top + (1 - value / maxLoss) * (height - top - bottom);

    let grid = "";
    for (let index = 0; index <= 4; index += 1) {
      const value = (maxLoss * index) / 4;
      grid += '<line x1="' + left + '" y1="' + y(value).toFixed(1) + '" x2="' + (width - right) +
        '" y2="' + y(value).toFixed(1) + '" stroke="#eef0f3"/><text x="' + (left - 7) + '" y="' +
        (y(value) + 3.5).toFixed(1) + '" text-anchor="end" fill="#6b7280">' + value.toFixed(2) + "</text>";
    }

    let paths = "";
    let legend = "";
    runs.forEach((run, index) => {
      const points = run.curve.filter((row) => row.val_loss != null);
      paths += '<path d="' + points
        .map((row, pointIndex) => (pointIndex ? "L" : "M") + x(row.epoch).toFixed(1) + " " + y(row.val_loss).toFixed(1))
        .join(" ") + '" fill="none" stroke="' + colors[index] + '" stroke-width="2"/>';
      const config = run.payload.config || {};
      legend += '<g transform="translate(' + (left + index * 210) + "," + (height - 14) + ')">' +
        '<line x1="0" y1="0" x2="20" y2="0" stroke="' + colors[index] + '" stroke-width="2"/>' +
        '<text x="26" y="4" fill="#111827">' +
        esc([config.activation, config.optimizer, "lr " + config.lr].filter(Boolean).join(" · ")) + "</text></g>";
    });

    overlay = '<div style="margin-top:14px"><svg viewBox="0 0 ' + width + " " + height +
      '" width="100%" role="img" aria-label="val_loss comparado">' + grid + paths + legend + "</svg>" +
      '<div class="hint">Curvas de <code>val_loss</code> superpuestas.</div></div>';
  }

  $("compare").innerHTML = banner + table + overlay +
    '<div style="margin-top:12px"><button class="btn ghost" id="clear-runs" type="button">Limpiar comparaciones</button></div>';
  document.querySelectorAll("#compare .remove-run").forEach((button) => {
    button.addEventListener("click", () => {
      runs.splice(Number(button.dataset.index), 1);
      renderCompare();
    });
  });
  $("clear-runs").addEventListener("click", () => {
    runs.length = 0;
    renderCompare();
  });
}

/* ============================================================
   Pestaña 2: evidencia real
   ============================================================ */
function renderCompareTab() {
  const evidence = state.evidence;
  if (!evidence) {
    $("exp-cards").innerHTML = '<div class="loading">No se pudo cargar la evidencia de las suites.</div>';
    return;
  }
  const experiments = evidence.experiments || [];

  $("exp-cards").innerHTML = experiments.length
    ? experiments.map((experiment) => {
        const bestFamily = (experiment.families || [])[0];
        return '<div class="card"><h2>' + esc(experiment.experiment_id) +
          '<span class="tag ok">suite completa</span></h2>' +
          '<p class="kv">Dataset: <b>' + esc(experiment.dataset) + "</b></p>" +
          "<div style='margin:10px 0'>" +
            stat("Corridas", experiment.runs) +
            stat("Grupos", experiment.groups) +
            stat("Semillas", experiment.seeds) +
            stat("Variantes cw", experiment.class_weight_variants) +
            stat("Tiempo total", experiment.hours >= 0.1 ? experiment.hours + " h CPU" : "< 0.1 h CPU") +
          "</div>" +
          (bestFamily
            ? '<p class="kv">Mejor familia: <b>' + esc(bestFamily.label) + "</b> · F1 " + fmt(bestFamily.f1_mean, 4) + "</p>"
            : "") +
          '<p style="margin:12px 0 0">' +
            (experiment.report_url ? '<a href="' + experiment.report_url + '" target="_blank">Informe HTML</a>' : "") +
            (experiment.report_url && experiment.pdf_url ? " · " : "") +
            (experiment.pdf_url ? '<a href="' + experiment.pdf_url + '" target="_blank">PDF</a>' : "") +
          "</p></div>";
      }).join("")
    : '<div class="empty">No hay suites completas exportadas en results/exports.</div>';

  let rows = "";
  experiments.forEach((experiment) => {
    (experiment.families || []).forEach((family) => {
      rows += "<tr" + (family.is_bottom ? " class='flag'" : "") + ">" +
        "<td>" + esc(experiment.dataset) + "</td>" +
        "<td><b>" + esc(family.label) + "</b>" +
          (family.failure_mode ? '<span class="tag bad">problemática</span>' : "") + "</td>" +
        "<td class='num'>" + family.n + "</td>" +
        "<td class='num'>" + fmt(family.f1_mean, 4) + "</td>" +
        "<td class='num'>" + fmt(family.accuracy_mean, 4) + "</td>" +
        "<td class='num'>" + fmt(family.epochs_mean, 1) + "</td>" +
        "<td class='num'>" + fmt(family.time_mean, 1) + " s</td>" +
        "<td>" + (family.failure_label ? esc(family.failure_label) : "—") + "</td></tr>";
    });
  });

  $("family-table").innerHTML = rows
    ? "<table><thead><tr><th>Dataset</th><th>Familia</th><th class='num'>n</th><th class='num'>F1</th>" +
      "<th class='num'>Accuracy</th><th class='num'>Épocas</th><th class='num'>Tiempo</th>" +
      "<th>Diagnóstico</th></tr></thead><tbody>" + rows + "</tbody></table>"
    : '<div class="empty">Sin familias para mostrar.</div>';

  const withSuite = new Set(evidence.datasets_with_suite || []);
  let coverage =
    "<table><thead><tr><th>Dataset</th><th class='num'>Muestras</th><th class='num'>Features</th>" +
    "<th class='num'>Clases</th><th class='num'>Desbalance</th><th>Cache</th><th>Suite completa</th></tr></thead><tbody>";
  (state.options ? state.options.datasets : []).forEach((dataset) => {
    coverage += "<tr><td><b>" + esc(dataset.name) + "</b><br><span class='pill'>" + esc(dataset.source) + "</span></td>" +
      "<td class='num'>" + (dataset.n_samples != null ? Number(dataset.n_samples).toLocaleString("es-AR") : "—") + "</td>" +
      "<td class='num'>" + (dataset.n_features != null ? dataset.n_features : "—") + "</td>" +
      "<td class='num'>" + (dataset.n_classes != null ? dataset.n_classes : "—") + "</td>" +
      "<td class='num'>" + (dataset.imbalance != null ? Number(dataset.imbalance).toFixed(2) : "—") + "</td>" +
      "<td>" + (dataset.cached ? '<span class="tag ok">sí</span>' : '<span class="tag">pendiente</span>') + "</td>" +
      "<td>" + (withSuite.has(dataset.name) ? '<span class="tag ok">disponible</span>' : '<span class="tag">pendiente</span>') +
      "</td></tr>";
  });
  $("coverage").innerHTML = coverage + "</tbody></table>";
}

/* ============================================================
   Historial persistido (results/ui-runs)
   ============================================================ */
let historyRows = [];
let historyFilterText = "";
let historyDetail = null;
let historyReturnFocus = null;

const HISTORY_STATE_CLASS = {
  COMPLETED: "ok",
  COMPLETED_WITH_WARNINGS: "warn",
  INCOMPLETE: "bad",
  FAILED: "bad",
};

function historyModalOpen() {
  return !$("history-modal").hidden;
}

function formatRunTime(value) {
  if (!value) return "—";
  const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/.exec(String(value));
  if (!match) return String(value);
  return match[3] + "/" + match[2] + "/" + match[1] + " " + match[4] + ":" + match[5] + ":" + match[6];
}

function historyStateTag(stateName) {
  const name = stateName || "DESCONOCIDO";
  return (
    '<span class="tag ' + (HISTORY_STATE_CLASS[name] || "") +
    '" title="execution_state: ' + esc(name) + '">' + esc(executionStateLabel(name)) + "</span>"
  );
}

function historyConfigLabel(row) {
  return [row.activation, row.optimizer, row.lr != null ? "lr " + row.lr : null, row.init]
    .filter(Boolean)
    .join(" · ");
}

function setHistoryDetailMode(isDetail) {
  $("history-toolbar").hidden = isDetail;
  $("history-hint").hidden = isDetail;
  $("history-detail-back").hidden = !isDetail;
  $("history-count").hidden = isDetail;
  $("history-title-text").textContent = isDetail ? "Detalle de la corrida" : "Historial de corridas";
}

function openHistory() {
  if (historyModalOpen()) return;
  historyReturnFocus = document.activeElement;
  historyFilterText = "";
  $("history-filter").value = "";
  $("history-modal").hidden = false;
  setHistoryDetailMode(false);
  document.body.style.overflow = "hidden";
  $("history-filter").focus();
  loadHistory();
}

function closeHistory() {
  if (!historyModalOpen()) return;
  $("history-modal").hidden = true;
  document.body.style.overflow = "";
  historyDetail = null;
  if (historyReturnFocus && typeof historyReturnFocus.focus === "function") {
    historyReturnFocus.focus();
  }
}

async function loadHistory() {
  $("history-body").innerHTML = '<div class="loading">Leyendo corridas persistidas…</div>';
  try {
    const body = await api("/api/history");
    historyRows = body.runs || [];
    historyDetail = null;
    $("history-hint").innerHTML =
      "Corridas persistidas en <code>" + esc(body.store_root || "results/ui-runs") +
      "</code> · experimento <code>" + esc(body.experiment_id || "—") +
      "</code> · fechas en UTC. Sobreviven a reinicios del servidor.";
    renderHistory();
  } catch (error) {
    $("history-body").innerHTML =
      '<div class="alert bad"><span class="t">No se pudo leer el historial</span>' + esc(error.message) + "</div>";
  }
}

function renderHistory() {
  if (historyDetail) {
    renderHistoryDetail(historyDetail);
    return;
  }
  setHistoryDetailMode(false);

  const needle = historyFilterText.trim().toLowerCase();
  const rows = historyRows.filter((row) => {
    if (!needle) return true;
    return [
      row.run_id,
      row.config_id,
      row.dataset,
      row.execution_state,
      historyConfigLabel(row),
      row.class_weight || "none",
      "seed " + (row.seed != null ? row.seed : ""),
    ]
      .join(" ")
      .toLowerCase()
      .includes(needle);
  });
  $("history-count").textContent = needle
    ? rows.length + " de " + historyRows.length
    : String(historyRows.length);

  if (!historyRows.length) {
    $("history-body").innerHTML =
      '<div class="empty">Todavía no hay corridas persistidas. Ejecutá una combinación y aparecerá acá.</div>';
    return;
  }
  if (!rows.length) {
    $("history-body").innerHTML = '<div class="empty">Ninguna corrida coincide con el filtro.</div>';
    return;
  }

  let table =
    "<div class='history-table'><table><thead><tr>" +
    "<th>Fecha (UTC)</th><th>Corrida</th><th>Dataset</th><th>Configuración</th>" +
    "<th>Estado</th><th class='num'>F1 val</th><th class='num'>Acc val</th>" +
    "<th class='num'>Épocas</th><th class='num'>Tiempo</th><th></th>" +
    "</tr></thead><tbody>";

  rows.forEach((row) => {
    table += "<tr>" +
      "<td>" + formatRunTime(row.start_time) + "</td>" +
      "<td><span class='history-run-id'>" + esc(row.run_id || "—") + "</span>" +
        (row.warning_count
          ? '<br><span class="pill warn">' + row.warning_count + " aviso(s)</span>"
          : "") + "</td>" +
      "<td>" + esc(row.dataset || "—") + "</td>" +
      "<td>" + esc(historyConfigLabel(row) || "—") +
        "<br><span class='pill'>cw " + esc(row.class_weight || "none") + "</span> " +
        "<span class='pill'>seed " + esc(row.seed != null ? row.seed : "—") + "</span></td>" +
      "<td>" + historyStateTag(row.execution_state) + "</td>" +
      "<td class='num'><b>" + fmt(row.f1_macro, 4) + "</b></td>" +
      "<td class='num'>" + fmt(row.accuracy, 4) + "</td>" +
      "<td class='num'>" + (row.epochs_executed != null ? row.epochs_executed : "—") +
        (row.max_epochs != null ? "<br><span class='pill'>tope " + esc(row.max_epochs) + "</span>" : "") +
        "</td>" +
      "<td class='num'>" + (row.training_time != null ? fmt(row.training_time, 1) + " s" : "—") + "</td>" +
      "<td class='num'>" + (row.run_id
        ? "<button class='btn ghost history-row-btn' type='button' data-run-id='" + esc(row.run_id) +
          "'>Detalle</button>"
        : "—") + "</td>" +
      "</tr>";
  });
  table += "</tbody></table></div>";
  $("history-body").innerHTML = table;

  document.querySelectorAll("#history-body .history-row-btn").forEach((button) => {
    button.addEventListener("click", () => showHistoryDetail(button.dataset.runId));
  });
}

async function showHistoryDetail(runId) {
  setHistoryDetailMode(true);
  $("history-body").innerHTML =
    '<div class="loading">Cargando corrida <code>' + esc(runId) + "</code>…</div>";
  try {
    const detail = await api("/api/history/" + encodeURIComponent(runId));
    if (!historyModalOpen()) return;
    historyDetail = detail;
    renderHistoryDetail(detail);
  } catch (error) {
    if (!historyModalOpen()) return;
    historyDetail = null;
    setHistoryDetailMode(false);
    $("history-body").innerHTML =
      '<div class="alert bad"><span class="t">No se pudo abrir la corrida</span>' + esc(error.message) + "</div>";
  }
}

function renderHistoryDetail(detail) {
  setHistoryDetailMode(true);
  $("history-detail-back").focus();
  const metadata = detail.metadata || {};
  const validation = metadata.validation_metrics || {};
  const warnings = metadata.warnings || [];
  const epochs = detail.epochs || [];
  const architecture = metadata.architecture || {};
  const preprocessing = metadata.preprocessing || {};

  let html =
    "<h3 style='margin:0;text-transform:none;font-size:14px;font-weight:600;color:var(--tinta)'>" +
      "<span class='history-run-id'>" + esc(detail.run_id || "—") + "</span> " +
      historyStateTag(metadata.execution_state || metadata.state) + "</h3>" +
    '<div class="hint">Inicio ' + formatRunTime(metadata.start_time) + " · fin " +
      formatRunTime(metadata.end_time) + " · config_id <code>" + esc(detail.config_id || "—") + "</code></div>" +
    '<div class="hint">Persistida en <code>' +
      esc(detail.path || "results/ui-runs/" + (detail.experiment_id || "") + "/" + (detail.run_id || "")) +
      "</code></div>";

  html += '<div class="metrics" style="margin-top:14px">' +
    metric("F1 macro (val)", fmt(validation.f1_macro, 4), "métrica de decisión", true) +
    metric("Accuracy (val)", fmt(validation.accuracy, 4), "set de validación") +
    metric("Balanced acc.", fmt(validation.balanced_accuracy, 4), "sensible a clases") +
    metric("Val loss", fmt(validation.loss, 4), "mejor época") +
    metric("Épocas", metadata.epochs_executed != null ? metadata.epochs_executed : "—",
      "tope " + (metadata.max_epochs != null ? metadata.max_epochs : "—")) +
    metric("Mejor época", validation.epoch != null ? validation.epoch : "—", "restore_best_weights") +
    metric("Tiempo", fmt(metadata.training_time, 1) + " s", "CPU") +
    metric("Parámetros",
      metadata.n_params != null ? Number(metadata.n_params).toLocaleString("es-AR") : "—",
      "MLP [" + (Array.isArray(architecture.hidden_layers) ? architecture.hidden_layers.join(",") : "") + "]") +
    "</div>";

  html += '<div class="hint" style="margin-top:12px">Configuración: <b>' +
    esc([metadata.activation, metadata.optimizer, metadata.lr != null ? "lr " + metadata.lr : null, metadata.init]
      .filter(Boolean).join(" · ") || "—") + "</b> · dataset <b>" +
    esc((metadata.dataset || {}).name || "—") + "</b> · cw <b>" + esc(preprocessing.class_weight || "none") +
    "</b> · seed <b>" + esc(metadata.seed != null ? metadata.seed : "—") + "</b>" +
    (metadata.batch_size != null ? " · batch_size <b>" + esc(metadata.batch_size) + "</b>" : "") + "</div>";

  if (epochs.length) {
    html += '<div style="margin-top:16px">' + lossCurveSvg(epochs, validation.epoch) +
      '<div class="hint">Curva reconstruida desde <code>epochs.csv</code>.</div></div>';
  } else {
    html += '<div class="empty" style="margin-top:16px">Esta corrida no tiene <code>epochs.csv</code> persistido.</div>';
  }

  if (warnings.length) {
    html += '<div class="alert warn" style="margin-top:14px"><span class="t">Advertencias</span>' +
      warnings.map(localizedWarning).map(esc).join(" · ") + "</div>";
  }

  html += "<details><summary>Metadata completa de la corrida (metadata.json)</summary><pre>" +
    esc(JSON.stringify(detail.metadata, null, 2)) + "</pre></details>";

  $("history-body").innerHTML = html;
}

function wireHistory() {
  $("history-btn").addEventListener("click", openHistory);
  $("history-close").addEventListener("click", closeHistory);
  $("history-refresh").addEventListener("click", loadHistory);
  $("history-detail-back").addEventListener("click", () => {
    historyDetail = null;
    renderHistory();
    $("history-filter").focus();
  });
  $("history-filter").addEventListener("input", (event) => {
    historyFilterText = event.target.value || "";
    if (historyDetail) historyDetail = null;
    renderHistory();
  });
  $("history-modal").addEventListener("click", (event) => {
    if (event.target === $("history-modal")) closeHistory();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && historyModalOpen()) closeHistory();
  });
}

/* ============================================================
   Navegación
   ============================================================ */
function selectTab(name) {
  ["run", "compare", "act"].forEach((tab) => {
    $("tab-" + tab).hidden = tab !== name;
    $("tab-btn-" + tab).setAttribute("aria-selected", String(tab === name));
  });
  if (name === "compare") renderCompareTab();
  if (window.location.hash !== "#" + name && (name === "run" || name === "compare")) {
    history.replaceState(null, "", "#" + name);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

wireHistory();
init().then(() => {
  const initial = window.location.hash.replace("#", "");
  if (initial === "compare") selectTab("compare");
});
