// ======================================================================
// ENGINES PAGE (UPGRADED, CLASSIC SCRIPT + DYNAMIC MODULE BRIDGE)
// Aligned with detection.js, alerts.js, dashboard.js, monitor.js
// - Works with existing engines.html (classic <script> tag)
// - Bridges to SocketCore and shared ui_core helpers when available
// - Reads/writes GlobalState so other pages stay coherent
// - Supports endpoint fallbacks for load + toggle actions
// ======================================================================

(function enginesPageBootstrap() {
  "use strict";

  const MODULE_PATHS = {
    socket: "./core/socket_core.js",
    ui: "./core/ui_core.js",
    utils: "./core/utils.js",
  };

  const ENGINES_ENDPOINTS = [
    "/api/engines",
    "/api/engines/list",
    "/api/engines/status",
    "/api/detection/engines",
  ];

  const state = {
    modules: {
      SocketCore: null,
      ui: {},
      utils: {},
    },
    engines: [],
    endpointUsed: null,
    loading: false,
    pendingToggles: new Set(),
    refreshTimer: null,
  };

  const els = {};

  const ENGINE_COPY = {
    ml_engine: {
      title: "ML Engine",
      description: "Machine-learning based verdicting using trained models and scoring features.",
      accent: "var(--blue, #3b82f6)",
    },
    signature: {
      title: "Signature Engine",
      description: "Rule and pattern matching for known traffic fingerprints and abuse sequences.",
      accent: "var(--red, #ef4444)",
    },
    anomaly: {
      title: "Anomaly Engine",
      description: "Statistical deviation analysis against learned baselines and expected distributions.",
      accent: "var(--amber, #f59e0b)",
    },
    threshold: {
      title: "Threshold Engine",
      description: "Deterministic threshold checks for flow rate, byte volume, and error-rate spikes.",
      accent: "var(--green, #10b981)",
    },
    threat_intel: {
      title: "Threat Intel Engine",
      description: "Reputation and enrichment checks using IP/domain intelligence sources.",
      accent: "#a855f7",
    },
    ti: {
      title: "Threat Intel Engine",
      description: "Reputation and enrichment checks using IP/domain intelligence sources.",
      accent: "#a855f7",
    },
    behavioral_engine: {
      title: "Behavioral Engine",
      description: "Contextual engine focused on service spread, probing patterns, and behaviour sequencing.",
      accent: "#14b8a6",
    },
    reputation_engine: {
      title: "Reputation Engine",
      description: "External-source scoring and reputation-driven risk uplift logic.",
      accent: "#6366f1",
    },
    signature_engine: {
      title: "Signature Engine",
      description: "Rule and pattern matching for known traffic fingerprints and abuse sequences.",
      accent: "var(--red, #ef4444)",
    },
    anomaly_engine: {
      title: "Anomaly Engine",
      description: "Statistical deviation analysis against learned baselines and expected distributions.",
      accent: "var(--amber, #f59e0b)",
    },
    threshold_engine: {
      title: "Threshold Engine",
      description: "Deterministic threshold checks for flow rate, byte volume, and error-rate spikes.",
      accent: "var(--green, #10b981)",
    },
  };

  function cacheDom() {
    els.grid = document.getElementById("engines-grid");
    els.emptyState = document.getElementById("empty-state");
    els.loading = document.getElementById("loading-spinner");
  }

  async function loadModules() {
    const [socketResult, uiResult, utilsResult] = await Promise.allSettled([
      import(MODULE_PATHS.socket),
      import(MODULE_PATHS.ui),
      import(MODULE_PATHS.utils),
    ]);

    if (socketResult.status === "fulfilled") {
      state.modules.SocketCore = socketResult.value.default || socketResult.value.SocketCore || null;
    }
    if (uiResult.status === "fulfilled") {
      state.modules.ui = uiResult.value || {};
    }
    if (utilsResult.status === "fulfilled") {
      state.modules.utils = utilsResult.value || {};
    }

    console.log("%c[ENGINES] Loaded", "color:#22c55e;font-weight:bold;", {
      socket: Boolean(state.modules.SocketCore),
      ui: Object.keys(state.modules.ui).length > 0,
      utils: Object.keys(state.modules.utils).length > 0,
    });
  }

  function init() {
    cacheDom();
    attachDomEvents();
    hydrateFromState();
    attachGlobalStateListeners();
    attachSocketHandlers();
    loadEngines();
  }

  function attachDomEvents() {
    if (!els.grid) return;

    els.grid.addEventListener("change", async function (event) {
      const input = event.target.closest('input[data-engine-id]');
      if (!input) return;
      const engineId = input.dataset.engineId;
      const enabled = Boolean(input.checked);
      await toggleEngine(engineId, enabled, input);
    });
  }

  function attachGlobalStateListeners() {
    if (!window.GlobalState || typeof window.GlobalState.subscribe !== "function") return;

    window.GlobalState.subscribe((appState) => {
      if (!appState || typeof appState !== "object") return;

      if (Array.isArray(appState.engines) && appState.engines.length > 0) {
        const normalized = appState.engines.map(normalizeEngine).filter(Boolean);
        if (normalized.length > 0 && !state.loading) {
          state.engines = normalized;
          renderEngines(normalized);
        }
      }

      if (appState.lastDetection && Array.isArray(state.engines) && state.engines.length > 0) {
        enrichFromLastDetection(appState.lastDetection);
      }
    });
  }

  function attachSocketHandlers() {
    const SocketCore = state.modules.SocketCore;
    if (!SocketCore || typeof SocketCore.on !== "function") return;

    SocketCore.on("connect", () => {
      showSuccess("Engines: real-time feed connected");
      scheduleRefresh(250);
    });

    SocketCore.on("disconnect", () => {
      showError("Engines: real-time feed lost — cached state active", true);
    });

    SocketCore.on("reconnect", () => {
      hideError();
      showSuccess("Engines: real-time feed restored");
      scheduleRefresh(250);
    });

    [
      "engine_update",
      "engine_status_change",
      "new_alert",
      "manual_detection_run",
      "block_update",
      "alert_update",
    ].forEach((eventName) => {
      SocketCore.on(eventName, () => {
        scheduleRefresh(300);
      });
    });
  }

  function hydrateFromState() {
    if (!window.GlobalState || !window.GlobalState.data) return;
    const appState = window.GlobalState.data;
    if (Array.isArray(appState.engines) && appState.engines.length > 0) {
      state.engines = appState.engines.map(normalizeEngine).filter(Boolean);
      renderEngines(state.engines);
    }
  }

  async function loadEngines() {
    state.loading = true;
    showLoading(true);
    hideError();

    try {
      const payload = await requestEngines();
      const engines = extractEnginesArray(payload).map(normalizeEngine).filter(Boolean);
      state.engines = engines;
      syncState(engines);
      renderEngines(engines);
    } catch (error) {
      console.error("[ENGINES] Error loading engines:", error);
      if (Array.isArray(state.engines) && state.engines.length > 0) {
        renderEngines(state.engines);
        showError("Failed to refresh engines from backend — showing cached state", true);
      } else {
        renderEngines([]);
        showError("Failed to load engines: " + ((error && error.message) || "Unknown error"), true);
      }
    } finally {
      state.loading = false;
      showLoading(false);
    }
  }

  async function requestEngines() {
    let lastError = null;

    for (const endpoint of ENGINES_ENDPOINTS) {
      try {
        const response = await fetch(endpoint, { method: "GET", headers: { "Accept": "application/json" } });
        if (!response.ok) {
          lastError = new Error(`HTTP ${response.status} from ${endpoint}`);
          continue;
        }
        const data = await response.json();
        state.endpointUsed = endpoint;
        return data;
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error("No engine endpoint responded successfully");
  }

  function extractEnginesArray(payload) {
    if (Array.isArray(payload)) return payload;
    if (payload && Array.isArray(payload.engines)) return payload.engines;
    if (payload && Array.isArray(payload.items)) return payload.items;
    if (payload && payload.data && Array.isArray(payload.data.engines)) return payload.data.engines;
    return [];
  }

  function normalizeEngine(engine) {
    if (!engine || typeof engine !== "object") return null;

    const engineId = String(
      engine.engine_id || engine.id || engine.key || engine.name || engine.slug || "unknown"
    );
    const engineType = String(
      engine.engine_type || engine.type || engine.category || engineId || "unknown"
    );

    const copy = ENGINE_COPY[engineId] || ENGINE_COPY[engineType] || null;
    const enabled = engine.enabled === undefined ? (engine.status !== "disabled") : engine.enabled === true;
    const ready = (engine.ready ?? engine.is_ready ?? engine.initialized ?? true) !== false;
    const health = String(engine.health || (ready ? "healthy" : "initializing"));
    const threshold = normalizeFraction(engine.confidence_threshold ?? engine.threshold ?? engine.min_confidence);
    const accuracy = normalizeFraction(engine.accuracy ?? engine.acc ?? engine.score_accuracy);

    return {
      id: engineId,
      type: engineType,
      title: String(engine.display_name || engine.label || (copy ? copy.title : humanize(engineId))),
      description: String(engine.description || (copy ? copy.description : "Detection engine")),
      accent: String(engine.accent || (copy ? copy.accent : "var(--border)")),
      enabled,
      ready,
      health,
      confidence_threshold: threshold,
      accuracy,
      detections_count: toInteger(engine.detections_count ?? engine.detections ?? engine.count),
      avg_latency_ms: toNumber(engine.avg_latency_ms ?? engine.latency_ms ?? engine.latency),
      weight: toNumber(engine.weight ?? engine.vote_weight ?? engine.voting_weight),
      last_updated: engine.last_updated || engine.updated_at || engine.last_seen || null,
      source: engine.source || state.endpointUsed || null,
      raw: engine,
    };
  }

  function renderEngines(engines) {
    if (!els.grid || !els.emptyState) return;

    if (!Array.isArray(engines) || engines.length === 0) {
      els.grid.style.display = "none";
      els.grid.innerHTML = "";
      els.emptyState.style.display = "block";
      return;
    }

    els.emptyState.style.display = "none";
    els.grid.style.display = "grid";
    els.grid.innerHTML = engines.map(createEngineCardHtml).join("");
  }

  function createEngineCardHtml(engine) {
    const statusLabel = engine.enabled ? "Enabled" : "Disabled";
    const readyLabel = engine.ready ? "Ready" : "Initializing";
    const healthLabel = humanize(engine.health || "healthy");
    const thresholdLabel = engine.confidence_threshold !== null
      ? `${Math.round(engine.confidence_threshold * 100)}%`
      : "—";
    const accuracyLabel = engine.accuracy !== null
      ? `${(engine.accuracy * 100).toFixed(1)}%`
      : "—";
    const detectionsLabel = engine.detections_count !== null ? String(engine.detections_count) : "—";
    const latencyLabel = engine.avg_latency_ms !== null ? `${engine.avg_latency_ms.toFixed(1)} ms` : "—";
    const weightLabel = engine.weight !== null ? engine.weight.toFixed(2) : "—";
    const lastUpdatedLabel = engine.last_updated ? formatTimestamp(engine.last_updated) : "N/A";

    return `
      <div class="engine-card ${engine.enabled ? "enabled" : "disabled"}" style="border-left-color:${escapeHtml(engine.accent)};">
        <div class="ds-card-body">
          <div class="engine-header">
            <div>
              <div class="engine-name">${escapeHtml(engine.title)}</div>
              <div class="engine-type">${escapeHtml(engine.type)}</div>
            </div>
            <label class="toggle" title="Toggle ${escapeHtml(engine.title)}">
              <input type="checkbox" data-engine-id="${escapeHtml(engine.id)}" ${engine.enabled ? "checked" : ""} ${state.pendingToggles.has(engine.id) ? "disabled" : ""}>
              <span class="slider"></span>
            </label>
          </div>

          <div class="engine-description">${escapeHtml(engine.description)}</div>

          <div class="engine-status">
            <span class="status-dot ${engine.enabled ? "on" : "off"}"></span>
            <span>${escapeHtml(statusLabel)}</span>
            <span>•</span>
            <span>${escapeHtml(readyLabel)}</span>
            <span>•</span>
            <span>${escapeHtml(healthLabel)}</span>
          </div>

          <div class="engine-details">
            <div><span>Confidence Threshold</span><strong>${escapeHtml(thresholdLabel)}</strong></div>
            <div><span>Accuracy</span><strong>${escapeHtml(accuracyLabel)}</strong></div>
            <div><span>Detections</span><strong>${escapeHtml(detectionsLabel)}</strong></div>
            <div><span>Avg Latency</span><strong>${escapeHtml(latencyLabel)}</strong></div>
            <div><span>Vote Weight</span><strong>${escapeHtml(weightLabel)}</strong></div>
            <div><span>Last Updated</span><strong>${escapeHtml(lastUpdatedLabel)}</strong></div>
          </div>
        </div>
      </div>
    `;
  }

  async function toggleEngine(engineId, enabled, inputEl) {
    if (!engineId) return;

    const current = state.engines.find((engine) => engine.id === engineId);
    if (!current) {
      showError(`Unknown engine: ${engineId}`);
      scheduleRefresh(0);
      return;
    }

    state.pendingToggles.add(engineId);
    renderEngines(state.engines);

    try {
      await requestToggle(engineId, enabled);

      state.engines = state.engines.map((engine) => engine.id === engineId
        ? { ...engine, enabled, ready: true, health: enabled ? "healthy" : "disabled" }
        : engine
      );
      syncState(state.engines);
      renderEngines(state.engines);

      emitEngineUpdate(engineId, enabled);
      showSuccess(`${current.title} ${enabled ? "enabled" : "disabled"} successfully`);
      scheduleRefresh(250);
    } catch (error) {
      console.error("[ENGINES] Error toggling engine:", error);
      showError(`Failed to toggle ${current.title}: ${((error && error.message) || "Unknown error")}`);
      if (inputEl) inputEl.checked = current.enabled;
      renderEngines(state.engines);
    } finally {
      state.pendingToggles.delete(engineId);
      renderEngines(state.engines);
    }
  }

  async function requestToggle(engineId, enabled) {
    const attempts = [
      {
        url: `/api/engines/${encodeURIComponent(engineId)}/toggle`,
        options: {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled }),
        },
      },
      {
        url: `/api/engines/${encodeURIComponent(engineId)}`,
        options: {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled }),
        },
      },
      {
        url: `/api/engines/${encodeURIComponent(engineId)}/${enabled ? "enable" : "disable"}`,
        options: {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        },
      },
    ];

    let lastError = null;
    for (const attempt of attempts) {
      try {
        const response = await fetch(attempt.url, attempt.options);
        if (!response.ok) {
          lastError = new Error(`HTTP ${response.status} from ${attempt.url}`);
          continue;
        }
        return await safeJson(response);
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error("Unable to toggle engine");
  }

  function emitEngineUpdate(engineId, enabled) {
    const SocketCore = state.modules.SocketCore;
    if (!SocketCore || typeof SocketCore.emit !== "function") return;

    try {
      SocketCore.emit("engine_update", {
        engine_id: engineId,
        enabled,
        source: "engines_page",
        updated_at: new Date().toISOString(),
      });
    } catch (error) {
      console.warn("[ENGINES] Unable to emit engine_update", error);
    }
  }

  function enrichFromLastDetection(lastDetection) {
    if (!lastDetection || !Array.isArray(state.engines) || state.engines.length === 0) return;

    const attackType = lastDetection.attack_type || lastDetection.verdict || null;
    if (!attackType) return;

    state.engines = state.engines.map((engine) => ({
      ...engine,
      last_updated: lastDetection.created_at || engine.last_updated,
      raw: { ...engine.raw, last_detection_type: attackType },
    }));
    renderEngines(state.engines);
  }

  function syncState(engines) {
    if (!window.GlobalState) return;

    const previous = window.GlobalState.data && typeof window.GlobalState.data === "object"
      ? window.GlobalState.data
      : {};

    const nextState = {
      ...previous,
      engines,
      engineSummary: {
        total: engines.length,
        enabled: engines.filter((engine) => engine.enabled).length,
        ready: engines.filter((engine) => engine.ready).length,
        endpoint: state.endpointUsed,
      },
    };

    if (typeof window.GlobalState.set === "function") {
      window.GlobalState.set(nextState);
    } else {
      window.GlobalState.data = nextState;
    }
  }

  function scheduleRefresh(delay) {
    window.clearTimeout(state.refreshTimer);
    state.refreshTimer = window.setTimeout(() => {
      loadEngines();
    }, typeof delay === "number" ? delay : 400);
  }

  function showLoading(show) {
    if (!els.loading) return;
    els.loading.style.display = show ? "block" : "none";
  }

  function hideError() {
    // This page uses transient toasts only. Kept for symmetry with other pages.
  }

  function showError(message, persist) {
    const uiShowError = state.modules.ui && state.modules.ui.showError;
    if (typeof uiShowError === "function") {
      try { uiShowError(String(message)); } catch (_) {}
    }

    toast(message, "danger", persist ? 7000 : 5000);
  }

  function showSuccess(message) {
    const uiShowSuccess = state.modules.ui && state.modules.ui.showSuccess;
    if (typeof uiShowSuccess === "function") {
      try {
        uiShowSuccess(String(message));
        return;
      } catch (_) {}
    }

    toast(message, "success", 3200);
  }

  function toast(message, type, timeout) {
    const node = document.createElement("div");
    node.className = `alert alert-${type === "danger" ? "danger" : "success"} position-fixed bottom-0 end-0 m-3`;
    node.style.zIndex = "9999";
    node.style.minWidth = "260px";
    node.style.maxWidth = "420px";
    node.style.boxShadow = "0 12px 30px rgba(0,0,0,.18)";
    node.innerHTML = `${escapeHtml(String(message || ""))}`;
    document.body.appendChild(node);
    window.setTimeout(() => node.remove(), timeout || 4000);
  }

  function humanize(value) {
    return String(value || "unknown")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (ch) => ch.toUpperCase());
  }

  function normalizeFraction(value) {
    const num = toNumber(value);
    if (num === null) return null;
    if (num > 1) return clamp(num / 100, 0, 1);
    return clamp(num, 0, 1);
  }

  function toNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function toInteger(value) {
    const num = toNumber(value);
    return num === null ? null : Math.round(num);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
  }

  async function safeJson(response) {
    try {
      return await response.json();
    } catch (_) {
      return {};
    }
  }

  function formatTimestamp(value) {
    if (!value) return "N/A";
    try {
      const date = new Date(value);
      if (Number.isNaN(+date)) throw new Error("Invalid date");
      return date.toLocaleString();
    } catch (_) {
      return String(value);
    }
  }

  function escapeHtml(text) {
    if (text === null || text === undefined) return "";
    return String(text).replace(/[&<>\"']/g, (m) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[m]);
  }

  window.toggleEngine = function (engineId, enabled) {
    return toggleEngine(engineId, enabled);
  };
  window.__EnginesPage = { loadEngines, toggleEngine, state };

  document.addEventListener("DOMContentLoaded", async function () {
    try {
      await loadModules();
    } catch (error) {
      console.warn("[ENGINES] Module bridge unavailable, continuing with local behaviour", error);
    } finally {
      init();
    }
  });
})();
``