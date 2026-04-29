// ======================================================================
// DETECTION PAGE (UPGRADED, CLASSIC SCRIPT + DYNAMIC MODULE BRIDGE)
// Aligned with alerts.js, actions.js, dashboard.js, monitor.js
// - Works with existing detection.html (no type="module" required)
// - Bridges into SocketCore / utils / ui_core when available
// - Updates GlobalState so other pages stay in sync
// - Falls back gracefully if backend endpoints are unavailable
// ======================================================================

(function detectionPageBootstrap() {
  "use strict";

  const MODULE_PATHS = {
    socket: "./core/socket_core.js",
    utils: "./core/utils.js",
    ui: "./core/ui_core.js",
  };

  const DETECTION_ENDPOINTS = [
    "/api/detect",
    "/api/detection",
    "/api/detection/run",
    "/api/detect/run",
    "/api/analyze",
  ];

  const state = {
    modules: {
      SocketCore: null,
      utils: {},
      ui: {},
    },
    initialized: false,
    lastResult: null,
    abortController: null,
  };

  const els = {};

  function cacheDom() {
    els.form = document.getElementById("detection-form");
    els.runBtn = document.getElementById("btn-detect");
    els.error = document.getElementById("error-message");
    els.loading = document.getElementById("loading-spinner");
    els.resultSection = document.getElementById("result-section");
    els.verdictContainer = document.getElementById("verdict-container");
    els.confidenceContainer = document.getElementById("confidence-container");
    els.confidenceBar = document.getElementById("confidence-bar");
    els.confidenceValue = document.getElementById("confidence-value");
    els.attackType = document.getElementById("attack-type");
    els.severity = document.getElementById("severity-result");
    els.enginesCount = document.getElementById("engines-count");
    els.enginesList = document.getElementById("engines-list");
    els.featuresDisplay = document.getElementById("features-display");

    els.inputs = {
      duration: document.getElementById("duration"),
      src_bytes: document.getElementById("src_bytes"),
      dst_bytes: document.getElementById("dst_bytes"),
      count: document.getElementById("count"),
      srv_count: document.getElementById("srv_count"),
      serror_rate: document.getElementById("serror_rate"),
      same_srv_rate: document.getElementById("same_srv_rate"),
      source_ip: document.getElementById("source_ip"),
    };
  }

  async function loadModules() {
    const [socketResult, utilsResult, uiResult] = await Promise.allSettled([
      import(MODULE_PATHS.socket),
      import(MODULE_PATHS.utils),
      import(MODULE_PATHS.ui),
    ]);

    if (socketResult.status === "fulfilled") {
      state.modules.SocketCore = socketResult.value.default || socketResult.value.SocketCore || null;
    }

    if (utilsResult.status === "fulfilled") {
      state.modules.utils = utilsResult.value || {};
    }

    if (uiResult.status === "fulfilled") {
      state.modules.ui = uiResult.value || {};
    }

    console.log(
      "%c[DETECTION] Loaded",
      "color:#8b5cf6;font-weight:bold;",
      {
        socket: Boolean(state.modules.SocketCore),
        utils: Object.keys(state.modules.utils).length > 0,
        ui: Object.keys(state.modules.ui).length > 0,
      }
    );
  }

  function init() {
    cacheDom();
    attachEventHandlers();
    hydrateFromState();
    attachGlobalStateListeners();
    attachSocketHandlers();
    state.initialized = true;
  }

  function attachEventHandlers() {
    if (els.form) {
      els.form.addEventListener("submit", function (e) {
        e.preventDefault();
        runDetection();
      });

      els.form.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
          const tag = ((e.target && e.target.tagName) || "").toLowerCase();
          if (tag !== "textarea") {
            e.preventDefault();
            runDetection();
          }
        }
      });
    }

    Object.values(els.inputs).forEach((input) => {
      if (!input) return;
      input.addEventListener("input", debounce(() => {
        hideError();
      }, 120));
    });
  }

  function attachGlobalStateListeners() {
    if (!window.GlobalState || typeof window.GlobalState.subscribe !== "function") return;

    window.GlobalState.subscribe((appState) => {
      if (!appState || typeof appState !== "object") return;

      const current = appState.current || {};
      const sourceIpInput = els.inputs.source_ip;
      if (sourceIpInput && !sourceIpInput.value && appState.lastAlert && appState.lastAlert.ip) {
        sourceIpInput.value = String(appState.lastAlert.ip);
      }

      if (current.flows !== undefined && els.inputs.count && !isUserEdited(els.inputs.count)) {
        const suggested = safeRound(current.flows, 0);
        if (suggested > 0) els.inputs.count.placeholder = String(suggested);
      }
    });
  }

  function attachSocketHandlers() {
    const SocketCore = state.modules.SocketCore;
    if (!SocketCore || typeof SocketCore.on !== "function") return;

    SocketCore.on("connect", () => {
      showSuccess("Detection: real-time bridge connected");
    });

    SocketCore.on("disconnect", () => {
      showError("Detection: real-time feed lost — local mode active", true);
    });

    SocketCore.on("reconnect", () => {
      hideError();
      showSuccess("Detection: real-time feed restored");
    });
  }

  function hydrateFromState() {
    if (!window.GlobalState || !window.GlobalState.data) return;

    const appState = window.GlobalState.data;
    if (appState.lastAlert && appState.lastAlert.features && !state.lastResult) {
      renderFeaturePreview(appState.lastAlert.features);
    }
  }

  function getInputFeatures() {
    return {
      duration: parseNumber(els.inputs.duration?.value),
      src_bytes: parseNumber(els.inputs.src_bytes?.value),
      dst_bytes: parseNumber(els.inputs.dst_bytes?.value),
      count: parseNumber(els.inputs.count?.value),
      srv_count: parseNumber(els.inputs.srv_count?.value),
      serror_rate: clamp(parseNumber(els.inputs.serror_rate?.value), 0, 1),
      same_srv_rate: clamp(parseNumber(els.inputs.same_srv_rate?.value), 0, 1),
      source_ip: sanitizeIp(els.inputs.source_ip?.value),
    };
  }

  function validateFeatures(features) {
    const hasSignal = [
      features.duration,
      features.src_bytes,
      features.dst_bytes,
      features.count,
      features.srv_count,
      features.serror_rate,
      features.same_srv_rate,
    ].some((value) => typeof value === "number" && !Number.isNaN(value) && value > 0);

    if (!hasSignal) {
      return "Please enter at least one non-zero feature value.";
    }

    if (features.source_ip && !isLikelyIp(features.source_ip)) {
      return "Source IP must be a valid IPv4/IPv6 value or left blank.";
    }

    return "";
  }

  async function runDetection() {
    const features = getInputFeatures();
    const validationError = validateFeatures(features);
    if (validationError) {
      showError(validationError);
      return;
    }

    hideError();
    showLoading(true);
    hideResults();

    if (state.abortController) {
      try {
        state.abortController.abort();
      } catch (_) {}
    }
    state.abortController = new AbortController();

    try {
      const result = await requestDetection(features, state.abortController.signal);
      state.lastResult = result;
      displayResults(result, features);
      syncApplicationState(result, features);
      emitRealtimeEvents(result, features);
    } catch (error) {
      console.error("[DETECTION] Error running detection:", error);
      showError("Detection failed: " + ((error && error.message) ? error.message : "Unknown error"));
    } finally {
      showLoading(false);
    }
  }

  async function requestDetection(features, signal) {
    let lastError = null;

    for (const endpoint of DETECTION_ENDPOINTS) {
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ features }),
          signal,
        });

        if (!response.ok) {
          lastError = new Error(`HTTP ${response.status} from ${endpoint}`);
          continue;
        }

        const data = await response.json();
        return normalizeResult(data, features, endpoint);
      } catch (error) {
        if (error && error.name === "AbortError") throw error;
        lastError = error;
      }
    }

    console.warn("[DETECTION] Falling back to local heuristic engine", lastError);
    return simulateDetection(features);
  }

  function normalizeResult(data, features, source) {
    const engineResults = normalizeEngineResults(data.engine_results || data.engines || data.results || null, features);
    const confidence = normalizeConfidence(data.confidence, engineResults);
    const verdict = normalizeVerdict(data.verdict || data.prediction || data.label, confidence);
    const attackType =
      data.attack_type ||
      data.attackType ||
      data.class_name ||
      inferAttackType(engineResults, features, verdict);
    const severity = normalizeSeverity(data.severity, confidence, verdict);

    return {
      id: data.id || buildAlertId(),
      verdict,
      confidence,
      attack_type: attackType,
      severity,
      engine_results: engineResults,
      prediction: data.prediction || verdict,
      reason: data.reason || buildReason(engineResults, features, verdict),
      created_at: data.created_at || new Date().toISOString(),
      ip: data.ip || features.source_ip || null,
      source: source || data.source || "backend",
      features: { ...features },
      raw: data,
    };
  }

  function simulateDetection(features) {
    const engineResults = {
      signature_engine: signatureEngine(features),
      anomaly_engine: anomalyEngine(features),
      behavioral_engine: behavioralEngine(features),
      threshold_engine: thresholdEngine(features),
      reputation_engine: reputationEngine(features),
    };

    const confidence = normalizeConfidence(null, engineResults);
    const verdict = normalizeVerdict(null, confidence);
    const attackType = inferAttackType(engineResults, features, verdict);
    const severity = normalizeSeverity(null, confidence, verdict);

    return {
      id: buildAlertId(),
      verdict,
      confidence,
      attack_type: attackType,
      severity,
      engine_results: engineResults,
      prediction: verdict,
      reason: buildReason(engineResults, features, verdict),
      created_at: new Date().toISOString(),
      ip: features.source_ip || null,
      source: "local_fallback",
      features: { ...features },
      raw: { fallback: true },
    };
  }

  function displayResults(data, inputFeatures) {
    const verdict = String(data.verdict || "unknown").toLowerCase();
    const confidence = clamp(Number(data.confidence) || 0, 0, 1);
    const attackType = data.attack_type || "Unknown";
    const severity = data.severity || "Unknown";
    const engineResults = data.engine_results || {};

    renderVerdictBadge(verdict, severity);

    const confidencePercent = Math.round(confidence * 100);
    if (els.confidenceBar) {
      els.confidenceBar.style.width = `${confidencePercent}%`;
      els.confidenceBar.style.background = confidenceGradient(confidencePercent, severity);
    }
    if (els.confidenceValue) {
      els.confidenceValue.textContent = `${confidencePercent}%`;
    }
    if (els.confidenceContainer) {
      els.confidenceContainer.style.display = "block";
    }

    if (els.attackType) els.attackType.textContent = String(attackType);
    if (els.severity) els.severity.textContent = String(severity).toUpperCase();
    if (els.enginesCount) {
      const triggeredCount = Object.values(engineResults).filter((r) => r && r.triggered).length;
      els.enginesCount.textContent = `${triggeredCount}/${Object.keys(engineResults).length}`;
    }

    renderEngineResults(engineResults);
    renderFeaturePreview(inputFeatures);
    showResults();
    showSuccess(`Detection complete (${String(data.source || "local").replace(/_/g, " ")})`);
  }

  function renderVerdictBadge(verdict, severity) {
    if (!els.verdictContainer) return;

    const label = verdict.toUpperCase();
    const sev = String(severity || "low").toLowerCase();
    const colorMap = {
      clean: "var(--green, #10b981)",
      benign: "var(--green, #10b981)",
      suspicious: "var(--amber, #f59e0b)",
      attack: "var(--red, #ef4444)",
      malicious: "var(--red, #ef4444)",
      unknown: "var(--text-muted, #94a3b8)",
    };

    const color = colorMap[verdict] || colorMap.unknown;
    els.verdictContainer.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <span style="display:inline-flex;align-items:center;gap:8px;padding:8px 14px;border-radius:999px;font-weight:700;letter-spacing:.08em;border:1px solid ${color};color:${color};background:rgba(255,255,255,.03);">
          <span style="width:9px;height:9px;border-radius:50%;background:${color};box-shadow:0 0 10px ${color};display:inline-block;"></span>
          ${escapeHtml(label)}
        </span>
        <span style="font-size:12px;color:var(--text-muted,#94a3b8);text-transform:uppercase;letter-spacing:.08em;">Severity: ${escapeHtml(sev)}</span>
      </div>
    `;
  }

  function renderEngineResults(engineResults) {
    if (!els.enginesList) return;

    const entries = Object.entries(engineResults || {});
    if (entries.length === 0) {
      els.enginesList.innerHTML = '<div class="engine-result clean">No engine results available.</div>';
      return;
    }

    const html = entries
      .sort((a, b) => Number(Boolean(b[1]?.triggered)) - Number(Boolean(a[1]?.triggered)))
      .map(([engineName, result]) => {
        const triggered = result && result.triggered === true;
        const confidence = clamp(Number((result && result.confidence)) || 0, 0, 1);
        const confidencePercent = Math.round(confidence * 100);
        const normalizedName = engineName
          .replace(/_/g, " ")
          .replace(/\w/g, (ch) => ch.toUpperCase());

        return `
          <div class="engine-result ${triggered ? "triggered" : "clean"}">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
              <div class="engine-name">${escapeHtml(normalizedName)}</div>
              <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                <span style="font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:${triggered ? "var(--red,#ef4444)" : "var(--green,#10b981)"};">${triggered ? "TRIGGERED" : "CLEAR"}</span>
                <span class="mono-val">${confidencePercent}%</span>
              </div>
            </div>
            <div style="margin-top:8px;color:var(--text-secondary,#cbd5e1);">${escapeHtml(result.reason || "No explanation provided")}</div>
            ${result.metric !== undefined ? `<div style="margin-top:6px;font-size:11px;color:var(--text-muted,#94a3b8);">Metric: ${escapeHtml(String(result.metric))}</div>` : ""}
          </div>
        `;
      })
      .join("");

    els.enginesList.innerHTML = html;
  }

  function renderFeaturePreview(inputFeatures) {
    if (!els.featuresDisplay) return;

    const rows = Object.entries(inputFeatures || {})
      .filter(([, value]) => value !== null && value !== "" && value !== undefined && !Number.isNaN(value))
      .map(([key, value]) => `${padRight(key, 16)} : ${String(value)}`);

    els.featuresDisplay.textContent = rows.length > 0 ? rows.join("\n") : "No features supplied";
  }

  function syncApplicationState(result, features) {
    if (!window.GlobalState) return;

    const previous = window.GlobalState.data && typeof window.GlobalState.data === "object"
      ? window.GlobalState.data
      : {};

    const alerts = Array.isArray(previous.alerts) ? previous.alerts.slice() : [];
    const shouldCreateAlert = isThreat(result.verdict);
    const alertItem = buildAlertItem(result, features);

    if (shouldCreateAlert) {
      alerts.unshift(alertItem);
    }

    const blockedTotal = Number(previous.blocked_total || previous.blockedTotal || 0);
    const alertsCount = alerts.length;
    const status = shouldCreateAlert
      ? ((result.severity.toLowerCase() === "critical" || result.severity.toLowerCase() === "high") ? "attack" : "suspicious")
      : "safe";

    const current = {
      ...(previous.current || {}),
      flows: Number(features.count || previous.current?.flows || 0),
      alerts_per_min: alertsCount,
      blocked_ips: blockedTotal,
      model_accuracy: Math.round((Number(result.confidence) || 0) * 100),
    };

    const metrics = {
      ...(previous.metrics || {}),
      alerts_total: alertsCount,
      prevention_actions_total: Number(previous.metrics?.prevention_actions_total || 0),
    };

    const nextState = {
      ...previous,
      status,
      current,
      metrics,
      alerts,
      alertsCount,
      blocked_total: blockedTotal,
      lastAlert: shouldCreateAlert ? alertItem : (previous.lastAlert || null),
      lastDetection: {
        verdict: result.verdict,
        confidence: result.confidence,
        attack_type: result.attack_type,
        severity: result.severity,
        source: result.source,
        created_at: result.created_at,
        features: { ...features },
      },
    };

    if (typeof window.GlobalState.set === "function") {
      window.GlobalState.set(nextState);
    } else {
      window.GlobalState.data = nextState;
    }
  }

  function emitRealtimeEvents(result, features) {
    const SocketCore = state.modules.SocketCore;
    if (!SocketCore || typeof SocketCore.emit !== "function") return;

    const alertItem = buildAlertItem(result, features);

    try {
      SocketCore.emit("manual_detection_run", {
        id: result.id,
        verdict: result.verdict,
        confidence: result.confidence,
        severity: result.severity,
        attack_type: result.attack_type,
        source: "detection_page",
        created_at: result.created_at,
        ip: result.ip || features.source_ip || null,
      });

      if (isThreat(result.verdict)) {
        SocketCore.emit("new_alert", alertItem);

        if (String(result.severity || "").toLowerCase() === "critical" && (result.ip || features.source_ip)) {
          SocketCore.emit("approval_request", {
            ip: result.ip || features.source_ip,
            reason: result.reason || `Critical manual detection: ${result.attack_type}`,
            severity: String(result.severity || "high").toLowerCase(),
            source: "detection_page",
            alert_id: result.id,
          });
        }
      }
    } catch (error) {
      console.warn("[DETECTION] Unable to emit real-time events", error);
    }
  }

  function buildAlertItem(result, features) {
    return {
      id: result.id,
      severity: String(result.severity || "medium").toLowerCase(),
      prediction: result.attack_type || result.verdict,
      verdict: result.verdict,
      confidence: clamp(Number(result.confidence) || 0, 0, 1),
      status: "open",
      created_at: result.created_at || new Date().toISOString(),
      ip: result.ip || features.source_ip || null,
      reason: result.reason || buildReason(result.engine_results, features, result.verdict),
      attack_type: result.attack_type || result.verdict,
      features: { ...features },
      engine_results: result.engine_results || {},
      source: result.source || "detection_page",
    };
  }

  function resetForm() {
    if (els.form) els.form.reset();

    const defaults = {
      duration: "10",
      src_bytes: "1000",
      dst_bytes: "2000",
      count: "50",
      srv_count: "40",
      serror_rate: "0.05",
      same_srv_rate: "0.8",
      source_ip: "",
    };

    Object.entries(defaults).forEach(([key, value]) => {
      if (els.inputs[key]) {
        els.inputs[key].value = value;
        els.inputs[key].dataset.userEdited = "false";
      }
    });

    hideError();
    hideResults();
    renderFeaturePreview(getInputFeatures());
  }

  function showLoading(show) {
    if (els.loading) {
      els.loading.style.display = show ? "block" : "none";
    }
    if (els.runBtn) {
      els.runBtn.disabled = !!show;
      els.runBtn.textContent = show ? "Running..." : "Run Detection";
    }
  }

  function hideError() {
    if (els.error) {
      els.error.style.display = "none";
      els.error.textContent = "";
      els.error.classList.remove("active");
    }
  }

  function hideResults() {
    if (els.resultSection) {
      els.resultSection.style.display = "none";
      els.resultSection.classList.remove("active");
    }
  }

  function showResults() {
    if (els.resultSection) {
      els.resultSection.style.display = "block";
      els.resultSection.classList.add("active");
    }
  }

  function showError(message, persist) {
    const uiShowError = state.modules.ui && state.modules.ui.showError;
    if (typeof uiShowError === "function") {
      try {
        uiShowError(String(message));
      } catch (_) {}
    }

    if (els.error) {
      els.error.textContent = String(message || "Unknown error");
      els.error.style.display = "block";
      els.error.classList.add("active");
    }

    if (!persist) {
      window.clearTimeout(showError._timer);
      showError._timer = window.setTimeout(() => {
        if (els.error && els.error.textContent === String(message)) hideError();
      }, 4500);
    }
  }

  function showSuccess(message) {
    const uiShowSuccess = state.modules.ui && state.modules.ui.showSuccess;
    if (typeof uiShowSuccess === "function") {
      try {
        uiShowSuccess(String(message));
        return;
      } catch (_) {}
    }
    console.info("[DETECTION]", message);
  }

  function signatureEngine(features) {
    const score = weightedScore([
      [normalizeRatio(features.serror_rate, 0.12), 0.42],
      [normalizeRatio(features.count, 85), 0.25],
      [normalizeRatio(features.srv_count, 70), 0.18],
      [normalizeRatio(features.duration, 20), 0.15],
    ]);
    return {
      triggered: score >= 0.58,
      confidence: score,
      reason: score >= 0.58
        ? "Traffic pattern matches SYN/connection abuse indicators."
        : "No strong signature match observed.",
      metric: safeRound(score * 100, 1) + "%",
    };
  }

  function anomalyEngine(features) {
    const byteRatio = normalizeRatio((features.src_bytes || 0) / Math.max(features.dst_bytes || 1, 1), 2.4);
    const score = weightedScore([
      [normalizeRatio(features.src_bytes, 6500), 0.30],
      [normalizeRatio(features.dst_bytes, 9000), 0.18],
      [byteRatio, 0.22],
      [normalizeRatio(features.count, 120), 0.16],
      [1 - clamp(features.same_srv_rate || 0, 0, 1), 0.14],
    ]);
    return {
      triggered: score >= 0.61,
      confidence: score,
      reason: score >= 0.61
        ? "Volume and distribution deviate from expected baseline."
        : "Feature distribution stays near learned baseline.",
      metric: safeRound(score * 100, 1) + "%",
    };
  }

  function behavioralEngine(features) {
    const score = weightedScore([
      [1 - clamp(features.same_srv_rate || 0, 0, 1), 0.34],
      [normalizeRatio(features.count - features.srv_count, 28), 0.24],
      [normalizeRatio(features.duration, 30), 0.16],
      [normalizeRatio(features.serror_rate, 0.08), 0.26],
    ]);
    return {
      triggered: score >= 0.55,
      confidence: score,
      reason: score >= 0.55
        ? "Behavior pattern suggests service spread or lateral probing."
        : "Behavior remains within typical service affinity.",
      metric: safeRound(score * 100, 1) + "%",
    };
  }

  function thresholdEngine(features) {
    const thresholdHits = [
      features.count > 120,
      features.srv_count > 90,
      features.serror_rate > 0.14,
      (features.src_bytes || 0) > 12000,
      (features.duration || 0) > 45,
    ].filter(Boolean).length;
    const score = clamp(thresholdHits / 4, 0, 1);
    return {
      triggered: thresholdHits >= 2,
      confidence: score,
      reason: thresholdHits >= 2
        ? `${thresholdHits} threshold conditions breached.`
        : "Static thresholds remain in normal range.",
      metric: `${thresholdHits} hit(s)`,
    };
  }

  function reputationEngine(features) {
    const privateIp = isPrivateIp(features.source_ip);
    const score = privateIp ? 0.16 : (features.source_ip ? 0.52 : 0.24);
    return {
      triggered: !privateIp && Boolean(features.source_ip) && score >= 0.5,
      confidence: score,
      reason: !features.source_ip
        ? "No source IP supplied for reputation enrichment."
        : privateIp
          ? "Source IP is private/internal; external reputation risk is low."
          : "External source IP supplied; reputation check recommended.",
      metric: features.source_ip || "N/A",
    };
  }

  function normalizeEngineResults(engineResults, features) {
    if (!engineResults || typeof engineResults !== "object" || Array.isArray(engineResults)) {
      return {
        signature_engine: signatureEngine(features),
        anomaly_engine: anomalyEngine(features),
        behavioral_engine: behavioralEngine(features),
        threshold_engine: thresholdEngine(features),
        reputation_engine: reputationEngine(features),
      };
    }

    const normalized = {};
    Object.entries(engineResults).forEach(([key, value]) => {
      const item = value && typeof value === "object" ? value : {};
      normalized[key] = {
        triggered: Boolean(item.triggered || item.hit || item.alert),
        confidence: clamp(
          typeof item.confidence === "number"
            ? item.confidence
            : typeof item.score === "number"
              ? item.score
              : typeof item.probability === "number"
                ? item.probability
                : 0,
          0,
          1
        ),
        reason: item.reason || item.message || item.detail || "No explanation provided",
        metric: item.metric !== undefined ? item.metric : (item.value !== undefined ? item.value : undefined),
      };
    });

    return normalized;
  }

  function normalizeConfidence(rawConfidence, engineResults) {
    if (typeof rawConfidence === "number") return clamp(rawConfidence, 0, 1);

    const values = Object.values(engineResults || {});
    if (values.length === 0) return 0;

    const weighted = values.reduce((sum, item) => sum + (Number(item.confidence) || 0), 0) / values.length;
    const bonus = values.filter((item) => item.triggered).length / Math.max(values.length * 5, 1);
    return clamp(weighted + bonus, 0, 1);
  }

  function normalizeVerdict(rawVerdict, confidence) {
    const verdict = String(rawVerdict || "").toLowerCase();
    if (["safe", "clean", "benign"].includes(verdict)) return "clean";
    if (["suspicious", "warning", "suspect"].includes(verdict)) return "suspicious";
    if (["attack", "malicious", "intrusion", "threat"].includes(verdict)) return "attack";
    if (confidence >= 0.74) return "attack";
    if (confidence >= 0.45) return "suspicious";
    return "clean";
  }

  function normalizeSeverity(rawSeverity, confidence, verdict) {
    const severity = String(rawSeverity || "").toLowerCase();
    if (["low", "medium", "high", "critical"].includes(severity)) return severity;

    if (verdict === "attack" && confidence >= 0.88) return "critical";
    if (verdict === "attack") return "high";
    if (verdict === "suspicious") return confidence >= 0.6 ? "medium" : "low";
    return "low";
  }

  function inferAttackType(engineResults, features, verdict) {
    if (verdict === "clean") return "Benign Traffic";

    const serror = Number(features.serror_rate || 0);
    const spread = (features.count || 0) - (features.srv_count || 0);
    const byteSkew = (features.src_bytes || 0) / Math.max(features.dst_bytes || 1, 1);

    if (serror >= 0.12 && (features.count || 0) > 80) return "SYN Flood / Connection Abuse";
    if (spread > 30 && (1 - (features.same_srv_rate || 0)) > 0.35) return "Reconnaissance / Port Sweep";
    if (byteSkew > 2.6 && (features.src_bytes || 0) > 8000) return "Data Exfiltration Pattern";

    const triggeredNames = Object.entries(engineResults || {})
      .filter(([, item]) => item && item.triggered)
      .map(([name]) => name);

    if (triggeredNames.some((name) => /reputation/i.test(name))) return "Suspicious External Source";
    if (triggeredNames.some((name) => /anomaly/i.test(name))) return "Anomalous Traffic Burst";
    return "Potential Intrusion";
  }

  function buildReason(engineResults, features, verdict) {
    const triggered = Object.entries(engineResults || {})
      .filter(([, item]) => item && item.triggered)
      .map(([name, item]) => `${name.replace(/_/g, " ")}: ${item.reason}`);

    if (triggered.length > 0) {
      return triggered.join(" | " );
    }

    if (verdict === "clean") {
      return "All engines remained below escalation thresholds.";
    }

    return `Detection confidence ${Math.round(normalizeConfidence(null, engineResults) * 100)}% based on supplied features.`;
  }

  function parseNumber(value) {
    if (value === null || value === undefined || value === "") return 0;
    const num = Number(value);
    return Number.isFinite(num) ? num : 0;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
  }

  function normalizeRatio(value, threshold) {
    if (!Number.isFinite(value) || threshold <= 0) return 0;
    return clamp(value / threshold, 0, 1);
  }

  function weightedScore(pairs) {
    const totalWeight = pairs.reduce((sum, [, weight]) => sum + weight, 0) || 1;
    const weightedSum = pairs.reduce((sum, [value, weight]) => sum + clamp(value, 0, 1) * weight, 0);
    return clamp(weightedSum / totalWeight, 0, 1);
  }

  function sanitizeIp(value) {
    const cleaned = String(value || "").trim();
    return cleaned ? cleaned : null;
  }

  function isLikelyIp(value) {
    if (!value) return false;
    const ipv4 = /^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$/;
    const ipv6 = /^[0-9a-fA-F:]+$/;
    return ipv4.test(value) || (value.includes(":") && ipv6.test(value));
  }

  function isPrivateIp(value) {
    if (!value || !isLikelyIp(value)) return false;
    if (value.includes(":")) {
      return value.startsWith("fc") || value.startsWith("fd") || value.startsWith("fe80") || value === "::1";
    }
    const parts = value.split(".").map((part) => Number(part));
    if (parts[0] === 10) return true;
    if (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) return true;
    if (parts[0] === 192 && parts[1] === 168) return true;
    if (parts[0] === 127) return true;
    return false;
  }

  function isThreat(verdict) {
    return ["attack", "malicious", "suspicious"].includes(String(verdict || "").toLowerCase());
  }

  function confidenceGradient(confidencePercent, severity) {
    const sev = String(severity || "low").toLowerCase();
    if (sev === "critical" || confidencePercent >= 88) {
      return "linear-gradient(90deg, #f59e0b, #ef4444)";
    }
    if (sev === "high" || confidencePercent >= 70) {
      return "linear-gradient(90deg, #facc15, #f97316)";
    }
    if (sev === "medium" || confidencePercent >= 45) {
      return "linear-gradient(90deg, #10b981, #f59e0b)";
    }
    return "linear-gradient(90deg, #10b981, #34d399)";
  }

  function escapeHtml(text) {
    if (text === null || text === undefined) return "";
    return String(text).replace(/[&<>"']/g, function (m) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[m];
    });
  }

  function buildAlertId() {
    const random = Math.random().toString(36).slice(2, 8).toUpperCase();
    return `ALERT-${Date.now()}-${random}`;
  }

  function safeRound(value, decimals) {
    const factor = Math.pow(10, decimals || 0);
    return Math.round((Number(value) || 0) * factor) / factor;
  }

  function debounce(fn, wait) {
    let timer = null;
    return function debounced() {
      const args = arguments;
      const context = this;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => fn.apply(context, args), wait);
    };
  }

  function padRight(text, width) {
    const str = String(text);
    if (str.length >= width) return str;
    return str + " ".repeat(width - str.length);
  }

  function isUserEdited(input) {
    if (!input) return false;
    if (!input.dataset.boundEditedFlag) {
      input.addEventListener("input", () => {
        input.dataset.userEdited = "true";
      });
      input.dataset.boundEditedFlag = "true";
    }
    return input.dataset.userEdited === "true";
  }

  window.runDetection = runDetection;
  window.resetForm = resetForm;
  window.__DetectionPage = { runDetection, resetForm, state };

  document.addEventListener("DOMContentLoaded", async function () {
    try {
      await loadModules();
    } catch (error) {
      console.warn("[DETECTION] Module bridge unavailable, continuing in local mode", error);
    } finally {
      init();
      renderFeaturePreview(getInputFeatures());
    }
  });
})();
