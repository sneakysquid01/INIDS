// ======================================================================
// ALERTS PAGE (ES MODULE VERSION)
// Aligned with dashboard.js, monitor.js, actions.js
// Uses: core/socket_core.js, core/utils.js, core/ui_core.js
// ======================================================================

import SocketCore from "./core/socket_core.js";
import { playAlertTone } from "./core/utils.js";
import {
  showError as coreShowError,
  showSuccess as coreShowSuccess,
} from "./core/ui_core.js";

console.log(
  "%c[ALERTS] Loaded (ES Module)",
  "color:#ef4444;font-weight:bold;"
);

// ======================================================================
// STATE
// ======================================================================

let currentAlerts = [];
let currentAlert = null;

let detailsModal = null;
let statusModal = null;

// ======================================================================
// INIT
// ======================================================================

(function init() {
  const detailsEl = document.getElementById("detailsModal");
  const statusEl = document.getElementById("statusModal");

  if (detailsEl && typeof bootstrap !== "undefined") {
    detailsModal = new bootstrap.Modal(detailsEl);
  }
  if (statusEl && typeof bootstrap !== "undefined") {
    statusModal = new bootstrap.Modal(statusEl);
  }

  // Initial render from GlobalState if available
  hydrateFromState();

  // Subscribe to GlobalState (same pattern as other pages)
  if (window.GlobalState) {
    window.GlobalState.subscribe((state) => {
      if (!state || !state.alerts) return;
      syncAlertsFromState(state.alerts);
    });
  }

  // Wire real-time socket events
  attachSocketHandlers();
})();

// ======================================================================
// SOCKET REAL-TIME HANDLERS
// ======================================================================

function attachSocketHandlers() {
  // New alert detected
  SocketCore.on("new_alert", (alert) => {
    console.log("%c[ALERTS] new_alert", "color:#ef4444;", alert);
    if (!alert) return;

    currentAlerts.unshift(alert);
    updateAlertCount(currentAlerts.length);
    renderAlerts(currentAlerts);
    playAlertTone(alert.severity || "high");
  });

  // Alert updated (status / assignee / close reason)
  SocketCore.on("alert_update", (updated) => {
    console.log("%c[ALERTS] alert_update", "color:#3b82f6;", updated);
    if (!updated?.id) return;

    const idx = currentAlerts.findIndex((a) => a.id === updated.id);
    if (idx !== -1) {
      currentAlerts[idx] = { ...currentAlerts[idx], ...updated };
      renderAlerts(currentAlerts);
    }
  });

  // Block executed from any page
  SocketCore.on("block_update", () => {
    console.log("%c[ALERTS] block_update", "color:#f59e0b;");
    // Refresh from backend/state
    hydrateFromState(true);
  });

  // Connection lifecycle
  SocketCore.on("connect", () => {
    console.log("%c[ALERTS] Socket connected", "color:#10b981;");
    coreShowSuccess("Alerts: real-time feed connected");
    hydrateFromState(true);
  });

  SocketCore.on("disconnect", () => {
    console.warn("[ALERTS] Socket disconnected");
    coreShowError("Alerts: real-time feed lost — using cached data");
  });

  SocketCore.on("reconnect", () => {
    console.log("%c[ALERTS] Socket reconnected", "color:#10b981;");
    coreShowSuccess("Alerts: real-time feed restored");
    hydrateFromState(true);
  });
}

// ======================================================================
// STATE → UI SYNC
// ======================================================================

function hydrateFromState(force = false) {
  if (!window.GlobalState || !window.GlobalState.data) return;

  const alerts = window.GlobalState.data.alerts;
  if (!Array.isArray(alerts)) return;

  if (force || alerts !== currentAlerts) {
    syncAlertsFromState(alerts);
  }
}

function syncAlertsFromState(alerts) {
  currentAlerts = filterAlerts(alerts);
  updateAlertCount(currentAlerts.length);
  renderAlerts(currentAlerts);
}

// ======================================================================
// FILTERING
// ======================================================================

function filterAlerts(alerts) {
  const sev = document.getElementById("severity-filter")?.value || "";
  const status = document.getElementById("status-filter")?.value || "";

  return alerts.filter((a) => {
    const s = (a.severity || "").toLowerCase();
    const st = (a.status || "").toLowerCase();

    if (sev && s !== sev.toLowerCase()) return false;
    if (status && st !== status.toLowerCase()) return false;

    return true;
  });
}

// ======================================================================
// RENDERING
// ======================================================================

function renderAlerts(alerts) {
  const wrapper = document.getElementById("alerts-wrapper");
  const tbody = document.getElementById("alerts-body");
  const empty = document.getElementById("empty-state");

  tbody.innerHTML = "";

  if (!alerts || alerts.length === 0) {
    wrapper.style.display = "none";
    empty.style.display = "block";
    return;
  }

  wrapper.style.display = "block";
  empty.style.display = "none";

  alerts.forEach((alert) => {
    tbody.appendChild(createAlertRow(alert));
  });
}

function createAlertRow(alert) {
  const row = document.createElement("tr");

  const sev = (alert.severity || "low").toLowerCase();
  const status = (alert.status || "open").toLowerCase();

  row.className =
    sev === "critical"
      ? "sev-critical-row"
      : sev === "high"
      ? "sev-high-row"
      : "";

  row.innerHTML = `
    <td class="mono">${escapeHtml(alert.id)}</td>
    <td class="uppercase ${sev}">${sev}</td>
    <td>${escapeHtml(alert.prediction || "unknown")}</td>
    <td class="mono">${Math.round((alert.confidence || 0) * 100)}%</td>
    <td class="mono">${status}</td>
    <td class="text-right">
      <button class="btn btn-sm btn-danger"
        onclick="blockAlert('${escapeHtml(alert.id)}')">
        Block
      </button>
    </td>
  `;

  row.addEventListener("click", (e) => {
    if (e.target.tagName === "BUTTON") return;
    showAlertDetails(alert);
  });

  return row;
}

// ======================================================================
// ACTIONS
// ======================================================================

async function blockAlert(alertId) {
  if (!confirm(`Block alert ${alertId}?`)) return;

  try {
    const res = await fetch(`/api/block/${encodeURIComponent(alertId)}`, {
      method: "POST",
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    // Emit so monitor + dashboard + actions update instantly
    SocketCore.emit("approval_response", {
      alert_id: alertId,
      action: "block",
      source: "alerts_page",
    });

    coreShowSuccess("Block action executed");
  } catch (err) {
    console.error(err);
    coreShowError("Failed to block alert");
  }
}

// ======================================================================
// MODALS
// ======================================================================

function showAlertDetails(alert) {
  currentAlert = alert;

  const body = document.getElementById("modal-body");
  body.innerHTML = `
    <div><strong>Alert ID:</strong> ${escapeHtml(alert.id)}</div>
    <div><strong>Severity:</strong> ${escapeHtml(alert.severity)}</div>
    <div><strong>Prediction:</strong> ${escapeHtml(alert.prediction)}</div>
    <div><strong>Confidence:</strong> ${Math.round(
      (alert.confidence || 0) * 100
    )}%</div>
    <div><strong>Status:</strong> ${escapeHtml(alert.status || "open")}</div>
    <div><strong>Time:</strong> ${formatTimestamp(alert.created_at)}</div>
  `;

  detailsModal.show();
}

// ======================================================================
// UI HELPERS
// ======================================================================

function updateAlertCount(count) {
  const el = document.getElementById("alert-count");
  if (el) el.textContent = count;
}

function formatTimestamp(ts) {
  if (!ts) return "N/A";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function escapeHtml(text) {
  if (text === null || text === undefined) return "";
  return String(text).replace(/[&<>\"']/g, (m) => ({
    "&": "&",
    "<": "<",
    ">": ">",
    '"': """,
    "'": "'",
  })[m]);
}

// ======================================================================
// EXPOSE FUNCTIONS USED BY INLINE HTML
// ======================================================================

window.blockAlert = blockAlert;
window.showAlertDetails = showAlertDetails;
``