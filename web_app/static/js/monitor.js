// ======================================================================
// MONITOR PAGE (ES MODULE VERSION)
// Uses core/socket_core.js, core/utils.js, core/ui_core.js
// ======================================================================

import SocketCore from "./core/socket_core.js";
import { smoothNumber, animateBar, fadeIn, playAlertTone } from "./core/utils.js";
import { showError, showSuccess } from "./core/ui_core.js";

console.log("%c[MONITOR] Loaded (ES Module)", "color:#00bcd4;font-weight:bold;");


// ======================================================================
// DOM ELEMENTS
// ======================================================================

const el = {
    statusCard: document.getElementById("status-card"),
    statusValue: document.getElementById("status-value"),

    alertCount: document.getElementById("alert-count"),
    blockedCount: document.getElementById("blocked-count"),

    flowsValue: document.getElementById("flows-value"),
    flowsBar: document.getElementById("flows-bar"),

    alertsMinuteValue: document.getElementById("alerts-minute-value"),
    alertsBar: document.getElementById("alerts-bar"),

    blockedIPsValue: document.getElementById("blocked-ips-value"),
    blockedBar: document.getElementById("blocked-bar"),

    accuracyValue: document.getElementById("accuracy-value"),
    accuracyBar: document.getElementById("accuracy-bar"),

    approvals: document.getElementById("approvals-container"),
    alerts: document.getElementById("alerts-container")
};


// ======================================================================
// GLOBALSTATE REACTIVITY
// (Monitor auto-updates when socket updates global state)
// ======================================================================

window.GlobalState.subscribe((state) => {
    if (!state) return;

    // ---- Status card ---------------------------------------------------
    if (state.status) {
        el.statusCard.classList.remove("safe", "suspicious", "attack");

        if (state.status.toLowerCase() === "safe") {
            el.statusCard.classList.add("safe");
            el.statusValue.textContent = "🟢 SAFE";
        } else if (state.status.toLowerCase() === "suspicious") {
            el.statusCard.classList.add("suspicious");
            el.statusValue.textContent = "🟡 SUSPICIOUS";
            playAlertTone("medium");
        } else {
            el.statusCard.classList.add("attack");
            el.statusValue.textContent = "🔴 ATTACK";
            playAlertTone("high");
        }
    }

    // ---- Metrics --------------------------------------------------------
    if (state.current) {
        const cur = state.current;

        smoothNumber(el.flowsValue, cur.flows || 0);
        animateBar(el.flowsBar, cur.flows || 0);

        smoothNumber(el.alertsMinuteValue, cur.alerts_per_min || 0);
        animateBar(el.alertsBar, cur.alerts_per_min || 0);

        smoothNumber(el.blockedIPsValue, cur.blocked_ips || 0);
        animateBar(el.blockedBar, cur.blocked_ips || 0);

        el.accuracyValue.textContent = (cur.model_accuracy || 0) + "%";
        animateBar(el.accuracyBar, cur.model_accuracy || 0);
    }

    // ---- Alert counts ---------------------------------------------------
    smoothNumber(el.alertCount, state.alertsCount || 0);
    smoothNumber(el.blockedCount, state.blocked_total || 0);

    // ---- Real-time alerts -----------------------------------------------
    if (state.lastAlert) {
        addRealtimeAlert(state.lastAlert);
    }
});


// ======================================================================
// REAL-TIME ALERT UI
// ======================================================================

function addRealtimeAlert(alert) {
    const div = document.createElement("div");
    div.className = `alert-item ${alert.severity}`;
    div.style.marginBottom = "6px";

    div.innerHTML = `
        <div class="alert-title">${alert.attack_type || alert.prediction}</div>
        <div class="alert-details">${alert.reason || "No details provided"}</div>
    `;

    fadeIn(div);
    el.alerts.prepend(div);

    playAlertTone(alert.severity);

    // auto-remove after 60s
    setTimeout(() => {
        div.style.opacity = 0;
        setTimeout(() => div.remove(), 400);
    }, 60000);
}


// ======================================================================
// REAL-TIME APPROVALS UI
// ======================================================================

SocketCore.on("approval_request", (item) => {
    el.approvals.innerHTML = "";

    const div = document.createElement("div");
    div.className = `approval-item ${item.severity || "low"}`;

    div.innerHTML = `
        <div class="approval-ip">${item.ip}</div>
        <div class="approval-reason">${item.reason}</div>
        <div class="approval-actions">
            <button class="ds-btn ds-btn-red">Block</button>
            <button class="ds-btn ds-btn-green">Allow</button>
        </div>
    `;

    fadeIn(div);
    el.approvals.appendChild(div);

    playAlertTone(item.severity);
});


// ======================================================================
// CONNECTION EVENTS
// ======================================================================

SocketCore.on("connect", () => {
    console.log("%c[MONITOR] Connected to WebSocket", "color: #4caf50;");
});

SocketCore.on("disconnect", () => {
    console.warn("[MONITOR] Socket disconnected — fallback active");
    showError("Real-time feed lost — using fallback polling…");
});
