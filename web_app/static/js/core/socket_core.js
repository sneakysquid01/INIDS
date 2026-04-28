// ======================================================
// INIDS SOCKET CORE (Unified WebSocket Manager)
// Refactored from your original socket.js but modernized.
// ======================================================

import { showError, showSuccess } from "./ui_core.js";

// Prevent double initialization
if (window.INIDSSocketManager) {
    console.warn("socket_core.js already loaded.");
}

const GlobalState = window.GlobalState;

if (!GlobalState || typeof window.io !== "function") {
    console.error("GlobalState or Socket.IO is unavailable.");
}

// ------------------------------------------------------
// Core socket connection (namespace: /events)
// ------------------------------------------------------
const socket = io("/events", {
    transports: ["websocket", "polling"],
    reconnection: true,
    reconnectionAttempts: 10,
    reconnectionDelay: 500
});

let fallbackTimer = null;


// ======================================================
// PAYLOAD NORMALIZATION HELPERS
// (Metrics, Pulse, Alerts)
// ======================================================

function normalizeMetricsPayload(payload) {
    const data = (payload && typeof payload === "object") ? payload : {};
    const pulse = (data.pulse && typeof data.pulse === "object") ? data.pulse : {};
    const current = data.current || pulse.current || {};
    const rollingAverages = data.rolling_averages || pulse.rolling_averages || {};

    return {
        ...data,
        lastAlert: null,
        pulse: pulse.current ? pulse : {
            current,
            rolling_averages: rollingAverages,
            status: data.status,
            pulse_strength: data.pulse_strength,
            timestamp: data.timestamp
        },
        current,
        rolling_averages: rollingAverages,
        status: data.status || pulse.status || "SAFE"
    };
}


function buildRealtimeAlert(payload) {
    const wrapper = payload && typeof payload === "object" ? payload : {};
    const raw = wrapper.data && typeof wrapper.data === "object" ? wrapper.data : wrapper;

    const timestamp = raw.timestamp || wrapper.timestamp || new Date().toISOString();
    const sourceIp = raw.source_ip || raw.source || "";
    const baseId = raw.id || `rt_${sourceIp}_${timestamp}`;

    return {
        id: String(baseId).replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64),
        timestamp,
        severity: String(raw.severity || "low").toLowerCase(),
        prediction: raw.prediction || "unknown",
        confidence: Number.isFinite(Number(raw.confidence)) ? Number(raw.confidence) : 0,
        status: raw.status || "open",
        profile: raw.profile || "N/A",
        source_ip: sourceIp,
        attack_type: raw.attack_type || "",
        reason: raw.reason || ""
    };
}


// ======================================================
// GLOBAL ALERT UPSERT
// (Keeps only 200 most recent)
// ======================================================
function upsertAlert(alert) {
    const alerts = Array.isArray(GlobalState.data.alerts)
        ? GlobalState.data.alerts.slice()
        : [];

    const idx = alerts.findIndex(a => a.id === alert.id);
    if (idx >= 0) alerts.splice(idx, 1);

    alerts.unshift(alert);
    return alerts.slice(0, 200);
}


// ======================================================
// API HELPERS (hydration, fallback polling)
// ======================================================

async function parseJson(response) {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

async function hydrateFromApi() {
    const [metricsRes, alertsRes, pulseRes] = await Promise.allSettled([
        fetch("/api/dashboard/metrics").then(parseJson),
        fetch("/api/alerts?limit=200").then(parseJson),
        fetch("/api/perception/pulse").then(parseJson)
    ]);

    const nextState = {};

    if (metricsRes.status === "fulfilled") {
        Object.assign(nextState, metricsRes.value);
    }

    if (alertsRes.status === "fulfilled") {
        nextState.alerts = alertsRes.value.alerts || [];
        nextState.alertsCount = alertsRes.value.count || nextState.alerts.length;
    }

    if (pulseRes.status === "fulfilled") {
        Object.assign(
            nextState,
            normalizeMetricsPayload({
                pulse: pulseRes.value,
                current: pulseRes.value.current || {},
                rolling_averages: pulseRes.value.rolling_averages || {},
                status: pulseRes.value.status,
                pulse_strength: pulseRes.value.pulse_strength,
                timestamp: pulseRes.value.timestamp
            })
        );
    }

    if (Object.keys(nextState).length > 0) {
        GlobalState.set(nextState);
    }
}

function startFallbackPolling() {
    if (fallbackTimer) return;

    hydrateFromApi().catch(console.error);

    fallbackTimer = setInterval(() => {
        fetch("/api/perception/pulse")
            .then(parseJson)
            .then(data => {
                GlobalState.set(
                    normalizeMetricsPayload({
                        pulse: data,
                        current: data.current || {},
                        rolling_averages: data.rolling_averages || {},
                        status: data.status,
                        pulse_strength: data.pulse_strength,
                        timestamp: data.timestamp
                    })
                );
            })
            .catch(err => console.error("fallback pulse failed:", err));
    }, 5000);
}

function stopFallbackPolling() {
    if (fallbackTimer) {
        clearInterval(fallbackTimer);
        fallbackTimer = null;
    }
}


// ======================================================
// SOCKET EVENT HANDLERS
// ======================================================

socket.on("connect", () => {
    console.log("%c[WEBSOCKET] Connected", "color: #4caf50");
    stopFallbackPolling();

    GlobalState.set({ socketConnected: true });

    socket.emit("subscribe_metrics");
    socket.emit("subscribe_alerts");
});

socket.on("disconnect", () => {
    console.warn("[WEBSOCKET] Disconnected. Starting fallback.");
    GlobalState.set({ socketConnected: false });
    startFallbackPolling();
});

socket.on("connect_error", () => {
    console.warn("[WEBSOCKET] Connection error.");
    GlobalState.set({ socketConnected: false });
    startFallbackPolling();
});


// Metrics events
socket.on("metrics.update", payload => {
    stopFallbackPolling();
    GlobalState.set(normalizeMetricsPayload(payload));
});

// Alerts events
socket.on("alert.new", payload => {
    const alert = buildRealtimeAlert(payload);
    const alerts = upsertAlert(alert);

    GlobalState.set({
        lastAlert: alert,
        alerts,
        alertsCount: alerts.length,
        active_alerts: Math.min(alerts.length, 99)
    });
});


// ======================================================
// EXPORT PUBLIC API
// ======================================================

export const SocketCore = {
    socket,
    hydrate: hydrateFromApi,
    startFallback: startFallbackPolling,
    stopFallback: stopFallbackPolling,
    on: (event, handler) => socket.on(event, handler),
    emit: (event, data) => socket.emit(event, data)
};

// Register globally (debug/dev)
window.INIDSSocketManager = SocketCore;

export default SocketCore;
