/**
 * Socket Manager v2 - Unified WebSocket + Fallback Polling
 * 
 * - Single socket connection per app lifecycle
 * - Graceful fallback to polling on disconnect
 * - Exponential backoff strategy
 * - Integrates with GlobalState for state management
 * 
 * Replaces: socket.js (legacy IIFE)
 * Used by: All pages via GlobalState subscriptions
 */

import { GlobalState } from "./global-state.js";

class SocketManager {
    constructor() {
        this.socket = null;
        this.pollTimer = null;
        this.pollDelay = 5000;  // Start: 5 seconds
        this.maxDelay = 60000;   // Max: 60 seconds
        this.isConnected = false;
        
        // Prevent duplicate initialization
        if (window.INIDSSocketManager) {
            console.warn("[SocketManager] Already initialized");
            return;
        }
        
        this.connect();
        window.INIDSSocketManager = this;
    }

    /**
     * Read the current in-memory JWT (set by HttpClient after login/refresh).
     * @private
     */
    _getToken() {
        return GlobalState.data["auth.token"] || null;
    }

    /**
     * Establish WebSocket connection to /events namespace
     */
    connect() {
        if (!window.io) {
            console.error("[SocketManager] Socket.IO client not loaded");
            this.startFallbackPolling();
            return;
        }

        const token = this._getToken();
        if (!token) {
            console.warn("[SocketManager] No JWT available — deferring connection until login");
            return;
        }

        this.socket = window.io("/events", {
            transports: ["websocket", "polling"],
            reconnection: true,
            reconnectionDelayMax: 8000,
            reconnectionAttempts: 10,
            auth: { token },
        });

        // Connection established
        this.socket.on("connect", () => {
            console.log("[SocketManager] Connected to /events");
            this.isConnected = true;
            this.stopFallbackPolling();
            
            // Notify GlobalState
            GlobalState.set("socket.connected", true);
        });

        // Connection lost
        this.socket.on("disconnect", (reason) => {
            console.warn("[SocketManager] Disconnected:", reason);
            this.isConnected = false;
            GlobalState.set("socket.connected", false);
            
            // Only fallback for certain disconnection reasons
            if (reason === "io server disconnect" || reason === "io client namespace disconnect") {
                console.log("[SocketManager] Starting fallback polling...");
                this.startFallbackPolling();
            }
        });

        // Reconnection attempt
        this.socket.on("reconnect_attempt", () => {
            console.log("[SocketManager] Attempting to reconnect...");
            GlobalState.set("socket.reconnecting", true);
        });

        // Connection error
        this.socket.on("error", (error) => {
            console.error("[SocketManager] Connection error:", error);
        });

        // ===== REAL-TIME EVENT HANDLERS =====

        /**
         * alert.new - New detection alert
         */
        this.socket.on("alert.new", (payload) => {
            try {
                const normalized = this._normalizeAlert(payload);
                GlobalState.push("alerts", normalized);
            } catch (err) {
                console.error("[SocketManager] Error processing alert.new:", err);
            }
        });

        /**
         * metrics.update - System metrics snapshot
         */
        this.socket.on("metrics.update", (payload) => {
            try {
                const normalized = this._normalizeMetrics(payload);
                GlobalState.set("metrics", normalized);
            } catch (err) {
                console.error("[SocketManager] Error processing metrics.update:", err);
            }
        });

        /**
         * module.update - Capability module state change
         */
        this.socket.on("module.update", (payload) => {
            try {
                if (payload && payload.module_id && payload.data) {
                    const modules = GlobalState.data.modules || {};
                    modules[payload.module_id] = payload.data;
                    GlobalState.set("modules", modules);
                }
            } catch (err) {
                console.error("[SocketManager] Error processing module.update:", err);
            }
        });

        /**
         * action.update - Response action execution
         */
        this.socket.on("action.update", (payload) => {
            try {
                const normalized = this._normalizeAction(payload);
                GlobalState.push("actions", normalized);
            } catch (err) {
                console.error("[SocketManager] Error processing action.update:", err);
            }
        });

        /**
         * policy.update - Security policy change
         */
        this.socket.on("policy.update", (payload) => {
            try {
                GlobalState.set("policy", payload || {});
            } catch (err) {
                console.error("[SocketManager] Error processing policy.update:", err);
            }
        });

        /**
         * health.update - System health status
         */
        this.socket.on("health.update", (payload) => {
            try {
                GlobalState.set("health", payload || {});
            } catch (err) {
                console.error("[SocketManager] Error processing health.update:", err);
            }
        });

        /**
         * engine.update - ML engine status change
         */
        this.socket.on("engine.update", (payload) => {
            try {
                if (payload && payload.engine_id) {
                    const engines = GlobalState.data.engines || [];
                    const idx = engines.findIndex(e => e.engine_id === payload.engine_id);
                    if (idx >= 0) {
                        engines[idx] = { ...engines[idx], ...payload };
                    } else {
                        engines.push(payload);
                    }
                    GlobalState.set("engines", engines);
                }
            } catch (err) {
                console.error("[SocketManager] Error processing engine.update:", err);
            }
        });
    }

    /**
     * Start exponential backoff polling when WebSocket is unavailable
     * Polls /api/perception/pulse for metrics + /api/alerts for recent alerts
     */
    startFallbackPolling() {
        if (this.pollTimer) {
            console.log("[SocketManager] Fallback polling already active");
            return;
        }

        console.log("[SocketManager] Starting fallback polling (delay: " + this.pollDelay + "ms)");
        const poll = async () => {
            try {
                // Fetch metrics from perception service
                const metricsRes = await fetch("/api/perception/pulse");
                if (!metricsRes.ok) throw new Error(`Metrics fetch failed: ${metricsRes.status}`);
                
                const metricsData = await metricsRes.json();
                const normalized = this._normalizeMetrics(metricsData);
                GlobalState.set("metrics", normalized);

                // Reset backoff on successful poll
                this.pollDelay = 5000;
                console.log("[SocketManager] Fallback poll successful, resetting backoff");

            } catch (err) {
                console.warn("[SocketManager] Fallback polling failed:", err.message);
                
                // Increase backoff delay with exponential strategy
                this.pollDelay = Math.min(this.pollDelay * 2, this.maxDelay);
                console.log("[SocketManager] Backoff increased to:", this.pollDelay + "ms");
            }

            // Schedule next poll
            this.pollTimer = setTimeout(poll, this.pollDelay);
        };

        // Start polling immediately
        poll();
    }

    /**
     * Stop fallback polling (called when WebSocket reconnects)
     */
    stopFallbackPolling() {
        if (this.pollTimer) {
            clearTimeout(this.pollTimer);
            this.pollTimer = null;
            this.pollDelay = 5000;  // Reset for next time
            console.log("[SocketManager] Fallback polling stopped, backoff reset");
        }
    }

    /**
     * Normalize alert payload to consistent schema
     */
    _normalizeAlert(payload) {
        const wrapper = (payload && typeof payload === "object") ? payload : {};
        const raw = (wrapper.data && typeof wrapper.data === "object") ? wrapper.data : wrapper;

        const timestamp = raw.timestamp || wrapper.timestamp || new Date().toISOString();
        const sourceIp = raw.source_ip || raw.source || "";
        const baseId = raw.id || `rt_${sourceIp}_${Date.now()}`;

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

    /**
     * Normalize metrics payload to consistent schema
     */
    _normalizeMetrics(payload) {
        const data = (payload && typeof payload === "object") ? payload : {};
        const pulse = (data.pulse && typeof data.pulse === "object") ? data.pulse : {};
        const current = data.current || pulse.current || {};
        const rollingAverages = data.rolling_averages || pulse.rolling_averages || {};

        return {
            ...data,
            current,
            rolling_averages: rollingAverages,
            pulse: pulse.current ? pulse : {
                current,
                rolling_averages: rollingAverages,
                status: data.status,
                pulse_strength: data.pulse_strength,
                timestamp: data.timestamp
            },
            status: data.status || pulse.status || "SAFE",
            timestamp: data.timestamp || new Date().toISOString()
        };
    }

    /**
     * Normalize action payload
     */
    _normalizeAction(payload) {
        const data = (payload && typeof payload === "object") ? payload : {};
        return {
            id: data.id || `action_${Date.now()}`,
            timestamp: data.timestamp || new Date().toISOString(),
            type: data.type || "unknown",
            target: data.target || "",
            status: data.status || "pending",
            result: data.result || null
        };
    }

    /**
     * Public API: Emit custom event to server
     */
    emit(event, data) {
        if (this.isConnected && this.socket) {
            this.socket.emit(event, data);
        } else {
            console.warn("[SocketManager] Cannot emit - socket not connected:", event);
        }
    }

    /**
     * Public API: Listen for custom events
     */
    on(event, callback) {
        if (this.socket) {
            this.socket.on(event, callback);
        }
    }

    /**
     * Public API: Remove event listener
     */
    off(event, callback) {
        if (this.socket) {
            this.socket.off(event, callback);
        }
    }

    /**
     * Public API: Check connection status
     */
    getStatus() {
        return {
            connected: this.isConnected,
            isPolling: !!this.pollTimer,
            pollDelay: this.pollDelay,
            socketReady: !!this.socket
        };
    }

    /**
     * Public API: (Re)connect after a token becomes available (called by HttpClient post-login).
     */
    reconnectWithToken() {
        if (this.isConnected) return;
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        this.connect();
    }
}

// Create and export singleton instance
export const Socket = new SocketManager();

// Make available globally for debugging
window.Socket = Socket;

console.log("[SocketManager] Initialized");
