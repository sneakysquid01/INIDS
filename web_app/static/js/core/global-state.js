/**
 * Global State v2 - Unified Application State Management
 * 
 * - Slice-based state organization (metrics, alerts, modules, etc.)
 * - Observer pattern with per-slice subscriptions
 * - Prevents race conditions via sliced observables
 * - Central hub for all data flowing through the app
 * 
 * Replaces: state.js (old global singleton)
 * Used by: SocketManager, Page controllers, Components
 */

export const GlobalState = {
    // ===== DATA SLICES =====
    data: {
        // Metrics & Health
        metrics: {},
        health: {},
        pulse: {},
        
        // Alerts & Actions
        alerts: [],
        actions: [],
        
        // Configuration
        policy: {},
        allowlist: [],
        models: [],
        
        // Modules & Engines
        modules: [],
        engines: [],
        
        // Honeypot
        honeypot: {},
        honeypots: [],

        // Investigations & Response
        investigations: [],
        playbooks: [],

        // Threat Intelligence (page uses this key)
        threat_intel: [],

        // Connection status
        socket: {
            connected: false,
            reconnecting: false
        }
    },

    // ===== LISTENERS =====
    // Map: slice_name -> [callback, callback, ...]
    listeners: {},

    /**
     * Subscribe to changes in a specific data slice
     * 
     * Returns: callback immediately with current value
     * 
     * @param {string} key - Data slice name (e.g., "alerts", "metrics")
     * @param {function} callback - Called with (value, key) when data changes
     * @returns {function} Unsubscribe function
     */
    subscribe(key, callback) {
        if (!this.listeners[key]) {
            this.listeners[key] = [];
        }

        // Store callback
        this.listeners[key].push(callback);

        // Call immediately with current value (snapshot)
        callback(this.data[key], key);

        // Return unsubscribe function
        return () => {
            const idx = this.listeners[key].indexOf(callback);
            if (idx >= 0) {
                this.listeners[key].splice(idx, 1);
            }
        };
    },

    /**
     * Set entire slice to new value
     * Notifies all subscribers to this slice
     * 
     * @param {string} key - Data slice name
     * @param {*} value - New value (replaces old)
     */
    set(key, value) {
        // Update data
        this.data[key] = value;

        // Notify all listeners for this slice
        if (this.listeners[key]) {
            this.listeners[key].forEach(callback => {
                try {
                    callback(value, key);
                } catch (err) {
                    console.error(`[GlobalState] Error in listener for '${key}':`, err);
                }
            });
        }

        // Log state changes (dev mode)
        if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === "development") {
            console.log(`[GlobalState] SET '${key}':`, value);
        }
    },

    /**
     * Merge partial update into existing slice
     * For objects: shallow merge
     * Notifies all subscribers to this slice
     * 
     * @param {string} key - Data slice name
     * @param {object} partialValue - Partial update object
     */
    update(key, partialValue) {
        // Only works with object-type slices
        if (typeof this.data[key] !== "object" || this.data[key] === null || Array.isArray(this.data[key])) {
            console.warn(`[GlobalState] Cannot update non-object slice '${key}'`);
            return;
        }

        // Shallow merge
        this.data[key] = { ...this.data[key], ...partialValue };

        // Notify all listeners for this slice
        if (this.listeners[key]) {
            this.listeners[key].forEach(callback => {
                try {
                    callback(this.data[key], key);
                } catch (err) {
                    console.error(`[GlobalState] Error in listener for '${key}':`, err);
                }
            });
        }

        // Log state changes (dev mode)
        if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === "development") {
            console.log(`[GlobalState] UPDATE '${key}':`, partialValue);
        }
    },

    /**
     * Push item to beginning of array slice
     * Array kept at max 200 items (LRU)
     * Notifies all subscribers to this slice
     * 
     * @param {string} key - Data slice name (must be array)
     * @param {*} entry - Item to prepend
     */
    push(key, entry) {
        // Only works with array-type slices
        if (!Array.isArray(this.data[key])) {
            console.warn(`[GlobalState] Cannot push to non-array slice '${key}'`);
            return;
        }

        // Prepend to array
        this.data[key].unshift(entry);

        // Keep max 200 items (LRU)
        if (this.data[key].length > 200) {
            this.data[key] = this.data[key].slice(0, 200);
        }

        // Notify all listeners for this slice
        if (this.listeners[key]) {
            this.listeners[key].forEach(callback => {
                try {
                    callback(this.data[key], key);
                } catch (err) {
                    console.error(`[GlobalState] Error in listener for '${key}':`, err);
                }
            });
        }

        // Log state changes (dev mode)
        if (typeof process !== 'undefined' && process.env && process.env.NODE_ENV === "development") {
            console.log(`[GlobalState] PUSH to '${key}':`, entry);
        }
    },

    /**
     * Clear slice (set to initial empty value)
     * 
     * @param {string} key - Data slice name
     */
    clear(key) {
        const initialValues = {
            metrics: {},
            health: {},
            alerts: [],
            actions: [],
            policy: {},
            allowlist: [],
            models: [],
            modules: [],
            engines: [],
            investigations: [],
            playbooks: [],
            honeypot: {},
            honeypots: [],
            threat_intel: [],
        };

        this.data[key] = initialValues[key] || null;

        if (this.listeners[key]) {
            this.listeners[key].forEach(callback => {
                try {
                    callback(this.data[key], key);
                } catch (err) {
                    console.error(`[GlobalState] Error in listener for '${key}':`, err);
                }
            });
        }
    },

    /**
     * Get snapshot of entire state
     * @returns {object} Shallow copy of data object
     */
    getSnapshot() {
        return { ...this.data };
    },

    /**
     * Get snapshot of specific slice
     * @param {string} key - Data slice name
     * @returns {*} Current value of slice
     */
    get(key) {
        return this.data[key];
    },

    /**
     * Reset all state to initial values
     */
    reset() {
        Object.keys(this.data).forEach(key => {
            this.clear(key);
        });
        if (window.__INIDS_DEBUG__) console.log("[GlobalState] Reset to initial state");
    },

    /**
     * Debug: Print current state
     */
    debug() {
        console.table({
            alerts: this.data.alerts.length,
            actions: this.data.actions.length,
            modules: Object.keys(this.data.modules).length,
            engines: this.data.engines.length,
            policy: !!this.data.policy.id,
            connected: this.data.socket.connected
        });
    }
};

// Make available globally for debugging
window.GlobalState = GlobalState;
