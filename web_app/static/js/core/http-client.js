/**
 * HTTP Client - Safe Fetch Wrapper
 * 
 * - Handles all API requests with consistent error handling
 * - Validates response status and content-type
 * - Integrates with GlobalState for notifications
 * - Supports request/response interceptors
 * - Automatic retry logic for certain failures
 * 
 * Used by: Page controllers, Components
 */

import { GlobalState } from "./global-state.js";

// In-memory token store — never written to localStorage (FIX-022).
let _token = null;
let _refreshTimer = null;
let _refreshInFlight = null; // single-flight promise

class HttpClient {
    constructor() {
        this.baseUrl = "";
        this.timeout = 30000;  // 30 seconds
        this.retryAttempts = 3;
        this.retryDelay = 1000; // 1 second
    }

    /**
     * Configure HTTP client
     */
    configure(options = {}) {
        if (options.baseUrl) this.baseUrl = options.baseUrl;
        if (options.timeout) this.timeout = options.timeout;
        if (options.retryAttempts) this.retryAttempts = options.retryAttempts;
    }

    /**
     * Store JWT after login and schedule proactive refresh at 80 % of TTL.
     * @param {string} token
     * @param {number} expiresIn  - seconds until expiry (default 3600)
     */
    setToken(token, expiresIn = 3600) {
        _token = token;
        GlobalState.set("auth.token", token);

        if (_refreshTimer) clearTimeout(_refreshTimer);
        const refreshDelay = Math.floor(expiresIn * 0.8) * 1000; // 80 % → ms
        _refreshTimer = setTimeout(() => this._proactiveRefresh(), refreshDelay);
        console.log(`[HttpClient] Token stored; refresh scheduled in ${Math.round(refreshDelay / 1000)}s`);

        // Trigger socket reconnect now that a token is available.
        if (window.Socket && typeof window.Socket.reconnectWithToken === "function") {
            window.Socket.reconnectWithToken();
        }
    }

    /** Return the current in-memory token (null if not logged in). */
    getToken() { return _token; }

    /** Clear the stored token (logout). */
    clearToken() {
        _token = null;
        GlobalState.set("auth.token", null);
        if (_refreshTimer) { clearTimeout(_refreshTimer); _refreshTimer = null; }
        _refreshInFlight = null;
    }

    /**
     * Proactive refresh — called by the scheduled timer.
     * @private
     */
    async _proactiveRefresh() {
        try {
            await this._doRefresh();
        } catch (err) {
            console.warn("[HttpClient] Proactive refresh failed — redirecting to login:", err.message);
            this.clearToken();
            window.location.href = "/login";
        }
    }

    /**
     * Perform the refresh call; single-flight so concurrent 401s share one request.
     * @private
     */
    _doRefresh() {
        if (_refreshInFlight) return _refreshInFlight;

        _refreshInFlight = (async () => {
            if (!_token) throw new Error("No token to refresh");
            const controller = new AbortController();
            const tid = setTimeout(() => controller.abort(), 5000);
            try {
                const res = await fetch("/api/auth/refresh", {
                    method: "POST",
                    headers: { "Authorization": `Bearer ${_token}` },
                    signal: controller.signal,
                });
                if (!res.ok) throw new Error(`Refresh returned ${res.status}`);
                const data = await res.json();
                this.setToken(data.token, data.expires_in || 3600);
                console.log("[HttpClient] Token refreshed successfully");
            } finally {
                clearTimeout(tid);
                _refreshInFlight = null;
            }
        })();

        return _refreshInFlight;
    }

    /**
     * Make GET request
     * @param {string} url
     * @param {object} options - fetch options
     * @returns {Promise<object>} Parsed JSON response
     */
    async get(url, options = {}) {
        return this._request(url, { ...options, method: "GET" });
    }

    /**
     * Make POST request
     * @param {string} url
     * @param {object} data - Request body (will be JSON stringified)
     * @param {object} options - fetch options
     * @returns {Promise<object>} Parsed JSON response
     */
    async post(url, data = {}, options = {}) {
        return this._request(url, {
            ...options,
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...options.headers
            },
            body: JSON.stringify(data)
        });
    }

    /**
     * Make PUT request
     * @param {string} url
     * @param {object} data - Request body
     * @param {object} options - fetch options
     * @returns {Promise<object>} Parsed JSON response
     */
    async put(url, data = {}, options = {}) {
        return this._request(url, {
            ...options,
            method: "PUT",
            headers: {
                "Content-Type": "application/json",
                ...options.headers
            },
            body: JSON.stringify(data)
        });
    }

    /**
     * Make DELETE request
     * @param {string} url
     * @param {object} options - fetch options
     * @returns {Promise<object>} Parsed JSON response
     */
    async delete(url, options = {}) {
        return this._request(url, { ...options, method: "DELETE" });
    }

    /**
     * Make PATCH request
     * @param {string} url
     * @param {object} data - Request body
     * @param {object} options - fetch options
     * @returns {Promise<object>} Parsed JSON response
     */
    async patch(url, data = {}, options = {}) {
        return this._request(url, {
            ...options,
            method: "PATCH",
            headers: {
                "Content-Type": "application/json",
                ...options.headers
            },
            body: JSON.stringify(data)
        });
    }

    /**
     * Build headers, injecting Bearer token when stored.
     * @private
     */
    _buildHeaders(extra = {}) {
        const headers = { ...extra };
        if (_token && !headers["Authorization"]) {
            headers["Authorization"] = `Bearer ${_token}`;
        }
        return headers;
    }

    /**
     * Core request method with retry logic and 401 JWT-refresh intercept (FIX-022).
     * @private
     */
    async _request(url, fetchOptions = {}, attempt = 1, _afterRefresh = false) {
        const fullUrl = this._buildUrl(url);

        // Inject auth header for every request
        fetchOptions = {
            ...fetchOptions,
            headers: this._buildHeaders(fetchOptions.headers || {}),
        };

        try {
            // Fetch with timeout
            const response = await Promise.race([
                fetch(fullUrl, fetchOptions),
                this._timeout(this.timeout)
            ]);

            // Handle HTTP error status
            if (!response.ok) {
                const errorBody = await this._parseResponse(response);

                // On 401 token_expired: refresh once then retry (FIX-022)
                if (
                    response.status === 401 &&
                    !_afterRefresh &&
                    _token &&
                    errorBody &&
                    typeof errorBody === "object" &&
                    errorBody.error === "token_expired"
                ) {
                    try {
                        await this._doRefresh();
                        return this._request(url, { ...fetchOptions, headers: {} }, attempt, true);
                    } catch {
                        this.clearToken();
                        window.location.href = "/login";
                        throw new HttpError("Session expired — redirecting to login", 401, errorBody);
                    }
                }

                const error = new HttpError(
                    `HTTP ${response.status}: ${response.statusText}`,
                    response.status,
                    errorBody
                );

                // Retry on 503, 504, 429
                if ([503, 504, 429].includes(response.status) && attempt < this.retryAttempts) {
                    console.warn(`[HttpClient] Retrying after ${this.retryDelay}ms (attempt ${attempt}/${this.retryAttempts})`);
                    await this._delay(this.retryDelay);
                    return this._request(url, fetchOptions, attempt + 1, _afterRefresh);
                }

                throw error;
            }

            // Parse response
            const data = await this._parseResponse(response);
            return data;

        } catch (err) {
            // Handle timeout
            if (err instanceof TimeoutError) {
                console.error(`[HttpClient] Request timeout: ${fullUrl}`);
                throw new HttpError("Request timeout", 0, null, err);
            }

            // Handle network error
            if (err instanceof TypeError && err.message.includes("fetch")) {
                console.error(`[HttpClient] Network error: ${fullUrl}`, err);
                throw new HttpError("Network error - please check your connection", 0, null, err);
            }

            // Handle custom HTTP errors
            if (err instanceof HttpError) {
                throw err;
            }

            // Handle parsing errors
            console.error(`[HttpClient] Request failed: ${fullUrl}`, err);
            throw new HttpError("Request failed", 0, null, err);
        }
    }

    /**
     * Build full URL
     * @private
     */
    _buildUrl(url) {
        if (url.startsWith("http")) {
            return url;
        }
        return (this.baseUrl || "") + url;
    }

    /**
     * Parse response (JSON or text)
     * @private
     */
    async _parseResponse(response) {
        const contentType = response.headers.get("content-type");

        if (contentType && contentType.includes("application/json")) {
            return response.json();
        } else {
            return response.text();
        }
    }

    /**
     * Timeout helper
     * @private
     */
    _timeout(ms) {
        return new Promise((_, reject) => {
            setTimeout(() => {
                reject(new TimeoutError(`Request timeout after ${ms}ms`));
            }, ms);
        });
    }

    /**
     * Delay helper (for retries)
     * @private
     */
    _delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Custom error class for HTTP errors
 */
class HttpError extends Error {
    constructor(message, status = 0, data = null, cause = null) {
        super(message);
        this.name = "HttpError";
        this.status = status;
        this.data = data;
        this.cause = cause;
    }
}

/**
 * Custom error class for timeouts
 */
class TimeoutError extends Error {
    constructor(message) {
        super(message);
        this.name = "TimeoutError";
    }
}

// Create and export singleton
export const HttpClient_Instance = new HttpClient();

// Also export class for testing
export { HttpClient, HttpError, TimeoutError };

// Make available globally; Socket reads window.HttpClient._token (FIX-005/FIX-022)
window.HttpClient = HttpClient_Instance;
Object.defineProperty(window.HttpClient, "_token", { get: () => _token });

console.log("[HttpClient] Initialized");
