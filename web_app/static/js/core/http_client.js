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

import { GlobalState } from "./global_state.js";

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
     * Core request method with retry logic
     * @private
     */
    async _request(url, fetchOptions = {}, attempt = 1) {
        const fullUrl = this._buildUrl(url);

        try {
            // Fetch with timeout
            const response = await Promise.race([
                fetch(fullUrl, fetchOptions),
                this._timeout(this.timeout)
            ]);

            // Handle HTTP error status
            if (!response.ok) {
                const errorBody = await this._parseResponse(response);
                const error = new HttpError(
                    `HTTP ${response.status}: ${response.statusText}`,
                    response.status,
                    errorBody
                );

                // Retry on 503, 504, 429
                if ([503, 504, 429].includes(response.status) && attempt < this.retryAttempts) {
                    console.warn(`[HttpClient] Retrying after ${this.retryDelay}ms (attempt ${attempt}/${this.retryAttempts})`);
                    await this._delay(this.retryDelay);
                    return this._request(url, fetchOptions, attempt + 1);
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

// Make available globally for debugging
window.HttpClient = HttpClient_Instance;

console.log("[HttpClient] Initialized");
