/**
 * Utility Helpers - Common functions across the app
 * 
 * - DOM query helpers
 * - Formatting utilities
 * - Date/time helpers
 * - String utilities
 * - Validation helpers
 * - Event helpers
 * - Animation helpers
 * 
 * Used by: All pages, components
 */

// ===== DOM QUERY HELPERS =====

/**
 * Safe querySelector wrapper
 * @param {string} selector - CSS selector
 * @param {Element} parent - Parent element (default: document)
 * @returns {Element|null}
 */
export function querySelector(selector, parent = document) {
    try {
        return parent.querySelector(selector);
    } catch (err) {
        console.warn(`[utils] Invalid selector: ${selector}`, err);
        return null;
    }
}

/**
 * Safe querySelectorAll wrapper
 * @param {string} selector - CSS selector
 * @param {Element} parent - Parent element (default: document)
 * @returns {NodeList}
 */
export function querySelectorAll(selector, parent = document) {
    try {
        return parent.querySelectorAll(selector);
    } catch (err) {
        console.warn(`[utils] Invalid selector: ${selector}`, err);
        return [];
    }
}

/**
 * Get element by ID
 * @param {string} id
 * @returns {Element|null}
 */
export function getElementById(id) {
    return document.getElementById(id);
}

/**
 * Create element with optional classes and attributes
 * @param {string} tag - HTML tag name
 * @param {object} options - {classes: [], attrs: {}, content: ""}
 * @returns {Element}
 */
export function createElement(tag, options = {}) {
    const el = document.createElement(tag);
    
    if (options.classes) {
        el.classList.add(...options.classes);
    }
    
    if (options.attrs) {
        Object.entries(options.attrs).forEach(([key, value]) => {
            el.setAttribute(key, value);
        });
    }
    
    if (options.content) {
        if (typeof options.content === "string") {
            el.textContent = options.content;
        } else {
            el.appendChild(options.content);
        }
    }
    
    return el;
}

/**
 * Add classes to element
 * @param {Element} el
 * @param {string|string[]} classes - Class name(s)
 */
export function addClass(el, classes) {
    if (!el) return;
    if (typeof classes === "string") {
        el.classList.add(classes);
    } else {
        el.classList.add(...classes);
    }
}

/**
 * Remove classes from element
 * @param {Element} el
 * @param {string|string[]} classes - Class name(s)
 */
export function removeClass(el, classes) {
    if (!el) return;
    if (typeof classes === "string") {
        el.classList.remove(classes);
    } else {
        el.classList.remove(...classes);
    }
}

/**
 * Toggle class on element
 * @param {Element} el
 * @param {string} className
 * @param {boolean} force - Optional force add/remove
 */
export function toggleClass(el, className, force) {
    if (!el) return;
    el.classList.toggle(className, force);
}

// ===== DATE/TIME HELPERS =====

/**
 * Format ISO timestamp to readable string
 * @param {string} isoString - ISO 8601 timestamp
 * @param {boolean} includeTime - Include time (default: true)
 * @returns {string} Formatted date
 */
export function formatDate(isoString, includeTime = true) {
    try {
        const date = new Date(isoString);
        if (isNaN(date.getTime())) return "Invalid date";
        
        const options = includeTime 
            ? { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
            : { month: "short", day: "numeric", year: "numeric" };
        
        return date.toLocaleDateString("en-US", options);
    } catch (err) {
        return "Invalid date";
    }
}

/**
 * Format timestamp to relative time (e.g., "2 hours ago")
 * @param {string|number} timestamp - ISO string or milliseconds
 * @returns {string} Relative time
 */
export function formatRelativeTime(timestamp) {
    try {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        
        const seconds = Math.floor(diffMs / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        
        if (seconds < 60) return "just now";
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        if (days < 7) return `${days}d ago`;
        
        return formatDate(timestamp, false);
    } catch (err) {
        return "Unknown";
    }
}

/**
 * Get current ISO timestamp
 * @returns {string}
 */
export function now() {
    return new Date().toISOString();
}

/**
 * Legacy: formatTimestamp (for backwards compatibility)
 */
export function formatTimestamp(ts) {
    if (!ts) return "N/A";
    try {
        const d = new Date(ts);
        return d.toLocaleString();
    } catch {
        return ts;
    }
}

// ===== STRING UTILITIES =====

/**
 * Capitalize first letter
 * @param {string} str
 * @returns {string}
 */
export function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Convert snake_case to camelCase
 * @param {string} str
 * @returns {string}
 */
export function toCamelCase(str) {
    return str.replace(/_([a-z])/g, (g) => g[1].toUpperCase());
}

/**
 * Convert camelCase to snake_case
 * @param {string} str
 * @returns {string}
 */
export function toSnakeCase(str) {
    return str.replace(/([A-Z])/g, "_$1").toLowerCase();
}

/**
 * Truncate string with ellipsis
 * @param {string} str
 * @param {number} maxLength
 * @returns {string}
 */
export function truncate(str, maxLength = 50) {
    if (!str || str.length <= maxLength) return str;
    return str.slice(0, maxLength - 3) + "...";
}

/**
 * Safe HTML escaping
 * @param {string} text
 * @returns {string}
 */
export function escapeHTML(text) {
    if (!text) return "";
    return text.replace(/[&<>"']/g, m => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#39;"
    }[m]));
}

/**
 * Sanitize HTML (escape special characters)
 * @param {string} str
 * @returns {string}
 */
export function escapeHtml(str) {
    return escapeHTML(str);  // Alias
}

// ===== NUMBER UTILITIES =====

/**
 * Format number as percentage
 * @param {number} value
 * @param {number} decimals - Decimal places
 * @returns {string}
 */
export function formatPercent(value, decimals = 1) {
    if (!Number.isFinite(value)) return "N/A";
    return (value * 100).toFixed(decimals) + "%";
}

/**
 * Format bytes to human-readable size
 * @param {number} bytes
 * @param {number} decimals
 * @returns {string}
 */
export function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return "0 B";
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

/**
 * Format number with commas
 * @param {number} num
 * @returns {string}
 */
export function formatNumber(num) {
    if (!Number.isFinite(num)) return "N/A";
    return num.toLocaleString();
}

// ===== VALIDATION HELPERS =====

/**
 * Validate IP address
 * @param {string} ip
 * @returns {boolean}
 */
export function isValidIp(ip) {
    const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}$/;
    if (!ipv4Regex.test(ip)) return false;
    
    const parts = ip.split(".");
    return parts.every(part => {
        const num = parseInt(part, 10);
        return num >= 0 && num <= 255;
    });
}

/**
 * Validate email
 * @param {string} email
 * @returns {boolean}
 */
export function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Check if object is empty
 * @param {object} obj
 * @returns {boolean}
 */
export function isEmpty(obj) {
    if (!obj) return true;
    if (typeof obj === "object") {
        return Object.keys(obj).length === 0;
    }
    return false;
}

/**
 * Deep clone object
 * @param {object} obj
 * @returns {object}
 */
export function deepClone(obj) {
    try {
        return JSON.parse(JSON.stringify(obj));
    } catch (err) {
        console.warn("[utils] Deep clone failed:", err);
        return obj;
    }
}

// ===== ARRAY UTILITIES =====

/**
 * Remove duplicates from array
 * @param {array} arr
 * @param {string} key - Optional: property to deduplicate by
 * @returns {array}
 */
export function unique(arr, key = null) {
    if (!Array.isArray(arr)) return [];
    
    if (key) {
        const seen = new Set();
        return arr.filter(item => {
            const val = item[key];
            if (seen.has(val)) return false;
            seen.add(val);
            return true;
        });
    }
    
    return [...new Set(arr)];
}

/**
 * Chunk array into smaller arrays
 * @param {array} arr
 * @param {number} size - Chunk size
 * @returns {array} Array of arrays
 */
export function chunk(arr, size) {
    const result = [];
    for (let i = 0; i < arr.length; i += size) {
        result.push(arr.slice(i, i + size));
    }
    return result;
}

// ===== EVENT HELPERS =====

/**
 * Debounce function execution
 * @param {function} func
 * @param {number} delay - Milliseconds
 * @returns {function} Debounced function
 */
export function debounce(func, delay = 300) {
    let timeoutId = null;
    return function debounced(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

/**
 * Throttle function execution
 * @param {function} func
 * @param {number} limit - Milliseconds
 * @returns {function} Throttled function
 */
export function throttle(func, limit = 300) {
    let inThrottle = false;
    return function throttled(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => {
                inThrottle = false;
            }, limit);
        }
    };
}

/**
 * Wait for element to appear in DOM
 * @param {string} selector
 * @param {number} timeout - Milliseconds (0 = no timeout)
 * @returns {Promise<Element|null>}
 */
export async function waitForElement(selector, timeout = 0) {
    return new Promise((resolve) => {
        const el = querySelector(selector);
        if (el) {
            resolve(el);
            return;
        }

        const observer = new MutationObserver(() => {
            const el = querySelector(selector);
            if (el) {
                observer.disconnect();
                clearTimeout(timeoutId);
                resolve(el);
            }
        });

        observer.observe(document, { subtree: true, childList: true });

        let timeoutId;
        if (timeout > 0) {
            timeoutId = setTimeout(() => {
                observer.disconnect();
                resolve(null);
            }, timeout);
        }
    });
}

// ===== ANIMATION HELPERS =====

/**
 * Smooth number transitions (for metrics, counters)
 * @param {Element} el
 * @param {number} target
 * @param {number} duration - Milliseconds
 */
export function smoothNumber(el, target, duration = 250) {
    if (!el) return;
    let current = Number(el.textContent) || 0;
    let step = (target - current) / (duration / 25);

    let i = 0;
    let timer = setInterval(() => {
        i++;
        current += step;
        el.textContent = Math.round(current);
        if (i >= 10) clearInterval(timer);
    }, 25);
}

/**
 * Fade-in animation for cards/rows
 * @param {Element} el
 * @param {number} duration - Milliseconds
 */
export function fadeIn(el, duration = 300) {
    if (!el) return;
    el.style.opacity = 0;
    el.style.transition = `opacity ${duration}ms ease`;
    requestAnimationFrame(() => el.style.opacity = 1);
}

/**
 * Metric bar animation
 * @param {Element} bar
 * @param {number} newValue - Percentage (0-100)
 * @param {number} duration - Milliseconds
 */
export function animateBar(bar, newValue, duration = 300) {
    if (!bar) return;
    bar.style.transition = `width ${duration}ms ease`;
    bar.style.width = Math.min(newValue, 100) + "%";
}

/**
 * Alert tones with severity
 * @param {string} level - "high", "medium", "low"
 */
export function playAlertTone(level) {
    const file =
        level === "high" ? "/static/sfx/alert_high.wav" :
        level === "medium" ? "/static/sfx/alert_med.wav" :
        "/static/sfx/alert_low.wav";
    try {
        const audio = new Audio(file);
        audio.volume = level === "high" ? 1 : 0.4;
        audio.play().catch(() => { /* noop — autoplay policy or missing file */ });
    } catch (_) { /* noop */ }
}
