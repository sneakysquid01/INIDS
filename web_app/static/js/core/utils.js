// ======================================================
// UTILITY FUNCTIONS (Shared across ALL INIDS pages)
// ======================================================

// Smooth number transitions (for metrics, counters)
export function smoothNumber(el, target) {
    let current = Number(el.textContent) || 0;
    let step = (target - current) / 10;

    let i = 0;
    let timer = setInterval(() => {
        i++;
        current += step;
        el.textContent = Math.round(current);
        if (i >= 10) clearInterval(timer);
    }, 25);
}

// Fade-in animation for cards/rows
export function fadeIn(el) {
    el.style.opacity = 0;
    el.style.transition = "opacity 0.3s ease";
    requestAnimationFrame(() => el.style.opacity = 1);
}

// Metric bar animation
export function animateBar(bar, newValue) {
    bar.style.transition = "width 0.3s ease";
    bar.style.width = Math.min(newValue, 100) + "%";
}

// Alert tones with severity
export function playAlertTone(level) {
    try {
        const file =
            level === "high" ? "/static/sfx/alert_high.mp3" :
            level === "medium" ? "/static/sfx/alert_med.mp3" :
            "/static/sfx/alert_low.mp3";

        const audio = new Audio(file);
        audio.volume = level === "high" ? 1 : 0.4;
        audio.play();
    } catch (err) {
        console.warn("Audio error:", err);
    }
}

// Safe HTML escaping
export function escapeHTML(text) {
    if (!text) return "";
    return text.replace(/[&<>"']/g, m => ({
        "&": "&",
        "<": "<",
        ">": ">",
        "\"": """,
        "'": "'"
    }[m]));
}

// Date formatter
export function formatTimestamp(ts) {
    if (!ts) return "N/A";
    try {
        const d = new Date(ts);
        return d.toLocaleString();
    } catch {
        return ts;
    }
}
