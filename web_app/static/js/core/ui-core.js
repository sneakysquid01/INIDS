// ======================================================
// UI CORE — Toasts, Modals, Loaders, DOM Helpers
// ======================================================

export function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `alert alert-${type} position-fixed bottom-0 end-0 m-3`;
    toast.style.zIndex = "9999";
    toast.innerHTML = `
        ${message}
        <button class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 6000);
}

export function showError(msg) {
    showToast(msg, "danger");
}

export function showSuccess(msg) {
    showToast(msg, "success");
}

export function hideLoader(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
}

export function showLoader(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = "block";
}
