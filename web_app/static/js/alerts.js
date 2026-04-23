/**
 * Alerts Page - SOC Enhanced Version
 */

let currentAlerts = [];
let currentAlert = null;
let alertsModal, statusModal;
let lastRealtimeAlertKey = null;

document.addEventListener('DOMContentLoaded', function () {
    alertsModal = new bootstrap.Modal(document.getElementById('detailsModal'));
    statusModal = new bootstrap.Modal(document.getElementById('statusModal'));

    document.getElementById('btn-refresh').addEventListener('click', requestSharedState);
    document.getElementById('severity-filter').addEventListener('change', loadAlerts);
    document.getElementById('status-filter').addEventListener('change', loadAlerts);

    document.getElementById('btn-update-status').addEventListener('click', () => {
        alertsModal.hide();
        statusModal.show();
    });

    document.getElementById('btn-save-status').addEventListener('click', saveAlertStatus);

    GlobalState.subscribe(data => {
        syncAlertsFromState(data);
    });

    loadAlerts();
});

/**
 * Load alerts
 */
function loadAlerts() {
    showLoading(true);

    if (Array.isArray(GlobalState.data.alerts)) {
        currentAlerts = filterAlerts(GlobalState.data.alerts);
        updateAlertCount(currentAlerts.length);
        renderAlerts(currentAlerts);
        showLoading(false);
        return;
    }

    requestSharedState();
}

function requestSharedState() {
    showLoading(true);

    if (window.INIDSSocketManager && typeof window.INIDSSocketManager.hydrate === 'function') {
        window.INIDSSocketManager.hydrate().catch(error => {
            console.error('Error hydrating shared alerts state:', error);
            showError('Failed to load alerts.');
            showLoading(false);
        });
        return;
    }

    showError('Failed to load alerts.');
    showLoading(false);
}

function syncAlertsFromState(data) {
    if (!data || typeof data !== 'object') {
        return;
    }

    if (Array.isArray(data.alerts)) {
        currentAlerts = filterAlerts(data.alerts);
        updateAlertCount(currentAlerts.length);
        renderAlerts(currentAlerts);
        showLoading(false);
    }

    if (data.lastAlert) {
        const key = `${data.lastAlert.id}:${data.lastAlert.timestamp}`;
        if (key !== lastRealtimeAlertKey) {
            lastRealtimeAlertKey = key;
            appendAlertRow(data.lastAlert);
        }
    }
}

function filterAlerts(alerts) {
    const severity = document.getElementById('severity-filter').value || '';
    const status = document.getElementById('status-filter').value || '';

    return (alerts || []).filter(alert => {
        const severityMatch = !severity || String(alert.severity || '').toLowerCase() === severity.toLowerCase();
        const statusMatch = !status || String(alert.status || '').toLowerCase() === status.toLowerCase();
        return severityMatch && statusMatch;
    });
}

/**
 * Render alerts (SOC upgraded)
 */
function renderAlerts(alerts) {
    const wrapper = document.getElementById('alerts-wrapper');
    const tbody = document.getElementById('alerts-body');
    const emptyState = document.getElementById('empty-state');

    tbody.innerHTML = '';

    if (!alerts || alerts.length === 0) {
        wrapper.style.display = 'none';
        emptyState.style.display = 'block';
        return;
    }

    wrapper.style.display = 'block';
    emptyState.style.display = 'none';

    alerts.forEach(alert => {
        tbody.appendChild(createAlertRow(alert));
    });
}

function createAlertRow(alert) {
    const row = document.createElement('tr');
    const severity = (alert.severity || 'low').toLowerCase();
    const status = (alert.status || 'open').toLowerCase();
    const confidence = parseFloat(alert.confidence || 0);

    if (severity === 'critical') {
        row.classList.add('sev-critical-row', 'pulse-soft');
    } else if (severity === 'high') {
        row.classList.add('sev-high-row');
    }

    row.onclick = () => showAlertDetails(alert);
    row.innerHTML = `
        <td class="mono-val">${escapeHtml(alert.id?.substring(0, 12))}</td>

        <td>${formatTimestamp(alert.timestamp)}</td>

        <td>
            <span class="sev-${severity}">
                ${severity}
            </span>
        </td>

        <td>${escapeHtml(alert.prediction || 'unknown')}</td>

        <td>
            <span class="mono-val">${(confidence * 100).toFixed(0)}%</span>
        </td>

        <td>
            <span class="status-${status}">
                ${status}
            </span>
        </td>

        <td>${escapeHtml(alert.profile || 'N/A')}</td>

        <td>
            <button class="btn-danger" onclick="event.stopPropagation(); blockAlert('${alert.id}')">
                Block
            </button>
        </td>
    `;

    return row;
}

function appendAlertRow(alert) {
    if (!alert) {
        return;
    }

    const normalizedAlert = {
        ...alert,
        confidence: Number.isFinite(Number(alert.confidence)) ? Number(alert.confidence) : 0,
        status: alert.status || 'open',
        profile: alert.profile || 'N/A'
    };

    if (!filterAlerts([normalizedAlert]).length) {
        return;
    }

    if (currentAlerts.some(item => item.id === normalizedAlert.id)) {
        return;
    }

    currentAlerts = [normalizedAlert, ...currentAlerts].slice(0, 200);
    updateAlertCount(currentAlerts.length);
    renderAlerts(currentAlerts);
}

/**
 * 🔥 Block alert (NEW)
 */
async function blockAlert(alertId) {
    if (!confirm(`Block alert ${alertId}?`)) return;

    try {
        const res = await fetch(`/api/block/${alertId}`, {
            method: 'POST'
        });

        if (!res.ok) throw new Error("Block failed");

        showSuccess("Alert blocked");
        requestSharedState();

    } catch (err) {
        console.error(err);
        showError("Failed to block alert");
    }
}

/**
 * Modal logic (unchanged)
 */
function showAlertDetails(alert) {
    currentAlert = alert;

    const modalBody = document.getElementById('modal-body');
    const severity = (alert.severity || 'unknown').toLowerCase();
    const prediction = (alert.prediction || 'unknown').toLowerCase();

    const confidence = parseFloat(alert.confidence || 0);
    const confidencePercent = (confidence * 100).toFixed(1);

    modalBody.innerHTML = `
        <div class="detail-row">
            <div class="detail-label">Alert ID</div>
            <div class="detail-value code-block">${escapeHtml(alert.id)}</div>
        </div>

        <div class="detail-row">
            <div class="detail-label">Severity</div>
            <div class="detail-value">
                <span class="sev-${severity}">${severity}</span>
            </div>
        </div>

        <div class="detail-row">
            <div class="detail-label">Prediction</div>
            <div class="detail-value">${escapeHtml(alert.prediction)}</div>
        </div>

        <div class="detail-row">
            <div class="detail-label">Confidence</div>
            <div class="detail-value">${confidencePercent}%</div>
        </div>
    `;

    alertsModal.show();
}

/**
 * Save alert status (unchanged)
 */
async function saveAlertStatus() {
    if (!currentAlert) return;

    const newStatus = document.getElementById('new-status').value;
    const newAssignee = document.getElementById('new-assignee').value;
    const closeReason = document.getElementById('close-reason').value;

    if (!newStatus) {
        alert('Please select a status');
        return;
    }

    try {
        const payload = { status: newStatus };
        if (newAssignee) payload.assignee = newAssignee;
        if (closeReason) payload.close_reason = closeReason;

        const response = await fetch(`/api/alerts/${encodeURIComponent(currentAlert.id)}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        statusModal.hide();
        showSuccess('Alert updated');

        requestSharedState();

    } catch (error) {
        console.error(error);
        showError('Failed to update alert');
    }
}

/**
 * UI helpers (unchanged)
 */
function updateAlertCount(count) {
    const elem = document.getElementById('alert-count');
    elem.textContent = `${count} ACTIVE`;
}

function showLoading(show) {
    document.getElementById('loading-spinner').classList.toggle('active', show);
}

function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'alert alert-danger position-fixed bottom-0 end-0 m-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `${escapeHtml(message)}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'alert alert-success position-fixed bottom-0 end-0 m-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `${escapeHtml(message)}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function formatTimestamp(ts) {
    if (!ts) return 'N/A';
    try {
        return new Date(ts).toLocaleString();
    } catch {
        return ts;
    }
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text).replace(/[&<>"']/g, m => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;',
        '"': '&quot;', "'": '&#039;'
    }[m]));
}
