/**
 * Alerts Page - Real API Integration
 * Handles all alert management functionality
 */

let currentAlerts = [];
let currentAlert = null;
let alertsModal, statusModal;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap modals
    alertsModal = new bootstrap.Modal(document.getElementById('detailsModal'));
    statusModal = new bootstrap.Modal(document.getElementById('statusModal'));
    
    // Event listeners
    document.getElementById('btn-refresh').addEventListener('click', loadAlerts);
    document.getElementById('severity-filter').addEventListener('change', loadAlerts);
    document.getElementById('status-filter').addEventListener('change', loadAlerts);
    document.getElementById('btn-update-status').addEventListener('click', () => {
        alertsModal.hide();
        statusModal.show();
    });
    document.getElementById('btn-save-status').addEventListener('click', saveAlertStatus);
    
    // Initial load
    loadAlerts();
    
    // Auto-refresh every 30 seconds
    setInterval(loadAlerts, 30000);
});

/**
 * Load alerts from API with filters
 */
async function loadAlerts() {
    const severity = document.getElementById('severity-filter').value || '';
    const status = document.getElementById('status-filter').value || '';
    
    try {
        showLoading(true);
        
        // Build query parameters
        const params = new URLSearchParams();
        params.append('limit', '200');
        if (severity) params.append('severity', severity);
        if (status) params.append('status', status);
        
        const response = await fetch(`/api/alerts?${params.toString()}`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        currentAlerts = data.alerts || [];
        
        updateAlertCount(data.count || 0);
        renderAlerts(currentAlerts);
        
    } catch (error) {
        console.error('Error loading alerts:', error);
        showError('Failed to load alerts. Check console for details.');
    } finally {
        showLoading(false);
    }
}

/**
 * Render alerts in table
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
        const row = document.createElement('tr');
        row.style.cursor = 'pointer';
        row.onclick = () => showAlertDetails(alert);
        
        const severity = (alert.severity || 'unknown').toLowerCase();
        const status = (alert.status || 'open').toLowerCase();
        const prediction = (alert.prediction || 'unknown').toLowerCase();
        const confidence = parseFloat(alert.confidence || 0);
        
        // Truncate ID for display
        const displayId = (alert.id || 'N/A').substring(0, 12);
        
        row.innerHTML = `
            <td>
                <code style="font-size: 0.8rem;">${escapeHtml(displayId)}</code>
            </td>
            <td>
                <div class="timestamp">
                    ${formatTimestamp(alert.timestamp)}
                </div>
            </td>
            <td>
                <span class="severity-badge severity-${severity}">
                    ${severity}
                </span>
            </td>
            <td>
                <span class="prediction-badge prediction-${prediction}">
                    ${prediction}
                </span>
            </td>
            <td>
                <div class="confidence-bar">
                    <div class="progress" style="width: 60px;">
                        <div class="progress-bar" style="width: ${confidence * 100}%"></div>
                    </div>
                    <span class="confidence-text">${(confidence * 100).toFixed(0)}%</span>
                </div>
            </td>
            <td>
                <span class="status-badge status-${status}">
                    ${status}
                </span>
            </td>
            <td>
                <small>${escapeHtml(alert.profile || 'N/A')}</small>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

/**
 * Show alert details in modal
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
            <div class="detail-value code-block">${escapeHtml(alert.id || 'N/A')}</div>
        </div>
        
        <div class="detail-row">
            <div class="detail-label">Timestamp</div>
            <div class="detail-value">${escapeHtml(alert.timestamp || 'N/A')}</div>
        </div>
        
        <div class="detail-row">
            <div class="detail-label">Severity</div>
            <div class="detail-value">
                <span class="severity-badge severity-${severity}">${severity}</span>
            </div>
        </div>
        
        <div class="detail-row">
            <div class="detail-label">Prediction</div>
            <div class="detail-value">
                <span class="prediction-badge prediction-${prediction}">${prediction}</span>
            </div>
        </div>
        
        <div class="detail-row">
            <div class="detail-label">Confidence Score</div>
            <div class="detail-value">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="flex: 1; min-width: 200px;">
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar" style="width: ${confidence * 100}%"></div>
                        </div>
                    </div>
                    <strong>${confidencePercent}%</strong>
                </div>
            </div>
        </div>
        
        <div class="detail-row">
            <div class="detail-label">Status</div>
            <div class="detail-value">
                <span class="status-badge status-${(alert.status || 'open').toLowerCase()}">
                    ${escapeHtml(alert.status || 'open')}
                </span>
            </div>
        </div>
        
        <div class="detail-row">
            <div class="detail-label">Profile</div>
            <div class="detail-value">${escapeHtml(alert.profile || 'N/A')}</div>
        </div>
        
        <div class="detail-row">
            <div class="detail-label">Reason</div>
            <div class="detail-value code-block">${escapeHtml(alert.reason || 'N/A')}</div>
        </div>
        
        ${alert.assignee ? `
        <div class="detail-row">
            <div class="detail-label">Assigned To</div>
            <div class="detail-value">${escapeHtml(alert.assignee)}</div>
        </div>
        ` : ''}
        
        ${alert.close_reason ? `
        <div class="detail-row">
            <div class="detail-label">Close Reason</div>
            <div class="detail-value">${escapeHtml(alert.close_reason)}</div>
        </div>
        ` : ''}
        
        ${alert.status_updated_at ? `
        <div class="detail-row">
            <div class="detail-label">Status Updated</div>
            <div class="detail-value">${escapeHtml(alert.status_updated_at)}</div>
        </div>
        ` : ''}
    `;
    
    alertsModal.show();
}

/**
 * Save alert status update
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
        const payload = {
            status: newStatus,
        };
        
        if (newAssignee) {
            payload.assignee = newAssignee;
        }
        
        if (closeReason) {
            payload.close_reason = closeReason;
        }
        
        const response = await fetch(`/api/alerts/${encodeURIComponent(currentAlert.id)}`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        // Close modal and reload
        statusModal.hide();
        
        // Clear form
        document.getElementById('new-status').value = '';
        document.getElementById('new-assignee').value = '';
        document.getElementById('close-reason').value = '';
        
        // Show success and reload
        showSuccess('Alert updated successfully');
        await loadAlerts();
        
    } catch (error) {
        console.error('Error updating alert:', error);
        showError('Failed to update alert: ' + error.message);
    }
}

/**
 * Update alert count display
 */
function updateAlertCount(count) {
    const elem = document.getElementById('alert-count');
    elem.innerHTML = `${count} Alert${count !== 1 ? 's' : ''}`;
}

/**
 * Show/hide loading spinner
 */
function showLoading(show) {
    document.getElementById('loading-spinner').classList.toggle('active', show);
}

/**
 * Show error message (simple toast)
 */
function showError(message) {
    const toast = document.createElement('div');
    toast.className = 'alert alert-danger position-fixed bottom-0 end-0 m-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `${escapeHtml(message)} <button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

/**
 * Show success message
 */
function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'alert alert-success position-fixed bottom-0 end-0 m-3';
    toast.style.zIndex = '9999';
    toast.innerHTML = `${escapeHtml(message)} <button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

/**
 * Format timestamp for display
 */
function formatTimestamp(ts) {
    if (!ts) return 'N/A';
    try {
        const date = new Date(ts);
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    } catch {
        return ts;
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}
