/**
 * Actions Page - Prevention Actions Management
 * Handles action approval workflow and incident response
 */

let currentAction = null;
let approvalModal;
let allActions = [];
let pendingActions = [];

document.addEventListener('DOMContentLoaded', function() {
    approvalModal = new bootstrap.Modal(document.getElementById('approvalModal'));
    
    // Load initial data
    loadAllActions();
    loadPendingActions();
    
    document.getElementById('btn-confirm-approve').addEventListener('click', approveAction);
    
    // Auto-refresh every 30 seconds
    setInterval(() => {
        loadAllActions();
        loadPendingActions();
    }, 30000);
});

/**
 * Switch between tabs
 */
function switchTab(tabName, evt) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Deactivate all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`tab-${tabName}`).classList.add('active');
    
    // Activate selected button
    evt.target.classList.add('active');
}

/**
 * Load all actions
 */
async function loadAllActions() {
    try {
        const response = await fetch('/api/actions?limit=200');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        allActions = data.actions || [];
        
        renderActions('all-actions-list', allActions);
        hideLoader('loading-all');
        
    } catch (error) {
        console.error('Error loading actions:', error);
        showError('Failed to load actions');
    }
}

/**
 * Load pending actions
 */
async function loadPendingActions() {
    try {
        const response = await fetch('/api/actions/pending');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        pendingActions = data.actions || [];
        
        // Update badge
        const badge = document.getElementById('pending-badge');
        if (pendingActions.length > 0) {
            badge.innerHTML = `<span class="badge bg-warning">${pendingActions.length}</span>`;
        } else {
            badge.innerHTML = '';
        }
        
        renderActions('pending-actions-list', pendingActions, true);
        hideLoader('loading-pending');
        
    } catch (error) {
        console.error('Error loading pending actions:', error);
        showError('Failed to load pending actions');
    }
}

/**
 * Filter actions by status
 */
function filterActionsByStatus(actions, status) {
    if (!status) return actions;
    return actions.filter(a => (a.status || '').toLowerCase() === status.toLowerCase());
}

/**
 * Render actions in list format
 */
function renderActions(containerId, actions, showApproveButton = false) {
    const container = document.getElementById(containerId);
    
    if (!actions || actions.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <h4>No Actions</h4>
                <p>No actions to display</p>
            </div>
        `;
        return;
    }
    
    // Filter further by tab
    let filtered = actions;
    const tabId = containerId.split('-')[0];
    if (tabId === 'executed') {
        filtered = filterActionsByStatus(actions, 'executed');
    } else if (tabId === 'failed') {
        filtered = filterActionsByStatus(actions, 'failed');
    }
    
    if (filtered.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <h4>No Actions</h4>
                <p>No actions in this category</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    
    filtered.forEach(action => {
        const status = (action.status || 'pending').toLowerCase();
        const isPending = ['pending', 'pending_approval'].includes(status);
        const actionType = action.action_type || action.action || 'unknown';
        
        html += `
            <div class="action-card">
                <div class="action-header">
                    <div>
                        <span class="action-type">${escapeHtml(actionType)}</span>
                        <span class="status-badge status-${status}">${status}</span>
                    </div>
                    <small class="action-timestamp">${formatTimestamp(action.created_at)}</small>
                </div>
                
                <div class="action-target">
                    🎯 ${escapeHtml(action.target || 'N/A')}
                </div>
                
                <div class="action-details">
                    <strong>ID:</strong> <code>${escapeHtml((action.id || 'N/A').substring(0, 16))}</code>
                    ${action.alert_id ? `<br><strong>Alert:</strong> <code>${escapeHtml((action.alert_id || 'N/A').substring(0, 16))}</code>` : ''}
                    ${action.reason ? `<br><strong>Reason:</strong> ${escapeHtml(action.reason)}` : ''}
                </div>
                
                ${action.created_by ? `<div class="action-details"><strong>Created by:</strong> ${escapeHtml(action.created_by)}</div>` : ''}
                ${action.approved_at ? `<div class="action-details"><strong>Approved:</strong> ${formatTimestamp(action.approved_at)}</div>` : ''}
                
                ${isPending && showApproveButton ? `
                    <button class="btn btn-sm btn-approve mt-2" onclick="openApprovalModal('${escapeHtml(JSON.stringify(action).replace(/'/g, "\\'"))}')">
                        ✔️ Review & Approve
                    </button>
                ` : ''}
            </div>
        `;
    });
    
    container.innerHTML = html;
}

/**
 * Open approval modal
 */
function openApprovalModal(actionJson) {
    try {
        currentAction = JSON.parse(actionJson);
        
        const detailsHtml = `
            <div class="detail-row">
                <div class="detail-label">Action ID</div>
                <div class="detail-value code-block">${escapeHtml(currentAction.id || 'N/A')}</div>
            </div>
            
            <div class="detail-row">
                <div class="detail-label">Action Type</div>
                <div class="detail-value"><span class="action-type-badge">${escapeHtml(currentAction.action_type || 'unknown')}</span></div>
            </div>
            
            <div class="detail-row">
                <div class="detail-label">Target</div>
                <div class="detail-value code-block">${escapeHtml(currentAction.target || 'N/A')}</div>
            </div>
            
            <div class="detail-row">
                <div class="detail-label">Related Alert</div>
                <div class="detail-value code-block">${escapeHtml(currentAction.alert_id || 'N/A')}</div>
            </div>
            
            <div class="detail-row">
                <div class="detail-label">Reason</div>
                <div class="detail-value">${escapeHtml(currentAction.reason || 'No reason specified')}</div>
            </div>
            
            <div class="detail-row">
                <div class="detail-label">Created</div>
                <div class="detail-value">${formatTimestamp(currentAction.created_at)}</div>
            </div>
            
            ${currentAction.created_by ? `
            <div class="detail-row">
                <div class="detail-label">Created By</div>
                <div class="detail-value">${escapeHtml(currentAction.created_by)}</div>
            </div>
            ` : ''}
            
            <div class="approval-notes">
                <strong>ℹ️ Note:</strong><br>
                Approving this action will immediately execute the ${escapeHtml(currentAction.action_type || 'action')} 
                on target <code>${escapeHtml(currentAction.target || 'unknown')}</code>.
            </div>
        `;
        
        document.getElementById('approval-details').innerHTML = detailsHtml;
        approvalModal.show();
        
    } catch (error) {
        console.error('Error parsing action:', error);
        showError('Error loading action details');
    }
}

/**
 * Approve action
 */
async function approveAction() {
    if (!currentAction) return;
    
    try {
        const response = await fetch(`/api/actions/${encodeURIComponent(currentAction.id)}/approve`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                notes: 'Approved via UI'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        approvalModal.hide();
        showSuccess('Action approved and executed successfully');
        
        // Reload data
        await loadAllActions();
        await loadPendingActions();
        
    } catch (error) {
        console.error('Error approving action:', error);
        showError('Failed to approve action: ' + error.message);
    }
}

/**
 * Hide loader
 */
function hideLoader(loaderId) {
    const loader = document.getElementById(loaderId);
    if (loader) {
        loader.style.display = 'none';
    }
}

/**
 * Show error message
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
 * Format timestamp
 */
function formatTimestamp(ts) {
    if (!ts) return 'N/A';
    try {
        const date = new Date(ts);
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return ts;
    }
}

/**
 * Escape HTML
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
