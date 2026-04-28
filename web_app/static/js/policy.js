/**
 * Policy Editor - Detection & Prevention Policy Configuration
 */

let currentPolicy = null;

document.addEventListener('DOMContentLoaded', function() {
    loadPolicy();
});

/**
 * Load policy from API
 */
async function loadPolicy() {
    try {
        const response = await fetch('/api/policy');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        currentPolicy = data;
        
        populateForm(currentPolicy);
        loadPolicyHistory();
        
        document.getElementById('loading-spinner').classList.remove('active');
        document.getElementById('policy-form').style.display = 'block';
        
    } catch (error) {
        console.error('Error loading policy:', error);
        showError('Failed to load policy');
        document.getElementById('loading-spinner').classList.remove('active');
    }
}

/**
 * Populate form with policy data
 */
function populateForm(policy) {
    // Mode
    const mode = policy.mode || 'detect';
    selectMode(mode);
    document.getElementById('mode').value = mode;
    
    // Thresholds
    document.getElementById('detection-threshold').value = 
        ((policy.confidence_block_threshold || policy.detection_threshold || 0.7) * 100);
    updateThresholdLabel(document.getElementById('detection-threshold').value);
    
    document.getElementById('alert-severity').value = 
        policy.min_severity || 'high';
    
    document.getElementById('escalation-threshold').value = 
        policy.escalation_threshold || 80;
    
    // Approval
    document.getElementById('approval-required').checked = 
        (policy.block_requires_approval !== false && policy.approval_required !== false);
    
    document.getElementById('approval-timeout').value = 
        policy.approval_timeout_minutes || 30;
    
    // Advanced
    document.getElementById('auto-escalate').checked = 
        policy.auto_escalate !== false;
    
    document.getElementById('anomaly-detection').checked = 
        policy.ml_enabled !== false;
    
    document.getElementById('log-level').value = 
        policy.log_level || 'info';
    
    // Engines
    renderEngineToggles(policy.engines || {});
}

/**
 * Select mode
 */
function selectMode(mode) {
    document.querySelectorAll('.mode-option').forEach(opt => {
        opt.classList.remove('selected');
    });
    
    if (mode === 'detect') {
        document.querySelectorAll('.mode-option')[0].classList.add('selected');
    } else {
        document.querySelectorAll('.mode-option')[1].classList.add('selected');
    }
    
    document.getElementById('mode').value = mode;
}

/**
 * Update threshold display
 */
function updateThresholdLabel(value) {
    document.getElementById('threshold-display').textContent = value + '%';
}

/**
 * Render engine toggles
 */
function renderEngineToggles(engines) {
    const container = document.getElementById('engine-toggles');
    
    const engineList = [
        { id: 'ml_engine', label: 'ML Engine' },
        { id: 'signature', label: 'Signature Engine' },
        { id: 'anomaly', label: 'Anomaly Engine' },
        { id: 'threshold', label: 'Threshold Engine' },
        { id: 'threat_intel', label: 'Threat Intel Engine' }
    ];
    
    container.innerHTML = '';
    
    engineList.forEach(engine => {
        const enabled = engines[engine.id] !== false;
        
        const toggle = document.createElement('div');
        toggle.className = 'engine-toggle';
        toggle.innerHTML = `
            <label style="flex: 1; margin: 0; cursor: pointer;">
                <input type="checkbox" class="engine-checkbox" data-engine="${engine.id}" 
                       ${enabled ? 'checked' : ''} style="cursor: pointer;">
                ${escapeHtml(engine.label)}
            </label>
        `;
        
        container.appendChild(toggle);
    });
}

/**
 * Load policy history
 */
async function loadPolicyHistory() {
    try {
        const response = await fetch('/api/policy/history');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        const history = data.history || [];
        
        renderPolicyHistory(history);
        
    } catch (error) {
        console.error('Error loading policy history:', error);
    }
}

/**
 * Render policy history
 */
function renderPolicyHistory(history) {
    const container = document.getElementById('policy-history');
    
    if (!history || history.length === 0) {
        container.innerHTML = '<p style="color: #999; margin: 0;">No policy history</p>';
        return;
    }
    
    let html = '';
    history.slice(0, 5).forEach(item => {
        const timestamp = new Date(item.timestamp).toLocaleString();
        html += `
            <div class="history-item">
                <strong>${escapeHtml(item.mode || 'unknown')}</strong>
                <span class="timestamp">${escapeHtml(timestamp)}</span>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

/**
 * Save policy
 */
async function savePolicy() {
    const mode = document.getElementById('mode').value;
    const detectionThreshold = parseFloat(document.getElementById('detection-threshold').value) / 100;
    const alertSeverity = document.getElementById('alert-severity').value;
    const escalationThreshold = parseFloat(document.getElementById('escalation-threshold').value);
    const approvalRequired = document.getElementById('approval-required').checked;
    const approvalTimeout = parseInt(document.getElementById('approval-timeout').value);
    const autoEscalate = document.getElementById('auto-escalate').checked;
    const anomalyDetection = document.getElementById('anomaly-detection').checked;
    const logLevel = document.getElementById('log-level').value;
    
    // Collect engine settings
    const engines = {};
    document.querySelectorAll('.engine-checkbox').forEach(checkbox => {
        const engineId = checkbox.dataset.engine;
        engines[engineId] = checkbox.checked;
    });
    
    const payload = {
        mode,
        confidence_block_threshold: detectionThreshold,
        min_severity: alertSeverity,
        escalation_threshold: escalationThreshold,
        block_requires_approval: approvalRequired,
        approval_timeout_minutes: approvalTimeout,
        auto_escalate: autoEscalate,
        ml_enabled: anomalyDetection,
        log_level: logLevel,
        engines
    };
    
    try {
        const response = await fetch('/api/policy', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        showSuccess('Policy saved successfully');
        
        // Reload policy
        setTimeout(loadPolicy, 1000);
        
    } catch (error) {
        console.error('Error saving policy:', error);
        showError('Failed to save policy: ' + error.message);
    }
}

/**
 * Reset policy to last saved
 */
function resetPolicy() {
    if (currentPolicy) {
        populateForm(currentPolicy);
        showSuccess('Policy reset to last saved version');
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
 * Escape HTML
 */
function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&',
        '<': '<',
        '>': '>',
        '"': '"',
        "'": '''
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}
