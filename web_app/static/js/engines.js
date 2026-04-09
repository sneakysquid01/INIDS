/**
 * Engine Management - Detection Engine Configuration
 */

document.addEventListener('DOMContentLoaded', function() {
    loadEngines();
});

/**
 * Load engines from API
 */
async function loadEngines() {
    try {
        const response = await fetch('/api/engines');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        const engines = data.engines || [];
        
        renderEngines(engines);
        hideLoader();
        
    } catch (error) {
        console.error('Error loading engines:', error);
        showError('Failed to load engines');
    }
}

/**
 * Render engines as cards
 */
function renderEngines(engines) {
    const grid = document.getElementById('engines-grid');
    const emptyState = document.getElementById('empty-state');
    
    if (!engines || engines.length === 0) {
        grid.style.display = 'none';
        emptyState.style.display = 'block';
        return;
    }
    
    grid.style.display = 'grid';
    emptyState.style.display = 'none';
    grid.innerHTML = '';
    
    engines.forEach(engine => {
        const card = createEngineCard(engine);
        grid.appendChild(card);
    });
}

/**
 * Create engine card element
 */
function createEngineCard(engine) {
    const card = document.createElement('div');
    card.className = 'engine-card';
    
    const engineId = engine.engine_id || engine.id || 'unknown';
    const engineType = engine.engine_type || engine.type || 'unknown';
    const isEnabled = engine.enabled === true;
    const isReady = (engine.ready ?? engine.is_ready) !== false;
    
    const descriptions = {
        'ml_engine': 'Machine Learning-based detection using trained models',
        'signature': 'Rule-based signature pattern matching',
        'anomaly': 'Statistical anomaly detection and behavior analysis',
        'threshold': 'Threshold-based statistical detection',
        'threat_intel': 'IP and domain reputation scoring',
        'ti': 'Threat Intelligence engine using external feeds'
    };
    
    const description = descriptions[engineId] || descriptions[engineType] || 'Detection engine';
    
    const statusIcon = isEnabled ? '🟢' : '🔘';
    const statusText = isEnabled ? 'Enabled' : 'Disabled';
    const readyText = isReady ? 'Ready' : 'Initializing';
    
    card.innerHTML = `
        <div class="engine-header">
            <div>
                <div class="engine-title">${escapeHtml(engineId)}</div>
                <span class="engine-type-badge">${escapeHtml(engineType)}</span>
            </div>
            <div class="toggle-switch">
                <label class="switch">
                    <input type="checkbox" id="toggle-${escapeHtml(engineId)}" 
                           ${isEnabled ? 'checked' : ''} 
                           ${!isReady ? 'disabled' : ''}
                           onchange="toggleEngine('${escapeHtml(engineId)}', this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        </div>
        
        <p class="engine-description">${escapeHtml(description)}</p>
        
        <div class="engine-status">
            <span class="status-indicator ${isEnabled ? 'status-enabled' : 'status-disabled'}"></span>
            <span>${statusText}</span>
        </div>
        
        <div class="engine-details">
            <div class="detail-line">
                <span class="detail-label">Status:</span>
                <span class="detail-value">${readyText}</span>
            </div>
            ${engine.confidence_threshold !== undefined ? `
            <div class="detail-line">
                <span class="detail-label">Confidence Threshold:</span>
                <span class="detail-value">${(parseFloat(engine.confidence_threshold) * 100).toFixed(0)}%</span>
            </div>
            ` : ''}
            ${engine.accuracy !== undefined ? `
            <div class="detail-line">
                <span class="detail-label">Accuracy:</span>
                <span class="detail-value">${(parseFloat(engine.accuracy) * 100).toFixed(1)}%</span>
            </div>
            ` : ''}
            ${engine.detections_count !== undefined ? `
            <div class="detail-line">
                <span class="detail-label">Detections:</span>
                <span class="detail-value">${engine.detections_count}</span>
            </div>
            ` : ''}
        </div>
    `;
    
    return card;
}

/**
 * Toggle engine on/off
 */
async function toggleEngine(engineId, enabled) {
    try {
        const response = await fetch(`/api/engines/${encodeURIComponent(engineId)}/toggle`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ enabled })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const status = enabled ? 'enabled' : 'disabled';
        showSuccess(`Engine ${engineId} ${status} successfully`);
        
        // Reload engines
        setTimeout(loadEngines, 500);
        
    } catch (error) {
        console.error('Error toggling engine:', error);
        showError('Failed to toggle engine: ' + error.message);
        
        // Revert toggle on error
        setTimeout(loadEngines, 500);
    }
}

/**
 * Hide loader
 */
function hideLoader() {
    document.getElementById('loading-spinner').classList.remove('active');
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
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}
