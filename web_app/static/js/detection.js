/**
 * Detection Console - Multi-Engine Detection
 * Run detection on custom input data
 */

document.addEventListener('DOMContentLoaded', function() {
    // Form submission via Enter key
    document.getElementById('detection-form').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            runDetection();
        }
    });
});

/**
 * Run detection
 */
async function runDetection() {
    // Get input values
    const features = {
        duration: parseFloat(document.getElementById('duration').value) || 0,
        src_bytes: parseFloat(document.getElementById('src_bytes').value) || 0,
        dst_bytes: parseFloat(document.getElementById('dst_bytes').value) || 0,
        count: parseFloat(document.getElementById('count').value) || 0,
        srv_count: parseFloat(document.getElementById('srv_count').value) || 0,
        serror_rate: parseFloat(document.getElementById('serror_rate').value) || 0,
        same_srv_rate: parseFloat(document.getElementById('same_srv_rate').value) || 0,
        source_ip: document.getElementById('source_ip').value || null
    };
    
    // Validate
    if (!features.duration && !features.src_bytes && !features.dst_bytes) {
        showError('Please enter at least some feature values');
        return;
    }
    
    try {
        showLoading(true);
        hideError();
        hideResults();
        
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data, features);
        
    } catch (error) {
        console.error('Error running detection:', error);
        showError('Detection failed: ' + error.message);
    } finally {
        showLoading(false);
    }
}

/**
 * Display detection results
 */
function displayResults(data, inputFeatures) {
    const verdict = (data.verdict || 'unknown').toLowerCase();
    const confidence = parseFloat(data.confidence || 0);
    const attackType = data.attack_type || 'Unknown';
    const severity = data.severity || 'Unknown';
    const engineResults = data.engine_results || {};
    
    // Verdict badge
    const verdictContainer = document.getElementById('verdict-container');
    verdictContainer.innerHTML = `
        <div><span class="verdict-badge verdict-${verdict}">${verdict.toUpperCase()}</span></div>
    `;
    
    // Confidence bar
    const confidenceContainer = document.getElementById('confidence-container');
    const confidencePercent = Math.round(confidence * 100);
    document.getElementById('confidence-bar').style.width = `${confidencePercent}%`;
    document.getElementById('confidence-value').textContent = `${confidencePercent}%`;
    confidenceContainer.style.display = 'flex';
    
    // Result cards
    document.getElementById('attack-type').textContent = escapeHtml(attackType);
    document.getElementById('severity-result').textContent = escapeHtml(severity);
    document.getElementById('engines-count').textContent = Object.keys(engineResults).length;
    
    // Engine results
    renderEngineResults(engineResults);
    
    // Features display
    const featuresDisplay = document.getElementById('features-display');
    let featuresHtml = '<strong>Input Features:</strong><br>';
    for (const [key, value] of Object.entries(inputFeatures)) {
        if (value !== null && value !== '') {
            featuresHtml += `<strong>${escapeHtml(key)}:</strong> ${escapeHtml(String(value))}<br>`;
        }
    }
    featuresDisplay.innerHTML = featuresHtml;
    
    // Show results
    showResults();
}

/**
 * Render engine results
 */
function renderEngineResults(engineResults) {
    const enginesList = document.getElementById('engines-list');
    
    if (!engineResults || Object.keys(engineResults).length === 0) {
        enginesList.innerHTML = '<p style="color: #999;">No engine results available</p>';
        return;
    }
    
    let html = '';
    
    for (const [engineName, result] of Object.entries(engineResults)) {
        const triggered = result.triggered === true;
        const confidence = parseFloat(result.confidence || 0);
        const confidencePercent = (confidence * 100).toFixed(1);
        
        html += `
            <div class="engine-result">
                <div class="engine-name">
                    ${escapeHtml(engineName)}
                    <span class="engine-status ${triggered ? 'engine-triggered' : 'engine-clean'}">
                        ${triggered ? '⚠️ TRIGGERED' : '✓ CLEAN'}
                    </span>
                    <span style="float: right; color: #0066cc; font-weight: 600;">
                        ${confidencePercent}%
                    </span>
                </div>
                <div class="engine-detail">
                    ${result.reason ? `<strong>Reason:</strong> ${escapeHtml(result.reason)}<br>` : ''}
                    ${result.attack_type ? `<strong>Type:</strong> ${escapeHtml(result.attack_type)}<br>` : ''}
                    ${result.severity ? `<strong>Severity:</strong> ${escapeHtml(result.severity)}` : ''}
                </div>
            </div>
        `;
    }
    
    enginesList.innerHTML = html;
}

/**
 * Reset form
 */
function resetForm() {
    document.getElementById('detection-form').reset();
    document.getElementById('duration').value = '10';
    document.getElementById('src_bytes').value = '1000';
    document.getElementById('dst_bytes').value = '2000';
    document.getElementById('count').value = '50';
    document.getElementById('srv_count').value = '40';
    document.getElementById('serror_rate').value = '0.05';
    document.getElementById('same_srv_rate').value = '0.8';
    hideResults();
    hideError();
}

/**
 * Show/hide UI elements
 */
function showLoading(show) {
    document.getElementById('loading-spinner').classList.toggle('active', show);
}

function hideError() {
    document.getElementById('error-message').classList.remove('active');
}

function hideResults() {
    document.getElementById('result-section').classList.remove('active');
}

function showResults() {
    document.getElementById('result-section').classList.add('active');
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = message;
    errorDiv.classList.add('active');
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
