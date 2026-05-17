const SAMPLE_ALERTS = [
    {
        id: 'alert-001',
        timestamp: new Date(Date.now() - 300000),
        title: 'Brute Force SSH Attack',
        severity: 'high',
        status: 'open',
        source_ip: '192.168.1.100',
        target_ip: '10.0.0.50',
        target_port: 22,
        confidence: 0.95,
        detection_engine: 'ML Model v2',
        attack_type: 'Brute Force',
        attempts: 1247,
        blocked: false,
        flags: 'Multiple failed auth'
    },
    {
        id: 'alert-002',
        timestamp: new Date(Date.now() - 600000),
        title: 'Suspicious DNS Query',
        severity: 'medium',
        status: 'flagged',
        source_ip: '172.16.0.25',
        target_ip: '8.8.8.8',
        target_port: 53,
        confidence: 0.72,
        detection_engine: 'Pattern Matcher',
        attack_type: 'DNS Exfiltration',
        requests: 342,
        blocked: false,
        flags: 'Uncommon TLD'
    },
    {
        id: 'alert-003',
        timestamp: new Date(Date.now() - 900000),
        title: 'Port Scan Detected',
        severity: 'low',
        status: 'blocked',
        source_ip: '203.0.113.45',
        target_ip: '10.0.0.0/24',
        target_port: 'Multiple',
        confidence: 0.68,
        detection_engine: 'Threshold Monitor',
        attack_type: 'Reconnaissance',
        ports_scanned: 1024,
        blocked: true,
        flags: 'Sequential ports'
    }
];

const socket = io('/events');

socket.on('connect', function() {
    console.log('Connected to real-time events');
});

function renderAlerts(alerts) {
    const container = document.getElementById('alerts-container');

    if (alerts.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">✓</div>
                <div>All clear! No alerts match your filters</div>
            </div>
        `;
        return;
    }

    container.innerHTML = alerts.map(alert => `
        <div class="alert-card ${alert.severity}" data-alert-id="${alert.id}">
            <div class="alert-header" onclick="toggleAlert('${alert.id}')">
                <div class="alert-header-left">
                    <div class="alert-time">${formatTime(alert.timestamp)}</div>
                    <div class="alert-title">${alert.title}</div>
                    <div class="alert-sources">
                        <div class="alert-source">
                            <span class="alert-source-label">From:</span>
                            <span>${alert.source_ip}</span>
                        </div>
                        <div class="alert-source">
                            <span class="alert-source-label">To:</span>
                            <span>${alert.target_ip}:${alert.target_port}</span>
                        </div>
                        <div class="alert-source">
                            <span class="alert-source-label">Type:</span>
                            <span>${alert.attack_type}</span>
                        </div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span class="alert-severity ${alert.severity}">${alert.severity}</span>
                    <span class="expand-icon">▼</span>
                </div>
            </div>

            <div class="alert-details">
                <div class="details-content">
                    <div class="detail-group">
                        <div class="detail-label">Detection Engine</div>
                        <div class="detail-value">${alert.detection_engine}</div>
                    </div>

                    <div class="detail-group">
                        <div class="detail-label">Confidence</div>
                        <div class="detail-value">${(alert.confidence * 100).toFixed(1)}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-bar-fill" style="width: ${alert.confidence * 100}%"></div>
                        </div>
                    </div>

                    <div class="detail-group">
                        <div class="detail-label">Status</div>
                        <div class="detail-value">${alert.blocked ? '🛑 BLOCKED' : '⚠️ ' + alert.status.toUpperCase()}</div>
                    </div>

                    <div class="detail-group">
                        <div class="detail-label">Additional Info</div>
                        <div class="detail-value">${alert.flags}</div>
                    </div>
                </div>

                <div class="actions-row">
                    <button class="btn-action mark-tp" onclick="markTruePositive('${alert.id}')">✓ True Positive</button>
                    <button class="btn-action mark-fp" onclick="markFalsePositive('${alert.id}')">✗ False Positive</button>
                    <button class="btn-action" onclick="viewDetails('${alert.id}')">View Full Details</button>
                </div>
            </div>
        </div>
    `).join('');
}

function toggleAlert(alertId) {
    const card = document.querySelector(`[data-alert-id="${alertId}"]`);
    card.classList.toggle('expanded');
}

function formatTime(date) {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);

    if (minutes < 1) return 'now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return date.toLocaleDateString();
}

function applyFilters() {
    const severity = document.getElementById('filter-severity').value;
    const status = document.getElementById('filter-status').value;
    const ip = document.getElementById('filter-ip').value.toLowerCase();

    let filtered = SAMPLE_ALERTS;

    if (severity) filtered = filtered.filter(a => a.severity === severity);
    if (status) filtered = filtered.filter(a => a.status === status || (status === 'blocked' && a.blocked));
    if (ip) filtered = filtered.filter(a => a.source_ip.includes(ip));

    renderAlerts(filtered);
}

function markTruePositive(alertId) {
    console.log('Marked as true positive:', alertId);
    socket.emit('feedback', { alert_id: alertId, feedback_type: 'true_positive' });
    alert('✓ Marked as True Positive - helps train the model!');
}

function markFalsePositive(alertId) {
    console.log('Marked as false positive:', alertId);
    socket.emit('feedback', { alert_id: alertId, feedback_type: 'false_positive' });
    alert('✓ Marked as False Positive - model will learn to avoid this pattern');
}

function viewDetails(alertId) {
    fetch(`/api/perception/confidence/${alertId}`)
        .then(response => response.json())
        .then(breakdown => showPerceptionBreakdown(breakdown))
        .catch(error => alert('Unable to load perception breakdown: ' + error));
}

function showPerceptionBreakdown(breakdown) {
    const message = `
🔬 CONFIDENCE BREAKDOWN

Confidence: ${(breakdown.overall_confidence * 100).toFixed(1)}%
Attack Type: ${breakdown.attack_type}

Top Factors:
${breakdown.top_features.map((f, i) =>
    `${i+1}. ${f.feature_name}: ${f.explanation}`
).join('\n')}

Summary: ${breakdown.summary}
    `;
    alert(message);
}

socket.on('alert.new', function(payload) {
    console.log('New alert received:', payload);
    applyFilters();
});

renderAlerts(SAMPLE_ALERTS);
