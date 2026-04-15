// Health Dashboard - Real-time System Monitoring

let healthChart = null;
let healthMetricsHistory = [];
let refreshInterval = null;

document.addEventListener('DOMContentLoaded', function() {
    loadHealthData();
    // Auto-refresh every 30 seconds
    refreshInterval = setInterval(loadHealthData, 30000);
});

async function loadHealthData() {
    try {
        // Load from API endpoints
        const healthRes = await fetch('/api/health');
        if (!healthRes.ok) {
            throw new Error('Failed to fetch health data');
        }

        const healthData = await healthRes.json();
        // Optional: fetch text metrics and parse if needed
        // const metricsText = await (await fetch('/api/metrics')).text();

        // Update UI with data
        updateSystemStatus(healthData);
        updateMetrics(healthData);
        updateEngineStatus(healthData);
        updateServiceStatus(healthData);
        updateResourceUsage(healthData);
        updateAlertsBreakdown(healthData);
        updateSystemInfo(healthData);
        updatePerformanceChart(healthData);

        // Update timestamp
        const now = new Date();
        document.getElementById('lastUpdate').textContent = `Last update: ${now.toLocaleTimeString()}`;

    } catch (error) {
        console.error('Error loading health data:', error);
        showErrorState();
    }
}

function updateSystemStatus(data) {
    const status = document.getElementById('statusText');
    const statusEmoji = document.getElementById('healthStatus');
    const uptime = document.getElementById('uptimeText');
    const startTime = document.getElementById('startTimeText');

    // Determine overall status
    let systemStatus = 'Healthy';
    let statusEmojis = '🟢';
    let statusColor = 'bg-success';

    if (data.status === 'degraded') {
        systemStatus = 'Degraded';
        statusEmojis = '🟡';
        statusColor = 'bg-warning';
    } else if (data.status === 'unhealthy') {
        systemStatus = 'Unhealthy';
        statusEmojis = '🔴';
        statusColor = 'bg-danger';
    }

    status.textContent = systemStatus;
    statusEmoji.textContent = statusEmojis;
    status.className = `mb-0 ${statusColor === 'bg-warning' ? 'text-warning' : statusColor === 'bg-danger' ? 'text-danger' : 'text-success'}`;

    // Format uptime
    if (data.uptime_seconds) {
        const uptimeFormatted = formatUptime(data.uptime_seconds);
        uptime.textContent = uptimeFormatted;
        startTime.textContent = `Started ${formatStartTime(data.uptime_seconds)} ago`;
    }
}

function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    if (days > 0) {
        return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
        return `${hours}h ${minutes}m`;
    } else {
        return `${minutes}m`;
    }
}

function formatStartTime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);

    if (days > 0) {
        return `${days} day${days > 1 ? 's' : ''}`;
    } else if (hours > 0) {
        return `${hours} hour${hours > 1 ? 's' : ''}`;
    } else {
        return 'less than an hour';
    }
}

function updateMetrics(data) {
    // CPU Usage
    const cpuPercent = data.cpu_percent || 0;
    document.getElementById('cpuPercent').textContent = `${cpuPercent.toFixed(1)}%`;
    const cpuBar = document.getElementById('cpuBar');
    cpuBar.style.width = `${cpuPercent}%`;
    cpuBar.className = `progress-bar ${cpuPercent > 80 ? 'bg-danger' : cpuPercent > 60 ? 'bg-warning' : 'bg-success'}`;

    // Key metrics
    document.getElementById('totalAlerts').textContent = (data.total_alerts || 0).toLocaleString();
    document.getElementById('totalActions').textContent = (data.total_actions || 0).toLocaleString();
    document.getElementById('avgResponseTime').textContent = `${(data.avg_response_time_ms || 0).toFixed(1)} ms`;
    document.getElementById('errorRate').textContent = `${(data.error_rate || 0).toFixed(2)}%`;

    // Store for chart
    healthMetricsHistory.push({
        timestamp: new Date(),
        cpu: cpuPercent,
        errors: data.error_rate || 0,
        responseTime: data.avg_response_time_ms || 0
    });

    // Keep only last 20 entries
    if (healthMetricsHistory.length > 20) {
        healthMetricsHistory.shift();
    }
}

function updateEngineStatus(data) {
    const enginesList = document.getElementById('enginesList');
    enginesList.innerHTML = '';

    const engines = data.detection_engines || data.engines || [];
    if (Array.isArray(engines)) {
        engines.forEach(engine => {
            const engineDiv = document.createElement('div');
            engineDiv.className = 'alert mb-2';
            
            const status = engine.enabled ? 'enabled' : 'disabled';
            const statusClass = engine.enabled ? 'alert-success' : 'alert-secondary';
            const statusEmoji = engine.enabled ? '✅' : '⚠️';

            engineDiv.classList.add(statusClass);
            engineDiv.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <strong>${statusEmoji} ${engine.name}</strong>
                        <br>
                        <small class="text-muted">Status: ${status}</small>
                    </div>
                    <div class="text-end">
                        <small>${engine.detections || 0} detections</small>
                    </div>
                </div>
            `;
            enginesList.appendChild(engineDiv);
        });
    }
}

function updateServiceStatus(data) {
    const servicesList = document.getElementById('servicesList');
    servicesList.innerHTML = '';

    const services = [
        { name: 'Alert Service', key: 'alert_service' },
        { name: 'Detection Service', key: 'detection_service' },
        { name: 'Prevention Service', key: 'prevention_service' },
        { name: 'Ingestion Service', key: 'ingestion_service' }
    ];

    services.forEach(service => {
        const isHealthy = data.services && data.services[service.key] !== false;
        const serviceDiv = document.createElement('div');
        serviceDiv.className = `alert mb-2 ${isHealthy ? 'alert-success' : 'alert-danger'}`;
        const emoji = isHealthy ? '✅' : '❌';

        serviceDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <strong>${emoji} ${service.name}</strong>
                    <br>
                    <small class="text-muted">Status: ${isHealthy ? 'Running' : 'Error'}</small>
                </div>
                <div class="badge ${isHealthy ? 'bg-success' : 'bg-danger'}">
                    ${isHealthy ? 'Online' : 'Offline'}
                </div>
            </div>
        `;
        servicesList.appendChild(serviceDiv);
    });
}

function updateResourceUsage(data) {
    // Memory
    const memoryUsed = data.memory_used_mb || 0;
    const memoryTotal = data.memory_total_mb || 1;
    const memoryPercent = (memoryUsed / memoryTotal) * 100;

    document.getElementById('memoryBar').style.width = `${memoryPercent}%`;
    document.getElementById('memoryBar').className = `progress-bar ${memoryPercent > 80 ? 'bg-danger' : memoryPercent > 60 ? 'bg-warning' : 'bg-info'}`;
    document.getElementById('memoryUsed').textContent = `${memoryUsed.toFixed(0)} MB`;
    document.getElementById('memoryTotal').textContent = `${memoryTotal.toFixed(0)} MB`;

    // Network
    document.getElementById('packetsPerSec').textContent = (data.packets_per_sec || 0).toLocaleString();
    document.getElementById('flowsAnalyzed').textContent = (data.flows_analyzed || 0).toLocaleString();

    // Database
    document.getElementById('dbAlerts').textContent = (data.db_alerts || 0).toLocaleString();
    document.getElementById('dbAudits').textContent = (data.db_audits || 0).toLocaleString();
    document.getElementById('dbSize').textContent = `${(data.db_size_mb || 0).toFixed(2)} MB`;
    
    const lastCleanup = data.last_cleanup ? new Date(data.last_cleanup).toLocaleDateString() : 'Never';
    document.getElementById('lastCleanup').textContent = lastCleanup;
}

function updateAlertsBreakdown(data) {
    const alerts = data.alerts_by_severity || { critical: 0, high: 0, medium: 0, low: 0 };

    document.getElementById('alertsCritical').textContent = (alerts.critical || 0).toLocaleString();
    document.getElementById('alertsHigh').textContent = (alerts.high || 0).toLocaleString();
    document.getElementById('alertsMedium').textContent = (alerts.medium || 0).toLocaleString();
    document.getElementById('alertsLow').textContent = (alerts.low || 0).toLocaleString();
}

function updateSystemInfo(data) {
    document.getElementById('systemVersion').textContent = data.version || '1.0.0';
    document.getElementById('pythonVersion').textContent = data.python_version || 'N/A';
    
    const now = new Date();
    document.getElementById('systemTime').textContent = now.toLocaleString();
}

function updatePerformanceChart(data) {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;

    const labels = healthMetricsHistory.map((_, i) => `${i + 1}`);
    const cpuData = healthMetricsHistory.map(m => m.cpu);
    const errorData = healthMetricsHistory.map(m => m.errors);
    const responseTimeData = healthMetricsHistory.map(m => m.responseTime / 10); // Scale down for visibility

    if (healthChart) {
        healthChart.destroy();
    }

    healthChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'CPU Usage (%)',
                    data: cpuData,
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    yAxisID: 'y',
                    tension: 0.4
                },
                {
                    label: 'Error Rate (%)',
                    data: errorData,
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                },
                {
                    label: 'Response Time (ms/10)',
                    data: responseTimeData,
                    borderColor: '#17a2b8',
                    backgroundColor: 'rgba(23, 162, 184, 0.1)',
                    yAxisID: 'y2',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'CPU %' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'center',
                    title: { display: true, text: 'Errors %' }
                },
                y2: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Response (ms/10)' }
                }
            }
        }
    });
}

function showErrorState() {
    document.getElementById('statusText').textContent = 'Error';
    document.getElementById('healthStatus').textContent = '⚠️';

    const errorAlert = document.createElement('div');
    errorAlert.className = 'alert alert-danger mt-3';
    errorAlert.textContent = 'Failed to load health data. Please check if the API is running.';
    document.querySelector('.container-fluid').prepend(errorAlert);
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    if (healthChart) {
        healthChart.destroy();
    }
});
