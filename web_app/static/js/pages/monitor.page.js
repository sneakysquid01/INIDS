/**
 * MONITOR PAGE — Real-Time Threat Oversight
 * 
 * Demonstrates complete Phase 3 page integration pattern:
 * - Component mounting to stable containers
 * - GlobalState subscriptions for reactive updates
 * - Socket connection status management
 * - Error boundaries and graceful degradation
 * 
 * Components used:
 * - MetricCard: System metrics (flows, alerts, blocked IPs, accuracy)
 * - AlertCard: Real-time security alerts with block action
 * - ActionCard: Recent security action history
 * - EngineCard: Detection engine status and performance
 * - ModuleCard: System health monitoring
 */

import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { AppToast } from '../components/app-toast.js';
import { MetricCard } from '../components/metric-card.js';
import { AlertCard } from '../components/alert-card.js';
import { ActionCard } from '../components/action-card.js';
import { EngineCard } from '../components/engine-card.js';
import { ModuleCard } from '../components/module-card.js';
import { LoadingSpinner } from '../components/loading-spinner.js';
import { debounce } from '../core/utils.js';

// ============================================================
// DOM CONTAINER REFERENCES
// ============================================================

const metricsGrid = document.getElementById('metrics-grid');
const alertsContainer = document.getElementById('alerts-container');
const actionsContainer = document.getElementById('actions-container');
const enginesGrid = document.getElementById('engines-grid');
const healthContainer = document.getElementById('health-container');

const connectionStatus = document.getElementById('connection-status');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const lastUpdateTime = document.getElementById('last-update-time');

// ============================================================
// CONNECTION STATUS MANAGEMENT
// ============================================================

/**
 * Update connection status indicator based on Socket state
 */
function updateConnectionStatus(state) {
    statusDot.classList.remove('bg-gray-500', 'bg-green-500', 'bg-amber-500', 'animate-pulse');
    
    switch (state) {
        case 'connected':
            statusDot.classList.add('bg-green-500');
            statusText.textContent = 'Connected';
            break;
        case 'connecting':
            statusDot.classList.add('bg-amber-500', 'animate-pulse');
            statusText.textContent = 'Connecting…';
            break;
        case 'disconnected':
            statusDot.classList.add('bg-gray-500');
            statusText.textContent = 'Disconnected';
            break;
        default:
            statusDot.classList.add('bg-gray-500');
            statusText.textContent = 'Unknown';
    }
}

// Monitor socket connection events
Socket.socket.on('connect', () => {
    updateConnectionStatus('connected');
    console.log('%c[Monitor] Socket connected', 'color:#10b981;font-weight:bold;');
});

Socket.socket.on('disconnect', () => {
    updateConnectionStatus('disconnected');
    console.warn('[Monitor] Socket disconnected');
});

Socket.socket.on('connect_error', (error) => {
    console.error('[Monitor] Connection error:', error);
    AppToast.warning('Connection issue — reconnecting…');
});

// Initial state
updateConnectionStatus('connecting');

// ============================================================
// METRICS RENDERING (MetricCard Grid)
// ============================================================

/**
 * Render system metrics in 4-column grid
 * Subscribes to metrics slice in GlobalState
 */
function renderMetrics() {
    GlobalState.subscribe('metrics', (metrics) => {
        // Clear skeleton loaders
        if (metricsGrid.querySelector('.animate-pulse')) {
            metricsGrid.innerHTML = '';
        }

        // Prepare metric array with standard keys
        const metricsList = [
            {
                label: 'Flows / sec',
                value: metrics.flows_per_second || 0,
                unit: '/sec',
                max: 5000,
                threshold: 4000,
                icon: 'arrow-lr',
                status: metrics.flows_per_second > 4000 ? 'warning' : 'normal'
            },
            {
                label: 'Alerts / min',
                value: metrics.alerts_per_minute || 0,
                unit: '/min',
                max: 100,
                threshold: 50,
                icon: 'exclamation-triangle',
                status: metrics.alerts_per_minute > 50 ? 'critical' : 'normal'
            },
            {
                label: 'Blocked IPs',
                value: metrics.blocked_ips_24h || 0,
                unit: 'today',
                max: 500,
                threshold: 250,
                icon: 'shield-slash',
                status: metrics.blocked_ips_24h > 250 ? 'warning' : 'normal'
            },
            {
                label: 'Model Accuracy',
                value: metrics.model_accuracy_percent || 0,
                unit: '%',
                max: 100,
                threshold: 85,
                icon: 'check-circle',
                status: metrics.model_accuracy_percent < 85 ? 'warning' : 'normal'
            }
        ];

        // Render each metric in grid
        metricsGrid.innerHTML = '';
        metricsList.forEach(metric => {
            const card = MetricCard(metric);
            metricsGrid.appendChild(card);
        });

        // Update last update time
        const now = new Date();
        lastUpdateTime.textContent = now.toLocaleTimeString();
    });
}

// ============================================================
// ALERTS RENDERING (AlertCard Stream)
// ============================================================

/**
 * Render real-time alerts in left panel
 * Shows most recent 10 alerts, scrollable
 * Each alert has block action (INT-002 fix)
 */
function renderAlerts() {
    GlobalState.subscribe('alerts', (alerts) => {
        // Clear skeleton if present
        if (alertsContainer.querySelector('.text-center')) {
            alertsContainer.innerHTML = '';
        }

        if (!alerts || alerts.length === 0) {
            alertsContainer.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="bi bi-inbox text-2xl mb-2 block"></i>
                    <p class="text-sm">No alerts yet</p>
                </div>
            `;
            return;
        }

        // Render most recent 10 alerts
        alertsContainer.innerHTML = '';
        const recentAlerts = alerts.slice(0, 10);
        
        recentAlerts.forEach(alert => {
            try {
                const card = AlertCard(alert);
                alertsContainer.appendChild(card);
            } catch (error) {
                console.error('[AlertCard] Render error:', error, alert);
                // Skip broken alerts, continue with next
            }
        });
    });
}

// ============================================================
// ACTIONS RENDERING (ActionCard Timeline)
// ============================================================

/**
 * Render recent security actions in right panel
 * Shows block, rate-limit, alert responses, etc.
 */
function renderActions() {
    GlobalState.subscribe('actions', (actions) => {
        // Clear skeleton if present
        if (actionsContainer.querySelector('.text-center')) {
            actionsContainer.innerHTML = '';
        }

        if (!actions || actions.length === 0) {
            actionsContainer.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="bi bi-inbox text-2xl mb-2 block"></i>
                    <p class="text-sm">No actions yet</p>
                </div>
            `;
            return;
        }

        // Render most recent 10 actions
        actionsContainer.innerHTML = '';
        const recentActions = actions.slice(0, 10);
        
        recentActions.forEach(action => {
            try {
                const card = ActionCard(action);
                actionsContainer.appendChild(card);
            } catch (error) {
                console.error('[ActionCard] Render error:', error, action);
                // Skip broken actions, continue with next
            }
        });
    });
}

// ============================================================
// ENGINES RENDERING (EngineCard Grid)
// ============================================================

/**
 * Render detection engine status cards
 * Shows: status, load, accuracy, model info, detection count
 */
function renderEngines() {
    GlobalState.subscribe('engines', (engines) => {
        // Clear skeletons if present
        if (enginesGrid.querySelector('.animate-pulse')) {
            enginesGrid.innerHTML = '';
        }

        if (!engines || Object.keys(engines).length === 0) {
            enginesGrid.innerHTML = `
                <div class="text-center py-8 text-gray-500 col-span-full">
                    <i class="bi bi-cpu text-2xl mb-2 block"></i>
                    <p class="text-sm">No engines available</p>
                </div>
            `;
            return;
        }

        // Render engine card for each engine
        enginesGrid.innerHTML = '';
        Object.entries(engines).forEach(([engineId, engineData]) => {
            try {
                const card = EngineCard({
                    id: engineId,
                    name: engineData.name || engineId,
                    status: engineData.status || 'unknown',
                    load: engineData.cpu_load_percent || 0,
                    accuracy: engineData.accuracy_percent || 0,
                    model: engineData.model_name || '—',
                    version: engineData.model_version || '—',
                    detections: engineData.detection_count || 0
                });
                enginesGrid.appendChild(card);
            } catch (error) {
                console.error('[EngineCard] Render error:', error, engineId);
                // Skip broken engines, continue with next
            }
        });
    });
}

// ============================================================
// HEALTH RENDERING (ModuleCard with Error Boundary)
// ============================================================

/**
 * Render system health status
 * Demonstrates ModuleCard error boundary pattern
 */
function renderHealth() {
    const healthCard = ModuleCard('system-health', {
        title: 'System Health',
        description: 'Real-time system health metrics',
        endpoint: '/api/health',
        refreshInterval: 10000
    });

    healthContainer.innerHTML = '';
    healthContainer.appendChild(healthCard);
}

// ============================================================
// SOCKET EVENT HANDLERS (Real-time Updates)
// ============================================================

/**
 * Socket events push data into GlobalState
 * Components automatically re-render via subscriptions
 */

// Metrics updates
Socket.on('metrics.update', (data) => {
    GlobalState.update('metrics', data);
    console.log('%c[Metrics Update]', 'color:#3b82f6;', data);
});

// New alerts
Socket.on('alert.new', (alert) => {
    GlobalState.push('alerts', alert);
    
    // Play alert tone based on severity
    if (alert.severity === 'critical' || alert.severity === 'high') {
        AppToast.error(`${alert.severity.toUpperCase()}: ${alert.title}`);
    } else {
        AppToast.warning(alert.title);
    }
    
    console.log('%c[Alert]', 'color:#ef4444;', alert);
});

// Action updates
Socket.on('action.update', (action) => {
    GlobalState.push('actions', action);
    console.log('%c[Action]', 'color:#10b981;', action);
});

// Engine updates
Socket.on('engine.update', (engineData) => {
    const engineId = engineData.id;
    GlobalState.update('engines', { [engineId]: engineData });
    console.log('%c[Engine Update]', 'color:#8b5cf6;', engineId, engineData);
});

// Health status updates
Socket.on('health.update', (healthData) => {
    GlobalState.update('health', healthData);
    console.log('%c[Health Update]', 'color:#f59e0b;', healthData);
});

// ============================================================
// PAGE INITIALIZATION
// ============================================================

/**
 * Initialize all components on page load
 * 1. Setup subscriptions (renders components on state change)
 * 2. Fetch initial data
 * 3. Listen for socket updates
 */

function initPage() {
    console.log('%c[Monitor] Page Initializing…', 'color:#0f1117;font-weight:bold;');

    // Step 1: Setup subscriptions (render functions called immediately with current state)
    renderMetrics();    // Mounts MetricCard grid
    renderAlerts();     // Mounts AlertCard stream
    renderActions();    // Mounts ActionCard timeline
    renderEngines();    // Mounts EngineCard grid
    renderHealth();     // Mounts ModuleCard with error boundary

    // Step 2: Trigger initial socket connection for real-time updates
    Socket.socket.emit('monitor:subscribe');
    
    console.log('%c[Monitor] Page Ready', 'color:#10b981;font-weight:bold;');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

// ============================================================
// UTILITY: Debounced last update timer
// ============================================================

const updateTimer = debounce(() => {
    const now = new Date();
    lastUpdateTime.textContent = now.toLocaleTimeString();
}, 1000);

// Update every time GlobalState changes
GlobalState.subscribe('metrics', updateTimer);
GlobalState.subscribe('alerts', updateTimer);
GlobalState.subscribe('actions', updateTimer);

