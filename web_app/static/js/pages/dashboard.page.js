/**
 * /static/js/pages/dashboard.page.js
 * 
 * Dashboard page controller — System modules and metrics overview
 * 
 * Architecture:
 * - Subscribes to GlobalState slices: modules, metrics
 * - Socket event handlers: module.update, metrics.update
 * - Renders ModuleCard grid (15+ modules)
 * - Renders metric quickstats
 * - Displays recent alerts and actions
 * - Error boundaries on each component
 */

import { ModuleCard } from '../components/module-card.js';
import { MetricCard } from '../components/metric-card.js';
import { AlertCard } from '../components/alert-card.js';
import { ActionCard } from '../components/action-card.js';
import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';

// ============================================================
// DOM REFERENCES
// ============================================================

const modulesGrid = document.getElementById('modules-grid');
const recentAlertsContainer = document.getElementById('recent-alerts');
const recentActionsContainer = document.getElementById('recent-actions');

const statIngested = document.getElementById('stat-ingested');
const statProcessed = document.getElementById('stat-processed');
const statAlerts = document.getElementById('stat-alerts');
const statBlocked = document.getElementById('stat-blocked');

const systemStatus = document.getElementById('system-status');
const statusIndicator = document.getElementById('status-indicator');
const systemHealth = document.getElementById('system-health');

// ============================================================
// RENDER FUNCTIONS
// ============================================================

/**
 * renderModules() — Mount ModuleCard components for each module
 * - Subscribes to GlobalState.modules
 * - Updates on every change
 * - Error boundary: try-catch catches individual module errors
 */
function renderModules() {
    GlobalState.subscribe('modules', (modules) => {
        modulesGrid.innerHTML = '';
        
        // Ensure modules is an array
        if (!Array.isArray(modules) || modules.length === 0) {
            modulesGrid.innerHTML = '<div class="text-gray-500 col-span-full text-center py-8">No modules configured</div>';
            return;
        }

        modules.forEach((module, index) => {
            try {
                const card = ModuleCard(module);
                modulesGrid.appendChild(card);
            } catch (err) {
                console.error(`[ModuleCard ${index}] Error:`, err);
                // Isolated error — other modules continue rendering
                const errorDiv = document.createElement('div');
                errorDiv.className = 'bg-[#151922] border border-red-500/50 rounded-lg p-4 text-red-400 text-sm';
                errorDiv.textContent = `Module ${index + 1} failed to load`;
                modulesGrid.appendChild(errorDiv);
            }
        });
    });
}

/**
 * renderMetrics() — Update quick stat badges
 * - Subscribes to GlobalState.metrics
 * - Updates counters: ingested, processed, alerts, blocked
 */
function renderMetrics() {
    GlobalState.subscribe('metrics', (metrics) => {
        try {
            const ingested = metrics.ingested_total || 0;
            const processed = metrics.processed_ingestion_total || 0;
            const alerts = metrics.alerts_total || 0;
            const blocked = metrics.prevention_actions_total || 0;

            statIngested.textContent = ingested.toLocaleString();
            statProcessed.textContent = processed.toLocaleString();
            statAlerts.textContent = alerts.toLocaleString();
            statBlocked.textContent = blocked.toLocaleString();

            // Update system health indicator
            if (alerts > 0) {
                statusIndicator.className = 'w-3 h-3 bg-red-500 rounded-full animate-pulse';
                systemHealth.className = 'text-sm font-mono text-red-400';
                systemHealth.textContent = 'Under Attack';
                systemStatus.className = 'flex items-center gap-2 px-3 py-2 bg-red-500/20 border border-red-500/50 rounded-lg';
            } else {
                statusIndicator.className = 'w-3 h-3 bg-green-500 rounded-full';
                systemHealth.className = 'text-sm font-mono text-green-400';
                systemHealth.textContent = 'Healthy';
                systemStatus.className = 'flex items-center gap-2 px-3 py-2 bg-green-500/20 border border-green-500/50 rounded-lg';
            }
        } catch (err) {
            console.error('[renderMetrics] Error:', err);
        }
    });
}

/**
 * renderAlerts() — Display recent alerts
 * - Subscribes to GlobalState.alerts
 * - Shows 5 most recent
 * - Each AlertCard has block action integrated
 */
function renderAlerts() {
    GlobalState.subscribe('alerts', (alerts) => {
        recentAlertsContainer.innerHTML = '';
        
        if (!alerts || alerts.length === 0) {
            recentAlertsContainer.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="bi bi-inbox text-2xl mb-2 block"></i>
                    <p class="text-sm">No recent alerts</p>
                </div>
            `;
            return;
        }

        // Show 5 most recent
        const recent = alerts.slice(0, 5);
        
        recent.forEach((alert, index) => {
            try {
                const card = AlertCard(alert);
                recentAlertsContainer.appendChild(card);
            } catch (err) {
                console.error(`[AlertCard ${index}] Error:`, err);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'text-red-400 text-xs p-2 bg-red-500/10 rounded';
                errorDiv.textContent = `Alert ${index + 1} failed to render`;
                recentAlertsContainer.appendChild(errorDiv);
            }
        });
    });
}

/**
 * renderActions() — Display recent prevention actions
 * - Subscribes to GlobalState.actions
 * - Shows 5 most recent
 * - ActionCard displays action type, target, status, reason
 */
function renderActions() {
    GlobalState.subscribe('actions', (actions) => {
        recentActionsContainer.innerHTML = '';
        
        if (!actions || actions.length === 0) {
            recentActionsContainer.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="bi bi-inbox text-2xl mb-2 block"></i>
                    <p class="text-sm">No recent actions</p>
                </div>
            `;
            return;
        }

        // Show 5 most recent
        const recent = actions.slice(0, 5);
        
        recent.forEach((action, index) => {
            try {
                const card = ActionCard(action);
                recentActionsContainer.appendChild(card);
            } catch (err) {
                console.error(`[ActionCard ${index}] Error:`, err);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'text-amber-400 text-xs p-2 bg-amber-500/10 rounded';
                errorDiv.textContent = `Action ${index + 1} failed to render`;
                recentActionsContainer.appendChild(errorDiv);
            }
        });
    });
}

// ============================================================
// SOCKET EVENT HANDLERS
// ============================================================

/**
 * Socket event: module.update
 * - Fired when a module status changes (e.g., enabled/disabled, error state)
 * - Updates GlobalState.modules → triggers renderModules()
 */
Socket.on('module.update', (data) => {
    console.log('[Socket] module.update:', data);
    GlobalState.update('modules', data);
});

/**
 * Socket event: metrics.update
 * - Fired when metrics change (ingested, processed, alerts, blocked counts)
 * - Updates GlobalState.metrics → triggers renderMetrics()
 */
Socket.on('metrics.update', (data) => {
    console.log('[Socket] metrics.update:', data);
    GlobalState.update('metrics', data);
});

/**
 * Socket event: alert.new
 * - Fired when a new alert is generated
 * - Pushes to GlobalState.alerts → triggers renderAlerts()
 */
Socket.on('alert.new', (data) => {
    console.log('[Socket] alert.new:', data);
    GlobalState.push('alerts', data);
});

/**
 * Socket event: action.update
 * - Fired when a new prevention action is logged
 * - Pushes to GlobalState.actions → triggers renderActions()
 */
Socket.on('action.update', (data) => {
    console.log('[Socket] action.update:', data);
    GlobalState.push('actions', data);
});

// ============================================================
// PAGE INITIALIZATION
// ============================================================

/**
 * initPage() — Setup all subscriptions and render functions
 * - Called once on page load
 * - Establishes GlobalState subscriptions
 * - Triggers initial render with latest data
 */
function initPage() {
    try {
        console.log('[Dashboard] Initializing...');
        
        // Mount component render functions
        renderModules();
        renderMetrics();
        renderAlerts();
        renderActions();
        
        console.log('[Dashboard] Page initialized successfully');
        AppToast.success('Dashboard loaded', 'System is ready for monitoring');
    } catch (err) {
        console.error('[Dashboard] Init error:', err);
        AppToast.error('Dashboard Error', 'Failed to initialize page. Check console for details.');
    }
}

// ============================================================
// STARTUP
// ============================================================

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}
