/**
 * /static/js/pages/alerts.page.js
 * 
 * Alerts page controller — Security alert management and filtering
 * 
 * Architecture:
 * - Subscribes to GlobalState.alerts
 * - Socket event handler: alert.new
 * - Filtering by severity (all, critical, high, medium, low)
 * - Search by classification/IP/message
 * - Bulk actions: block multiple, dismiss multiple
 * - Error boundaries on each component
 */

import { AlertCard } from '../components/alert-card.js';
import { AppToast } from '../components/app-toast.js';

// ============================================================
// STATE MANAGEMENT
// ============================================================

let currentFilter = 'all';
let searchQuery = '';
let selectedAlerts = new Set();

// ============================================================
// DOM REFERENCES
// ============================================================

const alertsList = document.getElementById('alerts-list');
const alertCount = document.getElementById('alert-count');
const severityFilters = document.querySelectorAll('.severity-filter');
const searchInput = document.getElementById('search-alerts');
const bulkActionsBar = document.getElementById('bulk-actions-bar');
const selectedCountSpan = document.getElementById('selected-count');
const bulkBlockBtn = document.getElementById('bulk-block-btn');
const bulkDismissBtn = document.getElementById('bulk-dismiss-btn');
const bulkCancelBtn = document.getElementById('bulk-cancel-btn');

// ============================================================
// FILTERING & SEARCH LOGIC
// ============================================================

/**
 * filterAlerts() — Apply severity and search filters
 * Returns: Filtered alert array
 */
function filterAlerts(alerts) {
    if (!alerts) return [];

    let filtered = alerts;

    // Apply severity filter
    if (currentFilter !== 'all') {
        filtered = filtered.filter(a => a.severity === currentFilter);
    }

    // Apply search filter
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(a => 
            (a.classification && a.classification.toLowerCase().includes(q)) ||
            (a.prediction && a.prediction.toLowerCase().includes(q)) ||
            (a.source_ip && a.source_ip.toLowerCase().includes(q)) ||
            (a.target_ip && a.target_ip.toLowerCase().includes(q)) ||
            (a.id && a.id.toString().includes(q))
        );
    }

    return filtered;
}

// ============================================================
// RENDER FUNCTIONS
// ============================================================

/**
 * renderAlerts() — Mount AlertCard components with filtering
 * - Subscribes to GlobalState.alerts
 * - Applies severity and search filters
 * - Updates on every change or filter change
 * - Error boundary: try-catch per card
 */
function renderAlerts() {
    GlobalState.subscribe('alerts', (alerts) => {
        const filtered = filterAlerts(alerts);
        alertCount.textContent = filtered.length;
        alertsList.innerHTML = '';

        if (!filtered || filtered.length === 0) {
            const emptyState = document.createElement('div');
            emptyState.className = 'text-center py-12 text-gray-500';
            emptyState.innerHTML = `
                <i class="bi bi-inbox text-4xl mb-4 block"></i>
                <p class="text-sm">${currentFilter === 'all' && !searchQuery ? 'No alerts' : 'No matching alerts'}</p>
            `;
            alertsList.appendChild(emptyState);
            return;
        }

        // Render each alert with error boundary
        filtered.forEach((alert, index) => {
            try {
                const card = AlertCard(alert);
                
                // Add selection checkbox
                const checkboxContainer = document.createElement('div');
                checkboxContainer.className = 'absolute top-4 left-4';
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.className = 'w-4 h-4 cursor-pointer';
                checkbox.dataset.alertId = alert.id;
                checkbox.addEventListener('change', (e) => {
                    if (e.target.checked) {
                        selectedAlerts.add(alert.id);
                    } else {
                        selectedAlerts.delete(alert.id);
                    }
                    updateBulkActionsBar();
                });
                checkboxContainer.appendChild(checkbox);
                
                // Insert checkbox into card (assumes card is wrapped)
                if (card.querySelector('[class*="border"]')) {
                    card.style.position = 'relative';
                    card.insertBefore(checkboxContainer, card.firstChild);
                }
                
                alertsList.appendChild(card);
            } catch (err) {
                console.error(`[AlertCard ${index}] Error:`, err);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'bg-[#151922] border border-red-500/50 rounded-lg p-4 text-red-400 text-sm';
                errorDiv.textContent = `Alert ${index + 1} failed to render`;
                alertsList.appendChild(errorDiv);
            }
        });
    });
}

/**
 * updateBulkActionsBar() — Show/hide bulk actions based on selection
 */
function updateBulkActionsBar() {
    if (selectedAlerts.size > 0) {
        bulkActionsBar.classList.remove('hidden');
        selectedCountSpan.textContent = `${selectedAlerts.size} selected`;
    } else {
        bulkActionsBar.classList.add('hidden');
        selectedAlerts.clear();
    }
}

// ============================================================
// EVENT HANDLERS — FILTERS & SEARCH
// ============================================================

/**
 * Severity filter buttons
 */
severityFilters.forEach(btn => {
    btn.addEventListener('click', () => {
        severityFilters.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.severity;
        
        // Force re-render by triggering GlobalState subscription
        const currentAlerts = GlobalState.data.alerts || [];
        GlobalState.set('alerts', currentAlerts);
    });
});

/**
 * Search input
 */
searchInput.addEventListener('input', (e) => {
    searchQuery = e.target.value;
    
    // Force re-render
    const currentAlerts = GlobalState.data.alerts || [];
    GlobalState.set('alerts', currentAlerts);
});

/**
 * Bulk block button
 */
bulkBlockBtn.addEventListener('click', async () => {
    if (selectedAlerts.size === 0) return;
    
    bulkBlockBtn.disabled = true;
    bulkBlockBtn.textContent = 'BLOCKING...';
    
    try {
        const blockRequests = Array.from(selectedAlerts).map(alertId => {
            // Find alert to get target IP
            const alerts = GlobalState.data.alerts || [];
            const alert = alerts.find(a => a.id === alertId);
            if (!alert) return null;
            
            return HttpClient.post('/api/actions', {
                alert_id: alertId,
                type: 'block',
                target_ip: alert.target_ip || alert.source_ip
            });
        }).filter(Boolean);
        
        await Promise.all(blockRequests);
        
        AppToast.success(`Blocked ${selectedAlerts.size} threat(s)`, 'Prevention actions queued');
        selectedAlerts.clear();
        updateBulkActionsBar();
    } catch (err) {
        console.error('[Bulk Block] Error:', err);
        AppToast.error('Block Failed', err.message);
    } finally {
        bulkBlockBtn.disabled = false;
        bulkBlockBtn.textContent = 'BLOCK ALL';
    }
});

/**
 * Bulk dismiss button
 */
bulkDismissBtn.addEventListener('click', async () => {
    if (selectedAlerts.size === 0) return;
    
    bulkDismissBtn.disabled = true;
    bulkDismissBtn.textContent = 'DISMISSING...';
    
    try {
        const dismissRequests = Array.from(selectedAlerts).map(alertId => {
            return HttpClient.post('/api/alerts/dismiss', {
                alert_id: alertId
            });
        });
        
        await Promise.all(dismissRequests);
        
        AppToast.success(`Dismissed ${selectedAlerts.size} alert(s)`, 'Alerts marked as reviewed');
        selectedAlerts.clear();
        updateBulkActionsBar();
    } catch (err) {
        console.error('[Bulk Dismiss] Error:', err);
        AppToast.error('Dismiss Failed', err.message);
    } finally {
        bulkDismissBtn.disabled = false;
        bulkDismissBtn.textContent = 'DISMISS';
    }
});

/**
 * Bulk cancel button
 */
bulkCancelBtn.addEventListener('click', () => {
    selectedAlerts.clear();
    updateBulkActionsBar();
    
    // Uncheck all checkboxes
    document.querySelectorAll('[data-alert-id]').forEach(cb => {
        cb.checked = false;
    });
});

// ============================================================
// SOCKET EVENT HANDLERS
// ============================================================

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
 * Socket event: alert.dismissed
 * - Fired when an alert is dismissed
 * - Removes from GlobalState.alerts
 */
Socket.on('alert.dismissed', (data) => {
    console.log('[Socket] alert.dismissed:', data);
    const alerts = GlobalState.data.alerts || [];
    const updated = alerts.filter(a => a.id !== data.alert_id);
    GlobalState.set('alerts', updated);
});

// ============================================================
// PAGE INITIALIZATION
// ============================================================

/**
 * initPage() — Setup all subscriptions and render functions
 */
function initPage() {
    try {
        console.log('[Alerts] Initializing...');
        
        // Mount component render function
        renderAlerts();
        
        console.log('[Alerts] Page initialized successfully');
        AppToast.success('Alerts loaded', 'Ready to review security events');
    } catch (err) {
        console.error('[Alerts] Init error:', err);
        AppToast.error('Alerts Error', 'Failed to initialize page. Check console for details.');
    }
}

// ============================================================
// STARTUP
// ============================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

