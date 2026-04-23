/**
 * INIDS Demo Platform - Dashboard Controller
 * Handles card interactions, module loading, and navigation
 */

class DashboardController {
    constructor() {
        this.moduleModal = null;
        this.currentModule = null;
        
        this.moduleRegistry = {
            'real-time-detection': {
                title: 'Real-Time Detection Panel',
                route: '/modules/real-time-detection',
                description: 'Live event stream showing real-time threat detection'
            },
            'multi-engine': {
                title: 'Multi-Engine Voting System',
                route: '/modules/multi-engine',
                description: 'Five detection engines voting on verdict consensus'
            },
            'risk-score': {
                title: 'Risk Score Visualizer',
                route: '/modules/risk-score',
                description: 'Multi-factor risk calculation with animated gauges'
            },
            'auto-blocking': {
                title: 'Automated Blocking',
                route: '/modules/auto-blocking',
                description: 'Detection to firewall block execution timeline'
            },
            'approval-workflow': {
                title: 'Approval Workflow',
                route: '/modules/approval-workflow',
                description: 'Human-in-the-loop review process'
            },
            'false-positive': {
                title: 'False Positive Learning',
                route: '/modules/false-positive',
                description: 'Feedback-driven system learning'
            },
            'threat-intel': {
                title: 'Threat Intelligence Enrichment',
                route: '/modules/threat-intel',
                description: 'External reputation checks and badging'
            },
            'anomaly-learning': {
                title: 'Anomaly Learning Engine',
                route: '/modules/anomaly-learning',
                description: 'Behavioral baseline and deviation detection'
            },
            'analytics': {
                title: 'Analytics Dashboard',
                route: '/modules/analytics',
                description: 'Security posture metrics and insights'
            },
            'escalation': {
                title: 'Escalation State Machine',
                route: '/modules/escalation',
                description: 'Per-IP escalation progression'
            },
            'pipeline-monitor': {
                title: 'Pipeline Monitor',
                route: '/modules/pipeline-monitor',
                description: 'Ingestion throughput and health metrics'
            },
            'policy-tuning': {
                title: 'Policy Tuning Simulator',
                route: '/modules/policy-tuning',
                description: 'Interactive policy parameter adjustment'
            },
            'alert-lifecycle': {
                title: 'Alert Lifecycle Manager',
                route: '/modules/alert-lifecycle',
                description: 'SOC workflow in Kanban board'
            },
            'engine-playground': {
                title: 'Engine Toggle Playground',
                route: '/modules/engine-playground',
                description: 'Disable engines to see coverage impact'
            },
            'pattern-detector': {
                title: 'Behavioral Pattern Detector',
                route: '/modules/pattern-detector',
                description: 'Network graph attack pattern visualization'
            }
        };

        this.demoMode = false;
        this.init();
    }

    // ==============================
    // 🔥 SOC THREAT STATE SYSTEM (FIXED POSITION)
    // ==============================

    syncThreatState(alertCount) {
        const statusStrip = document.querySelector('.status-strip');

        if (typeof window.updateThreatState === 'function') {
            window.updateThreatState(alertCount);
        }

        if (!statusStrip) {
            return;
        }

        statusStrip.classList.toggle('threat-active', alertCount > 0);
    }

    updateGlobalStatus(alertCount) {
        const statusTag = document.querySelector('.panel-tag');

        if (!statusTag) return;

        if (alertCount > 0) {
            statusTag.textContent = '🚨 UNDER ATTACK';
            statusTag.classList.remove('tag-green');
            statusTag.classList.add('tag-red', 'pulse-soft');
        } else {
            statusTag.textContent = '✅ NORMAL';
            statusTag.classList.remove('tag-red', 'pulse-soft');
            statusTag.classList.add('tag-green');
        }
    }

    init() {
        this.setupModal();
        this.attachCardListeners();
        this.attachControlListeners();
        this.subscribeToState();
        this.loadInitialMetrics();
    }

    subscribeToState() {
        GlobalState.subscribe(data => {
            this.updateDashboardMetrics(data);
        });
    }

    setupModal() {
        const modalElement = document.getElementById('moduleModal');
        if (modalElement) {
            this.moduleModal = new bootstrap.Modal(modalElement);
        }
    }

    attachCardListeners() {
        const cards = document.querySelectorAll('.capability-card');
        cards.forEach(card => {
            // Add keyboard accessibility (Enter key)
            card.setAttribute('tabindex', '0');
            card.setAttribute('role', 'button');
            
            card.addEventListener('click', (e) => {
                e.preventDefault();
                const module = card.dataset.module;
                this.openModule(module);
            });
            
            card.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const module = card.dataset.module;
                    this.openModule(module);
                }
            });
        });
    }

    attachControlListeners() {
        // Demo Mode Toggle
        const demoModeToggle = document.getElementById('demoModeToggle');
        if (demoModeToggle) {
            demoModeToggle.addEventListener('click', () => this.toggleDemoMode());
        }

        // Refresh Button
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshAllData());
        }

        // Settings Button (placeholder)
        const settingsBtn = document.getElementById('settingsBtn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.openSettings());
        }
    }

    openModule(moduleKey) {
        const moduleConfig = this.moduleRegistry[moduleKey];
        if (!moduleConfig) {
            console.error(`Unknown module: ${moduleKey}`);
            return;
        }

        this.currentModule = moduleKey;
        const modalTitle = document.getElementById('moduleTitle');
        const modalContent = document.getElementById('moduleContent');

        if (modalTitle) {
            modalTitle.textContent = moduleConfig.title;
        }

        // Show loading state
        if (modalContent) {
            modalContent.innerHTML = `
                <div class="text-center py-5">
                    <div class="spinner-border text-info" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2 text-muted">Loading ${moduleConfig.title}...</p>
                </div>
            `;
        }

        // Open modal
        if (this.moduleModal) {
            this.moduleModal.show();
        }

        // Load module content
        this.loadModuleContent(moduleConfig);
    }

    loadModuleContent(config, retryCount = 0) {
        const maxRetries = 2;
        const timeoutMs = 15000;
        
        // Create abort controller for timeout
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), timeoutMs);
        
        fetch(config.route, {
            method: 'GET',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': 'text/html'
            },
            signal: controller.signal
        })
        .then(response => {
            clearTimeout(timeout);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.text();
        })
        .then(html => {
            const modalContent = document.getElementById('moduleContent');
            if (modalContent) {
                modalContent.innerHTML = html;
                // Re-initialize any scripts/components in the loaded content
                this.initializeModuleScripts();
            }
        })
        .catch(error => {
            clearTimeout(timeout);
            console.error(`Error loading ${config.title}:`, error.message);
            
            const modalContent = document.getElementById('moduleContent');
            if (modalContent) {
                // Check if it's a network/timeout error
                const isNetworkError = error.name === 'AbortError' || !navigator.onLine;
                
                let retryButton = '';
                if (retryCount < maxRetries) {
                    retryButton = `
                        <button class="btn btn-sm btn-primary mt-3" 
                                onclick="window.dashboard.loadModuleContent(
                                    ${JSON.stringify(config)}, ${retryCount + 1}
                                )">
                            Retry (${maxRetries - retryCount} left)
                        </button>
                    `;
                }
                
                modalContent.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        <h4 class="alert-heading">⚠️ ${isNetworkError ? 'Connection Error' : 'Module Not Available'}</h4>
                        <p>${isNetworkError 
                            ? 'Could not connect to the server. Please check your connection.' 
                            : `The ${config.title} module is still in development.`
                        }</p>
                        <hr>
                        <p class="mb-0 text-muted" style="font-size: 0.9em;">
                            Error: ${error.message}
                        </p>
                        ${retryButton}
                    </div>
                `;
            }
        });
    }

    initializeModuleScripts() {
        // This method will be called to reinitialize any JavaScript in loaded modules
        // Useful for charts, animations, etc.
        console.log('Module scripts initialized');
    }

    toggleDemoMode() {
        this.demoMode = !this.demoMode;
        const demoModeText = document.getElementById('demoModeText');
        const demoModeToggle = document.getElementById('demoModeToggle');

        if (demoModeText) {
            demoModeText.textContent = this.demoMode ? 'Demo Mode: ON' : 'Demo Mode: OFF';
        }

        if (demoModeToggle) {
            demoModeToggle.classList.toggle('btn-outline-warning', !this.demoMode);
            demoModeToggle.classList.toggle('btn-warning', this.demoMode);
        }

        // Trigger demo traffic if enabled
        if (this.demoMode) {
            this.startDemoTraffic();
        } else {
            this.stopDemoTraffic();
        }
    }

    startDemoTraffic() {
        console.log('Starting demo traffic simulation...');
        // Call backend to start demo traffic
        fetch('/api/demo/start', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('Demo started:', data);
                this.showNotification('Demo Mode Activated', 'Generating synthetic attack traffic...', 'info');
            })
            .catch(error => {
                console.error('Error starting demo:', error);
                this.showNotification('Demo Mode Error', 'Could not start demo traffic', 'danger');
            });
    }

    stopDemoTraffic() {
        console.log('Stopping demo traffic simulation...');
        // Call backend to stop demo traffic
        fetch('/api/demo/stop', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('Demo stopped:', data);
                this.showNotification('Demo Mode Deactivated', 'Demo traffic stopped', 'info');
            })
            .catch(error => {
                console.error('Error stopping demo:', error);
            });
    }

    refreshAllData() {
        console.log('Refreshing all data...');
        // Show visual feedback
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '⟳ Refreshing...';
            refreshBtn.classList.add('pulse-soft');
        }

        // Call backend refresh endpoint
        fetch('/api/dashboard/refresh', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                GlobalState.set({ lastRefresh: data.timestamp || new Date().toISOString() });
                if (window.INIDSSocketManager && typeof window.INIDSSocketManager.hydrate === 'function') {
                    window.INIDSSocketManager.hydrate().catch(error => {
                        console.error('Error hydrating shared state:', error);
                    });
                }
                this.showNotification('Refresh Complete', 'All data updated', 'success');
            })
            .catch(error => {
                console.error('Error refreshing data:', error);
                this.showNotification('Refresh Failed', 'Could not refresh data', 'danger');
            })
            .finally(() => {
                if (refreshBtn) {
                    refreshBtn.disabled = false;
                    refreshBtn.innerHTML = '⟳ Refresh';
                    refreshBtn.classList.remove('pulse-soft');
                }
            });
    }

    loadInitialMetrics() {
        if (Object.keys(GlobalState.data || {}).length > 0) {
            this.updateDashboardMetrics(GlobalState.data);
            return;
        }

        if (window.INIDSSocketManager && typeof window.INIDSSocketManager.hydrate === 'function') {
            window.INIDSSocketManager.hydrate().catch(error => {
                console.error('Error loading shared dashboard state:', error);
                this.useMockMetrics();
            });
            return;
        }

        this.useMockMetrics();
    }

    updateDashboardMetrics(data) {
        if (!data || typeof data !== 'object') {
            return;
        }

        // Update system uptime
        if (data.system_uptime !== undefined) {
            const uptimeEl = document.getElementById('systemUptime');
            if (uptimeEl) uptimeEl.textContent = data.system_uptime;
        }

        // Update system health
        if (data.system_health !== undefined) {
            const healthEl = document.getElementById('systemHealth');
            if (healthEl) {
                healthEl.textContent = data.system_health + '%';
                healthEl.className = 'text-' + (data.system_health > 90 ? 'success' : 'warning');
            }
        }

        // Update system capacity
        if (data.system_capacity !== undefined) {
            const capacityEl = document.getElementById('systemCapacity');
            if (capacityEl) {
                capacityEl.textContent = data.system_capacity + '%';
                capacityEl.className = 'text-' + (data.system_capacity < 80 ? 'info' : 'warning');
            }
        }

        // Update threat counters
        if (data.active_attacks !== undefined) {
            const attacksEl = document.querySelector('.threat-counter:nth-child(1) .count-badge');
            if (attacksEl) attacksEl.textContent = data.active_attacks;
        }

        if (data.blocked !== undefined) {
            const blockedEl = document.querySelector('.threat-counter:nth-child(2) .count-badge');
            if (blockedEl) blockedEl.textContent = data.blocked;
        }

        if (data.active_alerts !== undefined) {
            const alertsEl = document.querySelector('.threat-counter:nth-child(3) .count-badge');

            if (alertsEl) {
                alertsEl.textContent = data.active_alerts;
                alertsEl.classList.toggle('pulse-soft', data.active_alerts > 0);
            }

            this.syncThreatState(data.active_alerts);
            this.updateGlobalStatus(data.active_alerts);

            const sidebarAlerts = document.querySelector('.sidebar-section .stat-mini:nth-child(2) .stat-mini-val');
            if (sidebarAlerts) {
                sidebarAlerts.textContent = data.active_alerts;
                sidebarAlerts.classList.toggle('pulse-soft', data.active_alerts > 0);
            }
        }

        if (data.under_review !== undefined) {
            const reviewEl = document.querySelector('.threat-counter:nth-child(4) .count-badge');
            if (reviewEl) reviewEl.textContent = data.under_review;
        }

        // Update last hour metrics
        if (data.last_hour_attacks !== undefined) {
            const lastHourEl = document.getElementById('lastHourAttacks');
            if (lastHourEl) lastHourEl.textContent = data.last_hour_attacks;
        }

        if (data.last_hour_blocks !== undefined) {
            const lastHourBlocksEl = document.getElementById('lastHourBlocks');
            if (lastHourBlocksEl) lastHourBlocksEl.textContent = data.last_hour_blocks;
        }

        if (data.fp_rate !== undefined) {
            const fpRateEl = document.getElementById('fpRate');
            if (fpRateEl) fpRateEl.textContent = data.fp_rate + '%';
        }
    }

    useMockMetrics() {
        // Fallback for development without backend
        const mockData = {
            system_uptime: '4.2h',
            system_health: 98,
            system_capacity: 45,
            active_attacks: 3,
            blocked: 5,
            active_alerts: 12,
            under_review: 2,
            last_hour_attacks: 127,
            last_hour_blocks: 23,
            fp_rate: 8
        };
        this.updateDashboardMetrics(mockData);
    }

    openSettings() {
        this.showNotification('Settings', 'Settings panel coming soon', 'info');
    }

    showNotification(title, message, type = 'info') {
        // Create toast notification
        const toastHtml = `
            <div class="toast align-items-center text-white bg-${this.mapAlertType(type)} border-0" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="d-flex">
                    <div class="toast-body">
                        <strong>${title}:</strong> ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        // Append to container (create if doesn't exist)
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(toastContainer);
        }

        const toastElement = document.createElement('div');
        toastElement.innerHTML = toastHtml;
        toastContainer.appendChild(toastElement.firstElementChild);

        const toast = new bootstrap.Toast(toastElement.firstElementChild);
        toast.show();

        // Auto-remove element after toast disappears
        toastElement.firstElementChild.addEventListener('hidden.bs.toast', () => {
            toastElement.firstElementChild.remove();
        });
    }

    mapAlertType(type) {
        const typeMap = {
            'info': 'info',
            'success': 'success',
            'warning': 'warning',
            'danger': 'danger'
        };
        return typeMap[type] || 'info';
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DashboardController();
    console.log('INIDS Dashboard initialized');
});
