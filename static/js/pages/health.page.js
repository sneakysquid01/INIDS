import { ModuleCard } from '../components/module-card.js';
import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global_state.js';
import { Socket } from '../core/socket_manager.js';
import { HttpClient } from '../core/http_client.js';

// DOM references
const healthModules = document.getElementById('health-modules');
const statHealthy = document.getElementById('stat-healthy');
const statIssues = document.getElementById('stat-issues');
const healthCpuBar = document.getElementById('health-cpu-bar');
const healthCpuText = document.getElementById('health-cpu-text');
const healthMemBar = document.getElementById('health-mem-bar');
const healthMemText = document.getElementById('health-mem-text');
const healthUptime = document.getElementById('health-uptime');
const healthLastcheck = document.getElementById('health-lastcheck');

/**
 * Render health modules
 */
function renderHealth() {
    GlobalState.subscribe('health', (health) => {
        healthModules.innerHTML = '';
        
        if (!health || Object.keys(health).length === 0) {
            healthModules.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        No health data available
                    </div>
                </div>
            `;
            return;
        }
        
        // Count healthy and issues
        let healthyCount = 0;
        let issueCount = 0;
        const modules = Object.entries(health);
        
        modules.forEach(([name, status]) => {
            if (status === 'healthy' || status === 'ok') {
                healthyCount++;
            } else if (status === 'warning' || status === 'error') {
                issueCount++;
            }
        });
        
        statHealthy.textContent = healthyCount;
        statIssues.textContent = issueCount;
        
        // Render module cards
        modules.forEach(([name, status]) => {
            try {
                const moduleData = {
                    name,
                    status: status === 'ok' ? 'healthy' : status,
                    timestamp: new Date()
                };
                
                const card = ModuleCard(moduleData);
                healthModules.appendChild(card);
            } catch (err) {
                console.error(`Error rendering module ${name}:`, err);
                // Show error card for this specific module
                const errorCard = document.createElement('div');
                errorCard.className = 'bg-[#151922] border border-[#ef4444] rounded-lg p-4 text-[#ef4444] text-sm';
                errorCard.textContent = `Failed to load module: ${name}`;
                healthModules.appendChild(errorCard);
            }
        });
    });
}

/**
 * Update system metrics
 */
function updateMetrics(metrics) {
    if (!metrics) return;
    
    // CPU usage
    const cpuUsage = metrics.cpu_usage || 0;
    healthCpuBar.style.width = cpuUsage + '%';
    healthCpuText.textContent = Math.round(cpuUsage) + '%';
    
    // Memory usage
    const memUsage = metrics.memory_usage || 0;
    healthMemBar.style.width = memUsage + '%';
    healthMemText.textContent = Math.round(memUsage) + '%';
    
    // Uptime
    const uptime = metrics.uptime_days || 0;
    healthUptime.textContent = uptime.toFixed(1);
    
    // Last check
    const lastCheck = metrics.last_check_seconds_ago || 0;
    if (lastCheck < 60) {
        healthLastcheck.textContent = Math.round(lastCheck) + 's';
    } else if (lastCheck < 3600) {
        healthLastcheck.textContent = Math.round(lastCheck / 60) + 'm';
    } else {
        healthLastcheck.textContent = Math.round(lastCheck / 3600) + 'h';
    }
}

/**
 * Initialize page
 */
async function initPage() {
    // Load initial health from API
    try {
        const response = await HttpClient.get('/api/health');
        const healthData = response.health || response || {};
        GlobalState.set('health', healthData);
        
        // Update metrics
        if (response.metrics) {
            updateMetrics(response.metrics);
        }
    } catch (err) {
        console.error('Failed to load health data:', err);
        AppToast.error('Failed to load system health');
    }
    
    // Render health modules
    renderHealth();
    
    // Setup socket handlers
    Socket.on('health.update', (health) => {
        GlobalState.set('health', health);
        renderHealth();
    });
    
    Socket.on('health.metrics', (metrics) => {
        updateMetrics(metrics);
    });
    
    Socket.on('module.status', (data) => {
        const health = GlobalState.state.health || {};
        health[data.module_name] = data.status;
        GlobalState.set('health', health);
        renderHealth();
    });
}

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, renderHealth, updateMetrics };
