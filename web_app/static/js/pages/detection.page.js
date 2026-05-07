import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient_Instance as HttpClient } from "../core/http-client.js";

// DOM references
const detectionInput = document.getElementById('detection-input');
const scanBtn = document.getElementById('scan-btn');
const loadingSpinner = document.getElementById('loading-spinner');
const resultsEmpty = document.getElementById('results-empty');
const resultsContainer = document.getElementById('results-container');
const statScans = document.getElementById('stat-scans');
const statThreats = document.getElementById('stat-threats');
const recentScans = document.getElementById('recent-scans');
const engineCheckboxes = document.querySelectorAll('.engine-select');
const inputTypeTabs = document.querySelectorAll('.input-type-tab');

// State
let currentInputType = 'ip';
let detectionHistory = [];

/**
 * Format detection result as card
 */
function formatDetectionResult(result) {
    const severity = result.severity || 'unknown';
    const severityColor = {
        'critical': '#ef4444',
        'high': '#f59e0b',
        'medium': '#3b82f6',
        'low': '#10b981',
        'clean': '#10b981'
    }[severity] || '#8f9099';
    
    const card = document.createElement('div');
    card.className = 'bg-[#0a0c10] border border-[#1a1f2e] rounded-lg p-4';
    card.innerHTML = `
        <div class="flex items-start justify-between mb-3">
            <div>
                <div class="text-white font-semibold text-sm">${result.target}</div>
                <div class="text-[#8f9099] text-xs mt-1">${result.analysis_type}</div>
            </div>
            <div class="flex items-center gap-2">
                <div class="w-3 h-3 rounded-full" style="background-color: ${severityColor}"></div>
                <span class="text-xs font-medium uppercase" style="color: ${severityColor}">${severity}</span>
            </div>
        </div>
        
        <div class="bg-[#151922] rounded px-3 py-2 mb-3 text-xs text-[#8f9099]">
            <div class="font-medium text-white mb-1">Detection Summary</div>
            <div>${result.summary || 'No threats detected'}</div>
        </div>
        
        <div class="grid grid-cols-3 gap-2 text-xs">
            <div class="bg-[#151922] rounded px-2 py-1.5">
                <div class="text-[#8f9099]">Confidence</div>
                <div class="text-white font-semibold">${result.confidence || 0}%</div>
            </div>
            <div class="bg-[#151922] rounded px-2 py-1.5">
                <div class="text-[#8f9099]">Engines Hit</div>
                <div class="text-white font-semibold">${result.engines_triggered || 0}</div>
            </div>
            <div class="bg-[#151922] rounded px-2 py-1.5">
                <div class="text-[#8f9099]">Time</div>
                <div class="text-white font-semibold">${new Date(result.timestamp).toLocaleTimeString()}</div>
            </div>
        </div>
        
        ${result.engine_details ? `
            <div class="mt-3 pt-3 border-t border-[#1a1f2e] text-xs">
                <div class="text-[#8f9099] font-medium mb-2">Engine Details</div>
                <div class="space-y-1">
                    ${Object.entries(result.engine_details).map(([engine, status]) => `
                        <div class="flex items-center gap-2">
                            <span class="w-1.5 h-1.5 rounded-full" style="background-color: ${status ? '#ef4444' : '#10b981'}"></span>
                            <span class="text-[#8f9099]">${engine}</span>
                            <span class="ml-auto font-semibold" style="color: ${status ? '#ef4444' : '#10b981'}">${status ? 'TRIGGERED' : 'CLEAN'}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        ` : ''}
    `;
    
    return card;
}

/**
 * Add scan to recent history
 */
function addToHistory(target, type, result) {
    const entry = {
        target,
        type,
        severity: result.severity,
        timestamp: new Date()
    };
    
    detectionHistory.unshift(entry);
    if (detectionHistory.length > 5) {
        detectionHistory.pop();
    }
    
    updateRecentScans();
}

/**
 * Update recent scans display
 */
function updateRecentScans() {
    recentScans.innerHTML = '';
    
    if (detectionHistory.length === 0) {
        recentScans.innerHTML = '<div class="text-[#6b7280] text-xs">No recent scans</div>';
        return;
    }
    
    detectionHistory.forEach(scan => {
        const item = document.createElement('div');
        const sevColor = {
            'critical': '#ef4444',
            'high': '#f59e0b',
            'medium': '#3b82f6',
            'low': '#10b981',
            'clean': '#10b981'
        }[scan.severity] || '#8f9099';
        
        item.className = 'p-2 bg-[#0a0c10] border border-[#1a1f2e] rounded cursor-pointer hover:border-[#3b82f6] transition-colors';
        item.innerHTML = `
            <div class="flex items-center gap-2 justify-between">
                <div class="flex-1 min-w-0">
                    <div class="text-white text-xs font-medium truncate">${scan.target}</div>
                    <div class="text-[#8f9099] text-xs mt-0.5">${scan.type}</div>
                </div>
                <div class="w-2 h-2 rounded-full flex-shrink-0" style="background-color: ${sevColor}"></div>
            </div>
        `;
        
        item.addEventListener('click', () => {
            detectionInput.value = scan.target;
            runScan();
        });
        
        recentScans.appendChild(item);
    });
}

/**
 * Run detection scan
 */
async function runScan() {
    const target = detectionInput.value.trim();
    
    if (!target) {
        AppToast.warning('Please enter a target to scan');
        return;
    }
    
    const selectedEngines = Array.from(engineCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);
    
    if (selectedEngines.length === 0) {
        AppToast.warning('Please select at least one detection engine');
        return;
    }
    
    // Show loading state
    scanBtn.disabled = true;
    loadingSpinner.style.display = 'block';
    resultsEmpty.style.display = 'none';
    resultsContainer.style.display = 'none';
    resultsContainer.innerHTML = '';
    
    try {
        // Call detection API
        const response = await HttpClient.post('/api/detection/analyze', {
            target,
            input_type: currentInputType,
            engines: selectedEngines
        });
        
        // Extract result (handle different response formats)
        const result = response.result || response;
        
        // Update stats
        statScans.textContent = parseInt(statScans.textContent || 0) + 1;
        if (result.severity && ['critical', 'high', 'medium'].includes(result.severity)) {
            statThreats.textContent = parseInt(statThreats.textContent || 0) + 1;
        }
        
        // Add to history
        addToHistory(target, currentInputType, result);
        
        // Display result
        resultsEmpty.style.display = 'none';
        resultsContainer.style.display = 'block';
        resultsContainer.innerHTML = '';
        
        const resultCard = formatDetectionResult(result);
        resultsContainer.appendChild(resultCard);
        
        // Show success message
        AppToast.success('Detection scan complete');
        
        // Emit socket event for real-time updates
        Socket.emit('detection.complete', result);
        
    } catch (err) {
        console.error('Detection scan failed:', err);
        AppToast.error(err.message || 'Detection scan failed');
        resultsEmpty.textContent = 'Scan failed. Please try again.';
        resultsEmpty.style.display = 'block';
    } finally {
        scanBtn.disabled = false;
        loadingSpinner.style.display = 'none';
    }
}

/**
 * Initialize page
 */
function initPage() {
    // Setup input type tabs
    inputTypeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            inputTypeTabs.forEach(t => {
                t.classList.remove('bg-[#3b82f6]');
                t.classList.add('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            });
            tab.classList.remove('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            tab.classList.add('bg-[#3b82f6]');
            
            currentInputType = tab.dataset.type;
            const placeholders = {
                ip: 'Enter IP address (e.g., 192.168.1.1)...',
                domain: 'Enter domain name (e.g., example.com)...',
                hash: 'Enter file hash (MD5, SHA1, SHA256)...'
            };
            detectionInput.placeholder = placeholders[currentInputType];
        });
    });
    
    // Setup scan button
    scanBtn.addEventListener('click', runScan);
    
    // Setup enter key on input
    detectionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            runScan();
        }
    });
    
    // Setup socket listeners
    Socket.on('detection.result', (result) => {
        // If detection result comes from socket, display it
        resultsEmpty.style.display = 'none';
        resultsContainer.style.display = 'block';
        resultsContainer.innerHTML = '';
        
        const resultCard = formatDetectionResult(result);
        resultsContainer.appendChild(resultCard);
    });
}

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, runScan };

