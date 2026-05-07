import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient_Instance as HttpClient } from "../core/http-client.js";

// DOM references
const threatsGrid = document.getElementById('threats-grid');
const typeFilters = document.querySelectorAll('.type-filter');
const confidenceFilters = document.querySelectorAll('.confidence-filter');
const searchInput = document.getElementById('search-threats');
const statThreats = document.getElementById('stat-threats');
const statCritical = document.getElementById('stat-critical');

// State
let currentTypeFilter = 'all';
let currentConfidenceFilter = 'all';
let searchQuery = '';

/**
 * Format threat as card
 */
function formatThreatCard(threat) {
    const severityColor = {
        'critical': '#ef4444',
        'high': '#f59e0b',
        'medium': '#3b82f6',
        'low': '#10b981'
    }[threat.severity] || '#8f9099';
    
    const card = document.createElement('div');
    card.className = 'bg-[#151922] border border-[#1a1f2e] rounded-lg p-4 hover:border-[#3b82f6] transition-colors';
    card.innerHTML = `
        <div class="flex items-start justify-between mb-3">
            <div class="flex-1">
                <div class="text-white font-semibold text-sm">${threat.name}</div>
                <div class="text-[#8f9099] text-xs mt-1">${threat.type}</div>
            </div>
            <div class="w-3 h-3 rounded-full flex-shrink-0 ml-2" style="background-color: ${severityColor}"></div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-3 line-clamp-2">${threat.description || 'No description'}</div>
        
        <div class="grid grid-cols-2 gap-2 mb-3 text-xs">
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Confidence</div>
                <div class="text-white font-semibold">${threat.confidence || 0}%</div>
            </div>
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Severity</div>
                <div class="font-semibold" style="color: ${severityColor}">${threat.severity || 'unknown'}</div>
            </div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-2">
            <strong>Source:</strong> ${threat.source || 'Unknown'}<br>
            <strong>Updated:</strong> ${threat.updated ? new Date(threat.updated).toLocaleDateString() : 'N/A'}
        </div>
        
        ${threat.indicators ? `
            <div class="mt-3 pt-3 border-t border-[#1a1f2e]">
                <div class="text-[#8f9099] text-xs font-medium mb-2">Indicators</div>
                <div class="space-y-1">
                    ${threat.indicators.slice(0, 2).map(ind => `
                        <div class="text-[#6b7280] text-xs truncate" title="${ind}">${ind}</div>
                    `).join('')}
                    ${threat.indicators.length > 2 ? `<div class="text-[#3b82f6] text-xs">+${threat.indicators.length - 2} more</div>` : ''}
                </div>
            </div>
        ` : ''}
    `;
    
    return card;
}

/**
 * Filter threats by type, confidence, and search
 */
function filterThreats(threats) {
    let filtered = threats;
    
    if (currentTypeFilter !== 'all') {
        filtered = filtered.filter(t => t.type === currentTypeFilter);
    }
    
    if (currentConfidenceFilter !== 'all') {
        const confLevel = { 'high': 75, 'medium': 50, 'low': 0 }[currentConfidenceFilter] || 0;
        filtered = filtered.filter(t => {
            const nextLevel = currentConfidenceFilter === 'high' ? 100 : (currentConfidenceFilter === 'medium' ? 75 : 50);
            return (t.confidence || 0) >= confLevel && (t.confidence || 0) < nextLevel;
        });
    }
    
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(t =>
            t.name?.toLowerCase().includes(q) ||
            t.description?.toLowerCase().includes(q) ||
            t.source?.toLowerCase().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render threats
 */
function renderThreats() {
    GlobalState.subscribe('threat_intel', (threats) => {
        const filtered = filterThreats(threats || []);
        threatsGrid.innerHTML = '';
        
        // Update stats
        statThreats.textContent = threats?.length || 0;
        statCritical.textContent = threats?.filter(t => t.severity === 'critical').length || 0;
        
        if (filtered.length === 0) {
            threatsGrid.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        ${(threats?.length || 0) === 0 ? 'No threats available' : 'No matching threats'}
                    </div>
                </div>
            `;
            return;
        }
        
        filtered.forEach(threat => {
            try {
                const card = formatThreatCard(threat);
                threatsGrid.appendChild(card);
            } catch (err) {
                console.error('Error rendering threat:', err);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    // Load initial threats from API
    try {
        const response = await HttpClient.get('/api/threat-intelligence');
        const threats = Array.isArray(response) ? response : response.threats || [];
        GlobalState.set('threat_intel', threats);
    } catch (err) {
        console.error('Failed to load threat intelligence:', err);
        AppToast.error('Failed to load threat intelligence');
    }
    
    renderThreats();
    
    // Setup type filter handlers
    typeFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            typeFilters.forEach(b => {
                b.classList.remove('bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            });
            btn.classList.remove('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            btn.classList.add('bg-[#3b82f6]');
            
            currentTypeFilter = btn.dataset.type;
            renderThreats();
        });
    });
    
    // Setup confidence filter handlers
    confidenceFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            confidenceFilters.forEach(b => {
                b.classList.remove('bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            });
            btn.classList.remove('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            btn.classList.add('bg-[#3b82f6]');
            
            currentConfidenceFilter = btn.dataset.confidence;
            renderThreats();
        });
    });
    
    // Setup search
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderThreats();
    });
    
    // Setup socket handlers
    Socket.on('threat_intel.update', (threat) => {
        const threats = GlobalState.state.threat_intel || [];
        const index = threats.findIndex(t => t.id === threat.id);
        if (index >= 0) {
            threats[index] = threat;
        } else {
            threats.push(threat);
        }
        GlobalState.set('threat_intel', threats);
        renderThreats();
    });
    
    Socket.on('threat_intel.new', (threat) => {
        const threats = GlobalState.state.threat_intel || [];
        threats.unshift(threat);
        GlobalState.set('threat_intel', threats);
        renderThreats();
        AppToast.warning('New threat intelligence received');
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterThreats, renderThreats };


