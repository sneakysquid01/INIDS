import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient } from '../core/http-client.js';

// DOM references
const honeypotsGrid = document.getElementById('honeypots-grid');
const statusFilters = document.querySelectorAll('.status-filter');
const typeFilters = document.querySelectorAll('.type-filter');
const searchInput = document.getElementById('search-honeypots');
const statTotal = document.getElementById('stat-total');
const statActive = document.getElementById('stat-active');
const statInteractions = document.getElementById('stat-interactions');
const statThreats = document.getElementById('stat-threats');

// State
let currentStatusFilter = 'all';
let currentTypeFilter = 'all';
let searchQuery = '';

/**
 * Format honeypot card
 */
function formatHoneypotCard(honeypot) {
    const statusColor = honeypot.active ? '#10b981' : '#6b7280';
    const statusText = honeypot.active ? 'Active' : 'Inactive';
    
    const typeMap = {
        'service': '#3b82f6',
        'log': '#8b5cf6',
        'file': '#f59e0b'
    };
    const typeColor = typeMap[honeypot.type] || '#8f9099';
    
    const card = document.createElement('div');
    card.className = 'bg-[#151922] border border-[#1a1f2e] rounded-lg p-4 hover:border-[#8b5cf6] transition-colors cursor-pointer group';
    card.innerHTML = `
        <div class="flex items-start justify-between mb-3">
            <div class="flex-1">
                <div class="text-white font-semibold text-sm group-hover:text-[#8b5cf6] transition-colors">${honeypot.name}</div>
                <div class="text-[#8f9099] text-xs mt-1">Type: <span class="capitalize" style="color: ${typeColor}">${honeypot.type || 'unknown'}</span></div>
            </div>
            <div class="px-2 py-1 rounded text-xs font-semibold" style="background-color: ${statusColor}33; color: ${statusColor}">${statusText}</div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-3 line-clamp-2">${honeypot.description || 'No description'}</div>
        
        <div class="grid grid-cols-2 gap-2 mb-3 text-xs">
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Port</div>
                <div class="font-semibold text-white">${honeypot.port || 'N/A'}</div>
            </div>
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Interactions</div>
                <div class="font-semibold text-white">${honeypot.interaction_count || 0}</div>
            </div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-3">
            <strong>Protocol:</strong> ${honeypot.protocol || 'TCP'}<br>
            <strong>Created:</strong> ${honeypot.created_date ? new Date(honeypot.created_date).toLocaleDateString() : 'N/A'}
        </div>
        
        <div class="grid grid-cols-2 gap-2 mb-3 text-xs">
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Threats</div>
                <div class="font-semibold" style="color: ${honeypot.threat_count > 0 ? '#ef4444' : '#10b981'}">${honeypot.threat_count || 0}</div>
            </div>
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Success Rate</div>
                <div class="font-semibold text-white">${honeypot.success_rate || 0}%</div>
            </div>
        </div>
        
        <div class="mt-3 pt-3 border-t border-[#1a1f2e]">
            <div class="flex items-center gap-2">
                <button class="toggle-btn flex-1 text-xs px-2 py-1 rounded transition-colors ${honeypot.active ? 'bg-[#ef4444] hover:bg-[#dc2626] text-white' : 'bg-[#10b981] hover:bg-[#059669] text-white'}">
                    ${honeypot.active ? 'Deactivate' : 'Activate'}
                </button>
                <span class="text-[#8b5cf6] group-hover:translate-x-1 transition-transform">→</span>
            </div>
        </div>
    `;
    
    const toggleBtn = card.querySelector('.toggle-btn');
    toggleBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
            await HttpClient.post(`/api/honeypots/${honeypot.id}/toggle`, {
                active: !honeypot.active
            });
            honeypot.active = !honeypot.active;
            AppToast.success(`Honeypot ${honeypot.active ? 'activated' : 'deactivated'}`);
        } catch (err) {
            AppToast.error('Failed to toggle honeypot');
        }
    });
    
    return card;
}

/**
 * Filter honeypots
 */
function filterHoneypots(honeypots) {
    let filtered = honeypots;
    
    if (currentStatusFilter !== 'all') {
        const active = currentStatusFilter === 'active';
        filtered = filtered.filter(h => h.active === active);
    }
    
    if (currentTypeFilter !== 'all') {
        filtered = filtered.filter(h => h.type === currentTypeFilter);
    }
    
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(h =>
            h.name?.toLowerCase().includes(q) ||
            h.description?.toLowerCase().includes(q) ||
            h.port?.toString().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render honeypots
 */
function renderHoneypots() {
    GlobalState.subscribe('honeypots', (honeypots) => {
        const filtered = filterHoneypots(honeypots || []);
        honeypotsGrid.innerHTML = '';
        
        // Update stats
        const total = honeypots?.length || 0;
        const active = honeypots?.filter(h => h.active).length || 0;
        const interactions = honeypots?.reduce((sum, h) => sum + (h.interaction_count || 0), 0) || 0;
        const threats = honeypots?.reduce((sum, h) => sum + (h.threat_count || 0), 0) || 0;
        
        statTotal.textContent = total;
        statActive.textContent = active;
        statInteractions.textContent = interactions;
        statThreats.textContent = threats;
        
        if (filtered.length === 0) {
            honeypotsGrid.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        ${total === 0 ? 'No honeypots configured' : 'No matching honeypots'}
                    </div>
                </div>
            `;
            return;
        }
        
        filtered.forEach(honeypot => {
            try {
                const card = formatHoneypotCard(honeypot);
                honeypotsGrid.appendChild(card);
            } catch (err) {
                console.error('Error rendering honeypot:', err);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    try {
        const response = await HttpClient.get('/api/honeypots');
        const honeypots = Array.isArray(response) ? response : response.honeypots || [];
        GlobalState.set('honeypots', honeypots);
    } catch (err) {
        console.error('Failed to load honeypots:', err);
        AppToast.error('Failed to load honeypots');
    }
    
    renderHoneypots();
    
    // Setup status filter handlers
    statusFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            statusFilters.forEach(b => {
                b.classList.remove('bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            });
            btn.classList.remove('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            btn.classList.add('bg-[#3b82f6]');
            
            currentStatusFilter = btn.dataset.status;
            renderHoneypots();
        });
    });
    
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
            renderHoneypots();
        });
    });
    
    // Setup search
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderHoneypots();
    });
    
    // Setup socket handlers
    Socket.on('honeypot.interaction', (data) => {
        const honeypots = GlobalState.state.honeypots || [];
        const honeypot = honeypots.find(h => h.id === data.honeypot_id);
        if (honeypot) {
            honeypot.interaction_count = (honeypot.interaction_count || 0) + 1;
            GlobalState.set('honeypots', honeypots);
            renderHoneypots();
        }
    });
    
    Socket.on('honeypot.threat', (honeypot) => {
        const honeypots = GlobalState.state.honeypots || [];
        const index = honeypots.findIndex(h => h.id === honeypot.id);
        if (index >= 0) {
            honeypots[index] = honeypot;
            GlobalState.set('honeypots', honeypots);
            renderHoneypots();
            AppToast.warning('Threat detected in honeypot');
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterHoneypots, renderHoneypots };

