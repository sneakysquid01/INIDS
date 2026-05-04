import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient } from '../core/http-client.js';

// DOM references
const investigationsGrid = document.getElementById('investigations-grid');
const statusFilters = document.querySelectorAll('.status-filter');
const severityFilters = document.querySelectorAll('.severity-filter');
const searchInput = document.getElementById('search-investigations');
const statActive = document.getElementById('stat-active');
const statCompleted = document.getElementById('stat-completed');
const statTotal = document.getElementById('stat-total');

// State
let currentStatusFilter = 'all';
let currentSeverityFilter = 'all';
let searchQuery = '';

/**
 * Format investigation card
 */
function formatInvestigationCard(investigation) {
    const severityColor = {
        'critical': '#ef4444',
        'high': '#f59e0b',
        'medium': '#3b82f6',
        'low': '#10b981'
    }[investigation.severity] || '#8f9099';
    
    const statusColor = investigation.status === 'open' ? '#3b82f6' : '#10b981';
    const statusText = investigation.status === 'open' ? 'Open' : 'Closed';
    
    const card = document.createElement('div');
    card.className = 'bg-[#151922] border border-[#1a1f2e] rounded-lg p-4 hover:border-[#3b82f6] transition-colors cursor-pointer group';
    card.innerHTML = `
        <div class="flex items-start justify-between mb-3">
            <div class="flex-1">
                <div class="text-white font-semibold text-sm group-hover:text-[#3b82f6] transition-colors">${investigation.title}</div>
                <div class="text-[#8f9099] text-xs mt-1">ID: ${investigation.id}</div>
            </div>
            <div class="w-3 h-3 rounded-full flex-shrink-0 ml-2" style="background-color: ${severityColor}"></div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-3 line-clamp-2">${investigation.description || 'No description'}</div>
        
        <div class="grid grid-cols-2 gap-2 mb-3 text-xs">
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Status</div>
                <div class="font-semibold" style="color: ${statusColor}">${statusText}</div>
            </div>
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Severity</div>
                <div class="font-semibold" style="color: ${severityColor}">${investigation.severity || 'unknown'}</div>
            </div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-3">
            <strong>Investigator:</strong> ${investigation.investigator || 'Unassigned'}<br>
            <strong>Started:</strong> ${investigation.created_date ? new Date(investigation.created_date).toLocaleDateString() : 'N/A'}
        </div>
        
        <div class="mt-3 pt-3 border-t border-[#1a1f2e]">
            <div class="flex items-center justify-between">
                <span class="text-[#6b7280] text-xs">${investigation.evidence_count || 0} evidence items</span>
                <span class="text-[#3b82f6] group-hover:translate-x-1 transition-transform">→</span>
            </div>
        </div>
    `;
    
    card.addEventListener('click', () => {
        AppToast.info(`Opened investigation: ${investigation.title}`);
    });
    
    return card;
}

/**
 * Filter investigations
 */
function filterInvestigations(investigations) {
    let filtered = investigations;
    
    if (currentStatusFilter !== 'all') {
        filtered = filtered.filter(i => i.status === currentStatusFilter);
    }
    
    if (currentSeverityFilter !== 'all') {
        filtered = filtered.filter(i => i.severity === currentSeverityFilter);
    }
    
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(i =>
            i.title?.toLowerCase().includes(q) ||
            i.description?.toLowerCase().includes(q) ||
            i.id?.toLowerCase().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render investigations
 */
function renderInvestigations() {
    GlobalState.subscribe('investigations', (investigations) => {
        const filtered = filterInvestigations(investigations || []);
        investigationsGrid.innerHTML = '';
        
        // Update stats
        const total = investigations?.length || 0;
        const active = investigations?.filter(i => i.status === 'open').length || 0;
        const completed = investigations?.filter(i => i.status === 'closed').length || 0;
        
        statTotal.textContent = total;
        statActive.textContent = active;
        statCompleted.textContent = completed;
        
        if (filtered.length === 0) {
            investigationsGrid.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        ${total === 0 ? 'No investigations yet' : 'No matching investigations'}
                    </div>
                </div>
            `;
            return;
        }
        
        filtered.forEach(investigation => {
            try {
                const card = formatInvestigationCard(investigation);
                investigationsGrid.appendChild(card);
            } catch (err) {
                console.error('Error rendering investigation:', err);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    try {
        const response = await HttpClient.get('/api/investigations');
        const investigations = Array.isArray(response) ? response : response.investigations || [];
        GlobalState.set('investigations', investigations);
    } catch (err) {
        console.error('Failed to load investigations:', err);
        AppToast.error('Failed to load investigations');
    }
    
    renderInvestigations();
    
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
            renderInvestigations();
        });
    });
    
    // Setup severity filter handlers
    severityFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            severityFilters.forEach(b => {
                b.classList.remove('bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            });
            btn.classList.remove('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            btn.classList.add('bg-[#3b82f6]');
            
            currentSeverityFilter = btn.dataset.severity;
            renderInvestigations();
        });
    });
    
    // Setup search
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderInvestigations();
    });
    
    // Setup socket handlers
    Socket.on('investigation.create', (investigation) => {
        const investigations = GlobalState.state.investigations || [];
        investigations.unshift(investigation);
        GlobalState.set('investigations', investigations);
        renderInvestigations();
        AppToast.success('New investigation created');
    });
    
    Socket.on('investigation.update', (investigation) => {
        const investigations = GlobalState.state.investigations || [];
        const index = investigations.findIndex(i => i.id === investigation.id);
        if (index >= 0) {
            investigations[index] = investigation;
            GlobalState.set('investigations', investigations);
            renderInvestigations();
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterInvestigations, renderInvestigations };

