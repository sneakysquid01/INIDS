import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient_Instance as HttpClient } from "../core/http-client.js";

// DOM references
const playbooksGrid = document.getElementById('playbooks-grid');
const statusFilters = document.querySelectorAll('.status-filter');
const typeFilters = document.querySelectorAll('.type-filter');
const searchInput = document.getElementById('search-playbooks');
const statPlaybooks = document.getElementById('stat-playbooks');
const statActive = document.getElementById('stat-active');
const statExecutions = document.getElementById('stat-executions');

// State
let currentStatusFilter = 'all';
let currentTypeFilter = 'all';
let searchQuery = '';

/**
 * Format playbook card
 */
function formatPlaybookCard(playbook) {
    const statusColor = playbook.enabled ? '#10b981' : '#6b7280';
    const statusText = playbook.enabled ? 'Enabled' : 'Disabled';
    
    const card = document.createElement('div');
    card.className = 'bg-[#151922] border border-[#1a1f2e] rounded-lg p-4 hover:border-[#3b82f6] transition-colors cursor-pointer group';
    card.innerHTML = `
        <div class="flex items-start justify-between mb-3">
            <div class="flex-1">
                <div class="text-white font-semibold text-sm group-hover:text-[#3b82f6] transition-colors">${playbook.name}</div>
                <div class="text-[#8f9099] text-xs mt-1">Type: <span class="capitalize">${playbook.type || 'custom'}</span></div>
            </div>
            <div class="px-2 py-1 rounded text-xs font-semibold" style="background-color: ${statusColor}33; color: ${statusColor}">${statusText}</div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-3 line-clamp-2">${playbook.description || 'No description'}</div>
        
        <div class="grid grid-cols-2 gap-2 mb-3 text-xs">
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Actions</div>
                <div class="font-semibold text-white">${playbook.action_count || 0}</div>
            </div>
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Executions</div>
                <div class="font-semibold text-white">${playbook.execution_count || 0}</div>
            </div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-3">
            <strong>Created:</strong> ${playbook.created_date ? new Date(playbook.created_date).toLocaleDateString() : 'N/A'}<br>
            <strong>Last Run:</strong> ${playbook.last_run_date ? new Date(playbook.last_run_date).toLocaleDateString() : 'Never'}
        </div>
        
        <div class="mt-3 pt-3 border-t border-[#1a1f2e]">
            <div class="flex items-center justify-between">
                <button class="execute-btn text-xs px-2 py-1 bg-[#3b82f6] hover:bg-[#2563eb] text-white rounded transition-colors">
                    Execute
                </button>
                <span class="text-[#3b82f6] group-hover:translate-x-1 transition-transform">→</span>
            </div>
        </div>
    `;
    
    const executeBtn = card.querySelector('.execute-btn');
    executeBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
            await HttpClient.post(`/api/playbooks/${playbook.id}/execute`);
            AppToast.success(`Executing playbook: ${playbook.name}`);
            playbook.execution_count = (playbook.execution_count || 0) + 1;
        } catch (err) {
            AppToast.error('Failed to execute playbook');
        }
    });
    
    return card;
}

/**
 * Filter playbooks
 */
function filterPlaybooks(playbooks) {
    let filtered = playbooks;
    
    if (currentStatusFilter !== 'all') {
        const enabled = currentStatusFilter === 'enabled';
        filtered = filtered.filter(p => p.enabled === enabled);
    }
    
    if (currentTypeFilter !== 'all') {
        filtered = filtered.filter(p => p.type === currentTypeFilter);
    }
    
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(p =>
            p.name?.toLowerCase().includes(q) ||
            p.description?.toLowerCase().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render playbooks
 */
function renderPlaybooks() {
    GlobalState.subscribe('playbooks', (playbooks) => {
        const filtered = filterPlaybooks(playbooks || []);
        playbooksGrid.innerHTML = '';
        
        // Update stats
        const total = playbooks?.length || 0;
        const active = playbooks?.filter(p => p.enabled).length || 0;
        const executions = playbooks?.reduce((sum, p) => sum + (p.execution_count || 0), 0) || 0;
        
        statPlaybooks.textContent = total;
        statActive.textContent = active;
        statExecutions.textContent = executions;
        
        if (filtered.length === 0) {
            playbooksGrid.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        ${total === 0 ? 'No playbooks yet' : 'No matching playbooks'}
                    </div>
                </div>
            `;
            return;
        }
        
        filtered.forEach(playbook => {
            try {
                const card = formatPlaybookCard(playbook);
                playbooksGrid.appendChild(card);
            } catch (err) {
                console.error('Error rendering playbook:', err);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    try {
        const response = await HttpClient.get('/api/playbooks');
        const playbooks = Array.isArray(response) ? response : response.playbooks || [];
        GlobalState.set('playbooks', playbooks);
    } catch (err) {
        console.error('Failed to load playbooks:', err);
        AppToast.error('Failed to load playbooks');
    }
    
    renderPlaybooks();
    
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
            renderPlaybooks();
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
            renderPlaybooks();
        });
    });
    
    // Setup search
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderPlaybooks();
    });
    
    // Setup socket handlers
    Socket.on('playbook.create', (playbook) => {
        const playbooks = GlobalState.state.playbooks || [];
        playbooks.unshift(playbook);
        GlobalState.set('playbooks', playbooks);
        renderPlaybooks();
        AppToast.success('New playbook created');
    });
    
    Socket.on('playbook.update', (playbook) => {
        const playbooks = GlobalState.state.playbooks || [];
        const index = playbooks.findIndex(p => p.id === playbook.id);
        if (index >= 0) {
            playbooks[index] = playbook;
            GlobalState.set('playbooks', playbooks);
            renderPlaybooks();
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterPlaybooks, renderPlaybooks };


