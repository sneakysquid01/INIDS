import { ActionCard } from '../components/action-card.js';
import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient } from '../core/http-client.js';

// DOM references
const actionsList = document.getElementById('actions-list');
const typeFilters = document.querySelectorAll('.type-filter');
const statusFilters = document.querySelectorAll('.status-filter');
const searchInput = document.getElementById('search-actions');
const statTotal = document.getElementById('stat-total');
const statSuccess = document.getElementById('stat-success');

// State
let currentTypeFilter = 'all';
let currentStatusFilter = 'all';
let searchQuery = '';

/**
 * Filter actions by type, status, and search
 */
function filterActions(actions) {
    let filtered = actions;
    
    // Apply type filter
    if (currentTypeFilter !== 'all') {
        filtered = filtered.filter(a => a.type === currentTypeFilter);
    }
    
    // Apply status filter
    if (currentStatusFilter !== 'all') {
        filtered = filtered.filter(a => a.status === currentStatusFilter);
    }
    
    // Apply search filter
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(a => 
            a.target?.toLowerCase().includes(q) ||
            a.reason?.toLowerCase().includes(q) ||
            a.type?.toLowerCase().includes(q) ||
            a.executor?.toLowerCase().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render actions with filtering applied
 */
function renderActions() {
    GlobalState.subscribe('actions', (actions) => {
        const filtered = filterActions(actions);
        actionsList.innerHTML = '';
        
        // Update stats
        statTotal.textContent = actions.length;
        statSuccess.textContent = actions.filter(a => a.status === 'success').length;
        
        if (filtered.length === 0) {
            actionsList.innerHTML = `
                <div class="text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        ${actions.length === 0 ? 'No actions' : 'No matching actions'}
                    </div>
                </div>
            `;
            return;
        }
        
        // Render actions in reverse chronological order (newest first)
        filtered.reverse().forEach((action, index) => {
            try {
                const card = ActionCard(action);
                actionsList.appendChild(card);
            } catch (err) {
                console.error(`Error rendering action ${index}:`, err);
                // Show error card for this specific action
                const errorCard = document.createElement('div');
                errorCard.className = 'bg-[#151922] border border-[#ef4444] rounded-lg p-4 text-[#ef4444] text-sm';
                errorCard.textContent = `Failed to load action: ${action.id}`;
                actionsList.appendChild(errorCard);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    // Load initial actions from API
    try {
        const response = await HttpClient.get('/api/actions');
        const actions = Array.isArray(response) ? response : response.actions || [];
        GlobalState.set('actions', actions);
    } catch (err) {
        console.error('Failed to load actions:', err);
        AppToast.error('Failed to load actions');
    }
    
    // Render actions
    renderActions();
    
    // Setup filter handlers - Type filters
    typeFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            typeFilters.forEach(b => {
                b.classList.remove('active:bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]');
            });
            btn.classList.remove('bg-[#0a0c10]');
            btn.classList.add('active:bg-[#3b82f6]');
            
            // Update filter and re-render
            currentTypeFilter = btn.dataset.type;
            renderActions();
        });
    });
    
    // Setup filter handlers - Status filters
    statusFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            statusFilters.forEach(b => {
                b.classList.remove('active:bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]');
            });
            btn.classList.remove('bg-[#0a0c10]');
            btn.classList.add('active:bg-[#3b82f6]');
            
            // Update filter and re-render
            currentStatusFilter = btn.dataset.status;
            renderActions();
        });
    });
    
    // Setup search handler
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderActions();
    });
    
    // Setup socket handlers
    Socket.on('action.update', (action) => {
        // Update action in GlobalState or replace if exists
        const actions = GlobalState.state.actions || [];
        const index = actions.findIndex(a => a.id === action.id);
        if (index >= 0) {
            actions[index] = action;
        } else {
            actions.push(action);
        }
        GlobalState.set('actions', actions);
        renderActions();
    });
    
    Socket.on('action.new', (action) => {
        const actions = GlobalState.state.actions || [];
        actions.push(action);
        GlobalState.set('actions', actions);
        renderActions();
    });
}

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterActions, renderActions };
