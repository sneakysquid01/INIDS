import { PolicyHistoryItem } from '../components/policy-history-item.js';
import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient } from '../core/http-client.js';
import { formatTimestamp } from '../core/utils.js';

// DOM references
const policiesList = document.getElementById('policies-list');
const policyFilters = document.querySelectorAll('.policy-filter');
const changeFilters = document.querySelectorAll('.change-filter');
const searchInput = document.getElementById('search-policies');
const statChanges = document.getElementById('stat-changes');
const statLast = document.getElementById('stat-last');

// State
let currentPolicyFilter = 'all';
let currentChangeFilter = 'all';
let searchQuery = '';

/**
 * Filter policy history by policy type, change type, and search
 */
function filterPolicies(policies) {
    let filtered = policies;
    
    // Apply policy type filter
    if (currentPolicyFilter !== 'all') {
        filtered = filtered.filter(p => p.policy_type === currentPolicyFilter);
    }
    
    // Apply change type filter
    if (currentChangeFilter !== 'all') {
        filtered = filtered.filter(p => p.change_type === currentChangeFilter);
    }
    
    // Apply search filter
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(p => 
            p.policy_name?.toLowerCase().includes(q) ||
            p.user?.toLowerCase().includes(q) ||
            p.description?.toLowerCase().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render policy history with filtering applied
 */
function renderPolicies() {
    GlobalState.subscribe('policy', (policies) => {
        const filtered = filterPolicies(policies || []);
        policiesList.innerHTML = '';
        
        // Update stats
        statChanges.textContent = policies?.length || 0;
        if (policies && policies.length > 0) {
            const lastChange = policies[0];
            statLast.textContent = formatTimestamp(lastChange.timestamp);
        } else {
            statLast.textContent = '—';
        }
        
        if (filtered.length === 0) {
            policiesList.innerHTML = `
                <div class="text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        ${(policies?.length || 0) === 0 ? 'No policy changes' : 'No matching policy changes'}
                    </div>
                </div>
            `;
            return;
        }
        
        // Render policies in reverse chronological order (newest first)
        filtered.reverse().forEach((policy, index) => {
            try {
                const item = PolicyHistoryItem(policy);
                policiesList.appendChild(item);
            } catch (err) {
                console.error(`Error rendering policy ${index}:`, err);
                // Show error card for this specific policy
                const errorCard = document.createElement('div');
                errorCard.className = 'bg-[#151922] border border-[#ef4444] rounded-lg p-4 text-[#ef4444] text-sm';
                errorCard.textContent = `Failed to load policy change: ${policy.id}`;
                policiesList.appendChild(errorCard);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    // Load initial policies from API
    try {
        const response = await HttpClient.get('/api/policy/history');
        const policies = Array.isArray(response) ? response : response.policy || [];
        GlobalState.set('policy', policies);
    } catch (err) {
        console.error('Failed to load policy history:', err);
        AppToast.error('Failed to load policy history');
    }
    
    // Render policies
    renderPolicies();
    
    // Setup filter handlers - Policy type filters
    policyFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            policyFilters.forEach(b => {
                b.classList.remove('active:bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]');
            });
            btn.classList.remove('bg-[#0a0c10]');
            btn.classList.add('active:bg-[#3b82f6]');
            
            // Update filter and re-render
            currentPolicyFilter = btn.dataset.policy;
            renderPolicies();
        });
    });
    
    // Setup filter handlers - Change type filters
    changeFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            changeFilters.forEach(b => {
                b.classList.remove('active:bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]');
            });
            btn.classList.remove('bg-[#0a0c10]');
            btn.classList.add('active:bg-[#3b82f6]');
            
            // Update filter and re-render
            currentChangeFilter = btn.dataset.change;
            renderPolicies();
        });
    });
    
    // Setup search handler
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderPolicies();
    });
    
    // Setup socket handlers
    Socket.on('policy.update', (policy) => {
        const policies = GlobalState.state.policy || [];
        // Insert new change at beginning (most recent first)
        policies.unshift(policy);
        GlobalState.set('policy', policies);
        renderPolicies();
        AppToast.success('Policy updated');
    });
    
    Socket.on('policy.change', (policy) => {
        const policies = GlobalState.state.policy || [];
        policies.unshift(policy);
        GlobalState.set('policy', policies);
        renderPolicies();
    });
}

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterPolicies, renderPolicies };
