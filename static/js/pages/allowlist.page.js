import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global_state.js';
import { Socket } from '../core/socket_manager.js';
import { HttpClient } from '../core/http_client.js';

// DOM references
const allowlistTable = document.getElementById('allowlist-table');
const allowlistEmpty = document.getElementById('allowlist-empty');
const typeFilters = document.querySelectorAll('.type-filter');
const statusFilters = document.querySelectorAll('.status-filter');
const searchInput = document.getElementById('search-allowlist');
const statEntries = document.getElementById('stat-entries');
const statIps = document.getElementById('stat-ips');
const statDomains = document.getElementById('stat-domains');

// State
let currentTypeFilter = 'all';
let currentStatusFilter = 'all';
let searchQuery = '';

/**
 * Format allowlist entry row
 */
function formatAllowlistRow(entry) {
    const statusColor = entry.active ? '#10b981' : '#8f9099';
    const statusText = entry.active ? 'Active' : 'Inactive';
    
    const row = document.createElement('tr');
    row.className = 'hover:bg-[#0a0c10] transition-colors';
    row.innerHTML = `
        <td class="px-4 py-3 font-mono text-[#e5e7eb]">${entry.value}</td>
        <td class="px-4 py-3">
            <span class="px-2 py-1 rounded text-xs font-medium ${entry.type === 'ip' ? 'bg-[#3b82f6] bg-opacity-20 text-[#3b82f6]' : 'bg-[#f59e0b] bg-opacity-20 text-[#f59e0b]'}">
                ${entry.type === 'ip' ? 'IP Address' : 'Domain'}
            </span>
        </td>
        <td class="px-4 py-3 text-[#d1d5db] text-sm">${entry.reason || '—'}</td>
        <td class="px-4 py-3 text-[#d1d5db] text-sm">${entry.added_by || 'System'}</td>
        <td class="px-4 py-3">
            <span class="inline-flex items-center gap-1 text-xs font-medium" style="color: ${statusColor}">
                <span class="w-2 h-2 rounded-full" style="background-color: ${statusColor}"></span>
                ${statusText}
            </span>
        </td>
        <td class="px-4 py-3 text-right">
            <button class="delete-btn px-2 py-1 text-xs text-[#ef4444] hover:bg-[#ef4444] hover:bg-opacity-10 rounded transition-colors" data-id="${entry.id}">
                Remove
            </button>
        </td>
    `;
    
    const deleteBtn = row.querySelector('.delete-btn');
    deleteBtn.addEventListener('click', async () => {
        try {
            await HttpClient.delete(`/api/allowlist/${entry.id}`);
            AppToast.success('Entry removed from allowlist');
            renderAllowlist();
        } catch (err) {
            console.error('Error deleting entry:', err);
            AppToast.error('Failed to remove allowlist entry');
        }
    });
    
    return row;
}

/**
 * Filter allowlist entries
 */
function filterAllowlist(entries) {
    let filtered = entries;
    
    if (currentTypeFilter !== 'all') {
        filtered = filtered.filter(e => e.type === currentTypeFilter);
    }
    
    if (currentStatusFilter !== 'all') {
        const isActive = currentStatusFilter === 'active';
        filtered = filtered.filter(e => e.active === isActive);
    }
    
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(e =>
            e.value?.toLowerCase().includes(q) ||
            e.reason?.toLowerCase().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render allowlist
 */
function renderAllowlist() {
    GlobalState.subscribe('allowlist', (entries) => {
        const filtered = filterAllowlist(entries || []);
        allowlistTable.innerHTML = '';
        
        // Update stats
        statEntries.textContent = entries?.length || 0;
        statIps.textContent = entries?.filter(e => e.type === 'ip').length || 0;
        statDomains.textContent = entries?.filter(e => e.type === 'domain').length || 0;
        
        if (filtered.length === 0) {
            allowlistTable.innerHTML = `
                <tr>
                    <td colspan="6" class="px-4 py-12 text-center text-[#8f9099] text-sm uppercase tracking-wider">
                        ${(entries?.length || 0) === 0 ? 'No allowlist entries' : 'No matching entries'}
                    </td>
                </tr>
            `;
            return;
        }
        
        filtered.forEach(entry => {
            try {
                const row = formatAllowlistRow(entry);
                allowlistTable.appendChild(row);
            } catch (err) {
                console.error('Error rendering entry:', err);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    // Load initial allowlist from API
    try {
        const response = await HttpClient.get('/api/allowlist');
        const entries = Array.isArray(response) ? response : response.allowlist || [];
        GlobalState.set('allowlist', entries);
    } catch (err) {
        console.error('Failed to load allowlist:', err);
        AppToast.error('Failed to load allowlist');
    }
    
    renderAllowlist();
    
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
            renderAllowlist();
        });
    });
    
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
            renderAllowlist();
        });
    });
    
    // Setup search
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderAllowlist();
    });
    
    // Setup socket handlers
    Socket.on('allowlist.add', (entry) => {
        const entries = GlobalState.state.allowlist || [];
        entries.unshift(entry);
        GlobalState.set('allowlist', entries);
        renderAllowlist();
        AppToast.success('New allowlist entry added');
    });
    
    Socket.on('allowlist.remove', (id) => {
        const entries = GlobalState.state.allowlist || [];
        const filtered = entries.filter(e => e.id !== id);
        GlobalState.set('allowlist', filtered);
        renderAllowlist();
    });
    
    Socket.on('allowlist.update', (entry) => {
        const entries = GlobalState.state.allowlist || [];
        const index = entries.findIndex(e => e.id === entry.id);
        if (index >= 0) {
            entries[index] = entry;
            GlobalState.set('allowlist', entries);
            renderAllowlist();
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterAllowlist, renderAllowlist };
