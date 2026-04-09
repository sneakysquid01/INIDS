let allAllowlist = [];
let filteredAllowlist = [];
let currentPage = 1;
let itemsPerPage = 10;
let currentDetailId = null;
let deleteId = null;

// Load allowlist on page load
document.addEventListener('DOMContentLoaded', loadAllowlist);

async function loadAllowlist() {
    try {
        const response = await fetch('/api/allowlist');
        if (!response.ok) throw new Error('Failed to load allowlist');
        
        const data = await response.json();
        allAllowlist = data.entries || [];
        filteredAllowlist = [...allAllowlist];
        currentPage = 1;
        
        updateStats();
        renderAllowlist();
        showSuccess('Allowlist loaded');
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to load allowlist: ' + error.message);
    }
}

function updateStats() {
    document.getElementById('totalCount').textContent = allAllowlist.length;
    
    const ips = allAllowlist.filter(a => isIP(a.entry)).length;
    const domains = allAllowlist.filter(a => !isIP(a.entry)).length;
    
    document.getElementById('ipCount').textContent = ips;
    document.getElementById('domainCount').textContent = domains;
    
    const now = new Date();
    document.getElementById('lastUpdated').textContent = now.toLocaleTimeString();
}

function isIP(entry) {
    const ipRegex = /^(\d{1,3}\.){3}\d{1,3}$/;
    return ipRegex.test(entry);
}

function filterAllowlist() {
    const search = document.getElementById('searchBox').value.toLowerCase();
    const typeFilter = document.getElementById('typeFilter').value;
    const reasonFilter = document.getElementById('reasonFilter').value;
    
    filteredAllowlist = allAllowlist.filter(item => {
        const matchSearch = item.entry.toLowerCase().includes(search);
        const matchType = !typeFilter || 
                         (typeFilter === 'ip' && isIP(item.entry)) ||
                         (typeFilter === 'domain' && !isIP(item.entry));
        const matchReason = !reasonFilter || item.reason === reasonFilter;
        
        return matchSearch && matchType && matchReason;
    });
    
    currentPage = 1;
    renderAllowlist();
}

function renderAllowlist() {
    const tbody = document.getElementById('allowlistTable');
    const start = (currentPage - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const pageItems = filteredAllowlist.slice(start, end);
    
    if (pageItems.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-4">No entries found</td></tr>';
        document.getElementById('pagination').style.display = 'none';
        return;
    }
    
    tbody.innerHTML = pageItems.map(item => `
        <tr style="cursor: pointer;" onclick="showDetails('${escapeHtml(item.entry)}')">
            <td>
                <strong>${escapeHtml(item.entry)}</strong>
                <br><small class="text-muted">${isIP(item.entry) ? '📍 IP Address' : '🌐 Domain'}</small>
            </td>
            <td><span class="badge ${getTypeBadge(item.entry)}">${getTypeLabel(item.entry)}</span></td>
            <td>${escapeHtml(item.reason || 'N/A')}</td>
            <td>${escapeHtml(item.added_by || 'system')}</td>
            <td><small>${new Date(item.added_at).toLocaleString()}</small></td>
            <td>
                <button class="btn btn-sm btn-outline-danger" onclick="event.stopPropagation(); openDeleteModal('${escapeHtml(item.entry)}');" title="Remove">
                    🗑️
                </button>
            </td>
        </tr>
    `).join('');
    
    updatePagination();
}

function getTypeLabel(entry) {
    return isIP(entry) ? 'IP' : 'Domain';
}

function getTypeBadge(entry) {
    return isIP(entry) ? 'bg-info' : 'bg-success';
}

function updatePagination() {
    const totalPages = Math.ceil(filteredAllowlist.length / itemsPerPage);
    
    if (totalPages <= 1) {
        document.getElementById('pagination').style.display = 'none';
        return;
    }
    
    document.getElementById('pagination').style.display = 'block';
    document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
    
    document.getElementById('prevBtn').classList.toggle('disabled', currentPage === 1);
    document.getElementById('nextBtn').classList.toggle('disabled', currentPage === totalPages);
}

function previousPage() {
    if (currentPage > 1) {
        currentPage--;
        renderAllowlist();
        window.scrollTo(0, 0);
    }
}

function nextPage() {
    const totalPages = Math.ceil(filteredAllowlist.length / itemsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        renderAllowlist();
        window.scrollTo(0, 0);
    }
}

function openAddModal() {
    document.getElementById('addForm').reset();
    new bootstrap.Modal(document.getElementById('addModal')).show();
}

async function saveEntry() {
    const entry = document.getElementById('entryInput').value.trim();
    const reason = document.getElementById('reasonSelect').value;
    const notes = document.getElementById('notesInput').value.trim();
    
    if (!entry || !reason) {
        showError('Please fill in all required fields');
        return;
    }
    
    // Validate entry format
    if (!isValidEntry(entry)) {
        showError('Invalid IP or domain format');
        return;
    }
    
    try {
        const response = await fetch('/api/allowlist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                entry: entry,
                reason: reason,
                notes: notes
            })
        });
        
        if (!response.ok) throw new Error(response.statusText);
        
        bootstrap.Modal.getInstance(document.getElementById('addModal')).hide();
        showSuccess('Entry added successfully');
        await loadAllowlist();
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to add entry: ' + error.message);
    }
}

function isValidEntry(entry) {
    // Check IP format
    const ipRegex = /^(\d{1,3}\.){3}\d{1,3}(\/(8|16|24|32))?$/;
    if (ipRegex.test(entry)) return true;
    
    // Check domain format
    const domainRegex = /^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$/;
    if (domainRegex.test(entry)) return true;
    
    return false;
}

function showDetails(entry) {
    const item = allAllowlist.find(a => a.entry === entry);
    if (!item) return;
    
    currentDetailId = entry;
    
    document.getElementById('detailEntry').textContent = escapeHtml(item.entry);
    document.getElementById('detailType').textContent = getTypeLabel(item.entry);
    document.getElementById('detailReason').textContent = escapeHtml(item.reason || 'N/A');
    document.getElementById('detailAddedBy').textContent = escapeHtml(item.added_by || 'system');
    document.getElementById('detailAddedDate').textContent = new Date(item.added_at).toLocaleString();
    document.getElementById('detailNotes').textContent = escapeHtml(item.notes || 'No notes');
    
    new bootstrap.Modal(document.getElementById('detailsModal')).show();
}

function openDeleteModal(entry) {
    deleteId = entry;
    const item = allAllowlist.find(a => a.entry === entry);
    if (!item) return;
    
    document.getElementById('deleteItemText').textContent = escapeHtml(item.entry);
    new bootstrap.Modal(document.getElementById('deleteModal')).show();
}

async function confirmDelete() {
    if (!deleteId) return;
    
    try {
        const response = await fetch(`/api/allowlist/${encodeURIComponent(deleteId)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error(response.statusText);
        
        bootstrap.Modal.getInstance(document.getElementById('deleteModal')).hide();
        if (bootstrap.Modal.getOrCreateInstance(document.getElementById('detailsModal')).toggle) {
            bootstrap.Modal.getInstance(document.getElementById('detailsModal')).hide();
        }
        
        showSuccess('Entry removed successfully');
        await loadAllowlist();
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to remove entry: ' + error.message);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show position-fixed';
    alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alert.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
    document.body.appendChild(alert);
    setTimeout(() => alert.remove(), 3000);
}

function showError(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show position-fixed';
    alert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alert.innerHTML = message + '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>';
    document.body.appendChild(alert);
}

// Auto-refresh every 30 seconds
setInterval(() => {
    loadAllowlist();
}, 30000);
