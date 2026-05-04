import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient } from '../core/http-client.js';

// DOM references
const modelsGrid = document.getElementById('models-grid');
const statusFilters = document.querySelectorAll('.status-filter');
const typeFilters = document.querySelectorAll('.type-filter');
const searchInput = document.getElementById('search-models');
const statModels = document.getElementById('stat-models');
const statActive = document.getElementById('stat-active');

// State
let currentStatusFilter = 'all';
let currentTypeFilter = 'all';
let searchQuery = '';

/**
 * Format model as card
 */
function formatModelCard(model) {
    const accuracyColor = (model.accuracy || 0) > 0.85 ? '#10b981' : (model.accuracy || 0) > 0.75 ? '#3b82f6' : '#f59e0b';
    const statusColor = model.active ? '#10b981' : '#8f9099';
    
    const card = document.createElement('div');
    card.className = 'bg-[#151922] border border-[#1a1f2e] rounded-lg p-4 hover:border-[#3b82f6] transition-colors';
    card.innerHTML = `
        <div class="flex items-start justify-between mb-3">
            <div class="flex-1">
                <div class="text-white font-semibold text-sm">${model.name}</div>
                <div class="text-[#8f9099] text-xs mt-1">${model.type || 'Unknown'}</div>
            </div>
            <div class="w-3 h-3 rounded-full flex-shrink-0 ml-2" style="background-color: ${statusColor}"></div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-3 line-clamp-2">${model.description || 'No description'}</div>
        
        <div class="grid grid-cols-2 gap-2 mb-3 text-xs">
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Accuracy</div>
                <div class="font-semibold" style="color: ${accuracyColor}">${(model.accuracy ? (model.accuracy * 100).toFixed(1) : '0')}%</div>
            </div>
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">F1 Score</div>
                <div class="text-white font-semibold">${(model.f1_score || 0).toFixed(3)}</div>
            </div>
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Precision</div>
                <div class="text-white font-semibold">${(model.precision ? (model.precision * 100).toFixed(0) : '0')}%</div>
            </div>
            <div class="bg-[#0a0c10] rounded px-2 py-1">
                <div class="text-[#8f9099]">Recall</div>
                <div class="text-white font-semibold">${(model.recall ? (model.recall * 100).toFixed(0) : '0')}%</div>
            </div>
        </div>
        
        <div class="text-[#8f9099] text-xs mb-2">
            <strong>Trained:</strong> ${model.trained_date ? new Date(model.trained_date).toLocaleDateString() : 'N/A'}<br>
            <strong>Version:</strong> ${model.version || '1.0'}
        </div>
        
        <div class="mt-3 pt-3 border-t border-[#1a1f2e]">
            <span class="inline-block px-2 py-1 rounded text-xs font-medium ${model.active ? 'bg-[#10b981] bg-opacity-20 text-[#10b981]' : 'bg-[#8f9099] bg-opacity-20 text-[#8f9099]'}">
                ${model.active ? 'Active' : 'Inactive'}
            </span>
        </div>
    `;
    
    return card;
}

/**
 * Filter models
 */
function filterModels(models) {
    let filtered = models;
    
    if (currentStatusFilter !== 'all') {
        const isActive = currentStatusFilter === 'active';
        filtered = filtered.filter(m => {
            if (currentStatusFilter === 'deprecated') return m.status === 'deprecated';
            return m.active === isActive;
        });
    }
    
    if (currentTypeFilter !== 'all') {
        filtered = filtered.filter(m => m.type === currentTypeFilter);
    }
    
    if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        filtered = filtered.filter(m =>
            m.name?.toLowerCase().includes(q) ||
            m.description?.toLowerCase().includes(q)
        );
    }
    
    return filtered;
}

/**
 * Render models
 */
function renderModels() {
    GlobalState.subscribe('models', (models) => {
        const filtered = filterModels(models || []);
        modelsGrid.innerHTML = '';
        
        // Update stats
        statModels.textContent = models?.length || 0;
        statActive.textContent = models?.filter(m => m.active).length || 0;
        
        if (filtered.length === 0) {
            modelsGrid.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        ${(models?.length || 0) === 0 ? 'No models available' : 'No matching models'}
                    </div>
                </div>
            `;
            return;
        }
        
        filtered.forEach(model => {
            try {
                const card = formatModelCard(model);
                modelsGrid.appendChild(card);
            } catch (err) {
                console.error('Error rendering model:', err);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    // Load initial models from API
    try {
        const response = await HttpClient.get('/api/models');
        const models = Array.isArray(response) ? response : response.models || [];
        GlobalState.set('models', models);
    } catch (err) {
        console.error('Failed to load models:', err);
        AppToast.error('Failed to load ML models');
    }
    
    renderModels();
    
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
            renderModels();
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
            renderModels();
        });
    });
    
    // Setup search
    searchInput.addEventListener('input', (e) => {
        searchQuery = e.target.value;
        renderModels();
    });
    
    // Setup socket handlers
    Socket.on('model.update', (model) => {
        const models = GlobalState.state.models || [];
        const index = models.findIndex(m => m.id === model.id);
        if (index >= 0) {
            models[index] = model;
        } else {
            models.push(model);
        }
        GlobalState.set('models', models);
        renderModels();
    });
    
    Socket.on('model.retrain', (model) => {
        const models = GlobalState.state.models || [];
        const index = models.findIndex(m => m.id === model.id);
        if (index >= 0) {
            models[index] = model;
        }
        GlobalState.set('models', models);
        renderModels();
        AppToast.info(`Model ${model.name} training completed`);
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterModels, renderModels };

