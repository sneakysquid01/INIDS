import { EngineCard } from '../components/engine-card.js';
import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global_state.js';
import { Socket } from '../core/socket_manager.js';
import { HttpClient } from '../core/http_client.js';

// DOM references
const enginesGrid = document.getElementById('engines-grid');
const statusFilters = document.querySelectorAll('.status-filter');
const statEngines = document.getElementById('stat-engines');
const statAccuracy = document.getElementById('stat-accuracy');

// State
let currentStatusFilter = 'all';

/**
 * Filter engines by status
 */
function filterEngines(engines) {
    let filtered = engines;
    
    if (currentStatusFilter !== 'all') {
        filtered = filtered.filter(e => {
            const engineStatus = getEngineStatus(e);
            return engineStatus === currentStatusFilter;
        });
    }
    
    return filtered;
}

/**
 * Get engine status based on properties
 */
function getEngineStatus(engine) {
    if (!engine.enabled) return 'offline';
    if (engine.error || engine.status === 'error') return 'error';
    if (engine.accuracy && engine.accuracy < 0.75) return 'warning';
    if (engine.load && engine.load > 0.9) return 'warning';
    return 'healthy';
}

/**
 * Render engines with filtering applied
 */
function renderEngines() {
    GlobalState.subscribe('engines', (engines) => {
        const filtered = filterEngines(engines || []);
        enginesGrid.innerHTML = '';
        
        // Update stats
        statEngines.textContent = engines?.length || 0;
        
        if (engines && engines.length > 0) {
            const avgAccuracy = engines.reduce((sum, e) => sum + (e.accuracy || 0), 0) / engines.length;
            statAccuracy.textContent = Math.round(avgAccuracy * 100) + '%';
        } else {
            statAccuracy.textContent = '0%';
        }
        
        if (filtered.length === 0) {
            enginesGrid.innerHTML = `
                <div class="col-span-full text-center py-12">
                    <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                        ${(engines?.length || 0) === 0 ? 'No engines available' : 'No matching engines'}
                    </div>
                </div>
            `;
            return;
        }
        
        // Render engine cards
        filtered.forEach((engine, index) => {
            try {
                const card = EngineCard(engine);
                enginesGrid.appendChild(card);
            } catch (err) {
                console.error(`Error rendering engine ${index}:`, err);
                // Show error card for this specific engine
                const errorCard = document.createElement('div');
                errorCard.className = 'bg-[#151922] border border-[#ef4444] rounded-lg p-4 text-[#ef4444] text-sm';
                errorCard.textContent = `Failed to load engine: ${engine.name}`;
                enginesGrid.appendChild(errorCard);
            }
        });
    });
}

/**
 * Initialize page
 */
async function initPage() {
    // Load initial engines from API
    try {
        const response = await HttpClient.get('/api/engines');
        const engines = Array.isArray(response) ? response : response.engines || [];
        GlobalState.set('engines', engines);
    } catch (err) {
        console.error('Failed to load engines:', err);
        AppToast.error('Failed to load detection engines');
    }
    
    // Render engines
    renderEngines();
    
    // Setup filter handlers
    statusFilters.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            statusFilters.forEach(b => {
                b.classList.remove('bg-[#3b82f6]');
                b.classList.add('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            });
            btn.classList.remove('bg-[#0a0c10]', 'border', 'border-[#1a1f2e]');
            btn.classList.add('bg-[#3b82f6]');
            
            // Update filter and re-render
            currentStatusFilter = btn.dataset.status;
            renderEngines();
        });
    });
    
    // Setup socket handlers
    Socket.on('engine.update', (engine) => {
        const engines = GlobalState.state.engines || [];
        const index = engines.findIndex(e => e.id === engine.id);
        if (index >= 0) {
            engines[index] = engine;
        } else {
            engines.push(engine);
        }
        GlobalState.set('engines', engines);
        renderEngines();
    });
    
    Socket.on('engine.status', (data) => {
        const engines = GlobalState.state.engines || [];
        const engine = engines.find(e => e.id === data.engine_id);
        if (engine) {
            engine.status = data.status;
            engine.load = data.load || 0;
            engine.accuracy = data.accuracy || 0;
            engine.detection_count = data.detection_count || 0;
            GlobalState.set('engines', engines);
            renderEngines();
        }
    });
}

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, filterEngines, renderEngines, getEngineStatus };
