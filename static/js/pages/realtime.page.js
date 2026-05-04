import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global_state.js';
import { Socket } from '../core/socket_manager.js';
import { HttpClient } from '../core/http_client.js';

// DOM references
const eventsFeed = document.getElementById('events-feed');
const typeFilters = document.querySelectorAll('.type-filter');
const severityFilters = document.querySelectorAll('.severity-filter');
const clearEventsBtn = document.getElementById('clear-events-btn');
const statTotalEvents = document.getElementById('stat-total-events');
const statAlerts = document.getElementById('stat-alerts');
const statActions = document.getElementById('stat-actions');
const statUpdates = document.getElementById('stat-updates');

// State
let currentTypeFilter = 'all';
let currentSeverityFilter = 'all';
let events = [];
const MAX_EVENTS = 500;

/**
 * Format event item
 */
function formatEventItem(event) {
    const typeMap = {
        'alert': { color: '#ef4444', label: 'Alert' },
        'action': { color: '#3b82f6', label: 'Action' },
        'update': { color: '#10b981', label: 'Update' },
        'policy': { color: '#f59e0b', label: 'Policy' },
        'detection': { color: '#8b5cf6', label: 'Detection' }
    };
    
    const type = event.type || 'update';
    const typeInfo = typeMap[type] || typeMap.update;
    
    const severityMap = {
        'critical': '#ef4444',
        'high': '#f59e0b',
        'medium': '#3b82f6',
        'low': '#10b981'
    };
    
    const severityColor = severityMap[event.severity] || '#8f9099';
    const timestamp = new Date(event.timestamp || Date.now());
    const timeStr = timestamp.toLocaleTimeString();
    
    const item = document.createElement('div');
    item.className = 'px-4 py-3 hover:bg-[#1a1f2e] transition-colors border-l-4';
    item.style.borderColor = typeInfo.color;
    
    let eventContent = `
        <div class="flex items-start gap-3">
            <div class="flex-shrink-0 w-3 h-3 rounded-full mt-1.5" style="background-color: ${typeInfo.color}"></div>
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2">
                    <span class="text-xs font-semibold uppercase tracking-wider" style="color: ${typeInfo.color}">${typeInfo.label}</span>
                    ${event.severity ? `<span class="text-xs px-2 py-1 rounded" style="background-color: ${severityColor}33; color: ${severityColor}">${event.severity}</span>` : ''}
                </div>
                <div class="text-white text-sm mt-1 font-medium">${event.title || event.message || 'Event'}</div>
                ${event.description ? `<div class="text-[#8f9099] text-xs mt-1">${event.description}</div>` : ''}
                <div class="text-[#6b7280] text-xs mt-2">${timeStr}</div>
            </div>
        </div>
    `;
    
    item.innerHTML = eventContent;
    return item;
}

/**
 * Filter events
 */
function filterEvents(eventList) {
    let filtered = eventList;
    
    if (currentTypeFilter !== 'all') {
        filtered = filtered.filter(e => e.type === currentTypeFilter);
    }
    
    if (currentSeverityFilter !== 'all') {
        filtered = filtered.filter(e => e.severity === currentSeverityFilter);
    }
    
    return filtered;
}

/**
 * Render events
 */
function renderEvents() {
    const filtered = filterEvents(events);
    eventsFeed.innerHTML = '';
    
    // Update stats
    const alertCount = events.filter(e => e.type === 'alert').length;
    const actionCount = events.filter(e => e.type === 'action').length;
    const updateCount = events.filter(e => e.type === 'update').length;
    
    statTotalEvents.textContent = events.length;
    statAlerts.textContent = alertCount;
    statActions.textContent = actionCount;
    statUpdates.textContent = updateCount;
    
    if (filtered.length === 0) {
        eventsFeed.innerHTML = `
            <div class="p-6 text-center">
                <div class="text-[#8f9099] text-sm uppercase tracking-wider">
                    ${events.length === 0 ? 'Waiting for events...' : 'No matching events'}
                </div>
            </div>
        `;
        return;
    }
    
    // Render newest events first
    [...filtered].reverse().forEach(event => {
        try {
            const item = formatEventItem(event);
            eventsFeed.appendChild(item);
        } catch (err) {
            console.error('Error rendering event:', err);
        }
    });
}

/**
 * Add event to feed
 */
function addEvent(eventData) {
    const event = {
        id: `event-${Date.now()}-${Math.random()}`,
        timestamp: new Date(),
        ...eventData
    };
    
    events.unshift(event);
    
    // Keep only last N events
    if (events.length > MAX_EVENTS) {
        events = events.slice(0, MAX_EVENTS);
    }
    
    renderEvents();
}

/**
 * Initialize page
 */
function initPage() {
    renderEvents();
    
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
            renderEvents();
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
            renderEvents();
        });
    });
    
    // Clear events button
    clearEventsBtn.addEventListener('click', () => {
        events = [];
        renderEvents();
        AppToast.info('Event history cleared');
    });
    
    // Setup socket handlers for all event types
    Socket.on('alert.new', (alert) => {
        addEvent({
            type: 'alert',
            severity: alert.severity || 'high',
            title: `Alert: ${alert.rule_name || 'Unknown Rule'}`,
            description: alert.summary || '',
            timestamp: alert.timestamp
        });
    });
    
    Socket.on('action.update', (action) => {
        addEvent({
            type: 'action',
            severity: 'medium',
            title: `Action: ${action.action_type || 'Unknown'} - ${action.status || 'pending'}`,
            description: action.target || '',
            timestamp: action.timestamp
        });
    });
    
    Socket.on('action.new', (action) => {
        addEvent({
            type: 'action',
            severity: 'medium',
            title: `New Action: ${action.action_type || 'Unknown'}`,
            description: action.target || '',
            timestamp: action.timestamp
        });
    });
    
    Socket.on('policy.change', (policy) => {
        addEvent({
            type: 'policy',
            severity: 'low',
            title: `Policy Updated: ${policy.name || 'Unknown'}`,
            description: policy.change_type || '',
            timestamp: policy.timestamp
        });
    });
    
    Socket.on('detection.result', (detection) => {
        addEvent({
            type: 'detection',
            severity: detection.severity || 'medium',
            title: `Detection: ${detection.target || 'Unknown Target'}`,
            description: detection.analysis_type || '',
            timestamp: detection.timestamp
        });
    });
    
    Socket.on('engine.update', (engine) => {
        addEvent({
            type: 'update',
            severity: 'low',
            title: `Engine Update: ${engine.name || 'Unknown'}`,
            description: engine.status || '',
            timestamp: engine.timestamp
        });
    });
    
    Socket.on('health.metrics', (health) => {
        addEvent({
            type: 'update',
            severity: 'low',
            title: 'System Health Update',
            description: `CPU: ${health.cpu_usage}%, Memory: ${health.memory_usage}%`,
            timestamp: health.timestamp
        });
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, renderEvents, addEvent };
