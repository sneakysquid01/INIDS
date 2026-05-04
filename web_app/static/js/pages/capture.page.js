import { AppToast } from '../components/app-toast.js';
import { GlobalState } from '../core/global-state.js';
import { Socket } from '../core/socket-manager.js';
import { HttpClient } from '../core/http-client.js';

// DOM references
const captureToggleBtn = document.getElementById('capture-toggle-btn');
const bpfFilterInput = document.getElementById('bpf-filter-input');
const interfaceSelect = document.getElementById('interface-select');
const packetLimitInput = document.getElementById('packet-limit-input');
const packetsTbody = document.getElementById('packets-tbody');
const packetCount = document.getElementById('packet-count');
const statStatus = document.getElementById('stat-status');
const statPackets = document.getElementById('stat-packets');
const statBytes = document.getElementById('stat-bytes');
const statDuration = document.getElementById('stat-duration');

// State
let isCapturing = false;
let packets = [];
let captureStartTime = null;
let totalBytes = 0;

/**
 * Format packet row
 */
function formatPacketRow(packet, index) {
    const row = document.createElement('tr');
    row.className = 'hover:bg-[#1a1f2e] transition-colors cursor-pointer group';
    
    const timeStr = new Date(packet.timestamp || Date.now()).toLocaleTimeString();
    const protocolColor = {
        'TCP': '#3b82f6',
        'UDP': '#10b981',
        'ICMP': '#f59e0b',
        'HTTP': '#8b5cf6',
        'HTTPS': '#ec4899'
    }[packet.protocol] || '#8f9099';
    
    row.innerHTML = `
        <td class="px-4 py-2 text-[#8f9099]">${index + 1}</td>
        <td class="px-4 py-2 text-[#8f9099] font-mono">${timeStr}</td>
        <td class="px-4 py-2 text-white font-mono group-hover:text-[#3b82f6] transition-colors truncate">${packet.src_ip || 'N/A'}:${packet.src_port || '-'}</td>
        <td class="px-4 py-2 text-white font-mono group-hover:text-[#3b82f6] transition-colors truncate">${packet.dst_ip || 'N/A'}:${packet.dst_port || '-'}</td>
        <td class="px-4 py-2">
            <span class="px-2 py-1 rounded text-xs font-semibold" style="background-color: ${protocolColor}33; color: ${protocolColor}">
                ${packet.protocol || 'Unknown'}
            </span>
        </td>
        <td class="px-4 py-2 text-[#8f9099] font-mono">${packet.length || 0} bytes</td>
        <td class="px-4 py-2 text-[#8f9099] text-xs truncate">${packet.info || 'N/A'}</td>
    `;
    
    row.addEventListener('click', () => {
        AppToast.info(`Packet #${index + 1}: ${packet.src_ip || 'Unknown'} → ${packet.dst_ip || 'Unknown'}`);
    });
    
    return row;
}

/**
 * Format bytes to readable size
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Update duration display
 */
function updateDuration() {
    if (!captureStartTime) return;
    
    const elapsed = Math.floor((Date.now() - captureStartTime) / 1000);
    const hours = Math.floor(elapsed / 3600);
    const minutes = Math.floor((elapsed % 3600) / 60);
    const seconds = elapsed % 60;
    
    statDuration.textContent = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

/**
 * Render packets
 */
function renderPackets() {
    packetsTbody.innerHTML = '';
    
    // Update stats
    statPackets.textContent = packets.length;
    statBytes.textContent = formatBytes(totalBytes);
    packetCount.textContent = packets.length;
    
    if (packets.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="7" class="px-4 py-6 text-center text-[#8f9099] text-sm">
                ${isCapturing ? 'Waiting for packets...' : 'No packets captured'}
            </td>
        `;
        packetsTbody.appendChild(row);
        return;
    }
    
    // Render newest packets first
    [...packets].reverse().forEach((packet, index) => {
        try {
            const row = formatPacketRow(packet, index);
            packetsTbody.appendChild(row);
        } catch (err) {
            console.error('Error rendering packet:', err);
        }
    });
}

/**
 * Add packet
 */
function addPacket(packet) {
    packets.unshift(packet);
    totalBytes += packet.length || 0;
    
    // Limit packets
    const limit = parseInt(packetLimitInput.value) || 1000;
    if (packets.length > limit) {
        const removed = packets.pop();
        totalBytes -= removed.length || 0;
    }
    
    renderPackets();
}

/**
 * Start/Stop capture
 */
async function toggleCapture() {
    try {
        if (!isCapturing) {
            // Start capture
            const bpfFilter = bpfFilterInput.value || '';
            const iface = interfaceSelect.value || 'eth0';
            
            if (!iface) {
                AppToast.error('Please select an interface');
                return;
            }
            
            await HttpClient.post('/api/capture/start', {
                interface: iface,
                bpf_filter: bpfFilter,
                packet_limit: parseInt(packetLimitInput.value) || 1000
            });
            
            isCapturing = true;
            packets = [];
            totalBytes = 0;
            captureStartTime = Date.now();
            
            captureToggleBtn.textContent = 'Stop Capture';
            captureToggleBtn.classList.remove('bg-[#10b981]', 'hover:bg-[#059669]');
            captureToggleBtn.classList.add('bg-[#ef4444]', 'hover:bg-[#dc2626]');
            
            statStatus.textContent = 'Capturing';
            statStatus.classList.remove('text-[#6b7280]');
            statStatus.classList.add('text-[#ef4444]');
            
            // Update duration every second
            const durationInterval = setInterval(() => {
                if (isCapturing) {
                    updateDuration();
                } else {
                    clearInterval(durationInterval);
                }
            }, 1000);
            
            AppToast.success(`Capture started on ${iface}`);
            renderPackets();
        } else {
            // Stop capture
            await HttpClient.post('/api/capture/stop');
            
            isCapturing = false;
            captureToggleBtn.textContent = 'Start Capture';
            captureToggleBtn.classList.remove('bg-[#ef4444]', 'hover:bg-[#dc2626]');
            captureToggleBtn.classList.add('bg-[#10b981]', 'hover:bg-[#059669]');
            
            statStatus.textContent = 'Idle';
            statStatus.classList.remove('text-[#ef4444]');
            statStatus.classList.add('text-[#6b7280]');
            
            AppToast.success(`Capture stopped - ${packets.length} packets captured`);
        }
    } catch (err) {
        console.error('Capture error:', err);
        AppToast.error(isCapturing ? 'Failed to stop capture' : 'Failed to start capture');
    }
}

/**
 * Initialize page
 */
function initPage() {
    renderPackets();
    
    // Setup capture button
    captureToggleBtn.addEventListener('click', toggleCapture);
    
    // Setup socket handlers for packet events
    Socket.on('packet.captured', (packet) => {
        if (isCapturing) {
            addPacket(packet);
        }
    });
    
    Socket.on('capture.complete', (data) => {
        if (isCapturing) {
            isCapturing = false;
            captureToggleBtn.textContent = 'Start Capture';
            captureToggleBtn.classList.remove('bg-[#ef4444]', 'hover:bg-[#dc2626]');
            captureToggleBtn.classList.add('bg-[#10b981]', 'hover:bg-[#059669]');
            
            statStatus.textContent = 'Idle';
            statStatus.classList.remove('text-[#ef4444]');
            statStatus.classList.add('text-[#6b7280]');
            
            AppToast.success(`Capture complete - ${packets.length} packets`);
        }
    });
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPage);
} else {
    initPage();
}

export { initPage, renderPackets, addPacket, toggleCapture };

