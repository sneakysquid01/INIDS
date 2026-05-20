/**
 * INIDS Demo Controller
 *
 * Orchestrates a centralized demo mode across all pages.
 * All fake data originates here — no module generates its own.
 *
 * Architecture:
 *   Scenario definitions → DemoController → GlobalState slices → Page subscribers
 *
 * Usage (browser console):
 *   window.DemoController.startScenario('ddos')
 *   window.DemoController.reset()
 *   window.DemoController.setSpeed(2)
 */

import { GlobalState } from '../core/global-state.js';
import { baseline }   from './scenarios/baseline.js';
import { portScan }   from './scenarios/portScan.js';
import { bruteForce } from './scenarios/bruteForce.js';
import { ddos }       from './scenarios/ddos.js';

// ─── Baseline idle state seeds ───────────────────────────────────────────────

const IDLE_METRICS = {
    flows_per_second: 92,
    alerts_per_minute: 0,
    blocked_ips_24h: 0,
    model_accuracy_percent: 97.4,
};

const IDLE_HEALTH = {
    status: 'healthy',
    cpu_percent: 8,
    memory_used_mb: 340,
    memory_total_mb: 1024,
    uptime_seconds: 2 * 86400 + 14 * 3600 + 22 * 60,
    avg_response_time_ms: 4.2,
    error_rate: 0,
    total_alerts: 0,
    total_actions: 0,
    packets_per_sec: 92,
    flows_analyzed: 0,
    db_alerts: 0,
    db_audits: 0,
    db_size_mb: 1.4,
    version: '3.0.0-demo',
    python_version: '3.11',
    services: {
        alert_service: true,
        detection_service: true,
        prevention_service: true,
        ingestion_service: true,
    },
    detection_engines: [
        { name: 'ML Engine',        enabled: true,  detections: 0 },
        { name: 'Signature Engine', enabled: true,  detections: 0 },
        { name: 'Anomaly Engine',   enabled: true,  detections: 0 },
        { name: 'Threshold Engine', enabled: true,  detections: 0 },
        { name: 'Honeypot Engine',  enabled: true,  detections: 0 },
        { name: 'TI Engine',        enabled: true,  detections: 0 },
    ],
    alerts_by_severity: { critical: 0, high: 0, medium: 0, low: 0 },
};

const IDLE_HONEYPOTS = [
    {
        id: 'hp-ssh',
        name: 'Fake SSH Service',
        type: 'service',
        active: true,
        protocol: 'TCP',
        port: 22,
        description: 'Decoy SSH listener — captures credential stuffers and scanners',
        interaction_count: 0,
        threat_count: 0,
        success_rate: 0,
        created_date: new Date(Date.now() - 5 * 86400000).toISOString(),
    },
    {
        id: 'hp-http',
        name: 'Fake Admin Portal',
        type: 'service',
        active: true,
        protocol: 'TCP',
        port: 8080,
        description: 'Decoy HTTP admin panel — attracts web scanners',
        interaction_count: 0,
        threat_count: 0,
        success_rate: 0,
        created_date: new Date(Date.now() - 3 * 86400000).toISOString(),
    },
];

const IDLE_PLAYBOOKS = [
    {
        id: 'pb-autoblock',
        name: 'Auto-Block Suspicious IPs',
        type: 'prevention',
        enabled: true,
        description: 'Automatically blocks IPs flagged by threat intelligence with confidence > 70%',
        action_count: 3,
        execution_count: 0,
        created_date: new Date(Date.now() - 10 * 86400000).toISOString(),
        last_run_date: null,
    },
    {
        id: 'pb-escalate',
        name: 'Escalate Critical Alerts',
        type: 'notification',
        enabled: true,
        description: 'Sends email and creates incident for CRITICAL severity alerts',
        action_count: 2,
        execution_count: 0,
        created_date: new Date(Date.now() - 7 * 86400000).toISOString(),
        last_run_date: null,
    },
    {
        id: 'pb-ratelimit',
        name: 'Rate-Limit Repeated Offenders',
        type: 'prevention',
        enabled: true,
        description: 'Rate-limits IPs with > 5 alerts in 10 minutes before full block',
        action_count: 2,
        execution_count: 0,
        created_date: new Date(Date.now() - 4 * 86400000).toISOString(),
        last_run_date: null,
    },
];

// ─── Relative timestamp helper ────────────────────────────────────────────────

function relTime(ts) {
    const diff = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
    if (diff < 5)   return 'just now';
    if (diff < 60)  return `${diff} sec ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
    return `${Math.floor(diff / 3600)} hr ago`;
}

// ─── Controller ───────────────────────────────────────────────────────────────

const SCENARIOS = { baseline, portScan, bruteForce, ddos };

class DemoController {
    constructor() {
        this._timers       = [];
        this._idleTimer    = null;
        this._tsTimer      = null;
        this._speed        = 1;
        this._scenario     = null;
        this._startMs      = null;
        this._uptimeSec    = 2 * 86400 + 14 * 3600 + 22 * 60;
        this._flows        = 142670;
        this._tiSyncSec    = 14;
        this._currentMetrics = { ...IDLE_METRICS };
        this._currentHealth  = { ...IDLE_HEALTH };
        this._blockedToday   = 0;
        this._alertsToday    = 0;
    }

    // ── Public API ──────────────────────────────────────────────────────────

    startScenario(id) {
        const s = SCENARIOS[id];
        if (!s) { console.warn('[Demo] Unknown scenario:', id); return; }
        this.reset();
        this._scenario = s;
        this._startMs  = Date.now();
        this._updateChip();
        if (s.timeline.length > 0) {
            this._playScenario(s);
        }
        console.log(`[Demo] Started scenario: ${s.name}`);
    }

    stopScenario() {
        this._clearScenarioTimers();
        this._scenario = null;
        this._startMs  = null;
        this._updateChip();
    }

    reset() {
        this._clearScenarioTimers();
        this._scenario    = null;
        this._startMs     = null;
        this._blockedToday = 0;
        this._alertsToday  = 0;
        this._currentMetrics = { ...IDLE_METRICS };
        this._currentHealth  = { ...IDLE_HEALTH };

        // Reset all relevant GlobalState slices
        GlobalState.set('alerts',         []);
        GlobalState.set('actions',        []);
        GlobalState.set('threat_intel',   []);
        GlobalState.set('investigations', []);
        GlobalState.set('honeypots',      JSON.parse(JSON.stringify(IDLE_HONEYPOTS)));
        GlobalState.set('playbooks',      JSON.parse(JSON.stringify(IDLE_PLAYBOOKS)));
        GlobalState.set('metrics',        { ...IDLE_METRICS });

        this._applyHealth(IDLE_HEALTH);
        this._updateChip();
        console.log('[Demo] Reset to idle baseline');
    }

    setSpeed(multiplier) {
        this._speed = Number(multiplier) || 1;
        const el = document.getElementById('demo-speed-select');
        if (el) el.value = String(this._speed);
    }

    getStatus() {
        if (!this._scenario || !this._startMs) return { running: false, scenarioId: null, elapsed: 0 };
        return {
            running:    true,
            scenarioId: this._scenario.id,
            elapsed:    Date.now() - this._startMs,
        };
    }

    // ── Idle ticker ─────────────────────────────────────────────────────────

    startIdle() {
        this._seedBaseline();
        this._idleTimer = setInterval(() => this._tickIdle(), 2000);
        // Relative timestamp refresh every 5s
        this._tsTimer   = setInterval(() => this._refreshTimestamps(), 5000);
    }

    _seedBaseline() {
        GlobalState.set('metrics',   { ...IDLE_METRICS });
        GlobalState.set('honeypots', JSON.parse(JSON.stringify(IDLE_HONEYPOTS)));
        GlobalState.set('playbooks', JSON.parse(JSON.stringify(IDLE_PLAYBOOKS)));
        this._applyHealth(IDLE_HEALTH);
    }

    _tickIdle() {
        this._uptimeSec += 2;
        this._flows     += Math.floor(Math.random() * 6) + 3;
        this._tiSyncSec += 2;
        if (this._tiSyncSec > 60) this._tiSyncSec = 5;

        // Jitter packets/sec and CPU in idle band
        const pktJitter  = Math.round((Math.random() - 0.5) * 12);
        const cpuJitter  = Math.round((Math.random() - 0.5) * 2);

        const newPkt = Math.max(75, Math.min(155,
            (this._currentHealth.packets_per_sec || 92) + pktJitter));
        const newCpu = Math.max(6, Math.min(14,
            (this._currentHealth.cpu_percent || 8) + cpuJitter));

        this._currentHealth = {
            ...this._currentHealth,
            cpu_percent:    newCpu,
            packets_per_sec: newPkt,
            flows_analyzed:  this._flows,
            uptime_seconds:  this._uptimeSec,
        };
        this._applyHealth(this._currentHealth);

        // Keep metrics in sync
        this._currentMetrics = {
            ...this._currentMetrics,
            flows_per_second: newPkt,
            blocked_ips_24h:  this._blockedToday,
        };
        GlobalState.update('metrics', this._currentMetrics);

        this._updateStatusPill();
    }

    // ── Scenario playback ────────────────────────────────────────────────────

    _playScenario(scenario) {
        scenario.timeline.forEach(event => {
            const delay = Math.round(event.at / this._speed);
            const t = setTimeout(() => {
                this._handleEvent(event);
                this._updateChip();
            }, delay);
            this._timers.push(t);
        });

        // Auto-return to idle when scenario ends
        const endDelay = Math.round((scenario.duration + 2000) / this._speed);
        const endTimer = setTimeout(() => {
            this._scenario = null;
            this._startMs  = null;
            this._updateChip();
            const sel = document.getElementById('demo-scenario-select');
            if (sel) sel.value = 'idle';
            console.log('[Demo] Scenario complete — returned to idle');
        }, endDelay);
        this._timers.push(endTimer);
    }

    _handleEvent(event) {
        const p = { ...event.payload, timestamp: new Date().toISOString() };

        switch (event.type) {
            case 'traffic_spike':
                this._currentMetrics = { ...this._currentMetrics, ...p };
                GlobalState.update('metrics', this._currentMetrics);
                this._currentHealth.packets_per_sec = p.flows_per_second || this._currentHealth.packets_per_sec;
                break;

            case 'alert':
                this._alertsToday++;
                this._currentHealth.total_alerts = this._alertsToday;
                this._currentHealth.alerts_by_severity = this._bumpSeverity(
                    this._currentHealth.alerts_by_severity, p.severity);
                GlobalState.push('alerts', p);
                this._currentMetrics.alerts_per_minute = Math.min(
                    (this._currentMetrics.alerts_per_minute || 0) + 2, 60);
                GlobalState.update('metrics', this._currentMetrics);
                window.RealtimeFeed?.addAlert(p);
                break;

            case 'threat_lookup':
                p.updated = new Date().toISOString();
                // Simulate 900ms lookup delay for realism
                setTimeout(() => {
                    GlobalState.push('threat_intel', p);
                }, 900 / this._speed);
                break;

            case 'auto_block':
                this._blockedToday++;
                this._currentHealth.total_actions = this._blockedToday;
                p.timestamp = new Date().toISOString();
                GlobalState.push('actions', p);
                this._currentMetrics.blocked_ips_24h = this._blockedToday;
                GlobalState.update('metrics', this._currentMetrics);
                // Update a playbook's last run / execution count
                this._bumpPlaybookExecution('pb-autoblock');
                window.RealtimeFeed?.addAction(p);
                break;

            case 'incident_created':
                p.created_date = new Date().toISOString();
                GlobalState.push('investigations', p);
                this._bumpPlaybookExecution('pb-escalate');
                break;

            case 'honeypot_hit':
                this._bumpHoneypot('hp-ssh', p.source_ip);
                window.RealtimeFeed?.addDetection({
                    source_ip:  p.source_ip,
                    prediction: 'attack',
                    confidence: '0.97',
                    suspicious: true,
                });
                break;

            case 'health_change':
                this._currentHealth = {
                    ...this._currentHealth,
                    cpu_percent: p.cpu_percent ?? this._currentHealth.cpu_percent,
                    error_rate:  p.error_rate  ?? this._currentHealth.error_rate,
                };
                this._applyHealth(this._currentHealth);
                break;

            case 'log_entry':
                // Map log entries to investigations (closest matching slice)
                p.created_date = new Date().toISOString();
                GlobalState.push('investigations', p);
                break;

            default:
                break;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    _applyHealth(data) {
        window.INIDS_DEMO_HEALTH = data;
        // If health.js is loaded (health page), trigger a refresh
        if (typeof loadHealthData === 'function') {
            loadHealthData();
        }
    }

    _bumpSeverity(current, severity) {
        const out = { ...current };
        if (severity === 'critical') out.critical = (out.critical || 0) + 1;
        else if (severity === 'high') out.high = (out.high || 0) + 1;
        else if (severity === 'medium') out.medium = (out.medium || 0) + 1;
        else out.low = (out.low || 0) + 1;
        return out;
    }

    _bumpHoneypot(id, sourceIp) {
        const honeypots = GlobalState.get('honeypots') || [];
        const idx = honeypots.findIndex(h => h.id === id);
        if (idx >= 0) {
            honeypots[idx] = {
                ...honeypots[idx],
                interaction_count: (honeypots[idx].interaction_count || 0) + 1,
                threat_count:      (honeypots[idx].threat_count      || 0) + 1,
            };
            GlobalState.set('honeypots', [...honeypots]);
        }
    }

    _bumpPlaybookExecution(id) {
        const playbooks = GlobalState.get('playbooks') || [];
        const idx = playbooks.findIndex(p => p.id === id);
        if (idx >= 0) {
            playbooks[idx] = {
                ...playbooks[idx],
                execution_count: (playbooks[idx].execution_count || 0) + 1,
                last_run_date:   new Date().toISOString(),
            };
            GlobalState.set('playbooks', [...playbooks]);
        }
    }

    _clearScenarioTimers() {
        this._timers.forEach(clearTimeout);
        this._timers = [];
    }

    _refreshTimestamps() {
        // Page controllers that render timestamps will re-read from state on next
        // GlobalState notification — force a no-op push to trigger re-render.
        // Only ping slices that have data to avoid polluting empty states.
        const alerts = GlobalState.get('alerts');
        if (alerts && alerts.length > 0) GlobalState.set('alerts', alerts);
    }

    // ── Toolbar UI ───────────────────────────────────────────────────────────

    mountToolbar() {
        const container = document.getElementById('demo-toolbar');
        if (!container) return;

        container.innerHTML = `
            <select id="demo-scenario-select" title="Demo scenario" style="
                background:#0a0c10;
                border:1px solid #1a1f2e;
                color:#e2e8f0;
                font-size:11px;
                font-family:var(--mono,monospace);
                border-radius:6px;
                padding:4px 8px;
                cursor:pointer;
                outline:none;
            ">
                <option value="idle">Demo: Idle</option>
                <option value="portScan">Demo: Port Scan</option>
                <option value="bruteForce">Demo: Brute Force</option>
                <option value="ddos">Demo: DDoS</option>
            </select>

            <select id="demo-speed-select" title="Playback speed" style="
                background:#0a0c10;
                border:1px solid #1a1f2e;
                color:#8f9099;
                font-size:11px;
                font-family:var(--mono,monospace);
                border-radius:6px;
                padding:4px 6px;
                cursor:pointer;
                outline:none;
                width:48px;
            ">
                <option value="1">1x</option>
                <option value="2">2x</option>
                <option value="5">5x</option>
            </select>

            <button id="demo-reset-btn" title="Reset demo" style="
                background:transparent;
                border:1px solid #1a1f2e;
                color:#8f9099;
                border-radius:6px;
                padding:4px 8px;
                cursor:pointer;
                font-size:13px;
                line-height:1;
            ">↺</button>

            <div id="demo-progress-chip" style="
                display:none;
                font-family:var(--mono,monospace);
                font-size:10px;
                color:#06b6d4;
                background:rgba(6,182,212,.1);
                border:1px solid rgba(6,182,212,.25);
                border-radius:9999px;
                padding:3px 10px;
                white-space:nowrap;
            "></div>
        `;

        document.getElementById('demo-scenario-select').addEventListener('change', (e) => {
            const val = e.target.value;
            if (val === 'idle') { this.stopScenario(); }
            else                { this.startScenario(val); }
        });

        document.getElementById('demo-speed-select').addEventListener('change', (e) => {
            this.setSpeed(e.target.value);
        });

        document.getElementById('demo-reset-btn').addEventListener('click', () => {
            this.reset();
            const sel = document.getElementById('demo-scenario-select');
            if (sel) sel.value = 'idle';
        });
    }

    _updateChip() {
        const chip = document.getElementById('demo-progress-chip');
        if (!chip) return;

        if (!this._scenario || !this._startMs) {
            chip.style.display = 'none';
            return;
        }

        const elapsed = Math.floor((Date.now() - this._startMs) / 1000);
        const total   = Math.ceil(this._scenario.duration / 1000);
        const eStr    = String(Math.floor(elapsed / 60)).padStart(2,'0') + ':' +
                        String(elapsed % 60).padStart(2,'0');
        const tStr    = String(Math.floor(total   / 60)).padStart(2,'0') + ':' +
                        String(total   % 60).padStart(2,'0');
        chip.textContent  = `${this._scenario.name} — ${eStr} / ${tStr}`;
        chip.style.display = 'block';
    }

    _updateStatusPill() {
        const ti = document.getElementById('demo-ti-sync');
        if (ti) ti.textContent = `TI sync: ${this._tiSyncSec}s ago`;
    }
}

// ─── Bootstrap ───────────────────────────────────────────────────────────────

const ctrl = new DemoController();

document.addEventListener('DOMContentLoaded', () => {
    ctrl.mountToolbar();
    ctrl.startIdle();
});

window.DemoController = ctrl;
export default ctrl;
