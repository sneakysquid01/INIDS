/**
 * INIDS Demo Controller
 *
 * Persists scenario state across page navigations via sessionStorage.
 * Navigating between modules continues the running scenario — only ↺ resets.
 *
 * Architecture:
 *   Scenario definitions → DemoController → GlobalState slices → Page subscribers
 *
 * Console shortcuts:
 *   window.DemoController.startScenario('ddos')
 *   window.DemoController.reset()
 *   window.DemoController.setSpeed(2)
 */

import { GlobalState } from '../core/global-state.js';
import { baseline }   from './scenarios/baseline.js';
import { portScan }   from './scenarios/portScan.js';
import { bruteForce } from './scenarios/bruteForce.js';
import { ddos }       from './scenarios/ddos.js';

// ─── sessionStorage keys ──────────────────────────────────────────────────────
const SK = {
    scenarioId: 'inids_demo_scenario',
    startMs:    'inids_demo_start_ms',
    speed:      'inids_demo_speed',
    savedAt:    'inids_demo_saved_at',
    uptimeSec:  'inids_demo_uptime_sec',
    flows:      'inids_demo_flows',
    tiSync:     'inids_demo_ti_sync',
    state:      'inids_demo_state',
};

// ─── Baseline seeds ───────────────────────────────────────────────────────────

const IDLE_METRICS = {
    flows_per_second:      92,
    alerts_per_minute:     0,
    blocked_ips_24h:       0,
    model_accuracy_percent: 97.4,
};

const IDLE_HEALTH = {
    status:              'healthy',
    cpu_percent:         8,
    memory_used_mb:      340,
    memory_total_mb:     1024,
    uptime_seconds:      2 * 86400 + 14 * 3600 + 22 * 60,
    avg_response_time_ms: 4.2,
    error_rate:          0,
    total_alerts:        0,
    total_actions:       0,
    packets_per_sec:     92,
    flows_analyzed:      0,
    db_alerts:           0,
    db_audits:           0,
    db_size_mb:          1.4,
    version:             '3.0.0-demo',
    python_version:      '3.11',
    services: {
        alert_service:     true,
        detection_service: true,
        prevention_service: true,
        ingestion_service: true,
    },
    detection_engines: [
        { name: 'ML Engine',        enabled: true, detections: 0 },
        { name: 'Signature Engine', enabled: true, detections: 0 },
        { name: 'Anomaly Engine',   enabled: true, detections: 0 },
        { name: 'Threshold Engine', enabled: true, detections: 0 },
        { name: 'Honeypot Engine',  enabled: true, detections: 0 },
        { name: 'TI Engine',        enabled: true, detections: 0 },
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

// ─── Controller ───────────────────────────────────────────────────────────────

const SCENARIOS = { baseline, portScan, bruteForce, ddos };

class DemoController {
    constructor() {
        this._timers         = [];
        this._idleTimer      = null;
        this._speed          = 1;
        this._scenario       = null;
        this._startMs        = null;
        this._uptimeSec      = 2 * 86400 + 14 * 3600 + 22 * 60;
        this._flows          = 142670;
        this._tiSyncSec      = 14;
        this._currentMetrics = { ...IDLE_METRICS };
        this._currentHealth  = { ...IDLE_HEALTH };
        this._blockedToday   = 0;
        this._alertsToday    = 0;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    startScenario(id) {
        const s = SCENARIOS[id];
        if (!s) { console.warn('[Demo] Unknown scenario:', id); return; }
        this._clearScenarioTimers();
        this._scenario = s;
        this._startMs  = Date.now();
        this._saveToStorage();
        this._updateChip();
        if (s.timeline.length > 0) {
            this._playRemainingTimeline(s, 0);
        }
        console.log(`[Demo] Started: ${s.name}`);
    }

    stopScenario() {
        this._clearScenarioTimers();
        this._scenario = null;
        this._startMs  = null;
        this._saveToStorage();
        this._updateChip();
    }

    reset() {
        this._clearScenarioTimers();
        this._scenario       = null;
        this._startMs        = null;
        this._blockedToday   = 0;
        this._alertsToday    = 0;
        this._currentMetrics = { ...IDLE_METRICS };
        this._currentHealth  = { ...IDLE_HEALTH };
        this._uptimeSec      = 2 * 86400 + 14 * 3600 + 22 * 60;
        this._flows          = 142670;
        this._tiSyncSec      = 14;

        GlobalState.set('alerts',         []);
        GlobalState.set('actions',        []);
        GlobalState.set('threat_intel',   []);
        GlobalState.set('investigations', []);
        GlobalState.set('honeypots',      JSON.parse(JSON.stringify(IDLE_HONEYPOTS)));
        GlobalState.set('playbooks',      JSON.parse(JSON.stringify(IDLE_PLAYBOOKS)));
        GlobalState.set('metrics',        { ...IDLE_METRICS });
        this._applyHealth(IDLE_HEALTH);

        this._clearStorage();
        this._updateChip();
        console.log('[Demo] Reset to idle');
    }

    setSpeed(multiplier) {
        this._speed = Number(multiplier) || 1;
        const el = document.getElementById('demo-speed-select');
        if (el) el.value = String(this._speed);
        this._saveToStorage();
    }

    getStatus() {
        if (!this._scenario || !this._startMs) return { running: false, scenarioId: null, elapsed: 0 };
        return { running: true, scenarioId: this._scenario.id, elapsed: Date.now() - this._startMs };
    }

    // ── sessionStorage persistence ────────────────────────────────────────────

    _saveToStorage() {
        try {
            sessionStorage.setItem(SK.scenarioId, this._scenario?.id || '');
            sessionStorage.setItem(SK.startMs,    this._startMs || '');
            sessionStorage.setItem(SK.speed,      this._speed);
            sessionStorage.setItem(SK.savedAt,    Date.now());
            sessionStorage.setItem(SK.uptimeSec,  this._uptimeSec);
            sessionStorage.setItem(SK.flows,      this._flows);
            sessionStorage.setItem(SK.tiSync,     this._tiSyncSec);

            const state = {
                alerts:         (GlobalState.get('alerts')         || []).slice(0, 30),
                actions:        (GlobalState.get('actions')        || []).slice(0, 30),
                threat_intel:   (GlobalState.get('threat_intel')   || []).slice(0, 20),
                investigations: (GlobalState.get('investigations') || []).slice(0, 20),
                honeypots:      GlobalState.get('honeypots')  || [],
                playbooks:      GlobalState.get('playbooks')  || [],
                metrics:        this._currentMetrics,
                health:         this._currentHealth,
                blockedToday:   this._blockedToday,
                alertsToday:    this._alertsToday,
            };
            sessionStorage.setItem(SK.state, JSON.stringify(state));
        } catch (e) { /* quota or unavailable */ }
    }

    _restoreFromStorage() {
        try {
            const scenarioId  = sessionStorage.getItem(SK.scenarioId);
            const startMs     = Number(sessionStorage.getItem(SK.startMs))   || 0;
            const speed       = Number(sessionStorage.getItem(SK.speed))     || 1;
            const savedAt     = Number(sessionStorage.getItem(SK.savedAt))   || 0;
            const savedUptime = Number(sessionStorage.getItem(SK.uptimeSec)) || this._uptimeSec;
            const savedFlows  = Number(sessionStorage.getItem(SK.flows))     || this._flows;
            const savedTiSync = Number(sessionStorage.getItem(SK.tiSync))    || this._tiSyncSec;
            const stateJson   = sessionStorage.getItem(SK.state);

            if (!savedAt) return; // nothing saved yet

            // How many seconds passed while we were on another page
            const gapSec = Math.floor((Date.now() - savedAt) / 1000);

            this._speed      = speed;
            this._uptimeSec  = savedUptime + gapSec;
            this._flows      = savedFlows  + gapSec * 5;
            this._tiSyncSec  = Math.max(2, (savedTiSync + gapSec) % 60);

            if (stateJson) {
                const s = JSON.parse(stateJson);
                GlobalState.set('alerts',         s.alerts         || []);
                GlobalState.set('actions',        s.actions        || []);
                GlobalState.set('threat_intel',   s.threat_intel   || []);
                GlobalState.set('investigations', s.investigations || []);
                GlobalState.set('honeypots',      s.honeypots      || JSON.parse(JSON.stringify(IDLE_HONEYPOTS)));
                GlobalState.set('playbooks',      s.playbooks      || JSON.parse(JSON.stringify(IDLE_PLAYBOOKS)));
                this._currentMetrics = s.metrics      || { ...IDLE_METRICS };
                this._currentHealth  = s.health       || { ...IDLE_HEALTH };
                this._blockedToday   = s.blockedToday || 0;
                this._alertsToday    = s.alertsToday  || 0;
                GlobalState.set('metrics', this._currentMetrics);
                this._applyHealth(this._currentHealth);
            }

            // Resume scenario if one was active
            if (scenarioId && scenarioId !== '' && startMs) {
                const scenario = SCENARIOS[scenarioId];
                if (scenario) {
                    const elapsedMs = Date.now() - startMs;
                    if (elapsedMs < scenario.duration) {
                        this._scenario = scenario;
                        this._startMs  = startMs;
                        this._playRemainingTimeline(scenario, elapsedMs);
                        console.log(`[Demo] Resumed: ${scenario.name} (${Math.round(elapsedMs/1000)}s in)`);
                    }
                    // else: scenario finished while we were away — leave state as-is
                }
            }
        } catch (e) {
            console.warn('[Demo] Restore failed:', e);
        }
    }

    _clearStorage() {
        Object.values(SK).forEach(k => sessionStorage.removeItem(k));
    }

    // ── Idle ticker ───────────────────────────────────────────────────────────

    startIdle() {
        // Seed baseline only if nothing was restored from storage
        if (!sessionStorage.getItem(SK.savedAt)) {
            GlobalState.set('metrics',   { ...IDLE_METRICS });
            GlobalState.set('honeypots', JSON.parse(JSON.stringify(IDLE_HONEYPOTS)));
            GlobalState.set('playbooks', JSON.parse(JSON.stringify(IDLE_PLAYBOOKS)));
            this._applyHealth(IDLE_HEALTH);
        }
        this._idleTimer = setInterval(() => this._tickIdle(), 2000);
    }

    _tickIdle() {
        this._uptimeSec += 2;
        this._flows     += Math.floor(Math.random() * 6) + 3;
        this._tiSyncSec += 2;
        if (this._tiSyncSec > 60) this._tiSyncSec = 5;

        const pktJitter = Math.round((Math.random() - 0.5) * 12);
        const cpuJitter = Math.round((Math.random() - 0.5) * 2);
        const newPkt = Math.max(75, Math.min(155, (this._currentHealth.packets_per_sec || 92) + pktJitter));
        const newCpu = Math.max(6,  Math.min(14,  (this._currentHealth.cpu_percent    || 8)  + cpuJitter));

        this._currentHealth = {
            ...this._currentHealth,
            cpu_percent:     newCpu,
            packets_per_sec: newPkt,
            flows_analyzed:  this._flows,
            uptime_seconds:  this._uptimeSec,
        };
        this._applyHealth(this._currentHealth);

        this._currentMetrics = {
            ...this._currentMetrics,
            flows_per_second: newPkt,
            blocked_ips_24h:  this._blockedToday,
        };
        GlobalState.update('metrics', this._currentMetrics);

        // Save every tick so page navigations pick up fresh counters
        this._saveToStorage();
        this._updateChip();
    }

    // ── Scenario playback ─────────────────────────────────────────────────────

    _playRemainingTimeline(scenario, elapsedMs) {
        // Only fire events that haven't happened yet
        const remaining = scenario.timeline.filter(e => e.at > elapsedMs);

        remaining.forEach(event => {
            const delay = Math.round((event.at - elapsedMs) / this._speed);
            const t = setTimeout(() => {
                this._handleEvent(event);
                this._saveToStorage();
                this._updateChip();
            }, delay);
            this._timers.push(t);
        });

        // Auto-return to idle when scenario ends
        const endDelay = Math.max(0, Math.round((scenario.duration - elapsedMs + 2000) / this._speed));
        const endTimer = setTimeout(() => {
            this._scenario = null;
            this._startMs  = null;
            this._saveToStorage();
            this._updateChip();
            const sel = document.getElementById('demo-scenario-select');
            if (sel) sel.value = 'idle';
            console.log('[Demo] Scenario complete — back to idle');
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
                setTimeout(() => {
                    GlobalState.push('threat_intel', p);
                }, 900 / this._speed);
                break;

            case 'auto_block':
                this._blockedToday++;
                this._currentHealth.total_actions = this._blockedToday;
                GlobalState.push('actions', p);
                this._currentMetrics.blocked_ips_24h = this._blockedToday;
                GlobalState.update('metrics', this._currentMetrics);
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
                p.created_date = new Date().toISOString();
                GlobalState.push('investigations', p);
                break;

            default:
                break;
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    _applyHealth(data) {
        window.INIDS_DEMO_HEALTH = data;
        if (typeof loadHealthData === 'function') loadHealthData();
    }

    _bumpSeverity(current, severity) {
        const out = { ...current };
        if      (severity === 'critical') out.critical = (out.critical || 0) + 1;
        else if (severity === 'high')     out.high     = (out.high     || 0) + 1;
        else if (severity === 'medium')   out.medium   = (out.medium   || 0) + 1;
        else                              out.low      = (out.low      || 0) + 1;
        return out;
    }

    _bumpHoneypot(id) {
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

    // ── Toolbar ───────────────────────────────────────────────────────────────

    mountToolbar() {
        const container = document.getElementById('demo-toolbar');
        if (!container) return;

        container.innerHTML = `
            <select id="demo-scenario-select" title="Demo scenario" style="
                background:#0a0c10;border:1px solid #1a1f2e;color:#e2e8f0;
                font-size:11px;font-family:var(--mono,monospace);border-radius:6px;
                padding:4px 8px;cursor:pointer;outline:none;">
                <option value="idle">Demo: Idle</option>
                <option value="portScan">Demo: Port Scan</option>
                <option value="bruteForce">Demo: Brute Force</option>
                <option value="ddos">Demo: DDoS</option>
            </select>
            <select id="demo-speed-select" title="Playback speed" style="
                background:#0a0c10;border:1px solid #1a1f2e;color:#8f9099;
                font-size:11px;font-family:var(--mono,monospace);border-radius:6px;
                padding:4px 6px;cursor:pointer;outline:none;width:48px;">
                <option value="1">1x</option>
                <option value="2">2x</option>
                <option value="5">5x</option>
            </select>
            <button type="button" id="demo-reset-btn" title="Reset demo" style="
                background:transparent;border:1px solid #1a1f2e;color:#8f9099;
                border-radius:6px;padding:4px 8px;cursor:pointer;font-size:13px;line-height:1;">↺</button>
            <div id="demo-progress-chip" style="
                display:none;font-family:var(--mono,monospace);font-size:10px;color:#06b6d4;
                background:rgba(6,182,212,.1);border:1px solid rgba(6,182,212,.25);
                border-radius:9999px;padding:3px 10px;white-space:nowrap;"></div>
        `;

        // Sync dropdown to any restored scenario
        const sel = document.getElementById('demo-scenario-select');
        if (sel && this._scenario) sel.value = this._scenario.id;

        // Sync speed selector
        const spd = document.getElementById('demo-speed-select');
        if (spd) spd.value = String(this._speed);

        sel.addEventListener('change', e => {
            const val = e.target.value;
            if (val === 'idle') this.stopScenario();
            else                this.startScenario(val);
        });

        document.getElementById('demo-speed-select').addEventListener('change', e => {
            this.setSpeed(e.target.value);
        });

        document.getElementById('demo-reset-btn').addEventListener('click', () => {
            this.reset();
            const s = document.getElementById('demo-scenario-select');
            if (s) s.value = 'idle';
        });

        this._updateChip();
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
        const fmt = s => String(Math.floor(s / 60)).padStart(2,'0') + ':' + String(s % 60).padStart(2,'0');
        chip.textContent  = `${this._scenario.name} — ${fmt(elapsed)} / ${fmt(total)}`;
        chip.style.display = 'block';
    }
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────

const ctrl = new DemoController();

document.addEventListener('DOMContentLoaded', () => {
    // Restore first so toolbar dropdown and GlobalState are correct immediately
    ctrl._restoreFromStorage();
    ctrl.mountToolbar();
    ctrl.startIdle();
});

window.DemoController = ctrl;
export default ctrl;
