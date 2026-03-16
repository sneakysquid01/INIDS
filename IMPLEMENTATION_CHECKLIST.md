# INIDS Demo Platform — Implementation Checklist

## Quick Reference: All Files to Create/Modify

---

## LAYER 1: DASHBOARD FOUNDATIONS (Week 1, Days 1-2)

### Task 1.1: Dashboard Layout
- [ ] **CREATE** `web_app/templates/dashboard_main.html`
  - Bootstrap 5 grid layout (6 columns)
  - 15 capability cards arranged in 3 rows x 5 cards
  - System status bar at top (3 cols: health, threats, metrics)
  - Dark theme CSS
  
- [ ] **CREATE** `web_app/static/css/dashboard.css`
  - Dark navy background (#1a1f3a)
  - Card hover effects
  - Red/orange/green alert colors
  - Smooth transitions
  
- [ ] **CREATE** `web_app/static/js/dashboard.js`
  - Click handlers: each card opens module in modal or sidebar
  - Demo mode toggle (triggers sample traffic)
  - Active module tracking
  
- [ ] **MODIFY** `web_app/app.py`
  - Add route: `@app.route('/dashboard')` → render dashboard_main.html
  - Add route: `@app.route('/module/<module_name>')` → render module template
  - Seed system status data

**Acceptance Criteria:**
- Dashboard loads without errors
- 15 card titles visible, clickable
- System status shows correct data
- Mobile responsive (tested on phone view)

---

### Task 1.2: WebSocket Infrastructure
- [ ] **INSTALL** flask-sock: `pip install flask-sock`

- [ ] **MODIFY** `web_app/app.py`
  - Import: `from flask_sock import Sock`
  - Initialize: `sock = Sock(app)`
  - Add route: `@sock.route('/ws/detections')`
    - Accept WebSocket connections
    - Iterate through `event_bus.subscribe('detection')`
    - Convert DetectionEvent to JSON
    - Send via WebSocket (flush=True for real-time)
    - Handle client disconnect gracefully

- [ ] **CREATE** `web_app/static/js/websocket_client.js`
  - Connect to `/ws/detections`
  - Handle incoming JSON events
  - Emit custom JS events: `window.dispatchEvent(new CustomEvent('detection', {detail: event}))`

**Acceptance Criteria:**
- WebSocket connected indicator shows green
- 10 test events successfully received
- Handles network interrupt gracefully

---

### Task 1.3: Module Template System
- [ ] **CREATE** `web_app/templates/modules/base_module.html`
  - Jinja template with blocks: `{% block title %}`, `{% block content %}`, `{% block controls %}`
  - Standard header with module name
  - Info bar with config
  - Footer with action buttons
  - Extend this for all 15 modules

- [ ] **CREATE** `web_app/templates/modules/README.md`
  - Template for developers
  - Shows how to extend base_module.html
  - Example: `{% extends "modules/base_module.html" %}`

**Acceptance Criteria:**
- Each module template properly extends base
- Consistent styling across all modules

---

### Task 1.4: existing API Verification
Run existing tests to ensure no regressions:

```bash
python -m pytest tests/ -v --tb=short -x
```

**Acceptance Criteria:**
- Total tests: 318 passed
- Exit code: 0
- No warnings

---

## LAYER 2: CORE DETECTION ENGINES (Week 1, Days 3-7)

### Task 2.1: Multi-Engine Detection Framework (Backend)

- [ ] **CREATE** `src/detection/engine_registry.py`
  ```python
  class DetectionEngine:
      """Base class for all detection engines"""
      def predict(self, features) -> DetectionResult:
          pass
      
  class EngineRegistry:
      """Central registry for all detection engines"""
      def __init__(self):
          self.engines = {}
      
      def register(self, name, engine: DetectionEngine):
          self.engines[name] = engine
      
      def predict_all(self, features) -> List[DetectionResult]:
          results = []
          for name, engine in self.engines.items():
              result = engine.predict(features)
              results.append((name, result))
          return results
      
      def aggregate(self, results) -> AggregatedDetectionResult:
          # Majority vote + weighted average confidence
          pass
  ```

- [ ] **CREATE** `src/detection/voting_ensemble.py`
  ```python
  class VotingEnsemble:
      @staticmethod
      def majority_vote(verdicts: List[str]) -> str:
          """Return most common verdict"""
          pass
      
      @staticmethod
      def weighted_average_confidence(results: List[tuple]) -> float:
          """Average confidence across engines"""
          pass
      
      @staticmethod
      def generate_consensus_result(all_results) -> DetectionResult:
          """Combine all engine results into consensus"""
          pass
  ```

- [ ] **MODIFY** `src/detection/service.py`
  - Import DetectionService class
  - Add method: `predict_multi_engine(features) -> dict`
    - Call engine registry
    - Return per-engine results + consensus
    - Cache results for dashboard display

- [ ] **MODIFY** `web_app/app.py`
  - Add route: `POST /api/predict/multi-engine`
  - Input: flow features (JSON)
  - Output: `{engines: [{name, verdict, confidence}], consensus: {...}}`

**Unit Tests:**
- [ ] `tests/test_multi_engine_voting.py`
  - Test each engine individually
  - Test voting logic (tie-breaking, weighted average)
  - Test consensus calculation

**Acceptance Criteria:**
- All 5 engines return verdicts
- Consensus result always correct
- API endpoint returns proper JSON

---

### Task 2.2: Risk Scoring API

- [ ] **MODIFY** `web_app/app.py`
  - Add route: `POST /api/predict/score-breakdown`
  - Input: detection event + features
  - Output:
    ```json
    {
      "total_risk_score": 78,
      "base_score": 50,
      "confidence_factor": 28,
      "confidence_pct": 87,
      "severity_score": 8,
      "severity_pct": 10,
      "frequency_score": 0,
      "frequency_pct": 0,
      "thresholds": {
        "alert": 40,
        "block": 70
      }
    }
    ```

**Acceptance Criteria:**
- Risk calculation matches RiskEngine output
- Thresholds configurable via API
- Percentages sum to 100%

---

### Task 2.3: Real-Time Detection Panel (Frontend)

- [ ] **CREATE** `web_app/templates/modules/real_time_detection.html`
  ```html
  {% extends "modules/base_module.html" %}
  
  {% block title %}Real-Time Detection Panel{% endblock %}
  
  {% block content %}
  <div id="event-feed" class="event-feed">
    <!-- Events will be injected here by JS -->
  </div>
  {% endblock %}
  ```

- [ ] **CREATE** `web_app/static/js/modules/real_time_detection.js`
  - Listen for 'detection' custom event from WebSocket
  - Create event card HTML:
    ```html
    <div class="event-card alert">
      <div class="verdict-badge">ATTACK</div>
      <div class="flow-info">192.168.1.42 → 10.0.0.1:443</div>
      <div class="risk-score">Risk: 78/100</div>
      <div class="confidence">ML: 87%</div>
      <div class="timestamp">Just now</div>
    </div>
    ```
  - Animate: slide in from bottom, fade color transition
  - Click card → show detail modal
  - Max 50 events visible (oldest fade out)

- [ ] **CREATE** `web_app/static/css/modules/real_time_detection.css`
  - Event feed layout (vertical scrolling)
  - Card styles (border, shadow, hover)
  - Animation keyframes (slide-in, color transition)

**Manual Test:**
```bash
# Terminal 1: Start web server
python -m web_app.app

# Terminal 2: In Python
from src.realtime_simulation import run_simulation
run_simulation(num_events=10, interval=1)

# Browser: Open dashboard → Real-Time Detection Panel
# Watch events appear in real-time
```

**Acceptance Criteria:**
- Events appear in real-time (< 100ms latency)
- Animations smooth
- Colors correct (red, orange, green)
- Detail modal shows full context

---

### Task 2.4: Multi-Engine Comparison UI

- [ ] **CREATE** `web_app/templates/modules/multi_engine.html`
  - Table layout: Engine | Verdict | Confidence | Badge
  - Consensus row (bold/highlighted)
  - Coverage % metric

- [ ] **CREATE** `web_app/static/js/modules/multi_engine.js`
  - Fetch endpoint: `POST /api/predict/multi-engine`
  - Populate table with results
  - Add toggle buttons to disable engines
  - Recalculate on toggle
  - Show: "Coverage with 5 engines: 95% | With 4 engines: 92%"

- [ ] **CREATE** `web_app/static/css/modules/multi_engine.css`
  - Table styling, dark theme
  - Consensus row highlighting

**Manual Test:**
```bash
# Upload sample attack PCAP
# View Multi-Engine panel
# Verify: all 5 engines show, consensus correct
# Toggle engines: consensus updates
```

**Acceptance Criteria:**
- All engines participate in voting
- Consensus calculation correct
- Toggle updates coverage % correctly

---

### Task 2.5: Risk Score Visualization UI

- [ ] **CREATE** `web_app/templates/modules/risk_score_viz.html`
  - Gauge container (Canvas element)
  - Factor breakdown (pie or stacked bar)
  - Threshold markers
  - Historical sparkline

- [ ] **CREATE** `web_app/static/js/modules/risk_gauge.js`
  ```javascript
  class RiskGauge {
      constructor(canvasId, maxScore = 100) {
          this.canvas = document.getElementById(canvasId);
          this.ctx = this.canvas.getContext('2d');
      }
      
      animate(fromScore, toScore, duration = 2000) {
          // Animate gauge fill from fromScore to toScore
          // Color gradient: green (0-40), yellow (40-70), red (70-100)
      }
      
      drawThresholds(alertThreshold, blockThreshold) {
          // Draw horizontal lines at thresholds
      }
  }
  ```

- [ ] **CREATE** `web_app/static/css/modules/risk_gauge.css`

**Manual Test:**
- Open module
- Enter sample attack
- Watch gauge animate from 0 to risk score
- Colors change at thresholds

**Acceptance Criteria:**
- Gauge animation smooth
- Colors accurate
- Thresholds visible and labeled

---

## LAYER 3: IPS & PREVENTION (Week 2)

### Task 3.1: Auto-Blocking Timeline

- [ ] **MODIFY** `src/prevention/action_executor.py`
  - Track timing of each stage:
    ```python
    class ActionTimeline:
        detection_time: float
        risk_score_time: float
        policy_decision_time: float
        execution_time: float
        completion_time: float
    ```
  - Store timeline with each action

- [ ] **MODIFY** `web_app/app.py`
  - Add route: `GET /api/actions/<id>/timeline`
  - Return detailed timeline JSON

- [ ] **CREATE** `web_app/templates/modules/auto_blocking.html`
  - Timeline visualization (horizontal, animated)
  - Stages: Detection → Risk → Policy → Firewall → BLOCKED
  - Status icons (waiting, processing, complete)
  - IP address prominently displayed
  - Firewall rule output panel

- [ ] **CREATE** `web_app/static/js/modules/auto_blocking.js`
  - Fetch action timeline
  - Animate stages progressing
  - Show firewall rule when complete

**Manual Test:**
```bash
# Set Policy to AUTO_BLOCK
# Trigger attack
# View Auto-Blocking panel
# Watch timeline execute in real-time
# Verify firewall rule added
```

**Acceptance Criteria:**
- Timeline executes smoothly
- All stages visible
- Firewall rule shown correctly

---

### Task 3.2: Approval Workflow UI

- [ ] **CREATE** `web_app/templates/modules/approval_workflow.html`
  - Pending actions list
  - For each: IP, alert details, risk score, engine verdicts
  - APPROVE / REJECT buttons
  - Comment field

- [ ] **CREATE** `web_app/static/js/modules/approval_workflow.js`
  - Fetch pending actions: `GET /api/actions?status=pending`
  - Show action detail modal on click
  - APPROVE button: `POST /api/actions/<id>/approve`
  - REJECT button: `POST /api/actions/<id>/reject`
  - Refresh list after action

**Manual Test:**
```bash
# Set Policy to APPROVE_BEFORE_BLOCK
# Trigger attack
# Open Approval Workflow
# Verify pending action shown
# Click APPROVE
# Verify action executed + firewall rule added
```

**Acceptance Criteria:**
- All pending actions listed
- APPROVE/REJECT buttons work
- Action executes after approval

---

### Task 3.3: False Positive Learning UI

- [ ] **CREATE** `web_app/templates/modules/false_positive.html`
  - List of recent alerts
  - "Mark as FP" button for each
  - Active suppressions list
  - Suppression statistics (# prevented this week)

- [ ] **CREATE** `web_app/static/js/modules/false_positive.js`
  - Fetch alerts: `GET /api/alerts`
  - "Mark FP" button: `POST /api/alerts/<id>/suppress`
  - Fetch suppressions: `GET /api/suppressions`
  - Show count of prevented alerts

**Manual Test:**
```bash
# Generate false positive alert
# Mark as FP
# Trigger same traffic again
# Verify: no alert generated
# Check suppression count increased
```

**Acceptance Criteria:**
- Alert marked as FP
- Suppression rule created
- Same traffic pattern blocked
- Count shows suppressed alerts

---

## LAYER 4: FEEDBACK & LEARNING (Week 3)

### Task 4.1: Threat Intelligence Integration

- [ ] **CREATE** `src/integrations/threat_intelligence.py`
  ```python
  class MockThreatIntelligence:
      def __init__(self):
          # Load static bad IP list
          self.bad_ips = {
              "203.0.113.100": {"score": 95, "feeds": ["AbuseIPDB", "OTX"]},
              # ... more
          }
      
      def check_ip(self, ip: str) -> dict:
          if ip in self.bad_ips:
              return {
                  "is_malicious": True,
                  "abuse_score": self.bad_ips[ip]["score"],
                  "sources": self.bad_ips[ip]["feeds"],
                  "risk_boost": 15
              }
          return {"is_malicious": False}
  ```

- [ ] **MODIFY** `web_app/app.py`
  - Add route: `GET /api/threat-intel/check/<ip>`
  - Return TI enrichment

- [ ] **MODIFY** `src/detection/service.py`
  - Call TI check in detection pipeline
  - Add boost to risk score if IP is known bad

- [ ] **CREATE** `web_app/templates/modules/threat_intel.html`
  - Show TI badge: "⚠️ KNOWN MALICIOUS"
  - Display abuse score
  - List threat feeds that flagged it

- [ ] **CREATE** `web_app/static/js/modules/threat_intel.js`

**Manual Test:**
```bash
# Ingest traffic from known-bad IP (e.g., 203.0.113.100)
# View alert
# TI badge shows with score + feeds
# Risk score is boosted
```

**Acceptance Criteria:**
- TI badge visible for known-bad IPs
- Abuse score displayed
- Risk boost applied

---

### Task 4.2: Anomaly Engine UI

- [ ] **MODIFY** `src/detection/engines/anomaly_engine.py`
  - Add status methods:
    - `get_training_progress() -> float` (0-1)
    - `is_enabled() -> bool`
    - `set_enabled(bool)`
    - `get_recent_anomalies() -> list`

- [ ] **CREATE** `web_app/templates/modules/anomaly_learning.html`
  - Status: Disabled | Learning | Enabled
  - Progress bar
  - Toggle button
  - Recent anomalies list

- [ ] **CREATE** `web_app/static/js/modules/anomaly_learning.js`
  - Fetch status: `GET /api/anomaly/status`
  - Toggle: `PATCH /api/anomaly/toggle`
  - Show progress bar filling as data collected
  - List recent anomalies

**Manual Test:**
```bash
# System starts with anomaly engine disabled
# Ingest normal traffic (50+ packets)
# Progress bar fills
# Engine auto-enables
# Ingest anomalous traffic
# Anomaly score shows as high
```

**Acceptance Criteria:**
- Progress bar fills correctly
- Engine enables at threshold
- Anomalies detected and listed

---

### Task 4.3: Escalation State Machine Visualizer

- [ ] **CREATE** `web_app/templates/modules/escalation_viz.html`
  - State machine diagram: DEFAULT → LOW → MEDIUM → HIGH → MAX
  - Per-IP escalation table
  - Timeline of state transitions for selected IP

- [ ] **CREATE** `web_app/static/js/modules/escalation_viz.js`
  - Fetch escalation summary: `GET /api/escalation/summary`
  - Draw state diagram
  - Populate per-IP table
  - Fetch per-IP timeline: `GET /api/escalation/history/<ip>`
  - Animate state transitions

**Manual Test:**
```bash
# Trigger multiple attacks from same IP
# Watch escalation level increase
# View per-IP table showing levels
# Click IP to see timeline
```

**Acceptance Criteria:**
- State diagram visible
- Per-IP table accurate
- Timeline shows progression

---

## LAYER 5: OPERATIONS & ANALYTICS (Week 4)

### Task 5.1: Analytics Dashboard

- [ ] **CREATE** `web_app/templates/modules/analytics_dashboard.html`
  - 5 charts: attacks/min, engine triggers, risk distribution, severity breakdown, blocks over time
  - Time range selector
  - Export buttons

- [ ] **CREATE** `web_app/static/js/modules/analytics_dashboard.js`
  - Initialize Chart.js instances
  - Fetch metrics: `GET /api/metrics`
  - Parse Prometheus data
  - Populate charts
  - Enable time range filtering

- [ ] **MODIFY** `src/observability/metrics.py`
  - Ensure all relevant counters exported
  - Add histograms for latency

**Manual Test:**
```bash
# Ingest 100 test events
# Open Analytics Dashboard
# Verify all charts populated with data
# Change time range → data updates
```

**Acceptance Criteria:**
- All 5 charts display
- Data accurate
- Time range selector works

---

### Task 5.2: Pipeline Monitor

- [ ] **CREATE** `web_app/templates/modules/pipeline_monitor.html`
  - Pipeline diagram (Ingestion → Queue → Detection → Policy → Action)
  - Throughput gauges at each stage
  - Queue depth bar
  - Latency percentiles (p50, p95, p99)
  - Health status per component

- [ ] **CREATE** `web_app/static/js/modules/pipeline_monitor.js`
  - Fetch pipeline metrics: `GET /api/pipeline/throughput`
  - Fetch latencies: `GET /api/pipeline/latency-percentiles`
  - Fetch queue: `GET /api/pipeline/queue-depth`
  - Animate gauge updates every 1 second

**Manual Test:**
```bash
# Ingest traffic at scale
# Open Pipeline Monitor
# Verify throughput, queue, latency displayed
# No bottlenecks shown
```

**Acceptance Criteria:**
- Pipeline metrics displayed
- Latency percentiles accurate
- Health status correct

---

### Task 5.3: Alert Lifecycle Kanban Board

- [ ] **CREATE** `web_app/templates/modules/alert_lifecycle.html`
  - 3-column Kanban: NEW | INVESTIGATING | CLOSED
  - Alert cards in each column
  - Drag-and-drop between columns
  - Detail modal on click

- [ ] **CREATE** `web_app/static/js/modules/alert_lifecycle.js`
  - Fetch alerts: `GET /api/alerts?status=new` (and investigating, closed)
  - Implement drag-drop (vanilla JS or HTMX)
  - Update alert status on drop: `PATCH /api/alerts/<id>/status`
  - Show metrics: avg time to close, FP rate, total closed

- [ ] **MODIFY** `web_app/app.py`
  - Add route: `PATCH /api/alerts/<id>/status`

**Manual Test:**
```bash
# Create sample alerts
# Open Kanban board
# Drag alert from NEW to INVESTIGATING
# Add comment (optional)
# Drag to CLOSED
# Verify status changed in DB
```

**Acceptance Criteria:**
- Kanban columns visible
- Drag-drop works smoothly
- Status updates persisted
- Metrics calculated correctly

---

## LAYER 6: ADVANCED FEATURES (Week 5)

### Task 6.1: Policy Tuning Simulator

- [ ] **CREATE** `web_app/templates/modules/policy_tuning.html`
  - Sliders: Risk threshold, Confidence min, Frequency weight
  - Real-time preview showing alert verdicts
  - Coverage % metric
  - Save/Revert buttons

- [ ] **CREATE** `web_app/static/js/modules/policy_tuning.js`
  - Slider event listeners (oninput)
  - For each change:
    - Simulate policy on recent alerts
    - Show verdict change
    - Calculate coverage %
  - Save button: `PATCH /api/policy` with new thresholds
  - Revert button: `PUT /api/policy` to restore

**Manual Test:**
```bash
# Adjust thresholds
# Watch alert verdicts change in real-time
# Save new policy
# Trigger new events
# Verify new policy applied
```

**Acceptance Criteria:**
- Sliders work smoothly
- Real-time preview accurate
- Coverage % calculated
- Policy persisted

---

### Task 6.2: Engine Toggle Playground

- [ ] **CREATE** `web_app/templates/modules/engine_playground.html`
  - Checkboxes for each engine
  - Real-time verdict display with each toggle
  - Coverage metric: "X% of attacks caught"

- [ ] **CREATE** `web_app/static/js/modules/engine_playground.js`
  - Checkbox event listeners
  - On toggle:
    - Recalculate consensus without disabled engines
    - Show verdict change
    - Update coverage %

**Manual Test:**
```bash
# All engines enabled: 95% coverage
# Disable ML: 92% coverage
# Disable Anomaly: 88% coverage
# Show dramatic drop in coverage
```

**Acceptance Criteria:**
- Engine toggles work
- Verdict updates correctly
- Coverage % accurate

---

### Task 6.3: Behavioral Pattern Detector

- [ ] **CREATE** `src/integrations/pattern_detector.py`
  ```python
  class PatternDetector:
      def detect_port_scan(self, flows) -> bool:
          # Count unique destination ports from same source
          pass
      
      def detect_ddos_pattern(self, flows) -> bool:
          # Count unique sources targeting same destination
          pass
  ```

- [ ] **CREATE** `web_app/templates/modules/pattern_detector.html`
  - SVG/Canvas for force-directed graph
  - Nodes: IPs, Edges: flows, Size/Color: risk level
  - Pattern badges: ⚠️ Port Scanning, ⚠️ DDoS, etc.
  - Hover tooltip with details
  - Timeline scrubber to replay pattern formation

- [ ] **CREATE** `web_app/static/js/modules/pattern_detector.js`
  - Import D3.js or Cytoscape.js
  - Fetch flow data: `GET /api/threats/patterns`
  - Build graph JSON
  - Render force-directed layout
  - Detect patterns from graph structure
  - Implement timeline scrubber (replay pattern formation)

**Manual Test:**
```bash
# Ingest port scan (one IP hitting 50 ports)
# View graph: one large node connected to many targets
# Pattern badge: ⚠️ PORT SCANNING
# Ingest DDoS (50 IPs hitting same target)
# View graph: one large node receiving from many sources
# Pattern badge: ⚠️ DDoS
```

**Acceptance Criteria:**
- Graph renders correctly
- Patterns detected and badged
- Timeline scrubber works
- Interactive tooltips

---

## POLISH PASS (Days 24-30)

### Task P1: Animations & Transitions
- [ ] Smooth fade-in for all panels
- [ ] Slide transitions between modules
- [ ] Chart animations on data update
- [ ] Toast notifications for actions (SAVED, BLOCKED, etc.)

### Task P2: Responsive Design
- [ ] Test on mobile (375px width)
- [ ] Test on tablet (768px width)
- [ ] Test on desktop (1920px width)
- [ ] Adjust layouts for smaller screens

### Task P3: Error Handling
- [ ] All API calls wrapped in try-catch
- [ ] User-friendly error messages
- [ ] Fallback UI when API unavailable
- [ ] Connection error handling for WebSocket

### Task P4: Demo Data & Scripts
- [ ] Create `data/demo/attack_sample.pcap` (5MB mix of attacks)
- [ ] Create `scripts/run_demo.sh` (replay PCAP, trigger scenarios)
- [ ] Create `docs/demo_script.md` (narration for 20-minute demo)
- [ ] Create `docs/15_modules_explained.md` (1-minute summary each)

### Task P5: Documentation
- [ ] API documentation (Swagger/OpenAPI comments)
- [ ] Module developer guide
- [ ] Architecture diagrams (ASCII and Mermaid)
- [ ] Troubleshooting guide

### Task P6: Testing & Validation
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Regression testing (all 318+ tests passing)
- [ ] Manual end-to-end test of all 15 modules
- [ ] Load testing: verify system handles 100 events/sec

---

## FILES SUMMARY TABLE

| File Path | Status | Type | Est. Lines |
|-----------|--------|------|-----------|
| web_app/templates/dashboard_main.html | CREATE | HTML | 150 |
| web_app/static/css/dashboard.css | CREATE | CSS | 300 |
| web_app/static/js/dashboard.js | CREATE | JS | 200 |
| web_app/static/js/websocket_client.js | CREATE | JS | 100 |
| web_app/templates/modules/base_module.html | CREATE | HTML | 50 |
| src/detection/engine_registry.py | CREATE | Python | 150 |
| src/detection/voting_ensemble.py | CREATE | Python | 100 |
| src/detection/service.py | MODIFY | Python | +100 |
| web_app/app.py | MODIFY | Python | +200 |
| web_app/templates/modules/real_time_detection.html | CREATE | HTML | 50 |
| web_app/static/js/modules/real_time_detection.js | CREATE | JS | 200 |
| web_app/static/css/modules/real_time_detection.css | CREATE | CSS | 150 |
| web_app/templates/modules/multi_engine.html | CREATE | HTML | 60 |
| web_app/static/js/modules/multi_engine.js | CREATE | JS | 150 |
| web_app/templates/modules/risk_score_viz.html | CREATE | HTML | 60 |
| web_app/static/js/modules/risk_gauge.js | CREATE | JS | 200 |
| web_app/templates/modules/auto_blocking.html | CREATE | HTML | 80 |
| web_app/static/js/modules/auto_blocking.js | CREATE | JS | 150 |
| web_app/templates/modules/approval_workflow.html | CREATE | HTML | 70 |
| web_app/static/js/modules/approval_workflow.js | CREATE | JS | 150 |
| web_app/templates/modules/false_positive.html | CREATE | HTML | 70 |
| web_app/static/js/modules/false_positive.js | CREATE | JS | 150 |
| src/integrations/threat_intelligence.py | CREATE | Python | 100 |
| web_app/templates/modules/threat_intel.html | CREATE | HTML | 60 |
| web_app/static/js/modules/threat_intel.js | CREATE | JS | 100 |
| src/detection/engines/anomaly_engine.py | MODIFY | Python | +50 |
| web_app/templates/modules/anomaly_learning.html | CREATE | HTML | 70 |
| web_app/static/js/modules/anomaly_learning.js | CREATE | JS | 150 |
| web_app/templates/modules/escalation_viz.html | CREATE | HTML | 80 |
| web_app/static/js/modules/escalation_viz.js | CREATE | JS | 200 |
| web_app/templates/modules/analytics_dashboard.html | CREATE | HTML | 70 |
| web_app/static/js/modules/analytics_dashboard.js | CREATE | JS | 250 |
| web_app/templates/modules/pipeline_monitor.html | CREATE | HTML | 80 |
| web_app/static/js/modules/pipeline_monitor.js | CREATE | JS | 200 |
| web_app/templates/modules/alert_lifecycle.html | CREATE | HTML | 100 |
| web_app/static/js/modules/alert_lifecycle.js | CREATE | JS | 250 |
| web_app/templates/modules/policy_tuning.html | CREATE | HTML | 90 |
| web_app/static/js/modules/policy_tuning.js | CREATE | JS | 200 |
| web_app/templates/modules/engine_playground.html | CREATE | HTML | 60 |
| web_app/static/js/modules/engine_playground.js | CREATE | JS | 150 |
| src/integrations/pattern_detector.py | CREATE | Python | 100 |
| web_app/templates/modules/pattern_detector.html | CREATE | HTML | 80 |
| web_app/static/js/modules/pattern_detector.js | CREATE | JS | 300 |
| | TOTAL | | ~6,500 lines |

---

## Testing Checklist

### Unit Tests (Per Module)
```bash
pytest tests/unit/ -v
```
- [ ] Engine registry + voting
- [ ] Risk scoring breakdown
- [ ] Escalation logic
- [ ] Pattern detection
- [ ] TI enrichment

### Integration Tests (APIs)
```bash
pytest tests/integration/ -v
```
- [ ] Multi-engine endpoint
- [ ] Risk breakdown endpoint
- [ ] TI enrichment API
- [ ] Action timeline API
- [ ] Alert lifecycle status updates

### End-to-End Tests
- [ ] Module 1 (Real-Time): WebSocket stream 10 events
- [ ] Module 2 (Multi-Engine): Voting engine consensus
- [ ] Module 4 (Blocking): Attack → block → verify firewall
- [ ] Module 6 (FP): Mark FP → suppress → verify suppression
- [ ] Full flow: Ingest → Detect → Decide → Act → Observe

### Manual Testing
- [ ] All 15 modules load without errors
- [ ] No console JS errors
- [ ] Dashboard responsive on mobile/tablet/desktop
- [ ] WebSocket reconnects on disconnect
- [ ] Error states handled gracefully

### Regression Testing
```bash
pytest tests/ -v --tb=short
# MUST: 318+ tests pass, 0 failures
```

---

## Success Metrics (End of Week 5)

- ✅ All 15 modules deployed and functional
- ✅ 318+ tests passing (no regressions)
- ✅ Dashboard feels polished and professional
- ✅ WebSocket real-time performance < 100ms latency
- ✅ Each module explainable in 60 seconds
- ✅ Animations smooth (60 FPS on modern browser)
- ✅ Colors consistent with dark security ops theme
- ✅ Demo script written and rehearsed (5-20 min duration)
- ✅ System handles 100+ events/sec without lag
- ✅ Full end-to-end demo runs from dashboard → detection → blocking → learning

