# INIDS Demo Platform Implementation Roadmap
## Quick Start Guide for Transformation

**Created**: 2026-03-17 | **Status**: Ready for Phase 1 Execution
**Estimated Total Effort**: 4-5 weeks (one developer), modular (can parallelize)

---

## PHASE 1: QUICK WINS (2-3 days)
### Objective: Prove all pieces are in place

Nothing to code here—just wire what exists.

#### Task 1.1: Create New Dashboard Structure
**Files to create:**
- `web_app/templates/dashboard_main.html` — Landing page with 15-card grid
- `web_app/static/css/dashboard.css` — Dark theme + card styles
- `web_app/static/js/dashboard.js` — Card click handlers

**What this does:**
- Main dashboard shows all 15 capabilities as clickable cards
- "Demo Mode" toggle at top (trigger sample attacks)
- System status bar (health, threat summary)

**Time**: 3-4 hours

#### Task 1.2: Create Module Template System
**Files to create:**
- `web_app/templates/modules/base_module.html` — Generic panel template
- `web_app/templates/modules/README.md` — Instructions for module developers

**What this does:**
- Every module follows same slide-in panel UX
- Consistent headers, info bars, action buttons
- Reduces duplication

**Time**: 1 hour

#### Task 1.3: Activate Existing Endpoints
**Endpoints already working:**
- ✅ `POST /api/predict` → detect attacks
- ✅ `GET /api/alerts` → retrieve alerts  
- ✅ `/api/policy` → get/set policy
- ✅ `/api/metrics` → get prometheus metrics
- ✅ `/api/escalation/summary` → escalation counts
- ✅ `/api/actions` → list actions

**Frontend to create:**
- Simple route handlers in Flask
- Wire to new dashboard routes

**Time**: 30 min

#### Task 1.4: Test All Existing APIs
**Command:** `python -m pytest tests/ -v`

**Goal:** Verify all 318 tests pass (establish baseline)

**Time**: 15 min run time

---

## PHASE 2: CORE DETECTION MODULES (Week 1)
### Objective: Get 3 demo-ready modules working

### Module 1: Real-Time Detection Panel

**Files to create:**
- `web_app/templates/modules/real_time_detection.html`
- `web_app/static/js/real_time_detection.js`
- `web_app/app.py` — Add WebSocket endpoint `/ws/detections`

**Backend tasks:**
1. Add WebSocket server to Flask app (use `flask-sock`)
2. Connect event bus to WebSocket client (publish detection events)
3. Serialize `DetectionEvent` to JSON

**Frontend tasks:**
1. Build event feed component (list of cards)
2. WebSocket client connects on page load
3. New events slide in from bottom with animation
4. Color-code: RED=Attack, ORANGE=Suspicious, GREEN=Normal
5. Show confidence % and risk score

**Demo flow:**
```
1. Open dashboard → click "Real-Time Detection"
2. Panel opens, WebSocket connects (show green indicator)
3. Run: python -m src.realtime_simulation (triggers sample attacks)
4. Watch events appear in real-time, ranked red→orange→green
5. Click event → detail modal with full flow breakdown
```

**API needed:**
- ✅ Partially: Use existing event bus, need WebSocket wrapper

**Testing:**
- Unit: WebSocket serialization
- Integration: Trigger 10 sample detections, verify all appear in feed
- UI: Manual - verify animations smooth, colors correct

**Time**: 5-6 hours

---

### Module 2: Multi-Engine Detection Comparison

**Files to create:**
- `src/detection/engine_registry.py` — Registry pattern for engines
- `src/detection/voting_ensemble.py` — Voting logic
- `web_app/templates/modules/multi_engine.html`
- `web_app/app.py` — Add `POST /api/predict/multi-engine`

**Backend tasks:**
1. Create `EngineRegistry` class
   - Register ML models: Random Forest, SVM, Decision Tree, Naive Bayes, Logistic Regression
   - Register rules engine (threshold-based)
   - Register anomaly engine
   - Register statistical threshold engine
2. Implement voting logic
   - Majority vote for NORMAL/SUSPICIOUS/ATTACK
   - Weighted average confidence
   - Return per-engine verdict + consensus
3. Endpoint `/api/predict/multi-engine` (POST):
   - Input: feature vector
   - Output: `{engines: [{name, verdict, confidence}], consensus: {verdict, confidence}}`

**Frontend tasks:**
1. Table layout: Engine | Verdict | Confidence | Badge
2. Consensus row (highlighted)
3. Toggle to show explanation: "Why did X engine vote this way?"
4. Example: "Random Forest: 92% confident → triggered by feature X"
5. Show coverage: "95% of attacks caught with all 5 engines"

**Demo flow:**
```
1. Open dashboard → click "Multi-Engine Comparison"
2. Load sample attack flow
3. Table shows:
   ├─ RF: ATTACK (92%)
   ├─ SVM: ALERT (68%)
   ├─ Statistical: ALERT (71%)
   ├─ Anomaly: NORMAL (25%)
   └─ CONSENSUS: ATTACK (80%)
4. Hover engine → explanation popup
5. Click "Disable Anomaly Engine" → consensus recalculates
```

**Testing:**
- Unit: Voting logic with edge cases
- Integration: All 5 engines return consistent predictions
- UI: Manual - verify table layout, toggle, coverage %

**Time**: 6-8 hours

---

### Module 3: Risk Score Visualization

**Files to create:**
- `web_app/templates/modules/risk_score_viz.html`
- `web_app/static/js/risk_gauge.js` (Canvas gauge animation)
- `web_app/app.py` — Add `POST /api/predict/score-breakdown`

**Backend tasks:**
1. Endpoint `/api/predict/score-breakdown` (POST):
   - Input: detection event + flow features
   - Output: `{total_score, base: X, confidence_factor: Y%, severity_factor: Z, frequency_factor: W, thresholds: {alert: 40, block: 70}}`
   - Expose RiskEngine calculation steps

**Frontend tasks:**
1. Gauge component (Canvas or SVG)
   - 0-100 scale
   - Color gradient: green (0-40), yellow (40-70), red (70-100)
   - Animated fill from 0 to score (1-2 second animation)
   - Display current score in center
2. Factor breakdown (percentage pies or stacked bar)
   - Confidence contribution: 45%
   - Severity contribution: 30%
   - Frequency contribution: 25%
3. Threshold markers
   - Horizontal line at 40: "Alert threshold"
   - Horizontal line at 70: "Block threshold"
4. Risk history (mini sparkline)
   - Show risk score over last 10 similar flows

**Demo flow:**
```
1. Open dashboard → click "Risk Score Visualization"
2. Enter a sample attack flow
3. Gauge animates from 0 to 78 over 2 seconds
4. Factor breakdown shows contribution
5. Gauge fills past "Alert" threshold (40) → color changes to orange
6. Gauge reaches "Block" threshold (70) → color changes to red
7. Student adjusts confidence slider → gauge recalculates in real-time
```

**Testing:**
- Unit: Risk scoring calculation matches RiskEngine
- Integration: Known flows get correct scores
- UI: Manual - verify gauge animation, formatting

**Time**: 4-5 hours

---

## WEEK-BY-WEEK BREAKDOWN

### Week 1: Foundations + Modules 1-3
- **Mon-Tue**: Task 1.1-1.4 (Quick wins)
- **Wed-Thu**: Module 1 (Real-Time Detection)
- **Fri**: Module 2 (Multi-Engine) - first half
- **Sat-Sun**: Module 2 - completion + Module 3

**By end of week**: 3 modules live, impressive detection workflow visible

### Week 2: IPS Modules 4-6
- Modules 4-6: Blocking, Approval, FP Learning
- Focus: Show IPS behavior (not just detection)
- Time: 4-6 hours each

### Week 3: Intelligence & Adaptation
- Modules 7-10: TI, Anomaly, Escalation
- Add mock threat feed
- Activate anomaly engine UI

### Week 4: Operations
- Modules 9, 11, 13: Analytics, Pipeline, Kanban
- Build chart components
- Pipeline metrics from existing system

### Week 5: Polish & Advanced
- Modules 12, 14, 15: Tuning, Playground, Patterns
- Final animations
- Demo script preparation

---

## IMMEDIATE ACTION ITEMS (Today)

### 1. Create Issue Tracker
```markdown
# Demo Platform Modules
- [ ] 1. Real-Time Detection Panel (2-3 days)
- [ ] 2. Multi-Engine Comparison (2-3 days)
- [ ] 3. Risk Visualization (1-2 days)
- [ ] 4. Auto-Blocking Timeline (1-2 days)
- [ ] 5. Approval Workflow (1 day)
- [ ] 6. FP Learning (1 day)
- [ ] 7. Threat Intelligence (2-3 days)
- [ ] 8. Anomaly Learning (1-2 days)
- [ ] 9. Analytics Dashboard (2-3 days)
- [ ] 10. Escalation Visualizer (1-2 days)
- [ ] 11. Pipeline Monitor (2-3 days)
- [ ] 12. Policy Tuning (2-3 days)
- [ ] 13. Alert Lifecycle Kanban (2-3 days)
- [ ] 14. Engine Playground (1-2 days)
- [ ] 15. Pattern Detector (2-3 days)
```

### 2. Set Up Dashboard Route
Create minimal HTML structure, verify page loads.

### 3. Choose Tech Stack for Visualizations
- **Real-time feed**: Vanilla JS (simple)
- **Charts**: Chart.js (simple) or Plotly.js (advanced)
- **Gauge**: Canvas (performant)
- **Force graph**: D3.js or Cytoscape.js (powerful)
- **Kanban**: Vanilla JS drag-drop or HTMX + CSS Grid

### 4. Set Up Demo Data
- 5 small attack PCAPsin `data/demo/`
- Script to replay them on demand
- Mix of clear attacks vs ambiguous detections

### 5. Begin Module 1 Implementation
- Wire WebSocket to event bus
- Build event feed component
- Test with sample data

---

## SUCCESS CRITERIA

### Week 5 End (Full Platform)
- ✅ All 15 modules deployed and functional
- ✅ 318+ tests passing (no regressions)
- ✅ Dashboard feels cohesive and professional
- ✅ Each module explainable in 60 seconds
- ✅ Animations smooth, colors consistent
- ✅ Demo script written and rehearsed
- ✅ Live system can handle 100+ events without lag

---

## Resource Links & Dependencies

**Python packages to install:**
```bash
pip install flask-sock  # WebSocket support
pip install plotly      # Advanced charts (opt)
pip install pydantic    # Data validation (opt)
```

**JavaScript libraries (via CDN in templates):**
```html
<!-- Charts -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<!-- Force-directed graph -->
<script src="https://d3js.org/d3.v7.min.js"></script>

<!-- Animations -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
```

---

## Questions to Resolve Before Starting

1. **Authentication for modules?**
   - Answer: Use existing API key auth, no changes needed for demo

2. **Real-time data generation?**
   - Answer: Use `src/realtime_simulation.py` to generate sample events
   - Or upload small PCAP files and replay them

3. **Multi-process WebSocket?**
   - Answer: Use single Flask process for demo (simplifies WebSocket)
   - Production would use proper message queue

4. **Database for persistent state?**
   - Answer: Use existing SQLite (`ops_store.db`)
   - No schema changes needed

---

## Git Strategy

```bash
# Feature branches per module
git checkout -b feat/module-1-realtime-detection
git checkout -b feat/module-2-multi-engine
# ... etc

# Merge to main only when module complete + tests pass
git pull origin main
git merge --no-ff feat/module-X
git push origin main
```

---

## Presentation Order (Top to Bottom)

**If you have 10 minutes:**
1. Dashboard overview (30 sec)
2. Real-Time Detection (60 sec)
3. Multi-Engine Voting (60 sec)
4. Risk Visualization (30 sec)
5. Auto-Blocking (60 sec)
6. Threat Intel (30 sec)
7. Analytics (30 sec)
8. Wrap-up (30 sec)

**If you have 20 minutes:** Add 6-8 more modules in demo order

**If you have 30+ minutes:** Show all 15 modules with deep explanations

---

## Checkpoint Validation

### After Week 1
```bash
# Run tests
python -m pytest tests/ -v --tb=short
# Should show: 318 passed

# Manual test
python -m web_app.app
# Open http://localhost:5000
# Verify: Dashboard loads, 3 modules clickable, no errors in console
```

### After Week 2
```bash
# Verify IPS demo
# Set Policy to AUTO_BLOCK, trigger attack, verify firewall rule added
```

### After Week 5
```bash
# Full platform demo
# Run through all 15 modules, verify smooth transitions, correct data
```

