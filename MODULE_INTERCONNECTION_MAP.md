# INIDS Demo Platform — Module Interconnection Map

## System Architecture (Simplified for Demo)

```
                       ┌─────────────────────────────────┐
                       │   LIVE DATA INGESTION           │
                       │  (PCAP Replay / Live Traffic)   │
                       └──────────────┬────────────────┏┛
                                      │                ┃
                    ╔═════════════════╩════════════════╝
                    ║ FEATURE ENGINEERING & VALIDATION
                    ║ (Normalize flow fields)
                    ╚═════════════════╦═════════════════╗
                                      ┃                 ┃
         ┌────────────────────────────╋─────────────────╋───────────────────────┐
         │                            │                 │                       │
    ┌────▼────┐                 ┌─────▼─────┐    ┌─────▼─────┐           ┌─────▼─────┐
    │   ML    │                 │ Statistical│   │  Anomaly  │           │   Rules   │
    │ Engines │                 │  Rules     │    │  Engine   │           │  Engine   │
    │(5 models)│               └─────┬─────┘    └─────┬─────┘           └─────┬─────┘
    └────┬────┘                      │               │                        │
         │                           │               │                        │
         └───────────────────────────┼───────────────┼────────────────────────┘
                                     │               │
                        ┌────────────┴───────────────┴────────────┐
                        │                                         │
                   ┌────▼──────────────────────────────────────┐  │
                   │ MODULE 2: MULTI-ENGINE VOTING           │  │
                   │ ┌──────────────────────────────────────┐ │  │
                   │ │ Vote consolidation: majority + weight │ │  │
                   │ │ Output: verdict (NORMAL/ALERT/ATTACK) │ │  │
                   │ └──────────────────────────────────────┘ │  │
                   └────┬──────────────────────────────────────┘  │
                        │                                         │
                   ┌────▼───────────────────────────────────┐    │
                   │ MODULE 1: REAL-TIME DETECTION PANEL   │◀───┘
                   │ ┌────────────────────────────────────┐ │
                   │ │ Live event feed (WebSocket stream)  │ │
                   │ │ Badges: NORMAL/SUSPICIOUS/ATTACK    │ │
                   │ └────────────────────────────────────┘ │
                   └────┬────────────────────────────────────┘
                        │
                   ┌────▼────────────────────────────────────────┐
                   │ MODULE 3: RISK SCORE VISUALIZATION          │
                   │ ┌──────────────────────────────────────────┐│
                   │ │ Animated gauge: confidence/severity/freq  ││
                   │ │ Thresholds: Alert@40, Block@70           ││
                   │ └──────────────────────────────────────────┘│
                   └────┬─────────────────────────────────────────┘
                        │
                   ┌────▼──────────────────────────────────────┐
                   │ MODULE 15: THREAT INTELLIGENCE ENRICHMENT │
                   │ ┌───────────────────────────────────────┐ │
                   │ │ External TI lookup: abuse score      │ │
                   │ │ Badge: "Known Malicious IP"         │ │
                   │ │ Boost: Risk +15 if flagged          │ │
                   │ └───────────────────────────────────────┘ │
                   └────┬──────────────────────────────────────┘
                        │
      ┌─────────────────┴─────────────────┐
      │                                   │
 ┌────▼──────┐             ┌──────────────▼────┐
 │  RISK >= │             │   RISK <          │
 │   Block  │             │   Block           │
 │ Threshold│             │  Threshold        │
 └────┬──────┘             └────┬──────────────┘
      │                         │
      │                    ┌────▼────────────────────────┐
      │                    │ STORE AS ALERT               │
      │                    │ (OPS_STORE.ALERTS)          │
      │                    └────┬──────────────────────────┘
      │                         │
      │ ┌────────────────────────┤
      │ │                        │
 ┌────▼─────────────────────────▼──────────┐
 │ MODULE 4 & POLICY DECISION              │
 │ ┌──────────────────────────────────────┐│
 │ │ Policy Engine evaluates risk         ││
 │ │ Decisions: ALLOW/ALERT/RATE_LIMIT/   ││
 │ │           TEMP_BLOCK/PENDING_BLOCK/  ││
 │ │           BLOCK                      ││
 │ └──────────────────────────────────────┘│
 └────┬───────────────────────────┬────────┘
      │                           │
 ┌────▼──────────────┐    ┌──────▼─────────────────┐
 │ PENDING_BLOCK:    │    │  AUTO_BLOCK/           │
 │ Wait for approval │    │  RATE_LIMIT:           │
 │                   │    │  Execute immediately   │
 │ MODULE 5:         │    │                        │
 │ APPROVAL WORKFLOW │    │  MODULE 4:             │
 │ ┌───────────────┐ │    │  AUTO-BLOCKING DEMO    │
 │ │ Action queue  │ │    │ ┌────────────────────┐ │
 │ │ APPROVE/REJECT│ │    │ │ Timeline showing:  │ │
 │ │ buttons       │ │    │ │ - Detection (50ms) │ │
 │ │ Audit trail   │ │    │ │ - Risk scoring     │ │
 │ │               │ │    │ │ - Policy decision  │ │
 │ └───────────────┘ │    │ │ - Firewall update  │ │
 └────┬──────────────┘    │ │ - IP BLOCKED       │ │
      │                   │ └────────────────────┘ │
      └────┬──────────────┴──────────────┬─────────┘
           │                            │
      ┌────▼──────────────────────────┬─┘
      │      EXECUTE ACTION            │
      │ ┌─────────────────────────────┐│
      │ │ ActionExecutor.block_ip()   ││
      │ │ Add firewall rule (nftables)││
      │ │ TTL set for temp blocks     ││
      │ └─────────────────────────────┘│
      └────┬──────────────────────────┘
           │
      ┌────▼─────────────────────────┐
      │ ESCALATION TRACKING           │
      │ ┌───────────────────────────┐ │
      │ │ Per-IP escalation level   │ │
      │ │ Track frequency           │ │
      │ │ Increase response severity│ │
      │ │                           │ │
      │ │ MODULE 10:                │ │
      │ │ ESCALATION VISUALIZER     │ │
      │ │ ┌─────────────────────────┐│
      │ │ │ State machine diagram   ││
      │ │ │ DEFAULT→LOW→MED→HIGH→MAX││
      │ │ │ Per-IP tracking table   ││
      │ │ │ Timeline of transitions ││
      │ │ └─────────────────────────┘│
      │ └───────────────────────────┘ │
      └─────────────────┬──────────────┘
                        │
              ┌─────────▼────────────┐
              │ STORE DECISION       │
              │ (OPS_STORE.ACTIONS)  │
              │ (OPS_STORE.AUDITS)   │
              └─────────┬────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼────┐    ┌───▼────┐    ┌───▼──────┐
    │ Analytics│    │ Lifecycle│   │ Feedback │
    │Metrics   │    │Board    │   │Learning  │
    └────┬────┘    └───┬────┘    └───┬──────┘
         │              │             │
    ┌────▼───────────────┼─────────────▼─────┐
    │                    │                   │
┌───▼──────┐      ┌──────▼───────┐    ┌─────▼──────┐
│ MODULE 9:│      │ MODULE 13:   │    │ MODULE 6:  │
│ Analytics│      │ Alert        │    │ FP Learning│
│Dashboard │      │ Lifecycle    │    │┌──────────┐│
│┌────────┐│      │┌───────────┐ │    ││Suppress  ││
││Attacks │││      ││Kanban     │ │    ││pattern  ││
││per min ││      ││NEW/INVSTG/││    ││Add to   ││
││Engine  ││      ││CLOSED     │ │    ││rules   ││
││triggers││      ││Timeline of││    ││Match   ││
││Risk    ││      ││closes     │ │    ││future  ││
││distrib ││      │└───────────┘ │    ││alerts  ││
││Severity││      │Resolution    │    ││No      ││
││pie     ││      │time metrics  │    ││action  ││
││chart   ││      └───────────────┘    │└──────────┘│
│└────────┘│                          └────────────┘
│Time      │                               │
│range     │                          ┌────▼──────────┐
│selector  │                          │ SUPPRESSION   │
│Export    │                          │ AUDIT LOG     │
│          │                          │ FP % metric   │
│          │                          └──────┬───────┘
│          │                                 │
│ MODULE 11│                          ┌──────▼──────────┐
│ Pipeline │                          │ Pattern visible│
│ Monitor  │                          │ in dashboards │
│ ┌───────┐│                          │ User can see  │
│ │Through├┼─ Ingestion rate         │ suppression  │
│ │put    ││ Processing rate         │ working      │
│ │Gauges ├┼─ Queue depth            │              │
│ │Latency├┼─ p50/p95/p99            │ IMPROVEMENT  │
│ │pctiles││ Worker health           │ LEARNING LOOP│
│ │        ││ Stream lag              │              │
│ │Feed    ├┼─ Bottleneck indicator  └──────────────┘
│ │back    ││                              ▲
│ │ops    ││                              │
│ │team   ││                              └──── Feedback cycle
│ │       ││                                  repeats
│ └───────┘│
│ (Current)│
└──────────┘

            ┌───────────────────────────────────┐
            │ INDEPENDENT MODULE DEMONSTRATIONS  │
            │ (Can show without full flow)       │
            └───────────────────────────────────┘
                        │
         ┌──────────────┼────────────────────┬──────────────┐
         │              │                    │              │
    ┌────▼────┐    ┌───▼────┐        ┌──────▼────┐    ┌───▼────┐
    │ MODULE 8:│    │MODULE14:│       │ MODULE 12: │    │MODULE15:│
    │Anomaly   │    │Engine   │       │ Policy     │    │ Pattern │
    │Learning  │    │Playground│      │ Tuning     │    │Detector │
    │┌────────┐│    │┌───────┐ │      │┌──────────┐│    │┌──────┐ │
    ││Train   ││    ││Engine ││       ││Risk      │││    ││Port  │ │
    ││baseline││    ││select ││       ││threshold││││    ││Scans ││
    ││score   ││    ││boxes  ││       ││sliders  │││    ││Graph ││
    ││anomaly││    ││Toggle││       ││What-if  │││    ││Effect││
    ││status ││    ││OFF   ││       ││preview  │││    ││Size= ││
    ││on/off ││    ││detection││       ││Coverage ││    ││Freq│
    ││toggle ││    ││changes ││       ││%        │││    ││Red= ││
    ││Progress││    ││in     ││       ││Save/    │││    ││High ││
    ││bar    ││    ││real-  ││       ││Revert   │││    ││Risk ││
    ││Recent ││    ││time   ││       │└──────────┘│    │└──────┘ │
    ││anomaly││    │└───────┘ │      │Interactive│    │Force-   │
    ││list   ││    │Coverage %│      │Sliders    │    │directed │
    │└────────┘│    │         │      │Policy     │    │graph    │
    │FEEDBACK: │    │Shows    │      │audit trail│    │Pattern  │
    │System    │    │multi-   │      │           │    │labels:  │
    │learns    │    │engine   │      │MODEL 12:  │    │⚠ Port  │
    │baseline  │    │necessity│      │Let users  │    │scanning │
    │           │    │         │      │tune       │    │⚠ DDoS   │
    └──────┬──┘    └────┬───┘       │sensitivity │    │       │
           │            │           │instantly  │    │Timeline│
           │ EDUCATIONAL
```

---

## Module Dependencies & Execution Order

```
LAYER 1: INFRASTRUCTURE (Required first)
├─ Dashboard Layout (foundations)
├─ WebSocket setup
└─ API endpoints active

LAYER 2: CORE DETECTION (Must work first)
├─ Module 1: Real-Time Detection
├─ Module 2: Multi-Engine Voting
└─ Module 3: Risk Visualization

LAYER 3: DECISION & ACTION (Prevention)
├─ Module 4: Auto-Blocking Timeline
├─ Module 5: Approval Workflow
├─ Module 10: Escalation Visualizer
└─ (All use PolicyEngine + ActionExecutor)

LAYER 4: FEEDBACK & LEARNING (Adaptation)
├─ Module 6: False Positive Learning
├─ Module 7: Threat Intelligence
├─ Module 8: Anomaly Engine Activation
└─ (All improve system over time)

LAYER 5: OPERATIONS & OBSERVABILITY (Visibility)
├─ Module 9: Analytics Dashboard
├─ Module 11: Pipeline Monitor
├─ Module 13: Alert Lifecycle Board
└─ (All show what system is doing)

LAYER 6: ADVANCED & TUNING (Control)
├─ Module 12: Policy Tuning Simulator
├─ Module 14: Engine Playground
└─ Module 15: Behavior al Pattern Detector
```

---

## Data Flow Through All Modules

### Simple Attack Scenario

```
USER ACTION: Upload attack PCAP
     ↓
  [INGEST MODULE]
  Replay packets from file
     ↓
  [FEATURE ENGINEERING]
  Extract flow: src=192.168.1.42, dst=10.0.0.1, dport=443, protocol=TCP
     ↓
  ├─────────────────────────────────────┬──────────────────────────────┐
  │ [MODULE 2] PARALLEL ENGINE SCORING  │                              │
  │ ├─ ML (RF): 92% ATTACK              │                              │
  │ ├─ ML (SVM): 68% ALERT              │                              │
  │ ├─ Statistical: 71% ALERT           │                              │
  │ ├─ Anomaly: 25% NORMAL              │                              │
  │ └─ Rules: 75% ALERT                 │                              │
  │                                     │                              │
  │ CONSENSUS: 80% ATTACK ◄─── [MODULE 2 OUTPUT]                      │
  │                          Voting aggregation                        │
  └────────────┬────────────────────────┴──────────────────────────────┘
               ↓
      [MODULE 3] RISK SCORING
      base_verdict: ATTACK
      + confidence_factor: 80% = +48 points
      + severity_weight: 12 = +12 points
      + frequency_score: (first sight) 0 = +0 points
      ─────────────────────────
      TOTAL RISK SCORE: 60 / 100
      ↓
      [MODULE 7] THREAT INTEL ENRICHMENT
      Lookup: Is 192.168.1.42 known bad?
      Result: NO (internal network)
      TI Boost: 0 (no boost)
      ↓
      FINAL RISK: 60 / 100 (below block threshold of 70)
      ↓
      [MODULE 4] POLICY DECISION
      Risk = 60 (alert threshold passed: 40)
      Policy = APPROVE_BEFORE_BLOCK
      Decision = PENDING_BLOCK (needs human approval)
      ↓
      [STORE ALERT]
      ops_store.alerts ← New alert created
      Status = PENDING_ACTION
      ↓
      [MODULE 5] APPROVAL WORKFLOW QUEUE
      Display pending action to analyst
      Analyst sees: IP 192.168.1.42 → 10.0.0.1:443
      Risk: 60/100, Confidence: 80%, 5 engines voting
      ↓
      ANALYST DECISION: "This is suspicious, APPROVE block"
      ↓
      [MODULE 4] EXECUTE ACTION
      ActionExecutor.block_ip('192.168.1.42')
      - Call nftables adapter
      - Rule added: iptables -A INPUT -s 192.168.1.42 -j DROP
      - Set TTL: 3600 seconds
      - Store in ops_store.actions
      ↓
      [MODULE 10] ESCALATION
      tracking.escalate('192.168.1.42')
      Previous state: DEFAULT
      New state: LOW (first alert)
      ↓
      [MODULE 1] LIVE FEED UPDATE
      Event card appears in real-time feed:
      ┌─────────────────────────┐
      │ 192.168.1.42 → 10.0.0.1 │
      │ BLOCKED in real-time    │
      │ Risk: 60/100, ML: 92%   │
      │ 3 min ago               │
      └─────────────────────────┘
      ↓
      [MODULE 9] ANALYTICS UPDATE
      - alerts_total += 1
      - blocks_total += 1
      - risk_score_histogram[60] += 1
      ↓
      [MODULE 11] PIPELINE METRICS
      - Events processed: +1
      - Process latency: 243ms
      - Queue depth: 0
      ↓
      [MODULE 13] LIFECYCLE
      Alert moves to INVESTIGATING column
      Analyst can now comment/decide: TRUE_POSITIVE or FALSE_POSITIVE
      ↓
      ANALYST ACTION: Mark TRUE_POSITIVE
      ↓
      [MODULE 6] NO SUPPRESSION
      (Not a false positive, so no suppression rule)
      ↓
      ALERT CLOSED
      Status: TRUE_POSITIVE
      Time to close: 2 minutes
      ↓
      [MODULE 13] LIFECYCLE STATS
      Metric updated: "Avg time to close: 23 min"
      ↓
      [DASHBOARDS UPDATED]
      All 9 modules show new data:
      ├─ Module 1: New event visible in feed ✅
      ├─ Module 9: Metrics updated ✅
      ├─ Module 11: Pipeline throughput increased ✅
      ├─ Module 13: Alert in CLOSED column ✅
      └─ All others refresh with new aggregate data ✅
```

---

## False Positive Scenario (showing Module 6)

```
DIFFERENT ATTACK: Internal scanner (192.168.1.50) scanning ports
     ↓
  [SAME DETECTION FLOW]
  Scores: 78/100, ML: 85% ATTACK, Engines agree
     ↓
  [MODULE 4] POLICY DECISION: BLOCK
     ↓
  [EXECUTED ACTION]
  IP 192.168.1.50 BLOCKED
     ↓
  [MODULE 1] Event in live feed
     ↓
  [MODULE 5] Analyst sees action
  "Wait, 192.168.1.50 is our network scanner!"
     ↓
  [MODULE 13] Alert moved to INVESTIGATING
     ↓
  [MODULE 6] FALSE POSITIVE LEARNING
  Analyst clicks: "MARK AS FALSE POSITIVE"
     ↓
  SUPPRESSION RULE CREATED:
  {
    source_ip: "192.168.1.50",
    port: "445",
    protocol: "TCP",
    pattern: "port_scan_like",
    reason: "Internal network scanner"
  }
  ↓
  Rule added to fp_suppression_rules
  ↓
  [MODULE 6] Suppression stats:
  - Active suppressions: 23
  - Alerts prevented this week: 67
  ↓
  [NEW EVENT] Same IP (192.168.1.50) hits port 22
  Detection runs → Score: 76/100, ATTACK verdict
     ↓
  SUPPRESSION CHECK:
  ✓ Matches suppression rule (192.168.1.50)
     ↓
  ACTION: NO ALERT GENERATED
  (Suppressed before alert creation)
     ↓
  [MODULE 6] Suppression count += 1
     ↓
  [MODULE 1] Event NOT in live feed
     ↓
  [MODULE 9] Analytics NOT affected
  (Suppressed events don't count toward detection rate)
     ↓
  Professor observes: "That IP would have been blocked, but the system
  remembered it's our scanner. The FP suppression prevents alert fatigue."
```

---

## Escalation Scenario (showing Module 10)

```
ATTACKER IP: 203.0.113.100 attacks 5 times in 10 minutes
     ↓
  EVENT 1: Detection and risk score
  ├─ Risk: 65/100
  ├─ Decision: ALERT
  ├─ [MODULE 10] Escalation: DEFAULT → LOW
  └─ Store alert
     ↓
  EVENT 2: Repeated source IP (5 min later)
  ├─ Risk: 72/100
  ├─ Decision: RATE_LIMIT (escalation increased policy severity)
  ├─ [MODULE 10] Escalation: LOW → MEDIUM
  ├─ Action: Rate-limit traffic from IP (500 pps max)
  └─ Store alert
     ↓
  EVENT 3: Same IP again (2 min later)
  ├─ Risk: 78/100
  ├─ Decision: TEMP_BLOCK (1 hour)
  ├─ [MODULE 10] Escalation: MEDIUM → HIGH
  ├─ Action: Temporary firewall block
  └─ Store alert
     ↓
  EVENT 4: Same IP (5 min later)
  ├─ Risk: 85/100
  ├─ Decision: BLOCK (permanent until manual intervention)
  ├─ [MODULE 10] Escalation: HIGH → MAX
  ├─ Action: Permanent firewall block
  └─ Store alert
     ↓
  [MODULE 10] VISUALIZER displays:
  State diagram:
  DEFAULT (0 alerts)
    ↓
  LOW (1 alert) ← Event 1 happened 9 min ago
    ↓
  MEDIUM (2 alerts) ← Event 2 happened 4 min ago
    ↓
  HIGH (3 alerts) ← Event 3 happened 2 min ago
    ↓
  MAX (4 alerts) ← Event 4 happened now
  
  Per-IP table shows:
  ┌─────────────────┬────────────┬────────────────────┐
  │ IP              │ Escalation │ Events in window   │
  ├─────────────────┼────────────┼────────────────────┤
  │ 203.0.113.100   │ MAX 🔴     │ 4 alerts / 10 min  │
  │ 192.168.1.42    │ MEDIUM    │ 2 alerts / 30 min  │
  │ 10.0.0.55       │ LOW        │ 1 alert / 60 min   │
  │ Others...       │ DEFAULT    │ 0 alerts           │
  └─────────────────┴────────────┴────────────────────┘
  
  Timeline for 203.0.113.100:
  ├─ t=0:   DEFAULT (first detection)
  ├─ t=+5m: LOW (second detection, escalate)
  ├─ t=+7m: MEDIUM (third detection, escalate again)
  ├─ t=+12m: HIGH (fourth detection, escalate again)
  └─ t=+17m: MAX (fifth detection, at max level, stay there)
```

---

## All 15 Modules Interaction Matrix

| Module # | Depends On | Feeds To | Data Source |
|----------|-----------|----------|-------------|
| 1. Real-Time | 2,3,10 | All | DetectionEvent stream |
| 2. Multi-Engine | Engines | 1,3,4,6,14 | Individual engine verdicts |
| 3. Risk Score | 2,7 | 4,5 | RiskEngine calculation |
| 4. Auto-Blocking | 3,10 | 1,5,9,11,13 | PolicyEngine + ActionExecutor |
| 5. Approval | 4 | 4,1,13 | ActionQueue storage |
| 6. FP Learning | 1,13 | 1,9 | User feedback → suppression rules |
| 7. TI Enrichment | Live data | 3,4 | External feed lookup |
| 8. Anomaly | Live data | 2,9,14 | AnomalyEngine fit() + predict()|
| 9. Analytics | 1,4,5,6 | Dashboard | MetricsService counters |
| 10. Escalation | 4,6 | 1,4,5,9 | EscalationTracker state |
| 11. Pipeline | 1,2,3,4,5 | None (observational) | System internals |
| 12. Policy Tuning | 3,4 | None (simulator) | PolicyEngine parameters |
| 13. Lifecycle | 1,4,6 | 9 | AlertStore + audit trail |
| 14. Engine Toggle | 2,3 | None (playground) | Engine registry |
| 15. Pattern Detect | 1,4,10 | None (visualization) | Flow database |

---

## Demo Flow Recommendations

### Conservative (3 modules, 5 minutes)
```
1. Show dashboard
2. Real-Time Detection (Module 1) + Multi-Engine (Module 2)
3. Risk Scorer (Module 3)
4. "That's the core intelligence. Now for prevention..."
5. Auto-Blocking (Module 4)
6. "That's an IDS/IPS."
```

### Standard (8 modules, 12 minutes)
```
Dashboard → Detection → Engines → Risk → Blocking → Approval → 
FP Learning → TI → Analytics Summary
```

### Comprehensive (15 modules, 20-25 minutes)  
All modules in presentation order:
1. Dashboard
2. Real-Time Detection
3. Multi-Engine
4. Risk Score
5. Auto-Blocking
6. Approval Workflow
7. False Positive
8. Threat Intel
9. Anomaly Learning
10. Escalation
11. Pipeline Monitor
12. Analytics
13. Alert Lifecycle
14. Policy Tuning
15. Engine Playground
16. Pattern Detector

