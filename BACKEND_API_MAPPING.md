# INIDS Backend API Complete Mapping

**Document Purpose:** Map all existing backend APIs to UI pages. This ensures the UI layer is built entirely on real, existing functionality.

---

## 📊 ALERTS MANAGEMENT

### Alert Operations
| API Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/api/alerts` | GET | List all alerts | Query: limit, offset | Array of alert objects |
| `/api/alerts/<alert_id>` | PATCH | Update alert status | JSON: status, notes | Updated alert |
| `/api/alerts/<alert_id>/feedback` | POST | Add feedback to alert | JSON: feedback_type, comment | Feedback record |
| `/api/detections/history` | GET | Detection history | Query: limit, time_range | Array of detections |

### Alert Data Structure (from code analysis)
- `alert_id`: Unique identifier
- `timestamp`: Event timestamp
- `severity`: low/medium/high/critical
- `prediction`: Attack/Normal/Suspicious
- `confidence`: 0-1 score
- `status`: open/acknowledged/closed
- `source_ip`: Source IP address
- `attack_type`: Type of attack detected
- `risk_score`: Numeric risk score
- `profile`: Detection profile used

---

## 🛡️ ACTIONS (IPS) - Incident Response

### Action Management
| API Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/api/actions` | GET | List all actions taken | Query: limit, status | Array of actions |
| `/api/actions/pending` | GET | Pending approval queue | - | Pending actions |
| `/api/actions/<action_id>/approve` | POST | Approve action | JSON: notes | Approval result |
| `/api/actions/cleanup` | POST | Clean up old actions | Query: older_than | Cleanup result |

### Action Data Structure
- `action_id`: Unique identifier
- `alert_id`: Related alert
- `action_type`: block_ip, rate_limit, drop_packet, etc.
- `target`: IP/Port/Domain targeted
- `status`: pending/approved/executed/failed
- `created_at`: Action creation time
- `approved_at`: Approval timestamp
- `created_by`: User who created it

---

## 🔍 DETECTION - Multi-Engine System

### Detection Operations
| API Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/api/detect` | POST | Multi-engine detection | JSON: features | Detection result |
| `/api/predict` | POST | ML-based prediction | JSON: features | Prediction + confidence |
| `/api/engines` | GET | List all engines | - | Engine configs |
| `/api/engines/<engine_id>/toggle` | POST | Enable/disable engine | - | Updated engine state |
| `/api/api/explain` | POST | Explain detection | JSON: alert_id/features | Explanation details |

### Detection Result Structure
- `verdict`: Attack/Normal/Suspicious
- `confidence`: 0-1 score
- `attack_type`: Type classification
- `severity`: low/medium/high/critical
- `engine_results`: Per-engine results
- `features_used`: Featured attributes
- `explanation`: Human-readable reason

### Engines Available
1. **ML Engine**: ML model-based detection
2. **Signature Engine**: Rule-based detection
3. **Anomaly Engine**: Behavior-based detection
4. **Threshold Engine**: Statistical thresholds
5. **Threat Intel Engine**: IP/Domain reputation

---

## 📋 POLICY & RISK MANAGEMENT

### Policy Operations
| API Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/api/policy` | GET | Get current policy | - | Policy configuration |
| `/api/policy` | POST | Update policy | JSON: policy_config | Updated policy |
| `/api/policy/history` | GET | Policy change history | - | Policy versions |
| `/api/policy/rollback` | POST | Rollback to previous policy | Query: version | Rollback result |

### Policy Configuration
- `mode`: detect/prevent
- `detection_threshold`: Confidence threshold
- `approval_required`: Boolean
- `auto_escalate`: Boolean
- `escalation_threshold`: Risk score
- `firewall_adapter`: Which firewall to use
- Engine enablement flags

---

## 🌍 THREAT INTELLIGENCE

### Threat Intel Operations
| API Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/api/threat-intel/stats` | GET | TI statistics | - | TI metrics |
| `/api/threat-intel/lookup` | POST | Lookup IP/domain | JSON: ip/domain | TI data for entity |

### TI Data Structure
- `entity`: IP or domain
- `reputation`: good/suspicious/malicious
- `threat_feeds`: Which feeds flagged it
- `last_seen`: Timestamp
- `incidents`: Related incidents

---

## 🚫 ALLOWLIST & SUPPRESSION

### Allowlist Management
| API Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/api/allowlist` | GET | List allowlist entries | Query: limit | Allowlist entries |
| `/api/allowlist` | POST | Add to allowlist | JSON: entry, reason | New entry |
| `/api/allowlist/<entry>` | DELETE | Remove from allowlist | - | Removal result |

### False Positive Management
| API Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/api/fp-suppressions` | GET | List FP suppressions | - | Suppressed rules |
| `/api/fp-suppressions` | POST | Add FP suppression | JSON: engine_id, rule_id | New suppression |
| `/api/fp-suppressions/<engine_id>/<rule_id>` | DELETE | Remove suppression | - | Removal result |
| `/api/fp-stats` | GET | FP statistics | - | FP metrics |

---

## 📈 SYSTEM HEALTH & OBSERVABILITY

### Health Checks
| API Endpoint | Method | Purpose | Output |
|---|---|---|---|
| `/api/health` | GET | System health status | Health object |
| `/api/health/live` | GET | Liveness probe | Simple OK |
| `/api/health/ready` | GET | Readiness probe | Simple OK |

### Metrics
| API Endpoint | Method | Purpose | Output |
|---|---|---|---|
| `/api/metrics` | GET | System metrics | Prometheus metrics (optional) |
| `/api/dashboard/metrics` | GET | Dashboard-friendly metrics | JSON metrics |

### Anomaly & Escalation
| API Endpoint | Method | Purpose | Output |
|---|---|---|---|
| `/api/anomaly/status` | GET | Anomaly engine status | Engine health |
| `/api/escalation/summary` | GET | Escalation summary | Escalation stats |
| `/api/escalation/evict` | POST | Clear escalation | Clear result |

---

## 📥 INGESTION & PIPELINE

### Data Ingestion
| API Endpoint | Method | Purpose | Input | Output |
|---|---|---|---|---|
| `/api/ingest` | POST | Ingest raw data | JSON/CSV: flow data | Ingestion result |
| `/api/ingest/log` | POST | Ingest logs | JSON: log lines | Ingestion result |
| `/api/ingest/process` | POST | Process ingested data | Queue identifier | Processing result |

### Supported Log Formats
- Zeek conn logs
- Suricata EVE logs
- Raw flow data
- Custom CSV

---

## 👥 AUDIT & COMPLIANCE

### Audit Trail
| API Endpoint | Method | Purpose | Output |
|---|---|---|---|
| `/api/audit` | GET | Audit log | Audit entries |
| `/api/siem/flush` | GET | Export to SIEM | SIEM-formatted events |

---

## 📦 MODEL MANAGEMENT

### Models
| API Endpoint | Method | Purpose | Output |
|---|---|---|---|
| `/api/models/registry` | GET | Model registry | Model metadata |

---

## 🎮 DEMO & BATCH

### Demo Mode
| API Endpoint | Method | Purpose | Input |
|---|---|---|---|
| `/api/demo/start` | POST | Start demo | - |
| `/api/demo/stop` | POST | Stop demo | - |

### Batch Processing
- GET `/batch` - Batch UI
- POST `/batch` - Submit batch job

---

## 🔌 REAL-TIME & WEBSOCKETS

### WebSocket Events
- `connect` - Client connects
- `disconnect` - Client disconnects
- `subscribe_module` - Subscribe to module updates
- `unsubscribe_module` - Unsubscribe from module
- `request_module_data` - Request specific module data

---

## 📱 EXISTING PAGES

### Templates Already Exist
- `home.html` - Homepage
- `dashboard.html` - Main dashboard
- `predict.html` - Prediction/Detection console
- `batch.html` - Batch processing
- `models.html` - Model management
- `realtime.html` - Real-time visualization
- `capture.html` - Packet capture UI
- `dashboard_main.html` - Alternate dashboard

### Module Pages (24 modules)
Each `/modules/<module_id>` has associated endpoint `/api/modules/<module_id>`

---

## 🎯 CRITICAL FINDINGS

### What Needs UI Pages:
1. ✅ **Alerts Page** - List, view, update, status change
2. ✅ **Actions Page** - List, pending queue, approval
3. ✅ **Detection Console** - Run detection, view results
4. ✅ **Engine Management** - Enable/disable engines
5. ✅ **Policy & Risk** - View/edit policy, history
6. ✅ **Allowlist** - Manage allowlist entries
7. ✅ **Threat Intelligence** - Stats, lookup
8. ✅ **System Health** - Health status, metrics
9. ✅ **Ingestion** - Submit data, monitor queue
10. ✅ **Audit Trail** - View audit logs
11. ✅ **False Positives** - Manage suppressions
12. ✅ **Escalation** - View escalation queue

### Missing/Minimal UI Pages:
- Alerts: Basic
- Actions: Minimal
- False Positive Management: Needs work
- Escalation Tracker: Needs display
- Full Policy Editor: Needs form

---

## Summary Stats
- **Total API Endpoints**: 83+
- **Functional Services**: 15+
- **Detection Engines**: 5
- **Existing Templates**: 7
- **Module Endpoints**: 24

---

## IMPLEMENTATION STRATEGY

### Priority Tiers

**Tier 1 (Critical)** - Must have first:
1. Alerts page (HIGH PRIORITY)
2. Actions approval page
3. Detection console
4. Engine management

**Tier 2 (Important)** - Core functionality:
5. Policy editor
6. Allowlist manager
7. Threat Intel lookup
8. System health

**Tier 3 (Nice to have)**:
9. Ingestion monitor
10. Audit viewer
11. False Positive manager
12. Escalation tracker

