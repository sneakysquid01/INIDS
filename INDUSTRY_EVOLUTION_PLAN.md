# INIDS — Industry-Grade IDS/IPS Evolution Plan

> **Author**: Principal Security Architect review
> **Date**: 2026-03-12
> **Scope**: Evolve the existing INIDS academic prototype into a production-valuable IDS/IPS platform
> **Constraint**: Build incrementally on existing codebase — NO full rewrites

---

## Table of Contents

1. [Step 1 — Capability Gap Analysis vs Industry Systems](#step-1--capability-gap-analysis-vs-industry-systems)
2. [Step 2 — Dataset Strategy & ML Lifecycle Expansion](#step-2--dataset-strategy--ml-lifecycle-expansion)
3. [Step 3 — Ordered Engineering Roadmap](#step-3--ordered-engineering-roadmap)
4. [Step 4 — Full Implementation Plan](#step-4--full-implementation-plan)
5. [Step 5 — Final Industry Architecture Vision & Production Readiness Checklist](#step-5--final-industry-architecture-vision--production-readiness-checklist)

---

# Step 1 — Capability Gap Analysis vs Industry Systems

## Methodology

Comparison baseline: Suricata/Snort (signature IDS), Zeek (behavioral), CrowdStrike Falcon (EDR/XDR), Palo Alto NGFW IPS, Elastic Security, and Wazuh HIDS. Each gap is rated by priority and risk of absence.

---

### 1.1 Multi-Engine Detection Framework

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Signature-based detection | Core in Suricata/Snort — pattern match against known attack signatures | **Missing entirely.** Detection is ML-only via scikit-learn classifiers. | Critical |
| Anomaly detection engine | Statistical baselines in Zeek, Elastic ML | **Missing.** No unsupervised or statistical outlier detection. | Critical |
| Behavioral detection | Session/flow profiling in Zeek, EDR behavioral engines | **Missing.** No user/entity behavior analytics (UEBA). | High |
| Statistical threshold detection | Rate-based, volume-based triggers in every NGFW | **Partial.** `RiskEngine.recent_activity_score()` counts per-IP frequency. No aggregate threshold rules (e.g., SYN flood > N/sec). | High |
| Protocol-aware detection | Deep packet inspection in Suricata/Snort | **Missing.** No protocol state machines. Log parsers (`log_parsers.py`) only map a handful of Zeek/Suricata flow fields. | Medium |
| ML-based classification | Elastic ML, Darktrace, custom SIEM models | **Present but limited.** 5 scikit-learn models on NSL-KDD only. No ensemble voting, no deep learning, no online learning. | High |
| Threat intelligence matching | IP/domain/hash lookup in all enterprise IDS | **Missing entirely.** No TI feed integration. | Critical |
| Pluggable engine architecture | Suricata has detection modules; Elastic has detection rules API | **Missing.** `DetectionService` is a monolith — one model → one prediction path. No engine registry, no multi-engine aggregation. | Critical |

**Risk of not addressing**: Single-engine detection is the #1 reason academic IDS fails in production. Any single technique has blind spots: ML misses zero-day signature matches; signatures miss novel attacks; statistical rules miss low-and-slow attacks.

**Priority**: 🔴 Critical — This is the highest-priority gap.

---

### 1.2 Real-Time Streaming Processing

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Message queue backbone | Kafka (Elastic Security), Redis Streams (Wazuh), NATS (cloud-native) | **Partial.** `RedisStreamIngestionQueue` exists but is optional and unused by default. Primary path is synchronous Flask request → detect → respond. | High |
| Backpressure handling | Circuit breakers, consumer group lag monitoring | **Missing.** `InMemoryIngestionQueue` silently drops oldest entries. No lag metrics, no pause signal. | High |
| Async processing pipeline | Non-blocking event processing in all production IDS | **Missing.** Everything runs synchronously in the Flask request thread. `EventBus.publish()` is synchronous and blocking. | Critical |
| Batch vs streaming modes | Dual-mode in Elastic, Splunk | **Partial.** Batch CSV upload exists. No formal batch inference pipeline separate from streaming. | Medium |
| Consumer groups / partitioning | Kafka consumer groups, Redis XREADGROUP | **Missing.** Redis queue reads with XREAD (single consumer), no consumer groups. | High |

**Risk**: Under load (>100 events/sec), the synchronous Flask path will queue requests at the WSGI layer, creating cascading latency and potential timeouts. Detection latency becomes unpredictable.

**Priority**: 🔴 Critical — Streaming is the backbone of any real-time IDS.

---

### 1.3 Prevention Decision Engine

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Risk scoring aggregation | Multi-signal weighted scoring in all NGFW IPS | **Present.** `RiskEngine` with confidence/severity/frequency weights. Solid foundation. | ✅ Adequate |
| Confidence weighting | Bayesian confidence in EDR engines | **Present.** Confidence feeds into `RiskEngine.calculate()`. | ✅ Adequate |
| Policy evaluation engine | Tiered policy (monitor/alert/block) in NGFW | **Present.** `PolicyEngine.decide()` with 5 decision levels. | ✅ Adequate |
| False positive mitigation | Allowlists, confidence decay, human-in-the-loop | **Missing.** No allowlist mechanism. No confidence decay over time. No HITL approval flow. | High |
| Progressive enforcement | Alert → throttle → temp-block → permanent block escalation | **Partial.** `PolicyEngine` has graded decisions but no stateful escalation tracking per-IP across time. One event = one decision. | High |
| Policy conflict resolution | Priority-based rule ordering in NGFW | **Missing.** Policies are single-object, not rule-lists. No priority ordering, no conflict detection. | Medium |
| Dual-path unification | — | **Architectural debt.** BOTH `PreventionService.evaluate()` AND the EventBus IPS pipeline run on every prediction from `app.py`. They operate independently with separate state. | High |

**Risk**: Without false-positive mitigation, production deployment will generate alert fatigue and erode operator trust. The dual prevention path is a correctness hazard — two subsystems making independent blocking decisions.

**Priority**: 🔴 High

---

### 1.4 Active Response / Enforcement Layer

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Block IP | All NGFW, iptables/nftables integration | **Present.** `ActionExecutor.block_ip()` + Mock/UFW/nftables adapters. | ✅ Adequate |
| Rate limit | Adaptive shaping in NGFW | **Partial.** `ActionExecutor.rate_limit()` delegates to block_ip with shorter TTL. Not true shaping. | Medium |
| Drop connection / TCP RST | Inline IPS capability (Suricata inline mode) | **Missing.** System operates at application layer, not inline on wire. | Low (out-of-scope for app-layer IDS) |
| Kill process | EDR endpoint capability | **Missing.** Out-of-scope for network IDS. | Low |
| Quarantine asset | NAC integration in enterprise IDS | **Missing.** No network access control integration. | Low |
| Webhook / SOAR trigger | All enterprise SIEM/SOAR platforms | **Missing.** No outbound webhook on action events. No SOAR integration. | High |
| Action reconciliation | Rule drift detection in NGFW | **Present.** `ActionExecutor.reconcile()` + `PreventionScheduler` cleanup loop. Good foundation. | ✅ Adequate |

**Risk**: No webhook/SOAR integration means the system is an island — it cannot participate in an organization's security orchestration.

**Priority**: 🟡 High (webhook) / Low (inline wire-level)

---

### 1.5 Threat Intelligence Integration

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| IP reputation feeds | AbuseIPDB, OTX, Emerging Threats | **Missing entirely.** | Critical |
| Domain/URL feeds | PhishTank, URLhaus | **Missing.** System doesn't process domain/URL data. | Medium |
| Malware hash feeds | VirusTotal, MalwareBazaar | **Not applicable** (network IDS, not file-based). | Low |
| CVE/vulnerability feeds | NVD, CISA KEV | **Not applicable** for flow-based IDS. | Low |
| Feed ingestion scheduler | Periodic pull + caching in all enterprise IDS | **Missing.** No scheduled background feed pull. | High |
| Enrichment pipeline | Enrich detection events with TI context before decision | **Missing.** Events go straight from detection → risk → policy with no enrichment step. | Critical |
| TI caching layer | Local cache to avoid per-event API latency | **Missing.** | High |

**Risk**: Without TI enrichment, the system cannot leverage community-curated intelligence. An IP known to be malicious by every threat feed in the world would still need to trigger the ML classifier before any action — wasting time and missing context that would boost detection confidence.

**Priority**: 🔴 Critical

---

### 1.6 Observability Stack

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Structured logging | JSON logs with correlation IDs in all production systems | **Missing.** `logging_config.py` uses basic `StreamHandler`. No structured format, no correlation IDs. | High |
| Metrics (latency, throughput, detection rate) | Prometheus/Grafana in all cloud-native IDS | **Partial.** `MetricsService` has 14 counters in Prometheus text format. Missing: histograms for latency, gauge for queue depth, detection rate per engine. | Medium |
| Distributed tracing | OpenTelemetry in modern security platforms | **Missing.** No request trace IDs, no span propagation. | Medium |
| Alert dashboards | Kibana (Elastic), Grafana (Wazuh) | **Partial.** Flask dashboard exists with multiple panels. Not a dedicated observability stack. | Medium |
| Audit trail | Immutable audit logs in all compliance-relevant systems | **Present.** `OpsStore.add_audit()` with timestamped records. | ✅ Adequate |
| Health checks / liveness probes | Required for any containerized deployment | **Partial.** `/api/health` endpoint exists. Missing readiness probe (model loaded? DB connected?). | Medium |

**Risk**: Without structured logging and tracing, production incident investigation is manual and slow. Missing latency histograms means no SLO enforcement.

**Priority**: 🟡 High (structured logs) / Medium (tracing)

---

### 1.7 High Availability & Scalability

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Horizontal workers | Suricata multi-thread, Elastic cluster | **Missing.** Single Flask process. `InMemoryAlertStore`, `InMemoryPreventionStore`, in-memory `RiskEngine._events_by_source` all prevent multi-worker deployment. | Critical |
| Stateless detection nodes | Microservice pattern in cloud IDS | **Missing.** Detection is coupled to in-memory state. | High |
| Shared state store | Redis/PostgreSQL for session state | **Partial.** `OpsStore` uses SQLite/PostgreSQL for actions/audit. But alert store, rate limiter, and risk engine frequency data are all in-memory per-process. | High |
| Leader election | Leader-based scheduler in distributed systems | **Missing.** `PreventionScheduler` runs in every process — no leader election. Multiple workers = duplicate cleanup runs. | High |
| Graceful degradation | Circuit breakers, fallback modes | **Missing.** If the model fails to load, the entire system crashes. No degraded mode (e.g., signature-only fallback). | High |

**Risk**: Cannot scale beyond a single process. Under heavy traffic, a single Flask worker (even with gunicorn multi-worker) will have inconsistent state across workers due to in-memory stores.

**Priority**: 🔴 Critical

---

### 1.8 Policy Management System

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Rule priority / ordering | All NGFW have ordered rule evaluation | **Missing.** Single `PolicyConfig` object, not a rule list. | Medium |
| Rule conflict resolution | Longest-match, priority-first in NGFW | **Missing.** | Medium |
| Policy versioning | Git-based or DB-versioned policies in enterprise IDS | **Missing.** Policy is runtime-only via `set_policy()`. No versioning, no rollback. | High |
| Staging vs production rules | Shadow mode in all enterprise IPS | **Missing.** `dry_run` flag exists but is global, not per-rule. No shadow evaluation. | High |
| Dynamic reload | Hot reload without restart in Suricata/Snort | **Missing.** Policy changes require API call to running process. No file-based reload, no config watcher. | Medium |
| Allowlist / denylist management | Core in every firewall and IPS | **Missing.** No persistent allowlist. | High |

**Risk**: Without policy versioning and staging, a bad policy pushed to production cannot be audited or rolled back.

**Priority**: 🟡 High

---

### 1.9 Security Hardening

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Input validation | Strict input sanitization at every boundary | **Partial.** `IngestionService.normalize_features()` validates numeric fields. CSV extension check on batch upload. But no request body size limit, no JSON schema validation. | High |
| Sandboxed execution | Process isolation for untrusted operations | **Missing.** Model inference runs in the same process as the web server. | Medium |
| Rate limiting | Per-client throttling | **Present.** `RateLimiter` with sliding window. But in-memory only — doesn't work across workers. | Medium |
| Auth / RBAC | Required for any multi-user system | **Present.** `AuthService` with 3 roles. API key-based. | ✅ Adequate |
| Secrets management | Vault, env vars, rotated credentials | **Partial.** Env var-based with `load_settings()`. No rotation, no vault integration. `dev-inids-secret` default is risky. | High |
| TLS / transport security | Required for any network-facing service | **Missing.** No TLS configuration. Relies on reverse proxy. | Medium |
| Content Security Policy | XSS prevention headers | **Missing.** No CSP headers on Flask responses. | Medium |
| `MAX_CONTENT_LENGTH` | Upload size enforcement | **Missing.** No limit on batch upload size. | High |

**Risk**: Missing request size limits = potential DoS via large payloads. Missing CSP headers = XSS risk on the dashboard.

**Priority**: 🟡 High

---

### 1.10 Dataset & Model Maturity

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| Multi-dataset training | Multiple public datasets for coverage | **Single dataset.** NSL-KDD only (released 2009). | Critical |
| Dataset normalization layer | Unified schema across datasets | **Partial.** `schema.py` defines NSL-KDD columns. No abstract schema that normalizes other datasets. | High |
| Feature engineering pipeline | Automated feature extraction, selection | **Basic.** `ColumnTransformer` with `StandardScaler` + `OneHotEncoder`. No feature selection, no derived features. | High |
| Class imbalance handling | SMOTE, undersampling, weighted loss | **Missing.** `train_test_split` with `stratify` only. | High |
| Concept drift strategy | Online drift detection, retraining triggers | **Partial.** `drift_monitor.py` computes PSI between train/test. No automated retraining trigger. | Medium |
| Online/incremental learning | Model update without full retraining | **Missing.** All models are batch-trained scikit-learn. | Medium |
| Model evaluation pipeline | Automated evaluation, comparison, promotion | **Partial.** `train_cli.py` saves results JSON. `model_registry.py` tracks versions. No automated A/B comparison. | Medium |
| Deep learning models | CNN/LSTM/Transformer for sequence detection | **Missing.** Scikit-learn classifiers only. | Medium |
| Ensemble / voting | Multi-model consensus in production ML pipelines | **Missing.** Models are individually trained and deployed. No ensemble voting layer. | High |
| Adversarial robustness | Model hardening against evasion attacks | **Missing.** | Medium |

**Risk**: NSL-KDD alone covers network flow patterns from the late 1990s. It cannot detect modern attacks (IoT botnets, encrypted C2, lateral movement). This is the single biggest credibility gap for the ML pipeline.

**Priority**: 🔴 Critical

---

### 1.11 Integration Gaps

| Capability | Industry Standard | INIDS Current State | Gap |
|---|---|---|---|
| SIEM export (Syslog / CEF) | Universal in IDS products | **Missing.** No syslog output, no CEF/LEEF formatting. | High |
| SOAR integration | Webhook + API in modern IDS/SOAR | **Missing.** No outbound webhooks or integration API. | High |
| Firewall API integration | Palo Alto, Fortinet, AWS Security Group APIs | **Limited.** Mock, UFW, nftables only. No cloud firewall adapters. | Medium |
| EDR integration | Endpoint context enrichment | **Missing.** File-based or API-based endpoint data ingestion. | Medium |
| Network TAP / SPAN | Passive capture in network IDS | **Partial.** `capture_live_traffic.py` uses scapy for capture. Not production-tuned. | Medium |
| PCAP replay | Testing and forensics workflow | **Missing.** No PCAP replay capability for offline analysis. | Medium |
| REST API completeness | Documented, versioned API for integrations | **Partial.** 12+ API endpoints but no versioning, no OpenAPI schema. | Medium |

**Risk**: Without SIEM export, this system cannot feed into an organization's central security monitoring. It operates in isolation.

**Priority**: 🟡 High (SIEM/SOAR) / Medium (cloud firewalls/EDR)

---

### Gap Summary Matrix

| # | Gap Category | Critical | High | Medium | Low |
|---|---|---|---|---|---|
| 1.1 | Multi-Engine Detection | 3 | 2 | 1 | 0 |
| 1.2 | Streaming Processing | 1 | 2 | 1 | 0 |
| 1.3 | Prevention Decision | 0 | 3 | 1 | 0 |
| 1.4 | Active Response | 0 | 1 | 1 | 3 |
| 1.5 | Threat Intelligence | 2 | 2 | 1 | 1 |
| 1.6 | Observability | 0 | 1 | 4 | 0 |
| 1.7 | HA & Scalability | 1 | 3 | 0 | 0 |
| 1.8 | Policy Management | 0 | 3 | 3 | 0 |
| 1.9 | Security Hardening | 0 | 3 | 4 | 0 |
| 1.10 | Dataset & Model Maturity | 1 | 3 | 4 | 0 |
| 1.11 | Integration | 0 | 2 | 4 | 0 |
| **Total** | | **8** | **25** | **24** | **4** |

---

# Step 2 — Dataset Strategy & ML Lifecycle Expansion

## 2.1 Public IDS Dataset Integration Plan

### Tier 1 — Must Integrate (Core Coverage)

| Dataset | Year | Why It Improves Detection | Attack Types Covered | Records |
|---|---|---|---|---|
| **NSL-KDD** (existing) | 2009 | Baseline. Binary + multiclass classification already working. Legacy flow-level attacks. | DoS, Probe, R2L, U2R | 148K train / 22K test |
| **CICIDS2017** | 2017 | Modern attacks generated using real tools (Heartbleed, botnet, DDoS, infiltration, port scan). Pcap + labeled CSV. Bridges the 2009→modern gap. | Brute Force, DDoS, DoS (Hulk/GoldenEye/Slowloris), Heartbleed, Infiltration, Botnet, PortScan, Web Attacks | ~2.8M flows |
| **CSE-CIC-IDS2018** | 2018 | Extends CICIDS2017 with more infrastructure variety (5 departments, 50 machines) and updated attack profiles. | Same as 2017 + refined scenarios | ~16M flows |
| **UNSW-NB15** | 2015 | Diverse attack categories specifically designed to replace NSL-KDD. Includes flow + packet-level features. | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms | ~2.5M records |

### Tier 2 — Should Integrate (Extended Coverage)

| Dataset | Year | Why It Improves Detection | Attack Types Covered |
|---|---|---|---|
| **TON_IoT** | 2020 | IoT-specific attacks — fills the gap for non-traditional network devices. | DDoS, Ransomware, Backdoor, Injection, XSS, MITM, Password attacks on IoT protocols |
| **Bot-IoT** | 2018 | Botnet-specific with IoT focus. Very high volume (73M+ records) for scalability testing. | DDoS, DoS, OS/Service Scan, Keylogging, Data Exfiltration |
| **CIC-DDoS2019** | 2019 | Focused DDoS attacks via modern reflection/amplification vectors. | LDAP, MSSQL, NetBIOS, NTP, DNS, SNMP, SSDP, UDP, SYN, TFTP, UDP-Lag |

### Tier 3 — Nice to Have (Specialized)

| Dataset | Year | Purpose |
|---|---|---|
| **EMBER** | 2018 | Malware detection (PE file features) — only if file inspection is added later. |
| **CTU-13** | 2013 | Botnet C&C traffic captures — good for behavioral engine validation. |

---

## 2.2 Dataset Normalization Architecture

All datasets must be normalized into a **Common Flow Schema** that extends the existing `schema.py`:

```
Current NSL-KDD Schema (41 features + 2 labels)
                         ↓
    ┌────────────────────────────────────────┐
    │       Common Flow Schema (CFS)         │
    │                                        │
    │  Core Fields (always present):         │
    │    flow_id, timestamp, src_ip,         │
    │    src_port, dst_ip, dst_port,         │
    │    protocol, duration, src_bytes,      │
    │    dst_bytes, packets_in, packets_out  │
    │                                        │
    │  Statistical Fields (computed):        │
    │    flow_rate, byte_ratio, pkt_ratio,   │
    │    inter_arrival_mean, inter_arrival_  │
    │    std, active_time, idle_time         │
    │                                        │
    │  Connection Fields (aggregated):       │
    │    conn_count, srv_count, error_rates, │
    │    same_srv_rate, diff_host_rate       │
    │                                        │
    │  Labels:                               │
    │    dataset_source, label_binary,       │
    │    label_multiclass, attack_family     │
    └────────────────────────────────────────┘
```

### Normalizer Modules Required

| Module | Input | Output |
|---|---|---|
| `normalizers/nsl_kdd.py` | Existing `KDDTrain+.txt` format | CFS DataFrame |
| `normalizers/cicids.py` | CICIDS2017/2018 CSV (80+ CICFlowMeter features) | CFS DataFrame |
| `normalizers/unsw_nb15.py` | UNSW-NB15 CSV (49 features) | CFS DataFrame |
| `normalizers/ton_iot.py` | TON_IoT CSV | CFS DataFrame |
| `normalizers/bot_iot.py` | Bot-IoT CSV | CFS DataFrame |
| `normalizers/base.py` | Abstract base class defining CFS contract | — |

Each normalizer maps dataset-specific columns to CFS fields and fills missing fields with sensible defaults. The key principle: **the detection engines see only CFS — never raw dataset columns**.

---

## 2.3 Feature Engineering Improvements

### Current State
- `ColumnTransformer`: `StandardScaler` for numeric, `OneHotEncoder` for 3 categorical features.
- No derived features. No feature selection. No feature importance-based pruning.

### Required Additions

| Feature Category | Features | Why |
|---|---|---|
| **Flow ratios** | `byte_ratio = src_bytes / (dst_bytes + 1)`, `pkt_ratio` | Exfiltration and C2 beacons have extreme ratios |
| **Time-window aggregates** | `conn_per_minute_src`, `unique_dst_per_src_5min` | Detect scan and brute-force patterns |
| **Entropy features** | `payload_entropy`, `dst_port_entropy` | Encrypted C2 and port-hopping detection |
| **Statistical moments** | `inter_arrival_std`, `packet_size_variance` | Normal traffic is predictable; attacks are noisy |
| **Interaction features** | `protocol × service`, `flag × error_rate` | Non-linear patterns that tree models exploit |

### Feature Selection Strategy
1. Train baseline model on all features.
2. Use `permutation_importance` (scikit-learn) to rank features.
3. Drop features with importance < 0.01 (noise features).
4. Re-train and compare F1 to ensure no regression.
5. Store selected feature set in `model_registry` metadata.

---

## 2.4 Class Imbalance Handling

### Problem
- NSL-KDD: ~53% normal, ~47% attack (reasonably balanced, but multiclass labels U2R and R2L are <1%)
- CICIDS2017: ~83% benign, ~17% attack (highly imbalanced)
- UNSW-NB15: ~87% normal (varying by category)

### Strategy

| Technique | When to Use | Implementation |
|---|---|---|
| **Stratified sampling** | Always (already present via `train_test_split(stratify=...)`) | Keep |
| **Class weights** | Binary and multiclass training | `class_weight='balanced'` on RF, GB, DT. `sample_weight` for AdaBoost. |
| **SMOTE** | When minority class <5% of total | `imblearn.over_sampling.SMOTE` on training set only (never on validation/test) |
| **Undersampling** | When majority class >10x minority | `imblearn.under_sampling.RandomUnderSampler` combined with SMOTE (SMOTE-ENN) |
| **Threshold tuning** | Always after training | Use precision-recall curve to pick optimal threshold per attack class |

---

## 2.5 Offline vs Online Training Strategy

### Offline (Batch) Training — Primary Path

```
Raw Datasets → Normalizers → CFS DataFrames → Feature Engineering
    → Train/Val/Test Split → Model Training → Evaluation → Registry
```

- **Frequency**: Retrain when drift PSI > 0.2 on any critical feature, or on new dataset addition.
- **Process**: `train_cli.py` extended with `--dataset` flag to select/combine datasets.
- **Artifacts**: Model pickle, preprocessor pickle, feature list JSON, evaluation report JSON.

### Online (Incremental) Learning — Future Path

- **River library** (`river` PyPI): Incremental scikit-learn-compatible models.
- **Use case**: Adapt to slow concept drift without full retraining.
- **Implementation**: Shadow online model running alongside batch model. When online model's rolling F1 exceeds batch model for N consecutive windows, promote it.
- **Risk**: Online models are susceptible to adversarial data poisoning. Always validate against held-out test set before promotion.

---

## 2.6 Model Evaluation Pipeline

### Current State
- `train_cli.py` saves JSON with accuracy, F1, precision, recall.
- `model_registry.py` tracks version metadata.

### Required Additions

| Component | Purpose |
|---|---|
| **Per-class metrics** | Precision/recall/F1 per attack family (DoS, Probe, R2L, U2R) |
| **ROC-AUC per class** | Understand confidence calibration across categories |
| **Confusion matrix persistence** | Save as JSON artifact alongside model |
| **Cross-dataset evaluation** | Train on Dataset A, evaluate on Dataset B |
| **Threshold sensitivity analysis** | Evaluate at multiple thresholds (0.5, 0.6, 0.7, 0.8, 0.9) |
| **Automated comparison** | `train_cli.py --compare v3 v4` to generate side-by-side report |
| **Promotion gate** | Only promote to production if F1 > current production model's F1 |

---

# Step 3 — Ordered Engineering Roadmap

## Dependency Graph

```
Phase 1: Detection Framework ──────────────────────────────────────┐
    │                                                               │
Phase 2: Streaming Pipeline ───────────────────────────┐           │
    │                                                   │           │
Phase 3: Prevention Engine Maturity ───────────┐       │           │
    │                                           │       │           │
Phase 4: Threat Intelligence ──────────┐       │       │           │
    │                                   │       │       │           │
Phase 5: Dataset + ML Pipeline ────────│───────│───────│───────────┘
    │                                   │       │       │
Phase 6: Observability + Policy ───────┘       │       │
    │                                           │       │
Phase 7: Horizontal Scalability + HA ──────────┘───────┘
```

### Why This Order

1. **Detection Framework** first because everything downstream (risk scoring, policy, actions) depends on structured detection output. Cannot add new engines without a pluggable framework.
2. **Streaming Pipeline** second because Phases 3-7 all need asynchronous event flow. Building on synchronous Flask requests will block progress.
3. **Prevention Engine Maturity** third because it builds on the new detection events and async pipeline.
4. **Threat Intelligence** fourth because it enriches detection events — needs the engine framework and pipeline in place.
5. **Dataset + ML Pipeline** fifth because it strengthens the ML engine within the detection framework. Can be partially parallelized with Phase 4.
6. **Observability + Policy** sixth because the system is now complex enough to need structured ops tooling.
7. **HA + Scalability** last because it requires all state to be externalized (a prerequisite from Phases 1-6) and all components to be stateless.

---

## Phase Summary

| Phase | Name | Builds On | Key Deliverable | System State After |
|---|---|---|---|---|
| 1 | Detection Framework | Existing `DetectionService` | Pluggable multi-engine detection with engine registry | Multi-engine detection, still synchronous |
| 2 | Streaming Pipeline | Existing `EventBus`, `IngestionService` | Async event pipeline with backpressure | Asynchronous event-driven processing |
| 3 | Prevention Engine Maturity | Existing `PolicyEngine`, `ActionExecutor` | Unified prevention path, allowlists, escalation | Production-safe IPS decisions |
| 4 | Threat Intelligence | Phase 1 engine framework, Phase 2 pipeline | TI feed ingestion, caching, enrichment | Context-enriched detections |
| 5 | Dataset + ML Pipeline | Existing `preprocess_train.py`, Phase 1 engines | Multi-dataset training, ensemble voting, evaluation pipeline | Broad attack coverage |
| 6 | Observability + Policy | Phase 2 pipeline, existing `MetricsService` | Structured logs, tracing, policy versioning | Ops-ready monitoring |
| 7 | HA + Scalability | All previous phases | Externalized state, horizontal workers, leader election | Production-deployable at scale |

---

# Step 4 — Full Implementation Plan

## Phase 1 — Detection Framework Modularization

### Goal
Transform the monolithic `DetectionService` into a pluggable detection engine framework where multiple engines (ML, signature, anomaly, threshold, TI) can register, run, and contribute to a unified detection result.

### Components to Add

#### 1.1 `src/detection/engine_base.py` — Abstract Detection Engine

```python
class DetectionEngine(ABC):
    """Base class for all detection engines."""
    
    @property
    @abstractmethod
    def engine_id(self) -> str: ...
    
    @property
    @abstractmethod
    def engine_type(self) -> str: ...  # "ml", "signature", "anomaly", "threshold", "ti"
    
    @abstractmethod
    def evaluate(self, flow: CommonFlowRecord) -> EngineResult: ...
    
    @abstractmethod
    def is_ready(self) -> bool: ...
```

#### 1.2 `src/detection/engine_registry.py` — Engine Registry

- Register/unregister engines at runtime.
- Enable/disable engines via configuration.
- Health-check each engine before including in evaluation.

#### 1.3 `src/detection/engines/ml_engine.py` — ML Classification Engine

- Wraps the existing `DetectionService.predict_from_features()` logic.
- Returns `EngineResult(engine_id="ml_classifier", verdict="attack", confidence=92.5, metadata={...})`.
- Backward compatible — same models, same preprocessor, same threshold profiles.

#### 1.4 `src/detection/engines/signature_engine.py` — Signature-Based Engine

- Load rules from YAML/JSON rule files.
- Rule format:

```yaml
rules:
  - id: SIG-001
    name: "SYN Flood Detection"
    condition:
      protocol_type: tcp
      flag: S0
      count: ">100"
      same_srv_rate: "<0.1"
    severity: high
    attack_type: dos
```

- Pattern match against flow features.
- Returns `EngineResult` with rule ID and matched condition.

#### 1.5 `src/detection/engines/anomaly_engine.py` — Statistical Anomaly Engine

- Use `IsolationForest` or `LocalOutlierFactor` from scikit-learn.
- Train on "normal" subset of CFS data.
- Flag flows that deviate from learned normal profile.
- Returns anomaly score (0.0–1.0).

#### 1.6 `src/detection/engines/threshold_engine.py` — Statistical Threshold Engine

- Rule-based aggregate checks:
  - "If SYN packets from `src_ip` > 500/min → alert"
  - "If unique `dst_port` from `src_ip` > 50/min → port scan alert"
- Maintains sliding counters per source (similar to existing `RiskEngine.recent_activity_score()` but generalized).

#### 1.7 `src/detection/aggregator.py` — Multi-Engine Aggregator

- Collects `EngineResult` from all enabled engines.
- Aggregation strategies:
  - **Unanimous**: All engines must agree (high precision, low recall).
  - **Majority vote**: >50% of engines flag it (balanced).
  - **Any-trigger**: Any engine flags it (high recall, lower precision).
  - **Weighted vote**: Each engine has a configurable weight.
- Produces a `DetectionEvent` (extends existing) with `engine_results: list[EngineResult]` field.

### Components to Extend

| Existing Component | Change |
|---|---|
| `src/detection_service.py` | Refactor to become `MLClassificationEngine`. Move `predict_from_features` logic into new engine. Thin wrapper calls engine registry. |
| `src/core/event_bus.py` | Add `EngineResult` dataclass. Extend `DetectionEvent` with `engine_results` field. |
| `src/schema.py` | Add CFS (Common Flow Schema) alongside existing NSL-KDD schema. NSL-KDD is one materializer of CFS. |

### Data Flow Change

```
BEFORE:
  Features → DetectionService.predict_from_features() → PredictionResult → EventBus

AFTER:
  Features → CommonFlowRecord
           → EngineRegistry.evaluate_all(flow)
               ├→ MLClassificationEngine.evaluate(flow) → EngineResult
               ├→ SignatureEngine.evaluate(flow)        → EngineResult
               ├→ AnomalyEngine.evaluate(flow)         → EngineResult
               └→ ThresholdEngine.evaluate(flow)        → EngineResult
           → Aggregator.aggregate(results) → DetectionEvent → EventBus
```

### Runtime Improvement
- Detection confidence is now multi-signal. An attack flagged by both ML and signature engines has higher confidence than ML alone.
- Signature engine catches known patterns instantly (<1ms) even if ML is uncertain.
- Anomaly engine catches novel (zero-day) attacks that have no signature and look unlike training data.

### New Risks
- Multiple engines increase per-event latency. Mitigation: engines run in parallel (thread pool for CPU-bound, async for I/O-bound).
- Signature rules need maintenance. Mitigation: start with a small curated rule set for the attack types in NSL-KDD/CICIDS.

### Backward Compatibility
- `DetectionService` API remains unchanged. Internally delegates to engine registry.
- `PredictionResult` still returned. `engine_results` is an additive field.
- All existing tests pass — ML engine wraps the same model pipeline.

### Testing Strategy
- Unit test each engine in isolation with known-good and known-bad flows.
- Integration test: engine registry with all engines enabled, assert aggregator produces correct `DetectionEvent`.
- Regression test: existing `/api/predict` endpoint returns same results via ML engine as before.

---

## Phase 2 — Streaming Pipeline Introduction

### Goal
Replace the synchronous Flask-request-driven detection path with an asynchronous event pipeline using Redis Streams (already partially implemented) as the backbone.

### Components to Add

#### 2.1 `src/pipeline/stream_processor.py` — Async Stream Consumer

- Redis Streams consumer group reader.
- Reads from `inids:ingestion` stream.
- Deserializes to `CommonFlowRecord`.
- Dispatches to engine registry.
- Publishes `DetectionEvent` to `inids:detection` stream.

#### 2.2 `src/pipeline/backpressure.py` — Backpressure Controller

- Monitors consumer lag (Redis `XINFO GROUPS`).
- When lag exceeds threshold:
  1. Log warning.
  2. If lag > high watermark: switch to sampling mode (process every Nth event).
  3. If lag > critical: reject new ingestion with 503.
- Exposes lag metric for Prometheus.

#### 2.3 `src/pipeline/worker.py` — Pipeline Worker Entry Point

- Standalone Python process (not Flask).
- Runs event loop: read from stream → detect → risk → policy → action.
- Configurable concurrency (multiple worker processes).

#### 2.4 Extend `src/core/event_bus.py` — Hybrid Local + Stream EventBus

- Local mode (current): in-process handler dispatch. Used within a single worker.
- Stream mode (new): publish events to Redis Streams for cross-process consumption.
- Auto-select based on configuration.

### Components to Extend

| Existing Component | Change |
|---|---|
| `src/ingestion_service.py` | `enqueue_record()` always writes to Redis Stream when available. Add consumer group setup on init. |
| `web_app/app.py` | API endpoints (`/api/predict`, `/api/ingest`) become producers into the ingestion stream. Synchronous prediction kept as "fast path" for single-request use. |
| `src/ips/scheduler.py` | Make scheduler stream-aware — read action events from stream instead of relying on in-process EventBus. |

### Data Flow Change

```
BEFORE (synchronous):
  HTTP POST /api/predict → Flask thread → DetectionService → RiskEngine 
    → PolicyEngine → ActionExecutor → HTTP Response (all synchronous)

AFTER (dual-mode):
  FAST PATH (synchronous, for single predictions):
    HTTP POST /api/predict → Flask → EngineRegistry → response
    (EventBus fires locally for prevention pipeline)

  STREAMING PATH (async, for bulk/continuous):
    Ingestion source → Redis Stream [inids:ingestion]
      → Worker reads → EngineRegistry.evaluate_all()
        → Redis Stream [inids:detection]
          → Risk Worker → Redis Stream [inids:risk]
            → Policy Worker → Redis Stream [inids:decision]
              → Action Worker → Redis Stream [inids:action]
```

### Runtime Improvement
- Detection throughput scales linearly with worker count.
- Flask process is no longer blocked by ML inference.
- Backpressure prevents system overload under burst traffic.

### New Risks
- Redis becomes a single point of failure. Mitigation: Redis Sentinel or Redis Cluster for HA.
- Event ordering is per-partition (stream). Within a single IP's events, ordering is preserved by consumer design.
- Exactly-once semantics are not guaranteed with Redis Streams. Mitigation: idempotent action execution (check if action already exists before creating).

### Backward Compatibility
- Flask synchronous path remains for backward compatibility and single-prediction web UI use.
- All existing API endpoints work identically.
- Streaming path is opt-in via configuration (`INIDS_PIPELINE_MODE=streaming`).

---

## Phase 3 — Prevention Engine Maturity

### Goal
Unify the dual prevention paths, add allowlists, implement progressive enforcement, and add false-positive mitigation.

### Components to Add

#### 3.1 `src/prevention/allowlist.py` — IP/Network Allowlist

- Persistent allowlist in `OpsStore` (new SQLite table).
- Supports IP addresses, CIDR ranges, and hostnames.
- Checked before any blocking action.
- CRUD API endpoints.

#### 3.2 `src/prevention/escalation_tracker.py` — Progressive Enforcement

- Per-IP state machine: `CLEAN → ALERT → RATE_LIMIT → TEMP_BLOCK → BLOCK`.
- Escalation requires N detections within a time window to advance.
- De-escalation: if no detections for M minutes, drop one level.
- State persisted in `OpsStore` (new table).

#### 3.3 `src/prevention/false_positive_manager.py` — FP Mitigation

- Record analyst feedback: "this was a false positive" on any alert/action.
- Auto-suppress future alerts matching the same signature + source + profile after N confirmed FPs.
- Suppression rules have TTL (auto-expire) and can be manually removed.

#### 3.4 Webhook / SOAR Adapter — `src/firewall_adapters.py` extension

- New `WebhookAdapter(FirewallAdapter)` that calls a configurable URL on block/unblock.
- Payload format matches common SOAR webhook schemas.

### Components to Refactor

| Existing Component | Change |
|---|---|
| `src/prevention_service.py` | **Deprecate.** Migrate functionality into `PolicyEngine` + `ActionExecutor`. The legacy `PreventionService.evaluate()` is replaced by the EventBus pipeline. |
| `web_app/app.py` | Remove dual prevention path. All prevention flows through EventBus → PolicyEngine → ActionExecutor. Legacy `PreventionService` wrapper kept for backward compat but delegates to new pipeline. |
| `src/ips/policy_engine.py` | Add allowlist check before decision. Add escalation state as input to decision. |

### Data Flow Change

```
BEFORE:
  Detection → RiskEngine → PolicyEngine.decide() → ActionExecutor
  (AND separately: Detection → PreventionService.evaluate() → direct block)

AFTER:
  Detection → AllowlistCheck → RiskEngine → EscalationTracker
    → PolicyEngine.decide(risk, escalation_state, suppression_rules)
      → ActionExecutor → WebhookAdapter (if configured)
      → FP feedback loop (analyst can flag false positive)
```

### Runtime Improvement
- Single prevention path eliminates inconsistent blocking decisions.
- Progressive enforcement reduces false-positive impact — first offense is alert only, not block.
- Allowlist prevents blocking of known-good infrastructure (e.g., monitoring systems, scanners).

---

## Phase 4 — Threat Intelligence Integration

### Goal
Add the ability to consume external threat intelligence feeds and use them to enrich detection events before risk scoring.

### Components to Add

#### 4.1 `src/threat_intel/feed_manager.py` — Feed Ingestion

- Scheduled background task (extends `PreventionScheduler` pattern).
- Pulls from configured feed URLs at configurable intervals (default: every 4 hours).
- Supported feed types:
  - **IP reputation**: CSV/JSON list of malicious IPs (AbuseIPDB, OTX, Emerging Threats).
  - **Domain reputation**: Similar format.
- Stores in local cache (SQLite table or Redis hash).

#### 4.2 `src/threat_intel/ti_cache.py` — TI Cache Layer

- In-memory LRU cache backed by persistent store.
- Lookup: `cache.lookup_ip("1.2.3.4") → TIRecord(score=85, source="abuseipdb", tags=["scanner"])`.
- Cache invalidation on feed refresh.
- Miss = unknown (not malicious, not clean).

#### 4.3 `src/detection/engines/ti_engine.py` — TI Detection Engine

- Registered in engine registry.
- Looks up source/destination IP in TI cache.
- If IP has reputation score > threshold → flag as suspicious.
- Returns `EngineResult` with TI context.

#### 4.4 `src/threat_intel/enrichment.py` — Event Enrichment

- Runs between detection and risk scoring.
- Adds TI context to `DetectionEvent`: `ti_tags`, `ti_score`, `ti_sources`.
- RiskEngine can use `ti_score` as an additional component.

### Data Flow Change

```
BEFORE:
  Detection → RiskEngine (confidence + severity + frequency)

AFTER:
  Detection → TI Enrichment (add ti_score, ti_tags) → RiskEngine (confidence + severity + frequency + ti_score)
```

### New Risks
- External feed availability: feeds may be rate-limited or unavailable. Mitigation: cache with TTL; stale data is better than no data.
- Feed poisoning: a compromised feed could cause false positives. Mitigation: require multiple TI sources to agree before boosting risk score significantly.

---

## Phase 5 — Dataset + ML Pipeline Strengthening

### Goal
Expand training data beyond NSL-KDD, implement proper feature engineering, add ensemble voting, and build an automated evaluation pipeline.

### Components to Add

#### 5.1 `src/training/normalizers/` — Dataset Normalizer Modules

- As described in Step 2.2. One normalizer per dataset.
- CLI: `python -m src.train_cli normalize --dataset cicids2017 --output data/normalized/cicids2017.parquet`

#### 5.2 `src/training/feature_engineer.py` — Feature Engineering Pipeline

- Compute derived features (ratios, entropies, windowed aggregates).
- Feature selection via permutation importance.
- Output: feature-engineered DataFrame + selected feature list.

#### 5.3 `src/training/imbalance.py` — Class Imbalance Handlers

- Wrappers around `imblearn` (SMOTE, SMOTE-ENN, class weights).
- Strategy selected per dataset based on class distribution analysis.

#### 5.4 `src/detection/engines/ensemble_engine.py` — Ensemble Voting Engine

- Wraps multiple ML models (existing 5 + new deep learning models if added).
- Voting strategies: hard majority, soft average, weighted.
- Returns `EngineResult` with per-model breakdown.

#### 5.5 `src/training/evaluation.py` — Automated Evaluation Pipeline

- Per-class metrics, ROC-AUC, confusion matrix.
- Cross-dataset evaluation (train on A, test on B).
- Threshold sensitivity analysis.
- Automated comparison against current production model.
- Promotion gate: must exceed current model's F1.

### Components to Extend

| Existing Component | Change |
|---|---|
| `src/preprocess_train.py` | Add `--dataset` and `--combine` flags. Support CFS input format. |
| `src/train_cli.py` | Add `normalize`, `evaluate`, `compare`, `promote` subcommands. |
| `src/model_registry.py` | Add evaluation artifacts, dataset provenance, feature list to metadata. |
| `src/drift_monitor.py` | Support CFS features (not just NSL-KDD). Alert on cross-dataset drift. |
| `src/schema.py` | Add CFS schema alongside KDDNSL schema. Both coexist. |

### Runtime Improvement
- Models trained on CICIDS2017 + UNSW-NB15 + NSL-KDD can detect modern attack types that pure NSL-KDD models miss.
- Ensemble voting reduces variance — individual model errors cancel out.
- Automated evaluation prevents regression when retraining.

---

## Phase 6 — Observability & Policy Control Plane

### Goal
Add structured logging, tracing, enhanced metrics, and a policy management system with versioning.

### Components to Add

#### 6.1 `src/logging_config.py` — Structured Logging (refactor existing)

- JSON-formatted log output with fields: `timestamp`, `level`, `logger`, `message`, `correlation_id`, `source_ip`, `engine`, `duration_ms`.
- Correlation ID propagated through the entire detection→risk→policy→action pipeline.
- Configurable output: stdout (development), file (production), syslog (SIEM integration).

#### 6.2 `src/observability/tracing.py` — Lightweight Request Tracing

- Generate trace ID at ingestion point.
- Attach to all event dataclasses (`trace_id` field).
- Log trace ID in every structured log message.
- Expose trace timeline in dashboard (optional).

#### 6.3 `src/metrics_service.py` — Enhanced Metrics (extend existing)

- Add histogram for detection latency (p50, p95, p99).
- Add gauge for queue depth, active blocks, engine health.
- Add per-engine detection rate counters.
- Add model inference latency histogram.

#### 6.4 `src/policy/policy_store.py` — Policy Versioning

- Store policy versions in `OpsStore` (new table).
- Every `set_policy()` call creates a new version, keeping the previous one.
- Rollback: `rollback_policy(version_id)`.
- Diff: compare two policy versions.

#### 6.5 `src/policy/rule_engine.py` — Rule-Based Policy (extend existing)

- Replace single `PolicyConfig` with ordered rule list.
- Each rule has priority, conditions, and actions.
- Evaluation: iterate rules top-to-bottom, first match wins.
- Support staging rules (evaluated but not enforced, results logged).

#### 6.6 SIEM Export — `src/integrations/siem_exporter.py`

- Format alerts as CEF (Common Event Format) or LEEF.
- Send via syslog (UDP/TCP).
- Configurable via settings.

### Components to Extend

| Existing Component | Change |
|---|---|
| `src/core/event_bus.py` | Add `trace_id` field to all event dataclasses. |
| `src/metrics_service.py` | Add histograms and gauges. Keep existing counter API. |
| `src/settings.py` | Add configuration for log format, syslog endpoint, tracing enable/disable. |
| `web_app/app.py` | Add readiness probe endpoint. Generate correlation ID per request. |

### Runtime Improvement
- Structured logs enable grep/filter across all events by source IP, trace ID, engine, etc.
- Latency histograms enable SLO monitoring (e.g., "99% of detections complete in <500ms").
- Policy versioning enables safe rollback after bad policy push.

---

## Phase 7 — Horizontal Scalability & High Availability

### Goal
Externalize all in-memory state, enable multi-worker deployment, and add leader election for singleton tasks.

### Components to Refactor

| Component | Current State | Target State |
|---|---|---|
| `InMemoryAlertStore` | In-process list | PostgreSQL/Redis-backed `AlertRepository` |
| `InMemoryPreventionStore` | In-process list | Remove (replaced by `OpsStore` actions table) |
| `InMemoryIngestionQueue` | In-process deque | Redis Streams only (Phase 2 completion) |
| `RiskEngine._events_by_source` | In-process dict | Redis sorted sets (per-IP timestamps) |
| `RateLimiter` | In-process dict | Redis-backed sliding window (`ZRANGEBYSCORE`) |
| `MetricsService._counters` | In-process dict | Redis counters or Prometheus pushgateway |

### Components to Add

#### 7.1 `src/ha/leader_election.py` — Leader Election

- Redis-based leader lock using `SET key value NX EX ttl`.
- Only the leader runs `PreventionScheduler` (cleanup + reconciliation).
- If leader fails, another worker acquires lock after TTL expires.
- Heartbeat renewal every TTL/3 seconds.

#### 7.2 `src/ha/health.py` — Enhanced Health Checks

- **Liveness**: Process is alive, can respond to HTTP.
- **Readiness**: Model loaded, DB connected, Redis reachable, at least one engine healthy.
- **Startup**: Initial model loading complete.

#### 7.3 Deployment Configuration

- Docker Compose for development (single node).
- Docker Compose with multiple workers for staging.
- Kubernetes manifests for production (optional, not required for MVP):
  - `Deployment` for API workers.
  - `Deployment` for pipeline workers.
  - `StatefulSet` for Redis.
  - `CronJob` for TI feed refresh.

### Data Flow Change

```
BEFORE (single process):
  Flask process = web server + detection + prevention + scheduler + metrics

AFTER (multi-process):
  API Worker 1..N: Flask web server, synchronous predict, write to Redis Streams
  Pipeline Worker 1..M: Read streams, detect, risk score, policy, action
  Scheduler Leader: Cleanup, reconciliation, TI feed refresh (one instance)
  Redis: Streams, state, counters, leader lock
  PostgreSQL: Persistent storage (alerts, actions, audit, policies, TI cache)
```

### New Risks
- Redis availability becomes critical. Mitigation: Redis Sentinel with 3 nodes minimum.
- PostgreSQL availability becomes critical. Mitigation: connection pooling, read replicas for dashboard queries.
- Distributed debugging is harder. Mitigation: trace IDs (Phase 6) propagated across all workers.

### Backward Compatibility
- Single-process mode remains the default (`INIDS_PIPELINE_MODE=local`).
- Multi-worker mode requires Redis and is opt-in (`INIDS_PIPELINE_MODE=streaming`).
- All configuration via environment variables, consistent with existing `settings.py` pattern.

---

# Step 5 — Final Industry Architecture Vision & Production Readiness Checklist

## 5.1 Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONTROL PLANE                                     │
│                                                                             │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  Policy Mgmt  │  │ Model Registry  │  │  TI Feed Mgr │  │  Auth/RBAC  │ │
│  │  versioned    │  │  promote/       │  │  schedule/    │  │  API keys   │ │
│  │  rules, allow-│  │  rollback/      │  │  cache/       │  │  roles      │ │
│  │  lists, staged│  │  compare        │  │  enrich       │  │             │ │
│  └──────┬───────┘  └────────┬────────┘  └──────┬───────┘  └─────────────┘ │
│         │                    │                   │                           │
│         └────────────────────┼───────────────────┘                           │
│                              │                                               │
├──────────────────────────────┼───────────────────────────────────────────────┤
│                          DATA PLANE                                          │
│                              │                                               │
│  ┌───────────────────────────▼────────────────────────────────────────────┐  │
│  │                    INGESTION LAYER                                     │  │
│  │                                                                        │  │
│  │   ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────────────────┐   │  │
│  │   │ Scapy   │  │ Zeek Log │  │ Suricata  │  │ API Ingestion      │   │  │
│  │   │ Capture  │  │ Parser   │  │ Eve Parser│  │ (REST / webhook)   │   │  │
│  │   └────┬────┘  └────┬─────┘  └─────┬─────┘  └────────┬───────────┘   │  │
│  │        │             │              │                  │               │  │
│  │        └─────────────┼──────────────┼──────────────────┘               │  │
│  │                      │              │                                  │  │
│  │                      ▼              ▼                                  │  │
│  │              ┌──────────────────────────────┐                         │  │
│  │              │  Feature Normalization        │                         │  │
│  │              │  (CFS Common Flow Schema)     │                         │  │
│  │              └──────────┬───────────────────┘                         │  │
│  │                         │                                              │  │
│  │                         ▼                                              │  │
│  │              ┌──────────────────────────────┐                         │  │
│  │              │  Redis Stream: inids:ingest   │ ←── Backpressure ctrl  │  │
│  │              └──────────┬───────────────────┘                         │  │
│  └─────────────────────────┼─────────────────────────────────────────────┘  │
│                            │                                                 │
│  ┌─────────────────────────▼─────────────────────────────────────────────┐  │
│  │                   DETECTION ENGINE CLUSTER                             │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    Engine Registry                                │  │  │
│  │  │                                                                   │  │  │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐   │  │  │
│  │  │  │ ML Engine │ │ Signature │ │  Anomaly  │ │  Threshold    │   │  │  │
│  │  │  │(ensemble) │ │  Engine   │ │  Engine   │ │  Engine       │   │  │  │
│  │  │  │RF,GB,DT,  │ │YAML rules│ │IsoForest/ │ │rate counters  │   │  │  │
│  │  │  │ADA,MLP    │ │pattern   │ │LOF        │ │aggregate      │   │  │  │
│  │  │  └─────┬─────┘ └─────┬────┘ └─────┬─────┘ └──────┬────────┘   │  │  │
│  │  │        │              │            │               │            │  │  │
│  │  │        └──────────────┼────────────┼───────────────┘            │  │  │
│  │  │                       │            │                            │  │  │
│  │  │                       ▼            ▼                            │  │  │
│  │  │              ┌─────────────────────────────┐                   │  │  │
│  │  │              │  Multi-Engine Aggregator     │                   │  │  │
│  │  │              │  (weighted vote / majority)  │                   │  │  │
│  │  │              └──────────┬──────────────────┘                   │  │  │
│  │  └────────────────────────┼──────────────────────────────────────┘  │  │
│  │                           │                                         │  │
│  │  ┌────────────────────────▼────────────────────┐                   │  │
│  │  │  TI Enrichment                               │                   │  │
│  │  │  (lookup IP in TI cache, attach tags/score)  │                   │  │
│  │  └────────────────────────┬────────────────────┘                   │  │
│  │                           │                                         │  │
│  │              Redis Stream: inids:detection                          │  │
│  └───────────────────────────┼─────────────────────────────────────────┘  │
│                              │                                             │
│  ┌───────────────────────────▼─────────────────────────────────────────┐  │
│  │                        RISK ENGINE                                   │  │
│  │                                                                      │  │
│  │  Inputs:                    │    Output:                             │  │
│  │  • ML confidence      (0.5)│    • risk_score (0.0–1.0)              │  │
│  │  • Severity mapping   (0.3)│    • component breakdown               │  │
│  │  • Frequency score    (0.2)│    • escalation state                  │  │
│  │  • TI score           (+)  │                                         │  │
│  │  • Escalation history (+)  │                                         │  │
│  │                                                                      │  │
│  │           Redis Stream: inids:risk                                   │  │
│  └───────────────────────────┬─────────────────────────────────────────┘  │
│                              │                                             │
│  ┌───────────────────────────▼─────────────────────────────────────────┐  │
│  │                      POLICY ENGINE                                   │  │
│  │                                                                      │  │
│  │  ┌──────────────┐  ┌───────────────┐  ┌────────────────────────┐   │  │
│  │  │ Allowlist     │  │ Rule Evaluator│  │ FP Suppression Check   │   │  │
│  │  │ Check         │→ │ (ordered,     │→ │ (auto-suppress known   │   │  │
│  │  │ (skip if      │  │  versioned)   │  │  false positives)      │   │  │
│  │  │  allowlisted) │  │              │  │                        │   │  │
│  │  └──────────────┘  └───────────────┘  └────────────────────────┘   │  │
│  │                                                                      │  │
│  │  Decisions: ALLOW | ALERT | RATE_LIMIT | TEMP_BLOCK | BLOCK          │  │
│  │                                                                      │  │
│  │           Redis Stream: inids:decision                               │  │
│  └───────────────────────────┬─────────────────────────────────────────┘  │
│                              │                                             │
│  ┌───────────────────────────▼─────────────────────────────────────────┐  │
│  │                    ACTION EXECUTOR                                   │  │
│  │                                                                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │  │
│  │  │ Mock     │  │ UFW      │  │ nftables │  │ Webhook/SOAR      │  │  │
│  │  │ Adapter  │  │ Adapter  │  │ Adapter  │  │ Adapter           │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └───────────────────┘  │  │
│  │                                                                      │  │
│  │  → Persist to OpsStore → Audit log → Reconciliation loop            │  │
│  │                                                                      │  │
│  │           Redis Stream: inids:action                                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                       OBSERVABILITY PLANE                                 │
│                                                                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ Structured │  │ Prometheus │  │ Trace IDs  │  │ Audit Trail        │ │
│  │ JSON Logs  │  │ Metrics    │  │ (request-  │  │ (OpsStore)         │ │
│  │ + Syslog   │  │ /metrics   │  │  scoped)   │  │                    │ │
│  │ CEF export │  │ endpoint   │  │            │  │                    │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────┘ │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                       PERSISTENCE LAYER                                   │
│                                                                           │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────────────┐   │
│  │ PostgreSQL   │  │ Redis         │  │ Filesystem                   │   │
│  │              │  │               │  │                              │   │
│  │ • alerts     │  │ • streams     │  │ • model artifacts (.pkl)    │   │
│  │ • actions    │  │ • TI cache    │  │ • signature rules (.yaml)   │   │
│  │ • audit logs │  │ • rate limits │  │ • evaluation reports (.json)│   │
│  │ • policies   │  │ • leader lock │  │ • normalized datasets       │   │
│  │ • TI feeds   │  │ • counters    │  │                              │   │
│  │ • escalation │  │ • risk freq   │  │                              │   │
│  │ • FP records │  │               │  │                              │   │
│  └──────────────┘  └───────────────┘  └─────────────────────────────┘   │
│                                                                           │
├───────────────────────────────────────────────────────────────────────────┤
│                       PRESENTATION LAYER                                  │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                      Flask Web Application                        │   │
│  │                                                                    │   │
│  │  Dashboard │ Predict │ Batch │ Models │ Realtime │ Capture │ About │   │
│  │                                                                    │   │
│  │  + REST API (versioned: /api/v1/...) + WebSocket (SocketIO)       │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 Target Module Structure

```
INIDS/
├── src/
│   ├── __init__.py
│   ├── settings.py                    # (extend) unified configuration
│   ├── schema.py                      # (extend) CFS + NSL-KDD schemas
│   ├── label_utils.py                 # (keep)
│   ├── logging_config.py             # (refactor) structured JSON logging
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── event_bus.py              # (extend) hybrid local+stream
│   │   └── events.py                 # (new) all event dataclasses extracted
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── engine_base.py            # (new) ABC for detection engines
│   │   ├── engine_registry.py        # (new) plugin registry
│   │   ├── aggregator.py             # (new) multi-engine result fusion
│   │   ├── detection_service.py      # (refactor from src/detection_service.py)
│   │   └── engines/
│   │       ├── __init__.py
│   │       ├── ml_engine.py          # (new) wraps existing ML pipeline
│   │       ├── signature_engine.py   # (new)
│   │       ├── anomaly_engine.py     # (new)
│   │       ├── threshold_engine.py   # (new)
│   │       └── ti_engine.py          # (new) Phase 4
│   │
│   ├── prevention/
│   │   ├── __init__.py
│   │   ├── allowlist.py              # (new)
│   │   ├── escalation_tracker.py     # (new)
│   │   ├── false_positive_manager.py # (new)
│   │   └── prevention_service.py     # (deprecate → thin wrapper)
│   │
│   ├── ips/
│   │   ├── __init__.py
│   │   ├── action_executor.py        # (keep)
│   │   ├── policy_engine.py          # (extend)
│   │   ├── risk_engine.py            # (extend with TI score)
│   │   └── scheduler.py             # (extend with leader election)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── stream_processor.py       # (new) Redis Streams consumer
│   │   ├── backpressure.py           # (new)
│   │   └── worker.py                # (new) standalone pipeline entry point
│   │
│   ├── threat_intel/
│   │   ├── __init__.py
│   │   ├── feed_manager.py           # (new)
│   │   ├── ti_cache.py              # (new)
│   │   └── enrichment.py            # (new)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── normalizers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # (new) ABC
│   │   │   ├── nsl_kdd.py           # (new) wraps existing preprocess
│   │   │   ├── cicids.py            # (new)
│   │   │   ├── unsw_nb15.py         # (new)
│   │   │   └── ton_iot.py           # (new)
│   │   ├── feature_engineer.py      # (new)
│   │   ├── imbalance.py             # (new)
│   │   ├── evaluation.py            # (new)
│   │   ├── preprocess_train.py      # (keep, extend)
│   │   └── train_cli.py            # (keep, extend)
│   │
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── siem_exporter.py         # (new) CEF/syslog
│   │   └── webhook_adapter.py       # (new)
│   │
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── policy_store.py          # (new) versioned policies
│   │   └── rule_engine.py           # (new) ordered rules
│   │
│   ├── ha/
│   │   ├── __init__.py
│   │   ├── leader_election.py       # (new)
│   │   └── health.py               # (new) liveness/readiness/startup
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   └── tracing.py              # (new)
│   │
│   ├── auth_service.py              # (keep)
│   ├── rate_limiter.py              # (extend → Redis-backed)
│   ├── metrics_service.py           # (extend → histograms/gauges)
│   ├── firewall_adapters.py         # (extend → webhook adapter)
│   ├── ingestion_service.py         # (extend → consumer groups)
│   ├── ops_store.py                 # (extend → new tables)
│   ├── model_registry.py            # (extend → evaluation artifacts)
│   ├── drift_monitor.py             # (extend → CFS support)
│   └── log_parsers.py              # (keep)
│
├── web_app/                          # (keep, extend)
│   ├── app.py                        # (refactor → remove dual prevention path)
│   ├── templates/                    # (keep)
│   └── static/                       # (keep)
│
├── tests/
│   ├── unit/                         # (reorganize existing)
│   ├── integration/                  # (new)
│   └── e2e/                         # (new)
│
├── deploy/
│   ├── compose/
│   │   ├── docker-compose.yml        # (extend → multi-service)
│   │   └── docker-compose.ha.yml     # (new → HA stack)
│   └── k8s/                          # (new, optional)
│
├── rules/                            # (new) signature rule files
│   └── default_rules.yaml
│
├── data/
│   ├── KDDTrain+.txt                 # (keep)
│   ├── KDDTest+.txt                  # (keep)
│   └── normalized/                   # (new) normalized dataset output
│
├── models/                           # (keep)
├── results/                          # (keep)
└── docs/                             # (extend)
```

---

## 5.3 Event Model Evolution

### Current Events (keep all, extend)

| Event | Fields Added |
|---|---|
| `DetectionEvent` | `engine_results: list[EngineResult]`, `trace_id: str`, `ti_tags: list[str]`, `ti_score: float` |
| `RiskScoreEvent` | `ti_component: float`, `escalation_level: str` |
| `PolicyDecisionEvent` | `rule_id: str`, `policy_version: int`, `suppressed: bool` |
| `ActionEvent` | `trace_id: str`, `webhook_response: dict | None` |
| `AuditEvent` | `trace_id: str`, `correlation_id: str` |

### New Events

| Event | Purpose |
|---|---|
| `EngineResult` | Individual engine's verdict + confidence + metadata |
| `TIEnrichmentEvent` | TI lookup result attached to detection |
| `EscalationEvent` | IP escalation state change |
| `FeedRefreshEvent` | TI feed update completed |
| `PolicyChangeEvent` | Policy version created/rolled back |

---

## 5.4 State Management Strategy

| State Type | Current Location | Target (Phase 7) | Migration Path |
|---|---|---|---|
| Alerts | `InMemoryAlertStore` (list) | PostgreSQL `alerts` table | Add `AlertRepository` with same interface, swap at init |
| Prevention actions | `OpsStore` (SQLite) | PostgreSQL `actions` table | Already supports PostgreSQL — just configuration |
| Audit logs | `OpsStore` (SQLite) | PostgreSQL `audit` table | Same as above |
| Rate limit windows | `RateLimiter` (dict) | Redis sorted sets | New `RedisRateLimiter` with same API |
| Risk frequency data | `RiskEngine._events_by_source` (dict) | Redis sorted sets | New `RedisFrequencyStore` |
| TI cache | — (doesn't exist) | Redis hash + PostgreSQL backup | New component, no migration |
| Policy config | Runtime object | PostgreSQL `policies` table | New component, backward compat via env vars |
| Escalation state | — (doesn't exist) | PostgreSQL `escalation` table | New component |
| Metrics counters | `MetricsService._counters` (dict) | Redis counters or keep in-memory with aggregation | Prometheus /metrics endpoint is already stateless scraping |

### Migration Safety
- Every new store implements the same interface as the in-memory version.
- Feature flag: `INIDS_STATE_BACKEND=local|redis|postgres`.
- Default remains `local` (full backward compatibility).
- Integration tests run against both backends.

---

## 5.5 Queue / Streaming Design

### Redis Streams Topology

```
Stream: inids:ingestion
  └→ Consumer Group: inids-detection-workers
       ├→ Worker-1 (reads, detects, writes to inids:detection)
       ├→ Worker-2
       └→ Worker-N

Stream: inids:detection
  └→ Consumer Group: inids-risk-workers
       └→ Worker-1 (enriches with TI, scores risk, writes to inids:risk)

Stream: inids:risk
  └→ Consumer Group: inids-policy-workers
       └→ Worker-1 (policy evaluation, writes to inids:decision)

Stream: inids:decision
  └→ Consumer Group: inids-action-workers
       └→ Worker-1 (execute actions, write to inids:action)

Stream: inids:action
  └→ Consumer Group: inids-audit-workers
       └→ Worker-1 (persist audit, export to SIEM)
```

### Key Design Decisions
- **Consumer groups** ensure each event is processed exactly once across workers in the same group.
- **Acknowledged processing**: Worker ACKs message after successful processing. Failed messages are retried via `XPENDING` + `XCLAIM`.
- **Dead letter**: After N retries, move to `inids:dead-letter` stream for manual investigation.
- **Trim policy**: `MAXLEN ~100000` on each stream to bound memory.
- **Serialization**: JSON (human-readable, debuggable). MessagePack as optional performance optimization later.

---

## 5.6 Fail-Safe and Rollback Thinking

| Failure Scenario | Fail-Safe Behavior | Rollback Strategy |
|---|---|---|
| ML model fails to load | System starts in **signature + anomaly only** mode. Health probe reports `degraded`. | Fix model artifact. Restart worker. Model hot-reload if file watcher detects new model. |
| Redis Stream unavailable | Fall back to **in-memory EventBus** (single-process mode). Log warning. | Redis recovery → workers auto-reconnect (Redis client retry). |
| PostgreSQL unavailable | Fall back to **SQLite** (already supported). Log warning. | Postgres recovery → reconnect. No data loss if WAL mode enabled. |
| TI feed unreachable | Use **stale cache** (last successful pull). Log warning. TI engine still contributes data from cache. | Feed recovery → next scheduled pull refreshes cache. |
| Bad policy pushed | Policy is **versioned**. Rollback to previous version via API or CLI. Audit trail records who pushed it. | `POST /api/policy/rollback?version=N` |
| Bad model promoted | Model registry has **version history**. Rollback to previous version. | `python -m src.train_cli promote --version N` |
| Action executor fails | Action marked as `FAILED` in OpsStore. Reconciliation loop retries. | Manual intervention via dashboard or API if auto-retry fails. |
| Worker crash | Consumer group **message pending list** retains unACKed messages. Another worker claims them after timeout. | Worker restart. Pending messages auto-redistributed. |
| Disk full | Streaming continues (Redis in-memory). Persistence writes fail gracefully with error logging. | Free disk. OpsStore resumes writing. No data loss for in-flight events. |

---

## 5.7 Production Readiness Checklist

### Scalability Readiness

| Item | Status After Phase 7 | Verification |
|---|---|---|
| Multi-worker API serving | ✅ Gunicorn with N workers, shared Redis state | Load test: 200 concurrent requests, <500ms p95 |
| Multi-worker pipeline processing | ✅ Redis Streams consumer groups | Throughput test: 1000 events/sec sustained |
| Stateless detection nodes | ✅ In-memory state externalized to Redis/PostgreSQL | Kill any worker, verify no state loss |
| No single-writer bottleneck | ✅ Multiple workers write to independent streams | Verify under load: no lock contention |
| Bounded memory per worker | ✅ No unbounded in-process collections | Memory profiling: <512MB per worker at 100K events/hour |

### Security Readiness

| Item | Status After Phase 7 | Verification |
|---|---|---|
| Authentication on all API endpoints | ✅ API key RBAC (viewer/analyst/admin) | Test: unauthenticated request → 401 |
| Input validation on all boundaries | ✅ JSON schema validation, CSV extension check, MAX_CONTENT_LENGTH | Fuzz test: random payloads don't crash |
| Rate limiting | ✅ Redis-backed sliding window | Test: exceed limit → 429 |
| Secrets management | ✅ Env vars, no hardcoded secrets, require SECRET_KEY in production | Config audit: no secrets in code/logs |
| CSP headers | ✅ Content-Security-Policy on all HTML responses | Header inspection |
| TLS | ✅ Reverse proxy (nginx/caddy) in deployment config | TLS scan |
| No command injection | ✅ `_validate_target_ip()` on all firewall commands | Test: inject `; rm -rf /` in IP field → rejected |
| Audit trail | ✅ Every policy/action change logged with actor + timestamp | Audit query: no untracked changes |

### Performance Readiness

| Item | Status After Phase 7 | Verification |
|---|---|---|
| Detection latency p95 < 500ms | ✅ Multi-engine parallel execution | Latency histogram |
| Throughput > 500 events/sec (single node) | ✅ Async pipeline with Redis Streams | Benchmark suite |
| Model inference < 50ms per prediction | ✅ Scikit-learn ensemble on preprocessed features | Model benchmark |
| No memory leaks | ✅ Bounded collections, LRU caches, stream trimming | 24-hour soak test |
| Graceful degradation under load | ✅ Backpressure controller, sampling mode | Overload test: 10x burst |

### Operability Readiness

| Item | Status After Phase 7 | Verification |
|---|---|---|
| Health checks (liveness + readiness + startup) | ✅ | Kubernetes probe test |
| Structured JSON logging | ✅ With correlation IDs | Log pipeline test |
| Prometheus /metrics endpoint | ✅ Counters + histograms + gauges | Grafana dashboard renders |
| Alerting on anomalies (via metrics) | ✅ Prometheus alerting rules | Alert fires on simulated failure |
| Runbook documentation | ✅ Existing + extended | Review |
| One-command deployment | ✅ `docker compose up` or `make deploy` | Fresh machine test |

### Maintainability Readiness

| Item | Status After Phase 7 | Verification |
|---|---|---|
| Test coverage > 80% | ✅ Unit + integration + e2e | `pytest --cov` |
| Type hints throughout | ✅ Existing coverage good, extend to new modules | `mypy --strict` |
| Modular architecture (no circular imports) | ✅ Clear dependency graph | Import analysis |
| Configuration via env vars (12-factor) | ✅ `settings.py` pattern | Deploy to different env with different config |
| Backward compatibility with single-process mode | ✅ Feature flags for streaming/HA | `INIDS_PIPELINE_MODE=local` runs identically to current |

### Integration Readiness

| Item | Status After Phase 7 | Verification |
|---|---|---|
| SIEM export (CEF/syslog) | ✅ | Send to syslog-ng, verify parse |
| Webhook/SOAR notifications | ✅ | Trigger webhook, verify payload delivery |
| REST API (versioned, documented) | ✅ /api/v1/ prefix | OpenAPI schema validation |
| Firewall adapters (mock + UFW + nftables + webhook) | ✅ | Test each adapter in isolation |
| External TI feed consumption | ✅ | Mock feed server, verify cache population |
| Multi-dataset training pipeline | ✅ | Train on CICIDS2017 + UNSW-NB15, evaluate |

---

## 5.8 What This System Is NOT (Scope Boundaries)

To keep the evolution grounded, these are explicitly **out of scope**:

| Out of Scope | Why |
|---|---|
| Inline wire-speed packet inspection | Requires kernel-level or DPDK integration. This is an application-layer IDS/IPS. |
| Full EDR endpoint agent | This is a network-focused system, not a host-based agent. |
| Commercial TI feed licensing | Use free/open feeds (OTX, AbuseIPDB community tier, Emerging Threats open). |
| Kubernetes operator / CRD | Over-engineering for current maturity. Docker Compose is the deployment target. K8s manifests are optional. |
| Real-time PCAP DPI at Gbps speed | Python is not the right language for wire-speed DPI. Zeek/Suricata handle that layer; INIDS consumes their output. |
| Multi-tenant SaaS architecture | Single-tenant deployment. Multi-tenancy adds massive complexity with little value for the current use case. |

---

## Summary

This evolution plan transforms INIDS from an **academic ML classifier demo** into a **production-valuable IDS/IPS platform** through 7 incremental phases. Each phase builds on the previous, the system remains runnable at every step, and backward compatibility is maintained through feature flags and interface preservation.

The most impactful changes, in order of urgency:

1. **Detection engine framework** — stops being a one-trick ML pony
2. **Streaming pipeline** — stops being a synchronous web app
3. **Multi-dataset training** — stops being an NSL-KDD-only toy
4. **Threat intelligence** — starts acting like a real security product
5. **Observability + policy management** — becomes operable
6. **HA + scalability** — becomes deployable at scale

The final system retains every line of currently working code as its foundation, but wraps it in the architectural patterns that industry IDS/IPS products require.
