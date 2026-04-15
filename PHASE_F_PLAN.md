# Phase F: Advanced Features - Implementation Plan

## Overview

Phase F adds 15+ advanced detection and enrichment features to INIDS, transforming it from a basic signature detector into a production-grade threat detection platform. Phase F builds on the optimized pipeline from Phases A-E.

**Scope**: 4-6 weeks, ~8,000-10,000 production lines  
**Target Users**: Enterprise security operations, threat researchers  
**Integration**: Seamless with Phases A-E pipeline

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase F: Advanced Features                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────┐      │
│  │ Part 1: GeoIP Lookup    │  │ Part 2: DNS Detection   │      │
│  │ • MaxMind/IP2Location   │  │ • Sinkhole Detection    │      │
│  │ • AS Number Lookup      │  │ • Tunneling Detection   │      │
│  │ • VPN/Proxy Detection   │  │ • DGA Detection         │      │
│  │ • Caching Layer         │  │ • DNS Policy Check      │      │
│  └─────────────────────────┘  └─────────────────────────┘      │
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────┐      │
│  │ Part 3: TLS Validation  │  │ Part 4: HTTP Patterns   │      │
│  │ • Cert Validation Chain │  │ • Signature Matching    │      │
│  │ • Expired Cert Check    │  │ • Header Analysis       │      │
│  │ • Pinning Validation    │  │ • Body Pattern Matching │      │
│  │ • OCSP Validation       │  │ • Content-Type Check    │      │
│  └─────────────────────────┘  └─────────────────────────┘      │
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ Part 5: ML & Advanced Features                      │        │
│  │ • Anomaly Detection (statistical)                   │        │
│  │ • Behavioral Profiling                              │        │
│  │ • Time-Series Analysis                              │        │
│  │ • Ensemble Learning                                 │        │
│  │ • Custom Rule Engine                                │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────┐
        │ EVE JSON Output (Phase D)         │
        │ With enriched metadata            │
        └──────────────────────────────────┘
```

---

## Part 1: GeoIP Enrichment

### Purpose
Enrich network events with geographic and ASN data for location-based detection and investigation.

### Features

1. **Geographic Lookup**
   - Country/region of IP source/destination
   - City-level geolocation
   - Timezone information
   - Coordinates (lat/long)

2. **ASN Lookup**
   - Autonomous System Number
   - AS name/organization
   - IPv4/IPv6 prefix

3. **Risk Detection**
   - VPN/Proxy detection
   - Datacenter/hosting detection
   - Tor exit node detection
   - Known malicious IP lists

4. **Caching**
   - LRU cache for recent lookups
   - Database pooling
   - Background updates

### Implementation Files

- `src/advanced/geoip_enrichment.py` (~1,200 lines)
  - GeoIPLookup class
  - GeoIPCache with eviction
  - ASNLookup integration
  - RiskDetector for VPN/proxy/Tor
  - Integration with Phase D (EVE JSON)

### Expected Features

```python
from src.advanced import GeoIPLookup

lookup = GeoIPLookup(database_path="/path/to/geoip.db")

# Lookup IP
geoip = lookup.lookup("203.0.113.42")
# Returns:
# {
#     "country": "US",
#     "city": "Mountain View",
#     "latitude": 37.4192,
#     "longitude": -122.0574,
#     "asn": "AS15169",
#     "org": "Google Inc.",
#     "timezone": "America/Los_Angeles",
#     "is_vpn": False,
#     "is_datacenter": True,
#     "is_tor": False
# }

# Add to EVE event
eve_event["geoip_source"] = geoip
```

### Dependencies

- Optional: `geoip2` (MaxMind), `ip2location` (IP2Location)
- Fallback: Free GeoIP databases (GeoLite2)
- ASN: RIPE NCC, ISC, or free ASN database

---

## Part 2: DNS Detection

### Purpose
Detect DNS-based attacks (sinkhole redirection, DGA, tunneling, policy violations).

### Features

1. **Sinkhole Detection**
   - Detect common sinkhole IP ranges
   - Compare DNS response to sinkhole list
   - Known malware C2 sinkholes

2. **Domain Generation Algorithm (DGA) Detection**
   - Domain entropy analysis
   - Pattern-based detection
   - Known DGA patterns
   - Length-based heuristics

3. **DNS Tunneling Detection**
   - Query rate analysis
   - Domain length analysis
   - Subdomain enumeration detection
   - Base64/hex encoding in domains

4. **DNS Policy Enforcement**
   - Blocklist/allowlist checking
   - RPZ (Response Policy Zone) rules
   - Suspicious TLD detection
   - Domain whitelist bypass detection

### Implementation Files

- `src/advanced/dns_detection.py` (~1,500 lines)
  - SinkholeDetector class
  - DGADetector with entropy analysis
  - DNSTunnelingDetector
  - PolicyEnforcer with RPZ support
  - Integration with Phase B (DNS parser)

### Expected Features

```python
from src.advanced import DNSDetector

detector = DNSDetector(
    sinkhole_ips=sinkhole_list,
    dga_threshold=4.0,
    policy_rules=rpz_rules
)

# Analyze DNS query
result = detector.analyze_query(dns_packet, response)
# Returns:
# {
#     "is_sinkhole": False,
#     "dga_score": 3.8,  # 0-5 scale, >4.0 is likely DGA
#     "is_tunneling": False,
#     "policy_violations": [],
#     "anomaly_score": 2.1
# }

# Add to EVE event
eve_event["dns_analysis"] = result
```

---

## Part 3: TLS Certificate Validation

### Purpose
Detect TLS/SSL attacks (self-signed certs, expired certs, certificate pinning violations, OCSP failures).

### Features

1. **Certificate Chain Validation**
   - Verify certificate chain from root CA
   - Check certificate validity dates
   - Detect self-signed certificates
   - Validate signature algorithms

2. **Certificate Pinning**
   - Check pinned certificates (HPKP)
   - Detect pin violations
   - Public key pinning validation

3. **OCSP Validation**
   - OCSP stapling verification
   - Revocation status checking
   - OCSP timeout handling

4. **Certificate Anomalies**
   - Subject/Issuer mismatches
   - Unusual certificate sizes
   - Weak encryption algorithms
   - Known bad certificates

### Implementation Files

- `src/advanced/tls_validation.py` (~1,000 lines)
  - CertificateValidator class
  - CertificatePinningValidator
  - OCSPValidator
  - AnomalyDetector for certificates
  - Integration with Phase B (SSL/TLS parser)

### Expected Features

```python
from src.advanced import CertificateValidator

validator = CertificateValidator(
    ca_bundle_path="/path/to/ca-bundle.crt",
    pinning_db=pinning_database,
    check_ocsp=True
)

# Validate certificate
result = validator.validate(cert_der, hostname)
# Returns:
# {
#     "valid": False,
#     "errors": ["expired", "self_signed"],
#     "chain_valid": False,
#     "pinning_violation": False,
#     "ocsp_status": "revoked",
#     "confidence": 0.95
# }

# Add to EVE event
eve_event["tls_validation"] = result
```

---

## Part 4: HTTP Signature Patterns

### Purpose
Match HTTP requests against signature database for known attack patterns and malware indicators.

### Features

1. **Signature Matching**
   - Header signature matching
   - Request body pattern matching
   - URL pattern matching
   - HTTP method anomalies

2. **HTTP Anomalies**
   - Unusual header combinations
   - Content-Type mismatches
   - Encoding detection
   - Suspicious User-Agent strings

3. **Malware Indicators**
   - Known malware user agents
   - C&C communication patterns
   - Exploit kit signatures
   - SQLi/XSS/RFI patterns

4. **Bot/Scanner Detection**
   - Automated scanner signatures
   - Web crawler detection
   - Security tool identification

### Implementation Files

- `src/advanced/http_patterns.py` (~1,200 lines)
  - SignatureDatabase class
  - SignatureMatchEngine
  - PatternAnalyzer for anomalies
  - BotDetector
  - Integration with Phase B (HTTP parser)

### Expected Features

```python
from src.advanced import HTTPPatternDetector

detector = HTTPPatternDetector(
    signatures=signature_db,
    enable_anomaly_detection=True
)

# Analyze HTTP request
result = detector.analyze_request(http_request, response)
# Returns:
# {
#     "signatures_matched": ["sql_injection_attempt", "malware_ua"],
#     "confidence": 0.92,
#     "severity": "HIGH",
#     "anomaly_scores": {
#         "headers": 3.2,
#         "body": 4.5,
#         "url": 2.1
#     },
#     "bot_detected": False
# }

# Add to EVE event
eve_event["http_analysis"] = result
```

---

## Part 5: Machine Learning & Advanced Features

### Purpose
Enable statistical anomaly detection and behavioral analysis.

### Features

1. **Statistical Anomaly Detection**
   - Baseline learning (normal traffic)
   - Deviation detection
   - Time-series analysis
   - Seasonal adjustment

2. **Behavioral Profiling**
   - Per-host profiles
   - Protocol behavior analysis
   - Port usage patterns
   - DNS query patterns

3. **Ensemble Methods**
   - Combine multiple detectors
   - Confidence scoring
   - False positive reduction
   - Weighted voting

4. **Custom Rule Engine**
   - YARA-style rules
   - Lua scripting support
   - Complex correlation rules
   - Temporal rules

### Implementation Files

- `src/advanced/ml_features.py` (~1,500 lines)
  - BaselineBuilder for learning normal behavior
  - AnomalyDetector with statistical methods
  - BehavioralProfiler per host/flow
  - EnsembleClassifier combining detectors
  - RuleEngine for custom rules
  - Integration with monitoring from Phase E

### Expected Features

```python
from src.advanced import AnomalyDetector, BehavioralProfiler

# Build baseline from normal traffic
builder = BaselineBuilder()
for event in normal_traffic:
    builder.add_event(event)
baseline = builder.build()

# Create detectors
anomaly = AnomalyDetector(baseline, threshold=3.0)
profiler = BehavioralProfiler()

# Analyze event
for event in live_traffic:
    profiler.update(event)
    
    anomaly_score = anomaly.score(event)
    behavior_score = profiler.score(event)
    
    if anomaly_score > 3.0 or behavior_score > 4.0:
        # Add to EVE event
        event["anomaly_score"] = anomaly_score
        event["behavior_score"] = behavior_score
```

---

## Integration Timeline

### Week 1: Foundation
- Part 1: GeoIP Enrichment
- Create advanced module structure
- Setup database management
- Create caching layer

### Week 2: Detection Engines
- Part 2: DNS Detection
- Part 3: TLS Validation
- Create signature database
- Implement pattern matching

### Week 3: HTTP & Patterns
- Part 4: HTTP Signatures
- Anomaly scoring
- Bot detection
- Testing with real traffic

### Week 4: ML & Advanced
- Part 5: ML Features
- Statistical baselines
- Behavioral profiling
- Ensemble classification

### Week 5: Integration & Testing
- Full system integration
- Performance testing
- Validation against known attack patterns
- Benchmarking with Phase E monitoring

### Week 6: Documentation & Hardening
- API documentation
- Deployment guide
- Configuration templates
- Database management guide

---

## File Structure

```
src/advanced/
  __init__.py                  # Module exports
  
  geoip_enrichment.py         # Part 1 (1,200 lines)
  dns_detection.py            # Part 2 (1,500 lines)
  tls_validation.py           # Part 3 (1,000 lines)
  http_patterns.py            # Part 4 (1,200 lines)
  ml_features.py              # Part 5 (1,500 lines)
  
  config.py                   # Config management
  database.py                 # Database abstraction
  caching.py                  # Caching layer
  
tests/
  test_phase_f_geoip.py       # Part 1 tests (300 lines)
  test_phase_f_dns.py         # Part 2 tests (400 lines)
  test_phase_f_tls.py         # Part 3 tests (300 lines)
  test_phase_f_http.py        # Part 4 tests (400 lines)
  test_phase_f_ml.py          # Part 5 tests (500 lines)

validate_phase_f.py            # Comprehensive validation

databases/
  geoip/
    geoip.db               # GeoIP database (external)
  dns/
    sinkholes.txt         # Known sinkhole IPs
    dga_patterns.db       # DGA pattern database
  http/
    signatures.db         # HTTP signature database
  ml/
    baseline.pkl          # Learned baseline (state)
```

---

## Key Design Decisions

### 1. Modular Architecture
- Each feature in separate module
- Minimal coupling between parts
- Can enable/disable independently
- Staged rollout possible

### 2. Performance First
- Use pool manager from Phase E
- Connection pooling for external lookups
- Lazy loading of databases
- Incremental processing

### 3. Confidence Scoring
- All detectors output confidence 0-1
- Ensemble combines scores
- Configurable thresholds
- False positive reduction

### 4. Flexible Databases
- Support multiple GeoIP providers
- Custom database formats
- External enrichment APIs
- Offline fallback support

### 5. Graceful Degradation
- If database missing: skip enrichment
- If lookup fails: use cached data
- If validator error: pass through
- System continues functioning

---

## Testing Strategy

### Unit Tests
- Each component in isolation
- Mocked external dependencies
- Known test cases (malware samples)

### Integration Tests
- Components working together
- Full pipeline with Phase A-E
- End-to-end with real packets
- Performance benchmarking

### Validation Tests
- Against known attack patterns
- False positive rates
- Detection accuracy
- Latency impact

---

## Deployment Checklist

Before Phase F goes to production:

- [ ] All databases downloaded and validated
- [ ] Caching layer performance tested
- [ ] External API connections configured
- [ ] Confidence thresholds tuned
- [ ] Ensemble weights optimized
- [ ] Database update strategy defined
- [ ] Fallback mechanisms tested
- [ ] Logging enabled for all components
- [ ] Alerts configured for failures
- [ ] Documentation reviewed

---

## Success Criteria

### Functionality
- ✅ 15+ advanced features implemented
- ✅ All integrated with Phases A-E
- ✅ All tested with real attack patterns
- ✅ Database updates automated

### Performance
- ✅ <5ms overhead per event
- ✅ Maintains 100K+ pps throughput
- ✅ Cache hit rate >90%
- ✅ Memory usage <500MB

### Reliability
- ✅ 99%+ detection accuracy on known attacks
- ✅ <1% false positive rate
- ✅ Graceful degradation on failures
- ✅ Automatic database updates

### Operability
- ✅ Easy configuration
- ✅ Clear logging/debugging
- ✅ Configurable thresholds
- ✅ Monitoring integration

---

## Next Steps

Phase F Part 1: GeoIP Enrichment implementation starting now.

Expected delivery: Full Phase F in 4-6 weeks with ~8,000-10,000 production lines.
