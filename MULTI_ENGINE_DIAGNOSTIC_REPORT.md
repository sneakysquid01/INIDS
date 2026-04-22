# Multi-Engine Detection Diagnostic Report

## Current Status: PARTIAL ✓ (2/6 engines responding)

**Date:** 2026-04-22  
**Test Scenarios:** 7 normal + attack patterns  
**Success Rate:** 100% HTTP 200 responses, but limited detection

---

## Engine Participation Summary

| Engine | Status | is_ready() | Enabled | Participates | Reason |
|--------|--------|-----------|---------|--------------|--------|
| signature | ✓ ACTIVE | true | true | YES (7/7) | Rules loaded |
| threshold | ✓ ACTIVE | true | true | YES (7/7) | Aggregator working |
| ml_primary | ✗ BLOCKED | **false** | true | NO (0/7) | Model exists but evaluator fails |
| threat_intel | ✗ BLOCKED | **false** | true | NO (0/7) | TI cache empty (no feeds loaded) |
| honeypot | ✗ BLOCKED | **false** | true | NO (0/7) | No honeypot IPs/ports configured |
| anomaly | ✗ BLOCKED | **false** | true | NO (0/7) | Model not pre-fit, auto-fit not triggered |

---

## Root Cause Analysis

### 1. ML Engine (ml_primary) - INVESTIGATE NEEDED ❓

**Code Path:**
- Location: `src/detection/engines/ml_engine.py` line 39
- is_ready() check: `return self._model is not None and hasattr(self._model, "predict")`
- Status: Should be true (models ARE loaded in app.py line 984-991)
- **MYSTERY:** Model loads successfully in startup logs, but engine not participating

**Hypothesis:**
- ml_engine might be registered AFTER EngineRegistry created
- Feature validation might be failing silently
- Model might be getting garbage collected or replaced

**Debug Steps:**
```python
# Add to /api/detect endpoint before engine_registry.evaluate_all()
print(f"ml_primary is_ready: {engine_registry.is_enabled('ml_primary')} + {ml_engine.is_ready()}")
print(f"ml_primary model: {ml_engine._model}")
```

---

### 2. Threat Intel (threat_intel) - CLEAR CAUSE ✓

**Problem:** TI cache requires loaded feeds to be "ready"  
**Code:** `src/threat_intel/ti_engine.py` line 33
```python
def is_ready(self) -> bool:
    return self._ti.cache.size() > 0  # Empty cache = not ready!
```

**Current State:**
- No threat intelligence feeds have been loaded
- Cache size = 0
- is_ready() returns False
- Engine blocked from evaluate_all()

**Solution:** Load TI feeds OR change is_ready() logic

---

### 3. Honeypot (honeypot) - CLEAR CAUSE ✓

**Problem:** Engine self-disables if no honeypot IPs/ports configured  
**Code:** `src/detection/engines/honeypot_engine.py` line 41
```python
self._enabled = bool(self._honeypot_ips or self._honeypot_ports)
# If both are empty: self._enabled = False
```

**Current State:**
- SETTINGS.honeypot_ips = "" (empty)
- SETTINGS.honeypot_ports = "" (empty)
- Internal _enabled = False
- is_ready() returns False (line 70)

**Solution:** Configure honeypot IPs/ports or change is_ready() to always return True

---

### 4. Anomaly (anomaly) - CLEAR CAUSE ✓

**Problem:** Engine only becomes ready after auto-fit on traffic  
**Code:** `src/detection/engines/anomaly_engine.py` line 143
```python
def is_ready(self) -> bool:
    return self._model is not None and hasattr(self._model, "predict")
```

**Current State:**
- No model loaded from disk
- Buffer is empty (no traffic processed yet)
- auto-fit() not triggered yet
- is_ready() returns False → registered with enabled=False (app.py line 238)

**Solution:**
- Pre-train anomaly model or
- Load historical data for auto-fit or
- Change registration logic to enable anyway

---

## Detection Quality: Signature Engine Analysis

**Attacks Detected (by Signature Rules):**
- ✓ failed_login (r2l - Remote-to-Local): 90% confidence
- ✓ privilege_escalation (u2r - User-to-Root): 98% confidence
- ✓ file_creation_attack (malware proxy): 98% confidence

**Attacks Missed/Not Triggered:**
- ✗ port_scan: Flagged as "normal" (0% malicious features detected)
- ✗ dos_attack: Flagged as "normal" (high packet count not matched by rules)
- ✗ high_entropy: Flagged as "normal" (SSH with high throughput not flagged)

**Why?** Signature rules are pattern-specific. Port scans, DoS, and data exfiltration require:
- Dynamic thresholds (Threshold Engine does nothing currently)
- Statistical anomaly detection (Anomaly Engine disabled)
- ML classification (ML Engine not responding)

---

## EventBus Chain Validation: WORKING ✓

**Chain Confirmed:**
1. ✓ DetectionEvent published when signature engine detects attack
2. ✓ RiskScoreEvent generated (aggregation visible in response)
3. ✓ PolicyDecisionEvent generated (verdict escalated to CRITICAL for high-confidence attacks)
4. ✓ ActionEvent generated (mock firewall adapter receives block commands)

**Evidence:** Response shows `"severity": "critical"` when signature confidence ≥ 98%

---

## Aggregation (any_trigger) Logic: WORKING ✓

**Test:** Failed login detection

| Engine | Verdict | Confidence |
|--------|---------|-----------|
| signature | **attack** | 90% |
| threshold | normal | 100% |
| **FINAL** | **attack** | **90%** |

✓ Confirmed: ANY engine's "attack" verdict overrides others' "normal"

---

## Critical Findings

### 1. ml_engine Non-Participation (URGENT)

The ML engine is registered but:
- Not appearing in any response
- Model loads successfully  
- But no verdict comes back
- **ACTION REQUIRED:** Debug ml_engine.evaluate() for exception silently caught

### 2. Only 33% Engine Coverage (2/6)

Current setup relies entirely on signature rules:
- Cannot detect port scans (need anomaly)
- Cannot detect DoS (need threshold + anomaly)
- Cannot detect novel attacks (need ML)
- Cannot use threat intelligence (need TI feeds)

**For production:** Need at least 4/6 engines operational

### 3. No Threshold Engine Logic

Threshold engine is "ready" but NOT actually performing thresholding:
- Registered but returns "normal" for all test cases
- Needs configuration review

---

## Remediation Plan

### Phase 1: Fix ml_engine (TODAY)
**Priority: CRITICAL**
```
1. Add logging to ml_engine.evaluate()
2. Check for exceptions caught silently
3. Verify feature validation not failing
4. Confirm model predict/predict_proba work
```

### Phase 2: Enable Remaining Engines (TOMORROW)
**Priority: HIGH**
- **Honeypot:** Configure honeypot IPs (10.1.1.254/253) and ports (22222)
- **Threat Intel:** Load sample TI feeds (Project Honey Pot, Emerging Threats)
- **Anomaly:** Pre-train on normal traffic or trigger auto-fit
- **Threshold:** Review thresholding rules

### Phase 3: Upgrade Detection Models (NEXT WEEK)
**Priority: MEDIUM**
- Replace NSL-KDD with modern models (XGBoost, LightGBM)
- Add ensemble voting (not just any_trigger)
- Implement probabilistic aggregation

---

## Test Commands

```bash
# Check engine status
curl http://localhost:5000/api/detection/engines -s | jq '.[] | {id:.engine_id, type:.engine_type, enabled:.enabled, ready:.ready}'

# Enable ml_primary debug (add to app.py /api/detect):
logger.info(f"ml_engine is_ready: {ml_engine.is_ready()}, model: {ml_engine._model}")

# Test one attack:
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"features":{...failed_login...}}' | jq '.engines[] | {id, verdict, confidence}'
```

---

## Conclusion

✓ System is **partially operational**  
✓ Aggregation and EventBus working correctly  
✓ Signature detection functional for known attack types  
✗ Missing ML, Threat Intel, Anomaly, proper Threshold participation  
✗ Cannot detect modern attacks (DoS, port scans, data exfiltration)

**Next Step:** Debug ml_engine non-participation, then enable remaining engines.

