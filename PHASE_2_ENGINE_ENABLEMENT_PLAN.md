# Phase 2: Enable Remaining 3 Detection Engines

## Current Status
✓ **Working (3/6):** signature, threshold, ml_primary
✗ **Blocked (3/6):** anomaly, honeypot, threat_intel

---

## Phase 2A: Enable Threat Intelligence Engine

### Current Behavior
- Status: `enabled=True, ready=False`
- Root Cause: `is_ready()` checks `self._ti.cache.size() > 0` - cache is empty
- No TI feeds loaded at startup

### Solution: Load Sample TI Feeds
Files involved:
- `src/threat_intel/threat_intel_manager.py` - TI cache management
- `src/detection/engines/ti_engine.py` - is_ready() check

**Implementation Options:**

**Option A (Quick): Mock TI Feed at Startup**
- Create a few mock threat entries in TI cache during app initialization
- Example IPs: 192.168.1.50, 10.0.0.10 (malicious); 8.8.8.8 (clean)
- Example domains: badactor.com, malicious.net
- Location: Add to `web_app/app.py` during EngineRegistry initialization

**Option B (Better): Load Real TI Feed**
- Download sample from Project Honey Pot, Emerging Threats, or abuse.ch
- Parse CSV/JSON into cache
- Location: Add to `load_models()` or new `load_threat_intel()` function

### Expected Outcome
- Threat Intel engine will report `ready=True`
- Will participate in detections and flag known bad IPs/domains
- Adds external reputation scoring layer

### Test Command
```python
# After enabling TI, test with known bad IP
test_payload = {
    "features": {...all 39 NSL-KDD features...},
    "source": "192.168.1.50"  # Known malicious
}
# Expect: threat_intel engine returns "attack" verdict
```

---

## Phase 2B: Enable Honeypot Detection Engine

### Current Behavior
- Status: `enabled=True, ready=False`  
- Root Cause: `_enabled=False` (no honeypot IPs/ports configured)
- `is_ready()` checks: `return self._enabled` which is `bool(self._honeypot_ips or self._honeypot_ports)`

### Solution: Configure Honeypot IPs/Ports
Files involved:
- `src/detection/engines/honeypot_engine.py` - Configuration and detection
- `src/settings.py` - Settings configuration
- `web_app/app.py` - Engine initialization

**Implementation: Add Configuration**

1. **Update `src/settings.py`:**
```python
@property
def honeypot_ips(self) -> set:
    """Honeypot IP addresses to monitor"""
    ips_str = os.getenv('INIDS_HONEYPOT_IPS', '10.1.1.254,10.1.1.253')
    return set(ip.strip() for ip in ips_str.split(',') if ip.strip())

@property
def honeypot_ports(self) -> set:
    """Honeypot ports to monitor"""
    ports_str = os.getenv('INIDS_HONEYPOT_PORTS', '22,23,3389,8080')
    ports = set()
    for p in ports_str.split(','):
        p = p.strip()
        if p.isdigit():
            ports.add(int(p))
    return ports
```

2. **Test with Environment Variables:**
```bash
$env:INIDS_HONEYPOT_IPS = "10.1.1.254,10.1.1.253"
$env:INIDS_HONEYPOT_PORTS = "22,23,3389,8080"
python start_flask_dev.py
```

3. **Update Engine Initialization in app.py:**
```python
honeypot_engine = HoneypotDetectionEngine(
    honeypot_ips=SETTINGS.honeypot_ips,
    honeypot_ports=SETTINGS.honeypot_ports
)
engine_registry.register(honeypot_engine, enabled=True)
```

### Expected Outcome
- Honeypot engine will report `ready=True`
- Will detect connections to honeypot IPs/ports as attacks
- Useful for early attack detection (attackers often probe known services)

### Test Command
```python
test_payload = {
    "features": {...all 39 NSL-KDD features...},
    "destination": "10.1.1.254",
    "destination_port": 22  # SSH on honeypot
}
# Expect: honeypot engine returns "attack" verdict with high confidence
```

---

## Phase 2C: Enable Anomaly Detection Engine

### Current Behavior
- Status: `enabled=False, ready=False`
- Root Cause: `is_ready()` checks `self._model is not None and hasattr(self._model, 'predict')`
- Model not pre-trained, auto-fit not configured

### Solution: Pre-Train or Auto-Fit Anomaly Model
Files involved:
- `src/detection/engines/anomaly_engine.py` - Model management and is_ready() check
- `models/anomaly_engine.pkl` - Pre-trained model (may not exist)

**Implementation Option A: Load Pre-Trained Model**
1. Check if `models/anomaly_engine.pkl` exists (created during setup)
2. If exists: Load via joblib during engine initialization
3. If not: Skip or use Option B

**Implementation Option B: Auto-Fit on First Training Data**
1. Buffer detections from normal traffic (first N samples)
2. Fit Isolation Forest or Local Outlier Factor model on normal data
3. Threshold set to detect top 5% as anomalies

**Implementation Option C (Quick): Mock Pre-Fit Model**
1. Create simple fitted Isolation Forest model at startup
2. Fit on synthetic normal traffic data (10 samples)
3. Save to `models/anomaly_engine.pkl`
4. Load at startup

### Recommended: Option B (Auto-fit from baseline)
- Create 30-second baseline collection period
- Collect normal traffic patterns
- Fit Isolation Forest on baseline
- Mark as ready after baseline collection

### Expected Outcome
- Anomaly engine will report `ready=True`
- Will detect statistical outliers in network behavior
- Complements signature/ML detection with unsupervised approach

### Test Command
```python
# Normal traffic (should be in baseline)
test_payload = {
    "features": {...standard HTTP traffic...},
    "source": "192.168.1.100"
}
# Expect: anomaly engine returns "normal" (similar to baseline)

# Anomalous traffic (unusual stats)
test_payload = {
    "features": {...extreme values: src_bytes=999999, dst_bytes=1...},
    "source": "192.168.1.200"
}
# Expect: anomaly engine returns "attack" (statistical outlier)
```

---

## Implementation Sequence

### Step 1: Threat Intelligence (15 minutes)
```bash
# Add 3-5 mock TI entries to cache
# Verify TI engine reports ready=True
# Re-run test suite
```

### Step 2: Honeypot Engine (10 minutes)
```bash
# Update settings.py with configuration
# Set env vars for honeypot IPs/ports
# Restart Flask
# Verify honeypot engine reports ready=True
# Re-run test suite
```

### Step 3: Anomaly Engine (20 minutes)
```bash
# Implement auto-fit baseline collection
# Wait 30 seconds for baseline
# Verify anomaly engine reports ready=True
# Re-run test suite
```

### Total Time Estimate: 45-60 minutes

---

## Success Criteria

After all 3 phases complete:
- [ ] 6/6 engines registered
- [ ] 6/6 engines reporting ready=True
- [ ] 6/6 engines participating in every /api/detect call
- [ ] Multi-engine test suite shows all 6 engines in each scenario
- [ ] Cross-validation: ensure different engines can identify attacks other engines miss

---

## Expected Final Test Results
```
Scenario: port_scan
Verdicts: 
  - signature: normal  (rule doesn't match)
  - threshold: normal  (traffic volume OK)
  - ml_primary: attack (ML trained on NSL-KDD port scan patterns)
  - threat_intel: normal (IPs not in threat database)
  - honeypot: attack   (destination port open on honeypot)
  - anomaly: attack    (unusual packet size distribution)
Final Verdict: ATTACK (any_trigger aggregation)
```

---

## Files to Create/Modify
- `src/settings.py` - Add honeypot IP/port configuration
- `web_app/app.py` - Load TI feeds, configure honeypot engine, enable anomaly baseline
- `src/detection/engines/ti_engine.py` - Ensure is_ready() works correctly
- `src/detection/engines/anomaly_engine.py` - Implement baseline auto-fit

## Optional: Model Upgrade Track
After all 6 engines working, implement the user's secondary request:
- [ ] Replace RandomForest (NSL-KDD) with XGBoost on modern dataset (CIC-IDS2017)
- [ ] Maintain backward compatibility with NSL-KDD feature schema
- [ ] A/B test: NSL-KDD vs XGBoost/LightGBM on real attacks
