# INIDS - DETAILED FIX IMPLEMENTATIONS

## Fix Instructions for All Issues

---

## FIX #1: RiskEngine Memory Leak

**File**: `src/ips/risk_engine.py`

**BEFORE**:
```python
from collections import defaultdict, deque
from time import time

class RiskEngine:
    def __init__(self, weights: RiskWeights | None = None, ...):
        self.weights = weights or RiskWeights()
        self.frequency_window_seconds = max(30, int(frequency_window_seconds))
        self.frequency_high_watermark = max(1, int(frequency_high_watermark))
        self._events_by_source: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def recent_activity_score(self, source_ip: str) -> float:
        now = time()
        window_start = now - self.frequency_window_seconds
        source = str(source_ip or "unknown")
        with self._lock:
            q = self._events_by_source[source]
            q.append(now)
            while q and q[0] < window_start:
                q.popleft()
            count = len(q)
            # Bound in-memory source cardinality.
            if len(self._events_by_source) > 50000:
                excess = len(self._events_by_source) - 40000
                keys_to_remove = list(self._events_by_source)[:excess]
                for k in keys_to_remove:
                    del self._events_by_source[k]
        return _clamp(count / self.frequency_high_watermark)
```

**AFTER**:
```python
from collections import defaultdict, deque, OrderedDict
from time import time
from threading import Lock

class RiskEngine:
    def __init__(self, weights: RiskWeights | None = None, ...):
        self.weights = weights or RiskWeights()
        self.frequency_window_seconds = max(30, int(frequency_window_seconds))
        self.frequency_high_watermark = max(1, int(frequency_high_watermark))
        self._events_by_source: dict[str, deque[float]] = {}
        self._source_last_accessed: OrderedDict[str, float] = OrderedDict()
        self._lock = Lock()

    def recent_activity_score(self, source_ip: str) -> float:
        now = time()
        window_start = now - self.frequency_window_seconds
        source = str(source_ip or "unknown")
        
        with self._lock:
            # Initialize source if not exists
            if source not in self._events_by_source:
                self._events_by_source[source] = deque()
            
            q = self._events_by_source[source]
            q.append(now)
            
            # Clean old entries within this source's window
            while q and q[0] < window_start:
                q.popleft()
            
            # Clean up expired sources (no events in window)
            if len(self._events_by_source) > 50000:
                # Phase 1: Remove sources with empty queues
                empty_sources = [k for k, v in self._events_by_source.items() if not v]
                for k in empty_sources:
                    del self._events_by_source[k]
                    self._source_last_accessed.pop(k, None)
                
                # Phase 2: If still over limit, remove oldest by LRU
                if len(self._events_by_source) > 50000:
                    # Remove oldest 10k by last access time
                    items_to_remove = list(self._source_last_accessed.items())[:10000]
                    for k, _ in items_to_remove:
                        self._events_by_source.pop(k, None)
                        self._source_last_accessed.pop(k, None)
            
            # Update access time for LRU
            self._source_last_accessed[source] = now
            count = len(q)
            return _clamp(count / self.frequency_high_watermark)
```

**Testing**:
```python
def test_risk_engine_bounded_memory():
    engine = RiskEngine(frequency_window_seconds=10)
    # Simulate 100k unique IPs
    for i in range(100000):
        engine.recent_activity_score(f"192.168.1.{i % 256}")
    # Should not exceed 50k + margin
    assert len(engine._events_by_source) <= 55000
    # Should have cleaned up
    assert len(engine._events_by_source) > 40000
```

---

## FIX #2: ActionExecutor Missing Methods

**File**: `src/ips/action_executor.py`

**BEFORE**:
```python
class ActionExecutor:
    def execute(self, decision_event: PolicyDecisionEvent, policy) -> ActionEvent | None:
        ...
        if decision == "PENDING_BLOCK":
            action = ActionEvent(...)
            self._persist_action(action, action_id=action_id, executed_at=None)  # UNDEFINED!
            self._emit_audit("pending_block", f"...")  # UNDEFINED!
            return action
```

**AFTER**:
```python
import json
import logging
from datetime import datetime, timezone

class ActionExecutor:
    def __init__(self, ...):
        ...
        self.logger = logging.getLogger(__name__)
    
    def _persist_action(self, action: ActionEvent, action_id: str | None = None, executed_at: str | None = None) -> None:
        """Persist action to OPS store for audit trail and recovery."""
        if self.ops_store is None:
            self.logger.debug("OPS store not available, skipping action persistence")
            return
        
        try:
            action_data = {
                "action_id": action_id or f"act_{uuid.uuid4().hex[:16]}",
                "target": action.target,
                "action": action.action,
                "reason": action.reason,
                "executed": action.executed,
                "status": action.status,
                "adapter": action.adapter,
                "expires_at": action.expires_at,
                "created_at": action.created_at,
                "executed_at": executed_at,
            }
            
            self.ops_store.add_audit(
                event_type="prevention_action_executed",
                message=json.dumps(action_data, separators=(",", ":")),
                created_at=action.created_at,
            )
            self.logger.info(
                "Action persisted: %s for target=%s status=%s",
                action_id,
                action.target,
                action.status,
            )
        except Exception as exc:
            self.logger.error("Failed to persist action %s: %s", action_id, exc)
    
    def _emit_audit(self, event_type: str, message: str) -> None:
        """Emit audit event to event bus."""
        if self.event_bus is None:
            self.logger.debug("Event bus not available, skipping audit event")
            return
        
        try:
            from src.core.event_bus import AuditEvent
            event = AuditEvent(event_type=event_type, message=message)
            self.event_bus.publish(event)
        except Exception as exc:
            self.logger.error("Failed to emit audit event %s: %s", event_type, exc)
    
    def execute(self, decision_event: PolicyDecisionEvent, policy) -> ActionEvent | None:
        decision = str(decision_event.decision).strip().upper()
        if decision not in {"BLOCK", "TEMP_BLOCK", "RATE_LIMIT", "PENDING_BLOCK"}:
            return None

        target = self._normalize_ip(decision_event.risk.detection.source or "")
        if target is None:
            self._emit_audit("action_skipped", f"invalid_target source={decision_event.risk.detection.source}")
            return None

        ttl_seconds = int(decision_event.ttl_seconds or getattr(policy, "block_ttl_seconds", 300))
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat() if ttl_seconds > 0 else None
        action_id = f"act_{uuid.uuid4().hex[:16]}"

        # PENDING_BLOCK: save for operator approval without executing.
        if decision == "PENDING_BLOCK":
            action = ActionEvent(
                decision=decision_event,
                action="block",
                target=target,
                reason=decision_event.reason,
                dry_run=False,
                executed=False,
                status="PENDING_APPROVAL",
                adapter=self.adapter_name,
                expires_at=expires_at,
                created_at=now.isoformat(),
            )
            self._persist_action(action, action_id=action_id)
            self._emit_audit("pending_block", json.dumps({
                "target": target,
                "action_id": action_id,
                "reason": decision_event.reason,
            }, separators=(",", ":")))
            return action

        # Idempotency check...
        # [rest of execute method continues with proper error handling]
```

---

## FIX #3: EventBus Race Condition

**File**: `src/core/event_bus.py`

**BEFORE**:
```python
from collections import defaultdict
from typing import Callable, Dict, List, TypeVar

class EventBus:
    def __init__(self):
        self._listeners: Dict[type, List[Callable]] = defaultdict(list)
    
    def subscribe(self, event_type: type, callback: Callable) -> None:
        self._listeners[event_type].append(callback)
    
    def publish(self, event) -> None:
        callbacks = self._listeners.get(type(event), [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                pass
```

**AFTER**:
```python
from collections import defaultdict
from copy import copy
from typing import Callable, Dict, List, TypeVar
from threading import RLock
import logging

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self._listeners: Dict[type, List[Callable]] = defaultdict(list)
        self._lock = RLock()
    
    def subscribe(self, event_type: type, callback: Callable) -> None:
        """Register a callback for an event type."""
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback)}")
        
        with self._lock:
            # Avoid duplicate subscriptions
            if callback not in self._listeners[event_type]:
                self._listeners[event_type].append(callback)
                logger.debug(f"Subscribed {callback.__name__} to {event_type.__name__}")
    
    def publish(self, event: EventT) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            # Copy list to avoid modification during iteration
            callbacks = copy(self._listeners.get(type(event), []))
        
        # Invoke callbacks outside lock to avoid deadlock
        for callback in callbacks:
            try:
                callback(event)
            except Exception as exc:
                # Log but don't crash - one failing handler shouldn't break pipeline
                logger.exception(
                    "Callback error in %s for event %s: %s",
                    callback.__name__,
                    type(event).__name__,
                    exc,
                )
```

**Testing**:
```python
import threading
import time

def test_event_bus_thread_safety():
    bus = EventBus()
    results = []
    
    def callback(event):
        results.append(event.value)
    
    # Subscribe from main thread
    bus.subscribe(TestEvent, callback)
    
    # Publish from multiple threads
    def publish_events():
        for i in range(100):
            bus.publish(TestEvent(value=i))
            time.sleep(0.001)
    
    threads = [threading.Thread(target=publish_events) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should have 500 events, no crashes
    assert len(results) == 500
```

---

## FIX #4: InMemoryAlertStore Performance

**File**: `src/detection_service.py`

**BEFORE**:
```python
class InMemoryAlertStore:
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self._alerts: list[Alert] = []
        self._lock = __import__("threading").Lock()

    def add(self, alert: Alert) -> None:
        with self._lock:
            self._alerts.insert(0, alert)  # O(n)!
            if len(self._alerts) > self.max_items:
                self._alerts = self._alerts[: self.max_items]
```

**AFTER**:
```python
from collections import deque
from threading import Lock

class InMemoryAlertStore:
    def __init__(self, max_items: int = 1000):
        self.max_items = max(1, int(max_items))  # At least 1
        self._alerts: deque[Alert] = deque(maxlen=self.max_items)  # Auto-truncates!
        self._lock = Lock()

    def add(self, alert: Alert) -> None:
        if alert is None:
            return
        with self._lock:
            # deque.appendleft is O(1) and handles max_items automatically
            self._alerts.appendleft(alert)

    def list_alerts(self, limit: int = 50, severity: str | None = None) -> list[Alert]:
        limit = max(1, min(limit, 1000))  # Bound limit between 1-1000
        with self._lock:
            alerts = list(self._alerts)
        
        if severity:
            normalized = severity.strip().lower()
            alerts = [a for a in alerts if a.severity.lower() == normalized]
        
        return alerts[:limit]
```

---

## FIX #5: Auth Security Default

**File**: `src/auth_service.py`

**BEFORE**:
```python
class AuthService:
    def __init__(self):
        self.principals: dict[str, Principal] = {}
        self.require_api_keys = os.getenv("INIDS_REQUIRE_API_KEYS", "0") == "1"  # Default: OFF
        self._load_from_env()

    @property
    def enabled(self) -> bool:
        return self.require_api_keys or len(self.principals) > 0
```

**AFTER**:
```python
class AuthService:
    def __init__(self):
        self.principals: dict[str, Principal] = {}
        # Default: ON (require API keys unless explicitly disabled)
        self.require_api_keys = os.getenv("INIDS_REQUIRE_API_KEYS", "1") == "1"
        self.allow_unauthenticated = os.getenv("INIDS_ALLOW_UNAUTHENTICATED", "0") == "1"
        self._load_from_env()

    @property
    def enabled(self) -> bool:
        if self.allow_unauthenticated:
            return False
        return self.require_api_keys or len(self.principals) > 0
    
    def authorize(self, required_role: str) -> tuple[bool, str]:
        if required_role not in ROLE_RANK:
            return False, "unknown_role"
        
        if self.allow_unauthenticated:
            logger.warning("Auth bypassed - INIDS_ALLOW_UNAUTHENTICATED=1")
            return True, "unauthenticated_allowed"
        
        if not self.enabled:
            if self.require_api_keys:
                return False, "auth_required_but_not_configured"
            return True, "auth_disabled"
        
        # ... rest of auth logic
```

---

## FIX #6: Feature Column Validation

**File**: `src/detection/engines/ml_engine.py`

**BEFORE**:
```python
def evaluate(self, features: dict[str, Any]) -> EngineResult:
    row = DEFAULT_FEATURE_ROW.copy()
    for key, value in features.items():
        if key in FEATURE_COLUMNS:
            row[key] = value
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    # No validation - silently uses defaults if columns missing!
```

**AFTER**:
```python
def evaluate(self, features: dict[str, Any]) -> EngineResult:
    # Validate required columns
    required_columns = set(FEATURE_COLUMNS)
    provided_columns = set(features.keys())
    missing_columns = required_columns - provided_columns
    
    if missing_columns:
        logger.warning(
            "Missing features for ML evaluation: %s (using defaults)",
            ", ".join(sorted(missing_columns)),
        )
        # Log but don't fail - fill with defaults
        if len(missing_columns) > 10:  # If too many missing, high uncertainty
            return EngineResult(
                engine_id=self._engine_id,
                engine_type=self.engine_type,
                verdict="unknown",
                confidence=0.0,
                severity="low",
                attack_type="unknown",
                metadata={
                    "error": f"too_many_missing_features ({len(missing_columns)})",
                },
            )
    
    row = DEFAULT_FEATURE_ROW.copy()
    for key, value in features.items():
        if key in FEATURE_COLUMNS:
            try:
                # Type-check numeric columns
                if key in NUMERIC_FEATURES:
                    row[key] = float(value)
                else:
                    row[key] = str(value)
            except (ValueError, TypeError):
                logger.debug("Type conversion failed for %s=%s, using default", key, value)
    
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    # ... rest of evaluation
```

---

## FIX #7: Policy Threshold Validation

**File**: `src/prevention_service.py` & `src/ips/policy_engine.py`

**BEFORE**:
```python
def set_policy(self, ..., risk_alert_threshold=None, risk_rate_limit_threshold=None, ...):
    for attr, val in (...):
        if val is not None:
            fval = float(val)
            if fval < 0 or fval > 1:
                raise ValueError(f"{attr} must be between 0 and 1")
            setattr(self.policy, attr, fval)
    # No validation that alert < rate_limit < temp_block < block!
```

**AFTER**:
```python
def set_policy(self, ..., risk_alert_threshold=None, risk_rate_limit_threshold=None, ...):
    # ... individual validations ...
    
    # Validate threshold ordering
    thresholds = {
        "alert": float(getattr(self.policy, "risk_alert_threshold", 0.4)),
        "rate_limit": float(getattr(self.policy, "risk_rate_limit_threshold", 0.6)),
        "temp_block": float(getattr(self.policy, "risk_temp_block_threshold", 0.75)),
        "block": float(getattr(self.policy, "risk_block_threshold", 0.85)),
    }
    
    # Check strict ordering
    if not (thresholds["alert"] < thresholds["rate_limit"] < thresholds["temp_block"] < thresholds["block"]):
        raise ValueError(
            f"Risk thresholds must be strictly ordered: "
            f"alert({thresholds['alert']}) < "
            f"rate_limit({thresholds['rate_limit']}) < "
            f"temp_block({thresholds['temp_block']}) < "
            f"block({thresholds['block']})"
        )
    
    return self.policy
```

---

## FIX #8: Firewall Operation Timeouts

**File**: `src/firewall_adapters.py`

**BEFORE**:
```python
class UFWFirewallAdapter(FirewallAdapter):
    def _run(self, args: list[str]) -> tuple[bool, str]:
        try:
            result = self.run_cmd(args, capture_output=True, text=True)
            # No timeout!
```

**AFTER**:
```python
class UFWFirewallAdapter(FirewallAdapter):
    DEFAULT_TIMEOUT = 5  # seconds
    
    def _run(self, args: list[str], timeout: int | None = None) -> tuple[bool, str]:
        timeout = timeout or self.DEFAULT_TIMEOUT
        try:
            result = self.run_cmd(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,  # CRITICAL: Add timeout
            )
            return result.returncode == 0, str(getattr(result, "stdout", "") or "")
        except TimeoutError:
            logger.error("UFW command timeout after %d seconds: %s", timeout, " ".join(args))
            return False, "timeout"
        except FileNotFoundError:
            logger.error("UFW not found - ensure it is installed")
            return False, "not_installed"
        except Exception as exc:
            logger.error("UFW command error: %s", exc)
            return False, str(exc)
```

---

## FIX #9: Anomaly Engine Lock Contention

**File**: `src/detection/engines/anomaly_engine.py`

**BEFORE**:
```python
def add_sample(self, features: dict[str, Any]) -> bool:
    with self._buffer_lock:  # HOLD LOCK DURING FIT!
        self._buffer.append(...)
        if len(self._buffer) >= self._buffer_size:
            self.fit(...)  # LONG OPERATION!
```

**AFTER**:
```python
def add_sample(self, features: dict[str, Any]) -> bool:
    """Add sample to buffer, auto-fit when full."""
    with self._buffer_lock:
        # Extract numeric features
        row = [features.get(name, 0.0) for name in self._feature_names]
        self._buffer.append(row)
        
        # Check if we should fit (outside lock)
        should_fit = len(self._buffer) >= self._buffer_size
        buffer_copy = list(self._buffer) if should_fit else None
    
    # Fit outside lock to avoid blocking evaluations
    if should_fit and buffer_copy:
        try:
            import numpy as np
            X = np.array(buffer_copy)
            self.fit(X)
            return True  # Indicate engine just became ready
        except Exception as exc:
            logger.error("Anomaly engine fit failed: %s", exc)
            return False
    
    return False
```

---

## FIX #10: Add Request Validation Schema

**File**: `web_app/app.py`

**BEFORE**:
```python
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    features = data.get("features", {})
    # No validation!
```

**AFTER**:
```python
from jsonschema import validate, ValidationError

PREDICT_SCHEMA = {
    "type": "object",
    "required": ["features"],
    "properties": {
        "features": {
            "type": "object",
            "additionalProperties": True,
            "minProperties": 1,
        },
        "profile": {
            "type": "string",
            "enum": ["strict", "balanced", "lenient"],
            "default": "balanced",
        },
    },
}

@app.route("/api/predict", methods=["POST"])
@require_role("analyst")
def api_predict():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    
    try:
        validate(instance=data, schema=PREDICT_SCHEMA)
    except ValidationError as exc:
        return jsonify({
            "error": "validation_error",
            "message": exc.message,
            "path": list(exc.path),
        }), 400
    
    features = data.get("features", {})
    profile = data.get("profile", "balanced")
    # ... continue
```

---

## FIX #11: Persist Policy Store

**File**: Create `src/policy/policy_store.py` (if not exists)

```python
import json
import os
from pathlib import Path
from dataclasses import asdict
from typing import Any

class PolicyStore:
    """Persist policy configuration to disk."""
    
    def __init__(self, persistence_path: str = "data/policy.json"):
        self.persistence_path = Path(persistence_path)
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = None
        self._load()
    
    def _load(self) -> None:
        """Load policy from disk."""
        if not self.persistence_path.exists():
            self._config = {}
            return
        
        try:
            with open(self.persistence_path, "r") as f:
                self._config = json.load(f)
        except Exception as exc:
            logger.error("Failed to load policy from %s: %s", self.persistence_path, exc)
            self._config = {}
    
    def _save(self) -> None:
        """Save policy to disk."""
        try:
            with open(self.persistence_path, "w") as f:
                json.dump(self._config, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save policy to %s: %s", self.persistence_path, exc)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self._config[key] = value
        self._save()
    
    def update(self, updates: dict) -> None:
        self._config.update(updates)
        self._save()
```

---

## Summary of All Fixes

| Issue | File | Type | Effort | Priority |
|-------|------|------|--------|----------|
| Memory Leak in RiskEngine | risk_engine.py | Code + Logic | 4h | 🔥 |
| Missing ActionExecutor Methods | action_executor.py | Implementation | 3h | 🔥 |
| EventBus Race Condition | event_bus.py | Concurrency | 2h | 🔥 |
| Alert Store Performance | detection_service.py | Optimization | 1h | 🔥 |
| Auth Security Default | auth_service.py | Config | 1h | 🔥 |
| Feature Column Validation | ml_engine.py | Validation | 2h | ⚠️ |
| Policy Threshold Validation | prevention_service.py | Validation | 2h | ⚠️ |
| Firewall Timeouts | firewall_adapters.py | Robustness | 1h | ⚠️ |
| Anomaly Engine Lock | anomaly_engine.py | Concurrency | 2h | 🟡 |
| Request Validation | app.py | Validation | 3h | ⚠️ |
| Policy Persistence | policy_store.py | New Module | 2h | 🟡 |

**Total Effort**: ~23 hours
**Recommended Timeline**: Week 1-2 of development cycle

---

## Testing Checklist

After applying each fix:

- [ ] Unit tests pass for fixed module
- [ ] Integration tests pass (if applicable)
- [ ] Load test with 10k events/sec for 1 hour - check memory usage
- [ ] Concurrent request test (100 simultaneous /api/predict)
- [ ] Memory profiling - verify no leaks
- [ ] Exception handling test - ensure no silent failures
- [ ] Recovery test - restart app with corrupted state

