from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any
import uuid

import pandas as pd

from src.core.event_bus import DetectionEvent, EventBus
from src.schema import DEFAULT_FEATURE_ROW, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


THRESHOLD_PROFILES = {
    "strict": 75.0,
    "balanced": 60.0,
    "lenient": 45.0,
}


@dataclass
class Alert:
    id: str
    timestamp: str
    severity: str
    prediction: str
    confidence: float
    profile: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    profile: str
    threshold: float
    suspicious: bool
    reason: str
    alert: Alert | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.alert is not None:
            payload["alert"] = self.alert.to_dict()
        return payload


class InMemoryAlertStore:
    def __init__(self, max_items: int = 1000):
        self.max_items = max(1, int(max_items))
        self._alerts: deque[Alert] = deque(maxlen=self.max_items)
        self._lock = Lock()
        self._dropped_count = 0  # Track silently dropped alerts
        self._logger = None

    def _get_logger(self):
        """Lazy import to avoid circular dependencies."""
        if self._logger is None:
            import logging
            self._logger = logging.getLogger(__name__)
        return self._logger

    def add(self, alert: Alert | None) -> None:
        if alert is None:
            return
        with self._lock:
            # Check if buffer is at capacity before append
            at_capacity = len(self._alerts) >= self.max_items
            
            # deque with maxlen automatically handles truncation
            self._alerts.appendleft(alert)
            
            # Log if an alert was dropped
            if at_capacity:
                self._dropped_count += 1
                logger = self._get_logger()
                logger.warning(
                    f"Alert buffer full: dropped alert (id={alert.id}, severity={alert.severity}). "
                    f"Total dropped: {self._dropped_count}. Consider increasing max_items ({self.max_items})"
                )

    def list_alerts(self, limit: int = 50, severity: str | None = None) -> list[Alert]:
        limit = max(1, min(limit, 1000))
        with self._lock:
            alerts = list(self._alerts)
        if severity:
            normalized = severity.strip().lower()
            alerts = [a for a in alerts if a.severity.lower() == normalized]
        return alerts[:limit]


class DetectionService:
    def __init__(
        self,
        model,
        alert_store: InMemoryAlertStore | None = None,
        event_bus: EventBus | None = None,
        ops_store=None,
    ):
        self.model = model
        self.alert_store = alert_store  # B-06 sub-deploy 1: dual-write shim; None after sub-deploy 3
        self.event_bus = event_bus
        self.ops_store = ops_store  # B-06: primary persistent store for alerts

    def predict_from_features(
        self,
        features: dict[str, Any],
        profile: str = "balanced",
        source_ip: str = "unknown",
        attack_type: str | None = None,
    ) -> PredictionResult:
        threshold = THRESHOLD_PROFILES.get(profile, THRESHOLD_PROFILES["balanced"])
        normalized_profile = profile if profile in THRESHOLD_PROFILES else "balanced"

        row = DEFAULT_FEATURE_ROW.copy()
        for key, value in features.items():
            if key in FEATURE_COLUMNS:
                row[key] = value

        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        pred = int(self.model.predict(df)[0])
        proba = self.model.predict_proba(df)[0]
        confidence = round(float(max(proba) * 100), 2)

        suspicious = confidence < threshold
        prediction = "Attack" if pred == 1 else "Normal"
        reason = "below_confidence_threshold" if suspicious else "model_prediction"
        severity = self._severity(prediction, confidence, threshold)

        alert = None
        if suspicious or prediction == "Attack":
            alert = Alert(
                id=f"al_{uuid.uuid4().hex[:10]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=severity,
                prediction=prediction,
                confidence=confidence,
                profile=normalized_profile,
                reason=reason,
            )
            if self.alert_store is not None:
                self.alert_store.add(alert)
            if self.ops_store is not None:
                try:
                    self.ops_store.save_alert({
                        **alert.to_dict(),
                        "source_ip": source_ip,
                        "attack_type": attack_type or ("attack" if prediction == "Attack" else "normal"),
                    })
                except Exception as exc:
                    logger.warning("B-06 ops_store.save_alert failed (dual-write): %s", exc)

        result = PredictionResult(
            prediction=prediction,
            confidence=confidence,
            profile=normalized_profile,
            threshold=threshold,
            suspicious=suspicious,
            reason=reason,
            alert=alert,
        )
        self._emit_detection_event(
            source_ip=source_ip,
            prediction=prediction,
            confidence=confidence,
            profile=normalized_profile,
            severity=severity,
            suspicious=suspicious,
            reason=reason,
            features=row,
            attack_type=attack_type,
        )
        return result

    def _emit_detection_event(
        self,
        *,
        source_ip: str,
        prediction: str,
        confidence: float,
        profile: str,
        severity: str,
        suspicious: bool,
        reason: str,
        features: dict[str, Any],
        attack_type: str | None,
    ) -> None:
        if self.event_bus is None:
            return
        derived_attack_type = attack_type or ("attack" if prediction == "Attack" else "normal")
        event = DetectionEvent(
            source_ip=str(source_ip or "unknown"),
            prediction=prediction,
            confidence=confidence,
            features=features,
            attack_type=str(derived_attack_type),
            profile=profile,
            severity=severity,
            suspicious=suspicious,
            reason=reason,
        )
        self.event_bus.publish(event)

    @staticmethod
    def _severity(prediction: str, confidence: float, threshold: float) -> str:
        if prediction == "Attack" and confidence >= 90:
            return "critical"
        if prediction == "Attack":
            return "high"
        if confidence < threshold:
            return "medium"
        return "low"


    @staticmethod
    def explain_features(features: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        """Simple heuristic explanation using distance from default feature values."""
        row = DEFAULT_FEATURE_ROW.copy()
        for key, value in features.items():
            if key in FEATURE_COLUMNS:
                row[key] = value

        contributions: list[dict[str, Any]] = []
        for key in FEATURE_COLUMNS:
            current = row[key]
            base = DEFAULT_FEATURE_ROW[key]
            if isinstance(base, (int, float)):
                try:
                    score = abs(float(current) - float(base))
                except Exception:
                    score = 0.0
            else:
                score = 0.0 if str(current) == str(base) else 1.0
            contributions.append({
                "feature": key,
                "current": current,
                "baseline": base,
                "contribution": round(float(score), 6),
            })

        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions[:max(1, top_k)]


def drain_to_ops_store(source_alert_store: InMemoryAlertStore, ops_store) -> int:
    """B-06 sub-deploy 3: drain all alerts from source_alert_store into ops_store.

    Returns the count of successfully drained records.
    Raises RuntimeError if any drain failures occur (per PLAN.md requirement).
    """
    alerts = source_alert_store.list_alerts(limit=10000)
    drained = 0
    failures = 0
    for alert in alerts:
        payload = alert.to_dict() if hasattr(alert, "to_dict") else dict(alert)
        try:
            ops_store.save_alert(payload)
            drained += 1
        except Exception as exc:
            logger.error("drain_to_ops_store: failed for alert %s: %s", payload.get("id"), exc)
            failures += 1
    logger.info("drain_to_ops_store: drained=%d failures=%d", drained, failures)
    if failures > 0:
        raise RuntimeError(f"drain_to_ops_store: {failures} drain failures — aborting store removal")
    return drained
