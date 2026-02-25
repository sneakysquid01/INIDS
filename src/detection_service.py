from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

import pandas as pd

from src.schema import DEFAULT_FEATURE_ROW, FEATURE_COLUMNS


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
        self.max_items = max_items
        self._alerts: list[Alert] = []

    def add(self, alert: Alert) -> None:
        self._alerts.insert(0, alert)
        if len(self._alerts) > self.max_items:
            self._alerts = self._alerts[: self.max_items]

    def list_alerts(self, limit: int = 50, severity: str | None = None) -> list[Alert]:
        alerts = self._alerts
        if severity:
            normalized = severity.strip().lower()
            alerts = [a for a in alerts if a.severity.lower() == normalized]
        return alerts[:limit]


class DetectionService:
    def __init__(self, model, alert_store: InMemoryAlertStore):
        self.model = model
        self.alert_store = alert_store

    def predict_from_features(self, features: dict[str, Any], profile: str = "balanced") -> PredictionResult:
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

        alert = None
        if suspicious or prediction == "Attack":
            severity = self._severity(prediction, confidence, threshold)
            alert = Alert(
                id=f"al_{uuid.uuid4().hex[:10]}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=severity,
                prediction=prediction,
                confidence=confidence,
                profile=normalized_profile,
                reason=reason,
            )
            self.alert_store.add(alert)

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            profile=normalized_profile,
            threshold=threshold,
            suspicious=suspicious,
            reason=reason,
            alert=alert,
        )

    @staticmethod
    def _severity(prediction: str, confidence: float, threshold: float) -> str:
        if prediction == "Attack" and confidence >= 90:
            return "critical"
        if prediction == "Attack":
            return "high"
        if confidence < threshold:
            return "medium"
        return "low"
