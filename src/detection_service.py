from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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


class DetectionService:
    def __init__(
        self,
        model,
        event_bus: EventBus | None = None,
        ops_store=None,
    ):
        self.model = model
        self.event_bus = event_bus
        self.ops_store = ops_store

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

        suspicious = pred == 1 and confidence < threshold
        prediction = "Attack" if pred == 1 else "Normal"
        reason = "below_confidence_threshold" if suspicious else "model_prediction"
        severity = self._severity(prediction, confidence, threshold)

        alert = None
        if suspicious or prediction == "Attack":
            alert = Alert(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=severity,
                prediction=prediction,
                confidence=confidence,
                profile=normalized_profile,
                reason=reason,
            )
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


