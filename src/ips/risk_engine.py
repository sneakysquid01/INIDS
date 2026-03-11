from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import time

from src.core.event_bus import DetectionEvent, RiskScoreEvent


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class RiskWeights:
    confidence: float = 0.5
    severity: float = 0.3
    frequency: float = 0.2


class RiskEngine:
    """Aggregates model and behavioral signals into a unified risk score."""

    def __init__(
        self,
        weights: RiskWeights | None = None,
        frequency_window_seconds: int = 300,
        frequency_high_watermark: int = 20,
    ):
        self.weights = weights or RiskWeights()
        self.frequency_window_seconds = max(30, int(frequency_window_seconds))
        self.frequency_high_watermark = max(1, int(frequency_high_watermark))
        self._events_by_source: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    ATTACK_TYPE_SEVERITY: dict[str, float] = {
        "u2r": 1.0,
        "r2l": 0.95,
        "dos": 0.85,
        "probe": 0.7,
        "attack": 0.8,
        "normal": 0.1,
    }

    def map_attack_severity(self, prediction: str, severity: str, attack_type: str | None = None) -> float:
        explicit = str(severity or "").strip().lower()
        if explicit == "critical":
            return 1.0
        if explicit == "high":
            return 0.85
        if explicit == "medium":
            return 0.6
        if explicit == "low":
            return 0.25
        normalized_attack_type = str(attack_type or "").strip().lower()
        if normalized_attack_type in self.ATTACK_TYPE_SEVERITY:
            return self.ATTACK_TYPE_SEVERITY[normalized_attack_type]
        return 0.8 if str(prediction).lower() == "attack" else 0.1

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

    def calculate(self, detection_event: DetectionEvent) -> RiskScoreEvent:
        raw_confidence = float(detection_event.confidence)
        confidence_score = _clamp(raw_confidence / 100.0 if raw_confidence > 1.0 else raw_confidence)
        severity_score = self.map_attack_severity(
            detection_event.prediction,
            detection_event.severity,
            getattr(detection_event, "attack_type", None),
        )
        frequency_score = self.recent_activity_score(detection_event.source)

        risk = (
            confidence_score * self.weights.confidence
            + severity_score * self.weights.severity
            + frequency_score * self.weights.frequency
        )
        components = {
            "confidence": round(confidence_score, 6),
            "severity": round(severity_score, 6),
            "frequency": round(frequency_score, 6),
        }
        return RiskScoreEvent(
            detection=detection_event,
            risk_score=round(_clamp(risk), 6),
            components=components,
        )
