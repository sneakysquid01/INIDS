from __future__ import annotations

from collections import defaultdict, deque, OrderedDict
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
        max_sources: int = 10000,
    ):
        self.weights = weights or RiskWeights()
        self.frequency_window_seconds = max(30, int(frequency_window_seconds))
        self.frequency_high_watermark = max(1, int(frequency_high_watermark))
        self.max_sources = max(100, int(max_sources))  # Enforce minimum
        self._events_by_source: dict[str, deque[float]] = {}
        self._source_last_accessed: OrderedDict[str, float] = OrderedDict()
        self._lock = Lock()
        self._cleanup_count = 0  # Metrics tracking

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
            # Initialize source if not exists
            if source not in self._events_by_source:
                self._events_by_source[source] = deque()
            
            q = self._events_by_source[source]
            q.append(now)
            
            # Clean old entries within this source's window
            while q and q[0] < window_start:
                q.popleft()
            
            # Continuous TTL-based eviction: remove stale sources on every call
            # A source is stale if its queue is empty (no activity in current window)
            empty_sources = [k for k, v in self._events_by_source.items() if not v]
            for k in empty_sources:
                del self._events_by_source[k]
                self._source_last_accessed.pop(k, None)
                self._cleanup_count += 1
            
            # Aggressive cleanup: if still over limit, remove by LRU
            if len(self._events_by_source) >= self.max_sources:
                # Remove oldest 20% of sources by last access time
                remove_count = max(1, len(self._events_by_source) // 5)
                items_to_remove = list(self._source_last_accessed.items())[:remove_count]
                for k, _ in items_to_remove:
                    self._events_by_source.pop(k, None)
                    self._source_last_accessed.pop(k, None)
                    self._cleanup_count += 1
                
                # Log aggressive cleanup at INFO level for visibility
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"RiskEngine: Memory pressure - removed {remove_count} sources; "
                    f"current_sources={len(self._events_by_source)}/{self.max_sources}; "
                    f"total_cleanups={self._cleanup_count}"
                )
            
            # Update access time for LRU
            self._source_last_accessed[source] = now
            count = len(q)
            return _clamp(count / self.frequency_high_watermark)

    def calculate(self, detection_event: DetectionEvent, weights_override: RiskWeights | None = None) -> RiskScoreEvent:
        weights = weights_override or self.weights
        raw_confidence = float(detection_event.confidence)
        confidence_score = _clamp(raw_confidence / 100.0 if raw_confidence > 1.0 else raw_confidence)
        severity_score = self.map_attack_severity(
            detection_event.prediction,
            detection_event.severity,
            getattr(detection_event, "attack_type", None),
        )
        frequency_score = self.recent_activity_score(detection_event.source)

        risk = (
            confidence_score * weights.confidence
            + severity_score * weights.severity
            + frequency_score * weights.frequency
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
