from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, TypeVar


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DetectionEvent:
    source_ip: str
    prediction: str
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)
    attack_type: str = "unknown"
    profile: str = "balanced"
    severity: str = "low"
    suspicious: bool = False
    reason: str = "model_prediction"
    timestamp: str = field(default_factory=utc_now_iso)

    @property
    def source(self) -> str:
        # Backward-compatible alias for existing risk/action modules.
        return self.source_ip

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source_ip
        return payload


@dataclass
class RiskScoreEvent:
    detection: DetectionEvent
    risk_score: float
    components: dict[str, float]
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["detection"] = self.detection.to_dict()
        return payload


@dataclass
class PolicyDecisionEvent:
    risk: RiskScoreEvent
    decision: str
    reason: str
    ttl_seconds: int | None = None
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["risk"] = self.risk.to_dict()
        return payload


@dataclass
class ActionEvent:
    decision: PolicyDecisionEvent
    action: str
    target: str
    reason: str
    dry_run: bool
    executed: bool
    status: str
    adapter: str
    expires_at: str | None
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.to_dict()
        return payload


@dataclass
class AuditEvent:
    event_type: str
    message: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EventT = TypeVar("EventT")


class EventBus:
    """Lightweight in-process event dispatcher."""

    def __init__(self):
        self._handlers: dict[type, list[Callable[[Any], None]]] = defaultdict(list)
        self._lock = Lock()

    def subscribe(self, event_type: type[EventT], handler: Callable[[EventT], None]) -> None:
        with self._lock:
            self._handlers[event_type].append(handler)

    def publish(self, event: Any) -> None:
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "EventBus handler %s failed for %s", handler.__name__, type(event).__name__
                )
