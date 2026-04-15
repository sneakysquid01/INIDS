from __future__ import annotations

from collections import defaultdict
from copy import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock, RLock
from typing import Any, Callable, TypeVar
import logging

logger = logging.getLogger(__name__)


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
        self._lock = RLock()

    def subscribe(self, event_type: type[EventT], handler: Callable[[EventT], None]) -> None:
        """Register a handler for an event type. Prevents duplicate subscriptions."""
        if not callable(handler):
            raise TypeError(f"handler must be callable, got {type(handler)}")
        
        with self._lock:
            # Avoid duplicate subscriptions
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)
                logger.debug(f"Subscribed {getattr(handler, '__name__', str(handler))} to {event_type.__name__}")

    def publish(self, event: Any) -> None:
        """Publish an event to all registered handlers. Safe for concurrent access."""
        with self._lock:
            # Create a copy of handlers list to avoid modification during iteration
            handlers = copy(self._handlers.get(type(event), []))
        
        # Invoke handlers outside lock to prevent deadlock
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "EventBus handler %s failed for %s", 
                    getattr(handler, '__name__', str(handler)), 
                    type(event).__name__
                )
