"""Abstract base class for all detection engines and shared result types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from src.core.event_bus import utc_now_iso


@dataclass
class EngineResult:
    """Result produced by a single detection engine."""

    engine_id: str
    engine_type: str  # "ml", "signature", "anomaly", "threshold", "ti"
    verdict: str  # "attack", "normal", "suspicious", "unknown"
    confidence: float  # 0.0–100.0
    severity: str = "low"  # "critical", "high", "medium", "low"
    attack_type: str = "unknown"
    rule_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DetectionEngine(ABC):
    """Base class for all detection engines.

    Every engine must implement:
    - ``engine_id``: unique identifier for this engine instance.
    - ``engine_type``: category string (``ml``, ``signature``, ``anomaly``,
      ``threshold``, ``ti``).
    - ``evaluate(features)``: run detection and return an ``EngineResult``.
    - ``is_ready()``: return True if the engine is fully initialized and can
      accept evaluation requests.
    """

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier for this engine."""

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """Category: ml | signature | anomaly | threshold | ti"""

    @abstractmethod
    def evaluate(self, features: dict[str, Any]) -> EngineResult:
        """Run detection on a single flow record and return an EngineResult."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the engine is fully loaded and operational."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.engine_id} type={self.engine_type} ready={self.is_ready()}>"
