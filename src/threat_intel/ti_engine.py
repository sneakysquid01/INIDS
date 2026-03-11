"""Threat Intelligence Detection Engine — checks flows against TI indicator cache."""
from __future__ import annotations

import logging
from typing import Any

from src.detection.engine_base import DetectionEngine, EngineResult
from src.threat_intel.feed_manager import ThreatIntelManager

logger = logging.getLogger(__name__)


class TIEngine(DetectionEngine):
    """Checks whether a flow's source IP (or other IoC fields) appear in the
    threat intelligence cache.

    If a match is found the engine returns an ``attack`` verdict with severity
    derived from the TI indicator's own severity rating.
    """

    def __init__(self, ti_manager: ThreatIntelManager, *, engine_id: str = "threat_intel") -> None:
        self._ti = ti_manager
        self._engine_id = engine_id

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def engine_type(self) -> str:
        return "ti"

    def is_ready(self) -> bool:
        return self._ti.cache.size() > 0

    def evaluate(self, features: dict[str, Any]) -> EngineResult:
        source_ip = str(features.get("source_ip", ""))
        match = self._ti.lookup_ip(source_ip) if source_ip else None

        if match is not None and not match.is_expired():
            return EngineResult(
                engine_id=self._engine_id,
                engine_type=self.engine_type,
                verdict="attack",
                confidence=90.0,
                severity=match.severity,
                attack_type="threat_intel_match",
                metadata={
                    "ti_source": match.source,
                    "ti_tags": match.tags,
                    "matched_value": match.value,
                },
            )

        return EngineResult(
            engine_id=self._engine_id,
            engine_type=self.engine_type,
            verdict="normal",
            confidence=100.0,
            severity="low",
            attack_type="normal",
        )
