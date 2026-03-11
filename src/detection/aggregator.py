"""Multi-engine result aggregator — fuses EngineResults into a final verdict."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.detection.engine_base import EngineResult

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """How multiple engine verdicts are fused into a final decision."""
    UNANIMOUS = "unanimous"      # all engines must agree on "attack"
    MAJORITY = "majority"        # >50% of engines say "attack"
    ANY_TRIGGER = "any_trigger"  # any single engine saying "attack" is enough
    WEIGHTED = "weighted"        # weighted vote by confidence


# Severity ranking for "worst-wins" merging.
_SEV_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}

# Verdict priority (higher → more alarming).
_VERDICT_RANK = {"attack": 3, "suspicious": 2, "unknown": 1, "normal": 0}


@dataclass
class AggregatedResult:
    """Final fused result consumed by the rest of the pipeline."""
    verdict: str
    confidence: float
    severity: str
    attack_type: str
    engine_results: list[EngineResult]
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "severity": self.severity,
            "attack_type": self.attack_type,
            "strategy": self.strategy,
            "engine_count": len(self.engine_results),
            "engines": [r.to_dict() for r in self.engine_results],
            "metadata": self.metadata,
        }


class EngineAggregator:
    """Fuse a list of ``EngineResult`` objects into one ``AggregatedResult``."""

    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.ANY_TRIGGER):
        self.strategy = strategy

    def aggregate(self, results: list[EngineResult]) -> AggregatedResult:
        if not results:
            return AggregatedResult(
                verdict="unknown",
                confidence=0.0,
                severity="low",
                attack_type="unknown",
                engine_results=[],
                strategy=self.strategy.value,
            )

        handler = {
            AggregationStrategy.UNANIMOUS: self._unanimous,
            AggregationStrategy.MAJORITY: self._majority,
            AggregationStrategy.ANY_TRIGGER: self._any_trigger,
            AggregationStrategy.WEIGHTED: self._weighted,
        }[self.strategy]

        return handler(results)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _any_trigger(self, results: list[EngineResult]) -> AggregatedResult:
        """If *any* engine says "attack" or "suspicious", the final verdict
        escalates to the worst-case."""
        worst_verdict = max(results, key=lambda r: _VERDICT_RANK.get(r.verdict, 0))
        return self._build(results, worst_verdict.verdict)

    def _majority(self, results: list[EngineResult]) -> AggregatedResult:
        attack_count = sum(1 for r in results if r.verdict == "attack")
        verdict = "attack" if attack_count > len(results) / 2 else "normal"
        return self._build(results, verdict)

    def _unanimous(self, results: list[EngineResult]) -> AggregatedResult:
        all_attack = all(r.verdict == "attack" for r in results)
        verdict = "attack" if all_attack else "normal"
        return self._build(results, verdict)

    def _weighted(self, results: list[EngineResult]) -> AggregatedResult:
        attack_weight = sum(r.confidence for r in results if r.verdict == "attack")
        total_weight = sum(r.confidence for r in results) or 1.0
        ratio = attack_weight / total_weight
        verdict = "attack" if ratio > 0.5 else "normal"
        return self._build(results, verdict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build(self, results: list[EngineResult], verdict: str) -> AggregatedResult:
        worst_sev = max(results, key=lambda r: _SEV_RANK.get(r.severity, 0))
        avg_conf = sum(r.confidence for r in results) / len(results)

        # Determine attack_type: prefer the most specific non-"unknown" type.
        attack_types = [r.attack_type for r in results if r.attack_type != "unknown"]
        attack_type = attack_types[0] if attack_types else "unknown"

        return AggregatedResult(
            verdict=verdict,
            confidence=round(avg_conf, 2),
            severity=worst_sev.severity,
            attack_type=attack_type,
            engine_results=results,
            strategy=self.strategy.value,
        )
