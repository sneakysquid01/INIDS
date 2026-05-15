"""Threshold / Rate-based Detection Engine — detects volumetric attacks via counters."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Any

from src.detection.engine_base import DetectionEngine, EngineResult

logger = logging.getLogger(__name__)


class _RateCounter:
    """Sliding-window counter for a single (source_ip, metric) pair."""

    __slots__ = ("_timestamps", "_lock")

    _MAX_TIMESTAMPS = 10_000  # cap to prevent unbounded memory growth

    def __init__(self) -> None:
        self._timestamps: list[float] = []
        self._lock = Lock()

    def record(self, now: float | None = None) -> None:
        now = now or time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            if len(self._timestamps) > self._MAX_TIMESTAMPS:
                self._timestamps = self._timestamps[-self._MAX_TIMESTAMPS:]

    def count(self, window_seconds: float, now: float | None = None) -> int:
        now = now or time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            return len(self._timestamps)


class ThresholdEngine(DetectionEngine):
    """Rate-based aggregate counters for volumetric attack patterns.

    Tracks per-IP connection rates and flags IPs that exceed configurable
    thresholds within a sliding time window.

    Default rules detect:
    - **SYN flood**: high ``count`` + ``serror_rate`` > 0.7
    - **Port scan**: high ``dst_host_srv_count`` + ``srv_diff_host_rate`` > 0.4
    - **High error rate**: ``rerror_rate`` > 0.6
    """

    def __init__(
        self,
        *,
        engine_id: str = "threshold",
        window_seconds: float = 60.0,
        connection_rate_limit: int = 200,
        fp_manager: Any = None,
    ) -> None:
        self._engine_id = engine_id
        self._window = window_seconds
        self._conn_limit = connection_rate_limit
        self._fp_manager = fp_manager
        self._counters: dict[str, _RateCounter] = defaultdict(_RateCounter)
        self._max_tracked_ips = 50_000
        self._lock = Lock()

        # Threshold rules expressed as (field, op, value, severity, attack_type)
        self._rules: list[dict[str, Any]] = [
            {
                "name": "SYN Flood",
                "attack_type": "dos",
                "severity": "critical",
                "conditions": [
                    ("count", ">", 200),
                    ("serror_rate", ">", 0.7),
                ],
            },
            {
                "name": "Port Scan",
                "attack_type": "probe",
                "severity": "high",
                "conditions": [
                    ("dst_host_srv_count", ">", 50),
                    ("srv_diff_host_rate", ">", 0.4),
                ],
            },
            {
                "name": "High Error Rate",
                "attack_type": "dos",
                "severity": "medium",
                "conditions": [
                    ("rerror_rate", ">", 0.6),
                ],
            },
        ]

    # ------------------------------------------------------------------
    # DetectionEngine interface
    # ------------------------------------------------------------------

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def engine_type(self) -> str:
        return "threshold"

    def is_ready(self) -> bool:
        return True  # stateless — always ready

    def evaluate(self, features: dict[str, Any]) -> EngineResult:
        source_ip = str(features.get("source_ip", "unknown"))

        # Check if source is a known false positive (suppressed)
        if self._fp_manager is not None and source_ip != "unknown":
            try:
                if self._fp_manager.is_suppressed(source_ip):
                    logger.debug("ThresholdEngine: source %s suppressed by FP manager", source_ip)
                    return EngineResult(
                        engine_id=self._engine_id,
                        engine_type=self.engine_type,
                        verdict="normal",
                        confidence=100.0,
                        severity="low",
                        attack_type="normal",
                        metadata={"suppressed_by_fp_manager": True},
                    )
            except Exception:
                logger.exception("FP manager suppression check failed for %s", source_ip)

        # Track connection rate per source IP.
        now = time.monotonic()
        counter = self._get_counter(source_ip)
        counter.record(now)
        conn_count = counter.count(self._window, now)

        # Check rate limit first.
        if conn_count > self._conn_limit:
            return EngineResult(
                engine_id=self._engine_id,
                engine_type=self.engine_type,
                verdict="attack",
                confidence=90.0,
                severity="critical",
                attack_type="dos",
                metadata={"reason": "connection_rate_exceeded", "count": conn_count, "limit": self._conn_limit},
            )

        # Check threshold rules.
        for rule in self._rules:
            if self._match(rule, features):
                return EngineResult(
                    engine_id=self._engine_id,
                    engine_type=self.engine_type,
                    verdict="attack",
                    confidence=85.0,
                    severity=rule["severity"],
                    attack_type=rule["attack_type"],
                    metadata={"rule": rule["name"], "source_ip": source_ip},
                )

        return EngineResult(
            engine_id=self._engine_id,
            engine_type=self.engine_type,
            verdict="normal",
            confidence=100.0,
            severity="low",
            attack_type="normal",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_counter(self, key: str) -> _RateCounter:
        with self._lock:
            if len(self._counters) >= self._max_tracked_ips and key not in self._counters:
                # Evict oldest entries to prevent unbounded growth.
                excess = len(self._counters) - self._max_tracked_ips + 1
                for old_key in list(self._counters.keys())[:excess]:
                    del self._counters[old_key]
            return self._counters[key]

    @staticmethod
    def _match(rule: dict[str, Any], features: dict[str, Any]) -> bool:
        for field, op, threshold in rule.get("conditions", []):
            val = features.get(field)
            if val is None:
                return False
            try:
                val = float(val)
            except (TypeError, ValueError):
                return False
            if op == ">" and not (val > threshold):
                return False
            if op == ">=" and not (val >= threshold):
                return False
            if op == "<" and not (val < threshold):
                return False
        return True
