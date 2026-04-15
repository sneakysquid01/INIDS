"""Temporal correlation engine for multi-stage attack detection.

Detects attack chains that span multiple flows/events over time.
Inspired by WatchAD's deferred detection pattern with 60-second correlation window.

Example patterns:
- Port scan (t=0) → SSH brute force (t=15s) → Successful login (t=45s)
- Connection to known C2 (t=0) → Data exfiltration (t=120s)
- Directory enumeration (t=0) → Registry access (t=30s) → Process execution (t=60s)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CorrelationPattern:
    """Defines a correlation pattern for multi-step attacks."""

    def __init__(
        self,
        pattern_id: str,
        description: str,
        steps: list[dict[str, Any]],
        time_window_seconds: int = 300,
        verdict: str = "attack",
        severity: str = "high",
        attack_type: str = "multi_stage",
    ) -> None:
        """Initialize correlation pattern.
        
        Args:
            pattern_id: Unique pattern identifier
            description: Human-readable description
            steps: List of detection steps, each with:
                - engine_id: "signature", "threshold", "anomaly", etc.
                - rule_id: Specific rule to match
                - field_conditions: Additional field matching rules
                - time_offset_min: Min seconds from previous step
                - time_offset_max: Max seconds from previous step
            time_window_seconds: Maximum time span for full pattern
            verdict: Detection verdict if pattern matches
            severity: Alert severity level
            attack_type: Classification of attack type
        """
        self.pattern_id = pattern_id
        self.description = description
        self.steps = steps
        self.time_window_seconds = time_window_seconds
        self.verdict = verdict
        self.severity = severity
        self.attack_type = attack_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "steps": self.steps,
            "time_window_seconds": self.time_window_seconds,
            "verdict": self.verdict,
            "severity": self.severity,
            "attack_type": self.attack_type,
        }


class TemporalStore:
    """In-memory storage for deferred events awaiting correlation."""

    _MAX_EVENTS_PER_IP = 1000  # Prevent unbounded memory growth

    def __init__(self) -> None:
        # Store: {source_ip: deque of events sorted by timestamp}
        self._events_by_ip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._last_cleanup = time.time()

    @property
    def events_by_ip(self) -> dict[str, list[dict[str, Any]]]:
        return self._events_by_ip

    def add_event(self, event: dict[str, Any]) -> None:
        """Add event to temporal store."""
        source_ip = event.get("source_ip", "unknown")
        timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())
        
        # Add to store
        events = self._events_by_ip[source_ip]
        events.append(event)
        
        # Trim old events beyond time window
        now = datetime.now(timezone.utc)
        events[:] = [
            e for e in events
            if self._parse_iso_datetime(e.get("timestamp")).timestamp() > (now.timestamp() - 600)  # 10-minute window
        ]
        
        # Cap per-IP storage
        if len(events) > self._MAX_EVENTS_PER_IP:
            events[:] = events[-self._MAX_EVENTS_PER_IP:]
        
        # Periodic cleanup
        now_time = time.time()
        if now_time - self._last_cleanup > 60:  # Cleanup every 60 seconds
            self._cleanup_old_events()
            self._last_cleanup = now_time

    def get_events_for_ip(self, source_ip: str, window_seconds: int = 600) -> list[dict[str, Any]]:
        """Get recent events for IP within time window."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - window_seconds
        
        events = self._events_by_ip.get(source_ip, [])
        return [
            e for e in events
            if self._parse_iso_datetime(e.get("timestamp")).timestamp() > cutoff
        ]

    def _cleanup_old_events(self) -> None:
        """Remove old events from all IPs."""
        now = datetime.now(timezone.utc)
        cutoff_timestamp = now.timestamp() - 600
        
        for source_ip in list(self._events_by_ip.keys()):
            events = self._events_by_ip[source_ip]
            events[:] = [
                e for e in events
                if self._parse_iso_datetime(e.get("timestamp")).timestamp() > cutoff_timestamp
            ]
            if not events:
                del self._events_by_ip[source_ip]

    @staticmethod
    def _parse_iso_datetime(iso_str: str | None) -> datetime:
        """Parse ISO datetime string."""
        if not iso_str:
            return datetime.now(timezone.utc)
        try:
            # Handle both with and without timezone
            if iso_str.endswith("Z"):
                return datetime.fromisoformat(iso_str[:-1] + "+00:00")
            return datetime.fromisoformat(iso_str)
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)


class TemporalCorrelationEngine:
    """Detects multi-stage attacks by correlating events over time."""

    def __init__(self) -> None:
        self._patterns: dict[str, CorrelationPattern] = {}
        self._temporal_store = TemporalStore()
        self._correlation_cache: dict[str, dict[str, Any]] = {}  # Per-IP correlation state

    @property
    def patterns(self) -> dict[str, CorrelationPattern]:
        return self._patterns

    @property
    def temporal_store(self) -> TemporalStore:
        return self._temporal_store

    def register_pattern(
        self,
        pattern: CorrelationPattern | str,
        steps: list[dict[str, Any]] | None = None,
        *,
        description: str | None = None,
        time_window_seconds: int = 300,
        verdict: str = "attack",
        severity: str = "high",
        attack_type: str = "multi_stage",
    ) -> None:
        """Register a correlation pattern.

        Accepts either a pre-built ``CorrelationPattern`` or a simplified
        ``(pattern_id, steps)`` pair used by the Week 1 API/tests.
        """
        if isinstance(pattern, CorrelationPattern):
            correlation_pattern = pattern
        else:
            if steps is None:
                raise ValueError("steps are required when registering by pattern id")
            correlation_pattern = CorrelationPattern(
                pattern_id=str(pattern),
                description=description or str(pattern),
                steps=steps,
                time_window_seconds=int(time_window_seconds),
                verdict=verdict,
                severity=severity,
                attack_type=attack_type,
            )
        self._patterns[correlation_pattern.pattern_id] = correlation_pattern
        logger.info("Registered correlation pattern: %s", correlation_pattern.pattern_id)

    def evaluate(self, event: dict[str, Any]) -> tuple[str, str] | None:
        """Evaluate event for correlation matches.
        
        Returns: (pattern_id, matched_pattern_description) if match found, None otherwise
        """
        source_ip = event.get("source_ip", "unknown")
        
        # Add event to temporal store
        self._temporal_store.add_event(event)
        
        # Check all patterns
        for pattern_id, pattern in self._patterns.items():
            if self._check_pattern(pattern, source_ip):
                logger.info(
                    "Correlation pattern matched: %s for IP %s",
                    pattern_id, source_ip
                )
                return pattern_id, pattern.description
        
        return None

    def _check_pattern(self, pattern: CorrelationPattern, source_ip: str) -> bool:
        """Check if correlation pattern matches for an IP."""
        events = self._temporal_store.get_events_for_ip(
            source_ip,
            window_seconds=pattern.time_window_seconds,
        )
        
        if len(events) < len(pattern.steps):
            return False  # Not enough events
        
        # Try to match pattern steps in sequence
        return self._match_steps(pattern.steps, events)

    def _match_steps(self, steps: list[dict[str, Any]], events: list[dict[str, Any]]) -> bool:
        """Try to match a sequence of pattern steps against events."""
        if not steps or not events:
            return False
        
        # Simple greedy matching: try to match each step in order
        step_idx = 0
        prev_event_time = None
        
        for event in events:
            if step_idx >= len(steps):
                return True  # All steps matched
            
            step = steps[step_idx]
            event_time = _parse_iso_datetime(event.get("timestamp"))
            
            # Check time offset constraints
            if prev_event_time is not None:
                time_delta = (event_time - prev_event_time).total_seconds()
                time_offset_min = step.get("time_offset_min", 0)
                time_offset_max = step.get("time_offset_max", step.get("time_offset_seconds", 300))
                
                if not (time_offset_min <= time_delta <= time_offset_max):
                    continue  # Time constraint not met
            
            if self._step_matches(step, event):
                step_idx += 1
                prev_event_time = event_time
        
        return step_idx >= len(steps)  # True if all steps matched

    @staticmethod
    def _step_matches(step: dict[str, Any], event: dict[str, Any]) -> bool:
        """Support both detailed and simplified step schemas."""
        step_type = step.get("type")
        if step_type is not None and event.get("type") != step_type:
            return False

        engine_id = step.get("engine_id")
        if engine_id is not None and event.get("engine_id") != engine_id:
            return False

        rule_id = step.get("rule_id")
        if rule_id is not None and event.get("rule_id") != rule_id:
            return False

        confidence_min = step.get("confidence_min")
        if confidence_min is not None:
            try:
                if float(event.get("confidence", 0.0)) < float(confidence_min):
                    return False
            except (TypeError, ValueError):
                return False

        return TemporalCorrelationEngine._check_field_conditions(step.get("field_conditions", {}), event)

    @staticmethod
    def _check_field_conditions(conditions: dict[str, Any], event: dict[str, Any]) -> bool:
        """Check if event matches field conditions."""
        if not conditions:
            return True
        
        for field, expected_value in conditions.items():
            actual_value = event.get(field)
            if actual_value != expected_value:
                return False
        
        return True

    def get_correlation_state(self, source_ip: str) -> dict[str, Any]:
        """Get current correlation state for an IP."""
        return self._correlation_cache.get(source_ip, {"patterns_in_progress": []})


def create_example_patterns() -> list[CorrelationPattern]:
    """Create example correlation patterns."""
    patterns = [
        CorrelationPattern(
            pattern_id="port_scan_to_brute_force",
            description="Port scan followed by SSH brute force",
            steps=[
                {
                    "engine_id": "threshold",
                    "rule_id": "port_scan",
                    "field_conditions": {"attack_type": "probe"},
                    "time_offset_min": 0,
                    "time_offset_max": 0,
                },
                {
                    "engine_id": "threshold",
                    "rule_id": "brute_force",
                    "field_conditions": {"dst_port": 22},
                    "time_offset_min": 5,
                    "time_offset_max": 60,
                },
            ],
            time_window_seconds=300,
            verdict="attack",
            severity="high",
            attack_type="reconnaissance_to_exploitation",
        ),
        CorrelationPattern(
            pattern_id="c2_communication_to_data_exfil",
            description="Known C2 connection followed by data exfil",
            steps=[
                {
                    "engine_id": "ti",
                    "rule_id": "known_c2_ip",
                    "field_conditions": {},
                    "time_offset_min": 0,
                    "time_offset_max": 0,
                },
                {
                    "engine_id": "signature",
                    "rule_id": "data_exfil",
                    "field_conditions": {"attack_type": "data_exfil"},
                    "time_offset_min": 10,
                    "time_offset_max": 300,
                },
            ],
            time_window_seconds=600,
            verdict="attack",
            severity="critical",
            attack_type="command_and_control_plus_exfil",
        ),
    ]
    return patterns


# Helper to parse ISO datetime
def _parse_iso_datetime(iso_str: str | None) -> datetime:
    """Parse ISO datetime string."""
    if not iso_str:
        return datetime.now(timezone.utc)
    try:
        if iso_str.endswith("Z"):
            return datetime.fromisoformat(iso_str[:-1] + "+00:00")
        return datetime.fromisoformat(iso_str)
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)
