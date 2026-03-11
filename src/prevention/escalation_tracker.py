"""Per-IP escalation state machine.

Tracks each source IP through progressive response levels:

    CLEAN → ALERT → RATE_LIMIT → TEMP_BLOCK → PERM_BLOCK

Each detection event moves the IP one level up. Severity of the detection can
cause "skip" escalation (e.g., a critical hit jumps straight to TEMP_BLOCK).
Cool-down periods naturally de-escalate IPs back toward CLEAN.
"""
from __future__ import annotations

import logging
import time
from enum import IntEnum
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class EscalationLevel(IntEnum):
    CLEAN = 0
    ALERT = 1
    RATE_LIMIT = 2
    TEMP_BLOCK = 3
    PERM_BLOCK = 4


# Map severity to minimum escalation jump.
_SEVERITY_JUMP: dict[str, int] = {
    "critical": 3,  # jump at least to TEMP_BLOCK
    "high": 2,
    "medium": 1,
    "low": 0,
}


class _IPState:
    __slots__ = ("level", "hit_count", "last_hit", "first_hit")

    def __init__(self) -> None:
        self.level = EscalationLevel.CLEAN
        self.hit_count = 0
        self.last_hit = 0.0
        self.first_hit = 0.0


class EscalationTracker:
    """Thread-safe per-IP escalation tracker.

    Parameters
    ----------
    cooldown_seconds:
        Seconds of inactivity before an IP de-escalates one level.
    max_level:
        Maximum reachable level (default PERM_BLOCK).
    """

    def __init__(
        self,
        *,
        cooldown_seconds: float = 300.0,
        max_level: EscalationLevel = EscalationLevel.PERM_BLOCK,
    ) -> None:
        self._states: dict[str, _IPState] = {}
        self._cooldown = cooldown_seconds
        self._max_level = max_level
        self._lock = Lock()

    def record_hit(self, ip: str, severity: str = "low") -> EscalationLevel:
        """Record a detection hit for ``ip`` and return the new escalation level."""
        now = time.monotonic()
        with self._lock:
            state = self._states.get(ip)
            if state is None:
                state = _IPState()
                state.first_hit = now
                self._states[ip] = state

            # De-escalate if enough time has passed since last hit.
            if state.last_hit and (now - state.last_hit) > self._cooldown:
                steps_down = int((now - state.last_hit) // self._cooldown)
                state.level = EscalationLevel(max(0, state.level - steps_down))

            # Escalate: jump is at least 1, or more for high severity.
            jump = max(1, _SEVERITY_JUMP.get(severity, 1))
            new_level = min(state.level + jump, self._max_level)
            state.level = EscalationLevel(new_level)
            state.hit_count += 1
            state.last_hit = now

            logger.debug("Escalation[%s]: level=%s hits=%d severity=%s", ip, state.level.name, state.hit_count, severity)
            return state.level

    def get_level(self, ip: str) -> EscalationLevel:
        """Return current escalation level (with automatic de-escalation)."""
        now = time.monotonic()
        with self._lock:
            state = self._states.get(ip)
            if state is None:
                return EscalationLevel.CLEAN
            if state.last_hit and (now - state.last_hit) > self._cooldown:
                steps_down = int((now - state.last_hit) // self._cooldown)
                state.level = EscalationLevel(max(0, state.level - steps_down))
            return state.level

    def reset(self, ip: str) -> None:
        with self._lock:
            self._states.pop(ip, None)

    def summary(self) -> list[dict[str, Any]]:
        """Return escalation summary for all tracked IPs."""
        now = time.monotonic()
        rows: list[dict[str, Any]] = []
        with self._lock:
            for ip, state in self._states.items():
                if state.last_hit and (now - state.last_hit) > self._cooldown:
                    steps_down = int((now - state.last_hit) // self._cooldown)
                    state.level = EscalationLevel(max(0, state.level - steps_down))
                rows.append({
                    "ip": ip,
                    "level": state.level.name,
                    "level_value": int(state.level),
                    "hit_count": state.hit_count,
                    "seconds_since_last_hit": round(now - state.last_hit, 1) if state.last_hit else None,
                })
        return rows
