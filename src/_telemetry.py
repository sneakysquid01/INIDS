"""Lightweight in-process counters for operational observability.

Exposed via /api/health as simple integer fields. Not a Prometheus replacement —
just enough to make failure rates visible without an external metrics stack.
"""
from __future__ import annotations

import threading


class _Counter:
    """Thread-safe monotonic counter."""

    __slots__ = ("_value", "_lock")

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._value += n

    def get(self) -> int:
        with self._lock:
            return self._value

    def reset(self) -> int:
        with self._lock:
            v = self._value
            self._value = 0
            return v


# Module-level counters

anomaly_add_sample_errors: _Counter = _Counter()
streamer_emit_errors: dict[str, _Counter] = {}
_streamer_lock = threading.Lock()


def get_streamer_errors(room: str) -> _Counter:
    with _streamer_lock:
        if room not in streamer_emit_errors:
            streamer_emit_errors[room] = _Counter()
        return streamer_emit_errors[room]


def health_snapshot() -> dict:
    """Return a dict of current counter values for /api/health."""
    with _streamer_lock:
        by_room = {r: c.get() for r, c in streamer_emit_errors.items()}
    return {
        "anomaly_add_sample_errors": anomaly_add_sample_errors.get(),
        "streamer_emit_errors_by_room": by_room,
    }
