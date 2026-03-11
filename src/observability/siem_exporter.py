"""SIEM export service — batches structured events for forwarding to SIEM platforms.

Accumulates detection/action/audit events and exports them periodically as
JSON lines (JSONL) bundles, suitable for shipping to Elasticsearch, Splunk HEC,
or a Syslog collector.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_MAX_BUFFER = 10_000


class SiemExporter:
    """Accumulates events and exports them as JSONL batches.

    Events are added via ``emit()`` and retrieved via ``flush()`` which
    returns the next N events as a list of dicts (or a JSONL string).
    """

    def __init__(self, *, max_buffer: int = _MAX_BUFFER) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_buffer)
        self._lock = Lock()
        self._total_emitted = 0

    def emit(self, event: dict[str, Any]) -> None:
        """Add an event to the export buffer."""
        with self._lock:
            self._buffer.append(event)
            self._total_emitted += 1

    def flush(self, max_items: int = 500) -> list[dict[str, Any]]:
        """Drain up to ``max_items`` events from the buffer."""
        batch: list[dict[str, Any]] = []
        with self._lock:
            for _ in range(min(max_items, len(self._buffer))):
                batch.append(self._buffer.popleft())
        return batch

    def flush_jsonl(self, max_items: int = 500) -> str:
        """Drain events as a JSON-Lines formatted string."""
        batch = self.flush(max_items)
        return "\n".join(json.dumps(e, default=str, separators=(",", ":")) for e in batch)

    def pending(self) -> int:
        with self._lock:
            return len(self._buffer)

    def total_emitted(self) -> int:
        with self._lock:
            return self._total_emitted

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._buffer),
                "total_emitted": self._total_emitted,
                "buffer_capacity": self._buffer.maxlen,
            }
