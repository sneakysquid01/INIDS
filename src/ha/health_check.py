"""Enhanced health-check service for production readiness.

Aggregates the health status of all subsystems (model, engines, Redis,
OpsStore, ingestion queue, etc.) into a structured report suitable for
load-balancer health probes and monitoring dashboards.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HealthCheck:
    """Aggregates health probes from registered subsystems.

    Usage::

        hc = HealthCheck()
        hc.register("model", lambda: {"ready": model is not None})
        hc.register("ops_db", lambda: {"ready": ops_store.ping()})
        report = hc.check()  # {"status": "healthy", "checks": {...}}
    """

    def __init__(self) -> None:
        self._probes: dict[str, Callable[[], dict[str, Any]]] = {}
        self._start_time = time.monotonic()

    def register(self, name: str, probe: Callable[[], dict[str, Any]]) -> None:
        self._probes[name] = probe

    def check(self) -> dict[str, Any]:
        """Run all probes and return an aggregate health report."""
        checks: dict[str, Any] = {}
        all_ready = True

        for name, probe in self._probes.items():
            try:
                result = probe()
                checks[name] = result
                if not result.get("ready", True):
                    all_ready = False
            except Exception as exc:
                checks[name] = {"ready": False, "error": str(exc)}
                all_ready = False

        return {
            "status": "healthy" if all_ready else "degraded",
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "checks": checks,
        }

    def is_healthy(self) -> bool:
        return self.check()["status"] == "healthy"
