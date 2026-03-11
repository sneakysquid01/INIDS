"""Engine registry — manages registration, health, and lifecycle of detection engines."""
from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from src.detection.engine_base import DetectionEngine, EngineResult

logger = logging.getLogger(__name__)


class EngineRegistry:
    """Thread-safe registry that holds all active detection engines.

    Engines are registered by ``engine_id`` and can be enabled/disabled at
    runtime.  ``evaluate_all`` runs every *ready and enabled* engine against a
    feature dict and returns the list of ``EngineResult`` objects.
    """

    def __init__(self) -> None:
        self._engines: dict[str, DetectionEngine] = {}
        self._enabled: dict[str, bool] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, engine: DetectionEngine, *, enabled: bool = True) -> None:
        with self._lock:
            eid = engine.engine_id
            if eid in self._engines:
                logger.warning("Replacing existing engine %s", eid)
            self._engines[eid] = engine
            self._enabled[eid] = enabled
            logger.info("Registered engine %s (type=%s, enabled=%s)", eid, engine.engine_type, enabled)

    def unregister(self, engine_id: str) -> bool:
        with self._lock:
            removed = self._engines.pop(engine_id, None)
            self._enabled.pop(engine_id, None)
        if removed:
            logger.info("Unregistered engine %s", engine_id)
        return removed is not None

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def set_enabled(self, engine_id: str, enabled: bool) -> bool:
        with self._lock:
            if engine_id not in self._engines:
                return False
            self._enabled[engine_id] = enabled
        return True

    def is_enabled(self, engine_id: str) -> bool:
        with self._lock:
            return self._enabled.get(engine_id, False)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, features: dict[str, Any]) -> list[EngineResult]:
        """Run all ready+enabled engines and collect results.

        Each engine runs in the calling thread.  A failing engine logs the
        exception and is skipped — it does **not** prevent other engines from
        executing.
        """
        with self._lock:
            active = [
                (eid, eng)
                for eid, eng in self._engines.items()
                if self._enabled.get(eid, False) and eng.is_ready()
            ]

        results: list[EngineResult] = []
        for eid, engine in active:
            try:
                result = engine.evaluate(features)
                results.append(result)
            except Exception:
                logger.exception("Engine %s failed during evaluate", eid)
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_engines(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "engine_id": eid,
                    "engine_type": eng.engine_type,
                    "enabled": self._enabled.get(eid, False),
                    "ready": eng.is_ready(),
                }
                for eid, eng in self._engines.items()
            ]

    def get_engine(self, engine_id: str) -> DetectionEngine | None:
        with self._lock:
            return self._engines.get(engine_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._engines)
