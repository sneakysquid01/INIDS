"""Backpressure controller for the streaming pipeline.

Monitors consumer lag and applies progressive backpressure:
1. **Normal** — process all flows.
2. **Sampling** — drop a fraction of low-risk flows to catch up.
3. **Shedding** — reject new submissions with HTTP 503.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class BackpressureLevel(str, Enum):
    NORMAL = "normal"
    SAMPLING = "sampling"
    SHEDDING = "shedding"


class BackpressureController:
    """Monitors lag metrics and exposes the current backpressure level.

    Parameters
    ----------
    sampling_threshold:
        Lag count above which we enter SAMPLING mode.
    shedding_threshold:
        Lag count above which we enter SHEDDING mode.
    sample_rate:
        Fraction of flows to process during SAMPLING (0.0–1.0).
    cooldown_seconds:
        Minimum time between level changes to avoid oscillation.
    """

    def __init__(
        self,
        *,
        sampling_threshold: int = 5_000,
        shedding_threshold: int = 20_000,
        sample_rate: float = 0.25,
        cooldown_seconds: float = 10.0,
    ) -> None:
        self.sampling_threshold = sampling_threshold
        self.shedding_threshold = shedding_threshold
        self.sample_rate = max(0.01, min(1.0, sample_rate))
        self.cooldown_seconds = cooldown_seconds

        self._level = BackpressureLevel.NORMAL
        self._last_change = 0.0
        self._lock = Lock()
        self._counter = 0

    @property
    def level(self) -> BackpressureLevel:
        with self._lock:
            return self._level

    def update(self, current_lag: int) -> BackpressureLevel:
        """Re-evaluate backpressure level based on the latest lag metric."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_change < self.cooldown_seconds:
                return self._level

            if current_lag >= self.shedding_threshold:
                new_level = BackpressureLevel.SHEDDING
            elif current_lag >= self.sampling_threshold:
                new_level = BackpressureLevel.SAMPLING
            else:
                new_level = BackpressureLevel.NORMAL

            if new_level != self._level:
                logger.warning(
                    "Backpressure level changed: %s → %s (lag=%d)",
                    self._level.value,
                    new_level.value,
                    current_lag,
                )
                self._level = new_level
                self._last_change = now

            return self._level

    def should_process(self) -> bool:
        """Return True if the current flow should be processed.

        In NORMAL mode, always True.
        In SAMPLING mode, only ``sample_rate`` fraction passes.
        In SHEDDING mode, always False.
        """
        with self._lock:
            level = self._level
            self._counter += 1
            counter = self._counter

        if level == BackpressureLevel.NORMAL:
            return True
        if level == BackpressureLevel.SHEDDING:
            return False

        # Deterministic sampling using modular arithmetic.
        step = max(1, int(1.0 / self.sample_rate))
        return (counter % step) == 0

    def status(self) -> dict:
        with self._lock:
            return {
                "level": self._level.value,
                "sampling_threshold": self.sampling_threshold,
                "shedding_threshold": self.shedding_threshold,
                "sample_rate": self.sample_rate,
            }
