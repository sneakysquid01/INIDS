from __future__ import annotations

import logging
from threading import Event, Thread
from time import sleep

from src.ips.action_executor import ActionExecutor


class PreventionScheduler:
    """Background worker for expiry cleanup and reconciliation."""

    def __init__(
        self,
        executor: ActionExecutor,
        *,
        interval_seconds: int = 15,
        reconcile_every: int = 20,
    ):
        self.executor = executor
        self.interval_seconds = max(5, int(interval_seconds))
        self.reconcile_every = max(1, int(reconcile_every))
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._tick = 0
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="inids-prevention-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._tick += 1
            try:
                removed = self.executor.cleanup_expired_actions()
                if removed:
                    self.logger.info("scheduler cleanup removed=%s", removed)
                if self._tick % self.reconcile_every == 0:
                    summary = self.executor.reconcile()
                    self.logger.info("scheduler reconcile summary=%s", summary)
            except Exception:
                self.logger.exception("scheduler loop failed")
            sleep(self.interval_seconds)

