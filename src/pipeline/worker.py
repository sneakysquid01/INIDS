"""Standalone pipeline worker that ties together stream processing, backpressure,
and the detection engine registry.

Usage (as a separate process)::

    python -m src.pipeline.worker --redis-url redis://localhost:6379 --consumer worker-1

Or imported and wired into the Flask application for in-process operation.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class PipelineWorker:
    """Orchestrates a StreamProcessor with backpressure monitoring.

    Can run as a background thread inside the app process or as a standalone
    process consuming from Redis Streams.
    """

    def __init__(
        self,
        stream_processor,
        backpressure_controller,
        *,
        lag_poll_interval: float = 5.0,
    ) -> None:
        self.processor = stream_processor
        self.bp = backpressure_controller
        self.lag_poll_interval = lag_poll_interval
        self._thread: threading.Thread | None = None
        self._lag_thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker and lag monitor in background threads."""
        if self._running:
            return
        self._running = True

        self._lag_thread = threading.Thread(target=self._lag_loop, daemon=True, name="bp-lag-monitor")
        self._lag_thread.start()

        self._thread = threading.Thread(target=self._run, daemon=True, name="pipeline-worker")
        self._thread.start()

        logger.info("PipelineWorker started")

    def stop(self) -> None:
        self._running = False
        self.processor.stop()
        logger.info("PipelineWorker stopping")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self.processor.run()
        except Exception:
            logger.exception("StreamProcessor exited with error")

    def _lag_loop(self) -> None:
        while self._running:
            try:
                lag = self.processor.lag()
                self.bp.update(lag)
            except Exception:
                logger.exception("Lag poll error")
            time.sleep(self.lag_poll_interval)

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "backpressure": self.bp.status(),
            "pending": self.processor.pending_count() if self._running else 0,
        }


def _main() -> None:  # pragma: no cover
    """CLI entry-point for standalone worker process."""
    parser = argparse.ArgumentParser(description="INIDS pipeline worker")
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--stream-key", default="inids:flows")
    parser.add_argument("--group", default="inids-workers")
    parser.add_argument("--consumer", default="worker-1")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        import redis as redis_lib
    except ImportError:
        logger.error("redis package not installed — run: pip install redis")
        sys.exit(1)

    from src.detection.aggregator import EngineAggregator, AggregationStrategy
    from src.detection.engine_registry import EngineRegistry
    from src.pipeline.stream_processor import StreamProcessor
    from src.pipeline.backpressure import BackpressureController

    rc = redis_lib.from_url(args.redis_url)
    registry = EngineRegistry()
    aggregator = EngineAggregator(AggregationStrategy.ANY_TRIGGER)

    # Engines will be registered by the caller or config loader in production.
    processor = StreamProcessor(
        rc,
        registry,
        aggregator,
        stream_key=args.stream_key,
        group_name=args.group,
        consumer_name=args.consumer,
        batch_size=args.batch_size,
    )

    bp = BackpressureController()
    worker = PipelineWorker(processor, bp)

    def _shutdown(sig, frame):
        worker.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    worker.start()
    # Keep main thread alive.
    try:
        while worker._running:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()


if __name__ == "__main__":
    _main()
