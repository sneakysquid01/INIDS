"""Stream processor — Redis Streams consumer group reader for the detection pipeline.

Reads flows from a Redis stream using consumer groups (at-least-once delivery),
runs them through the multi-engine detection registry, and publishes results to
the EventBus.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from src.detection.aggregator import AggregatedResult, EngineAggregator
from src.detection.engine_registry import EngineRegistry
from src.feature_engineering import enrich_single_row

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Consumes flows from a Redis Stream and runs multi-engine detection.

    Uses consumer groups so that multiple workers can share the load while
    guaranteeing at-least-once delivery.
    """

    def __init__(
        self,
        redis_client: Any,
        engine_registry: EngineRegistry,
        aggregator: EngineAggregator,
        *,
        stream_key: str = "inids:flows",
        group_name: str = "inids-workers",
        consumer_name: str = "worker-1",
        batch_size: int = 50,
        block_ms: int = 2000,
        result_callback: Callable[[AggregatedResult, dict[str, Any]], None] | None = None,
    ) -> None:
        self.redis = redis_client
        self.engine_registry = engine_registry
        self.aggregator = aggregator
        self.stream_key = stream_key
        self.group_name = group_name
        self.consumer_name = consumer_name
        self.batch_size = batch_size
        self.block_ms = block_ms
        self.result_callback = result_callback
        self._running = False

        self._ensure_group()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_group(self) -> None:
        """Create consumer group if it doesn't already exist."""
        try:
            self.redis.xgroup_create(self.stream_key, self.group_name, id="0", mkstream=True)
            logger.info("Created consumer group '%s' on stream '%s'", self.group_name, self.stream_key)
        except Exception as exc:
            if "BUSYGROUP" in str(exc):
                logger.debug("Consumer group '%s' already exists", self.group_name)
            else:
                raise

    def run(self) -> None:
        """Blocking loop: read → detect → ack. Call ``stop()`` to exit."""
        self._running = True
        logger.info("StreamProcessor %s starting on %s/%s", self.consumer_name, self.stream_key, self.group_name)

        while self._running:
            try:
                entries = self.redis.xreadgroup(
                    self.group_name,
                    self.consumer_name,
                    {self.stream_key: ">"},
                    count=self.batch_size,
                    block=self.block_ms,
                )
                if not entries:
                    continue

                for _stream, messages in entries:
                    for msg_id, fields in messages:
                        self._process_message(msg_id, fields)

            except Exception:
                logger.exception("StreamProcessor read error, retrying after 1 s")
                time.sleep(1)

    def stop(self) -> None:
        self._running = False
        logger.info("StreamProcessor %s stopping", self.consumer_name)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process_message(self, msg_id: Any, fields: dict) -> None:
        try:
            features = self._decode_fields(fields)
            try:
                engine_features = enrich_single_row(features)
            except Exception:
                logger.warning("Feature enrichment failed in StreamProcessor; falling back to raw features", exc_info=True)
                engine_features = features

            results = self.engine_registry.evaluate_all(engine_features)
            aggregated = self.aggregator.aggregate(results)

            if self.result_callback is not None:
                self.result_callback(aggregated, engine_features)

            # ACK only after successful processing.
            self.redis.xack(self.stream_key, self.group_name, msg_id)
        except Exception:
            logger.exception("Failed to process message %s", msg_id)

    @staticmethod
    def _decode_fields(fields: dict) -> dict[str, Any]:
        """Decode Redis byte fields into a Python dict."""
        decoded: dict[str, Any] = {}
        for key, value in fields.items():
            k = key.decode() if isinstance(key, bytes) else str(key)
            v = value.decode() if isinstance(value, bytes) else str(value)
            # Try to parse JSON payload (bulk features might be stored as one JSON blob).
            if k == "payload":
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            try:
                decoded[k] = float(v)
            except ValueError:
                decoded[k] = v
        return decoded

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def pending_count(self) -> int:
        """Return number of pending (un-ACKed) messages for this consumer group."""
        try:
            info = self.redis.xpending(self.stream_key, self.group_name)
            return int(info["pending"]) if isinstance(info, dict) else 0
        except Exception:
            return 0

    def lag(self) -> int:
        """Approximate lag: stream length minus delivered count."""
        try:
            stream_len = self.redis.xlen(self.stream_key)
            info = self.redis.xinfo_groups(self.stream_key)
            for g in info:
                name = g.get("name", b"")
                if isinstance(name, bytes):
                    name = name.decode()
                if name == self.group_name:
                    delivered = g.get("last-delivered-id", b"0-0")
                    if isinstance(delivered, bytes):
                        delivered = delivered.decode()
                    # Rough estimate.
                    return max(0, stream_len - self.pending_count())
            return stream_len
        except Exception:
            return 0
