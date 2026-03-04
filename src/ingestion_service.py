from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from typing import Any

import pandas as pd

from src.schema import FEATURE_COLUMNS, NUMERIC_FEATURES, DEFAULT_FEATURE_ROW


@dataclass
class IngestionRecord:
    source: str
    payload: dict[str, Any]


class InMemoryIngestionQueue:
    def __init__(self, max_items: int = 10000):
        self.max_items = max_items
        self._queue: deque[IngestionRecord] = deque()

    def enqueue(self, record: IngestionRecord) -> None:
        self._queue.append(record)
        while len(self._queue) > self.max_items:
            self._queue.popleft()

    def dequeue(self) -> IngestionRecord | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def size(self) -> int:
        return len(self._queue)


class RedisStreamIngestionQueue:
    """Redis stream-backed ingestion queue with best-effort decoding."""

    def __init__(
        self,
        redis_client,
        *,
        stream_key: str = "inids:ingestion",
        max_items: int = 10000,
    ):
        self.redis_client = redis_client
        self.stream_key = stream_key
        self.max_items = max_items
        self._last_id = "0-0"

    @staticmethod
    def from_url(
        redis_url: str,
        *,
        stream_key: str = "inids:ingestion",
        max_items: int = 10000,
    ) -> "RedisStreamIngestionQueue":
        import redis

        client = redis.from_url(redis_url)
        return RedisStreamIngestionQueue(client, stream_key=stream_key, max_items=max_items)

    def enqueue(self, record: IngestionRecord) -> None:
        payload = {
            "source": record.source,
            "payload": json.dumps(record.payload),
        }
        self.redis_client.xadd(self.stream_key, payload, maxlen=self.max_items, approximate=True)

    def dequeue(self) -> IngestionRecord | None:
        rows = self.redis_client.xread({self.stream_key: self._last_id}, count=1, block=1)
        if not rows:
            return None
        _, events = rows[0]
        if not events:
            return None
        event_id, values = events[0]
        self._last_id = event_id.decode() if isinstance(event_id, bytes) else str(event_id)

        source = values.get("source", "ingestion")
        payload = values.get("payload", "{}")
        if isinstance(source, bytes):
            source = source.decode()
        if isinstance(payload, bytes):
            payload = payload.decode()
        decoded_payload = json.loads(payload)
        if not isinstance(decoded_payload, dict):
            decoded_payload = {}
        return IngestionRecord(source=str(source), payload=decoded_payload)

    def size(self) -> int:
        return int(self.redis_client.xlen(self.stream_key))


class IngestionService:
    def __init__(
        self,
        queue: InMemoryIngestionQueue,
        redis_stream_queue: RedisStreamIngestionQueue | None = None,
    ):
        self.queue = queue
        self.redis_stream_queue = redis_stream_queue

    def normalize_features(self, raw: dict[str, Any]) -> dict[str, Any]:
        row = DEFAULT_FEATURE_ROW.copy()
        for key, value in raw.items():
            if key in FEATURE_COLUMNS:
                row[key] = value

        for col in NUMERIC_FEATURES:
            row[col] = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(row[col]):
                raise ValueError(f"Invalid numeric value for '{col}'")

        for col in FEATURE_COLUMNS:
            if col not in NUMERIC_FEATURES:
                row[col] = str(row[col])

        return row

    def enqueue_record(self, payload: dict[str, Any], source: str = "ingestion") -> None:
        normalized = self.normalize_features(payload)
        record = IngestionRecord(source=source, payload=normalized)
        if self.redis_stream_queue is not None:
            try:
                self.redis_stream_queue.enqueue(record)
                return
            except Exception:
                # Fallback path keeps ingestion alive when Redis is unavailable.
                pass
        self.queue.enqueue(record)

    def enqueue_batch(self, rows: list[dict[str, Any]], source: str = "batch") -> int:
        for row in rows:
            self.enqueue_record(row, source=source)
        return len(rows)

    def process_one(self, handler) -> dict[str, Any] | None:
        record = None
        if self.redis_stream_queue is not None:
            try:
                record = self.redis_stream_queue.dequeue()
            except Exception:
                record = None
        if record is None:
            record = self.queue.dequeue()
        if record is None:
            return None
        result = handler(record.payload, record.source)
        return {
            "source": record.source,
            "result": result,
        }

    def process_all(self, handler, max_items: int = 100) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _ in range(max_items):
            item = self.process_one(handler)
            if item is None:
                break
            results.append(item)
        return results
