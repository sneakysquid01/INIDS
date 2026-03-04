from __future__ import annotations

from collections import deque
from dataclasses import dataclass
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
from threading import Lock
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
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
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        self._lock = Lock()

    def enqueue(self, record: IngestionRecord) -> None:
        with self._lock:
            self._queue.append(record)
            while len(self._queue) > self.max_items:
                self._queue.popleft()

    def dequeue(self) -> IngestionRecord | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def size(self) -> int:
        with self._lock:
            return len(self._queue)
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs

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
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs


class IngestionService:
    def __init__(self, queue: InMemoryIngestionQueue):
        self.queue = queue

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
        self.queue.enqueue(IngestionRecord(source=source, payload=normalized))

    def enqueue_batch(self, rows: list[dict[str, Any]], source: str = "batch") -> int:
        for row in rows:
            self.enqueue_record(row, source=source)
        return len(rows)

    def process_one(self, handler) -> dict[str, Any] | None:
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
