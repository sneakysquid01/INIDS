from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class ModelRegistryEntry:
    name: str
    model_path: str
    accuracy: float
    f1_score: float
    training_time: float
    model_version: str
    training_date: str
    dataset_hash: str
    drift_alert: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    def __init__(self, registry_file: str):
        self.registry_file = registry_file
        os.makedirs(os.path.dirname(registry_file), exist_ok=True)

    def _read(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.registry_file):
            return []
        with open(self.registry_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, entries: list[dict[str, Any]]) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    def register(
        self,
        name: str,
        model_path: str,
        accuracy: float,
        f1_score: float,
        training_time: float,
        model_version: str | None = None,
        training_date: str | None = None,
        dataset_hash: str | None = None,
        drift_alert: str | None = None,
    ) -> None:
        entries = self._read()
        normalized_name = str(name)
        if model_version is None:
            existing = [e for e in entries if e.get("name") == normalized_name]
            model_version = f"v{len(existing) + 1}"
        if training_date is None:
            training_date = datetime.now(timezone.utc).date().isoformat()
        entries.append(
            ModelRegistryEntry(
                name=normalized_name,
                model_path=model_path,
                accuracy=float(accuracy),
                f1_score=float(f1_score),
                training_time=float(training_time),
                model_version=str(model_version),
                training_date=str(training_date),
                dataset_hash=str(dataset_hash or ""),
                drift_alert=drift_alert,
                created_at=datetime.now(timezone.utc).isoformat(),
            ).to_dict()
        )
        self._write(entries)

    def list_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        entries = self._read()
        entries = sorted(entries, key=lambda x: x.get("created_at", ""), reverse=True)
        return entries[:limit]
