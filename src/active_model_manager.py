"""Step 41 — ACTIVE_MODELS manifest, staged model loading, and atomic hot-reload.

Design:
  models/active_models.json    — which version of each model slot is live
  models/staged/<name>.pkl     — operator drops a candidate here; manager verifies/promotes

Promotion flow:
  1. Operator places new model at models/staged/<slot_name>.pkl
  2. Call ActiveModelManager.promote(slot_name, sha256=expected_hash)
  3. Manager verifies checksum of staged file
  4. Atomically replaces active file via os.replace() (POSIX-atomic on same FS)
  5. Updates active_models.json manifest
  6. Notifies registered hot-reload callbacks so live engines swap in the new model
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "active_models.json"
_STAGED_SUBDIR = "staged"
_FORMAT_VERSION = "1"


@dataclass
class ModelSlot:
    """A single entry in the ACTIVE_MODELS manifest."""
    slot_name: str
    model_path: str       # path to the currently active .pkl
    model_version: str    # human-readable version tag
    sha256: str           # hex digest expected at activation time
    activated_at: str     # ISO-8601 timestamp

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSlot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ChecksumMismatch(Exception):
    """Raised when a staged model's SHA-256 does not match the declared hash."""


class ActiveModelManager:
    """Manages the ACTIVE_MODELS manifest and staged-model promotion.

    Thread-safe: a single RLock protects all manifest I/O and promotion steps.

    Usage::

        mgr = ActiveModelManager(models_dir="/app/models")

        # Register an existing model slot
        mgr.register_slot("rf_primary", "models/rf_nsl_kdd.pkl", "v1", sha256="...")

        # During operation, operator drops a new file at models/staged/rf_primary.pkl
        # Then promote it:
        new_ctx = mgr.promote("rf_primary", expected_sha256="<new hash>")

        # Engines listening via add_reload_callback() are notified
    """

    def __init__(self, models_dir: str | Path) -> None:
        self._dir = Path(models_dir)
        self._staged_dir = self._dir / _STAGED_SUBDIR
        self._manifest_path = self._dir / _MANIFEST_FILENAME
        self._lock = threading.RLock()
        self._reload_callbacks: dict[str, list[Callable[[str], None]]] = {}

        self._dir.mkdir(parents=True, exist_ok=True)
        self._staged_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _read_manifest(self) -> dict[str, Any]:
        if not self._manifest_path.exists():
            return {"format_version": _FORMAT_VERSION, "models": {}}
        try:
            with open(self._manifest_path, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data.get("models"), dict):
                data["models"] = {}
            return data
        except Exception:
            logger.exception("Failed to read active_models.json; returning empty manifest")
            return {"format_version": _FORMAT_VERSION, "models": {}}

    def _write_manifest(self, data: dict[str, Any]) -> None:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self._manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, self._manifest_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_slot(
        self,
        slot_name: str,
        model_path: str | Path,
        model_version: str,
        sha256: str,
    ) -> ModelSlot:
        """Register (or update) a model slot in the manifest.

        Call this once per model during application startup after verifying
        the checksum of the currently active model file.
        """
        slot = ModelSlot(
            slot_name=slot_name,
            model_path=str(model_path),
            model_version=model_version,
            sha256=sha256,
            activated_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            data = self._read_manifest()
            data["models"][slot_name] = slot.to_dict()
            self._write_manifest(data)
        logger.info("ActiveModelManager: registered slot %s → %s (%s)", slot_name, model_path, model_version)
        return slot

    def get_slot(self, slot_name: str) -> ModelSlot | None:
        """Return the manifest entry for slot_name, or None if not registered."""
        with self._lock:
            data = self._read_manifest()
        entry = data["models"].get(slot_name)
        return ModelSlot.from_dict(entry) if entry else None

    def list_slots(self) -> list[ModelSlot]:
        """Return all registered model slots."""
        with self._lock:
            data = self._read_manifest()
        return [ModelSlot.from_dict(v) for v in data["models"].values()]

    def promote(
        self,
        slot_name: str,
        expected_sha256: str,
        new_version: str | None = None,
    ) -> ModelSlot:
        """Promote a staged model to active after checksum verification.

        The staged file must be at: <models_dir>/staged/<slot_name>.pkl

        Raises:
            FileNotFoundError: staged file does not exist
            ChecksumMismatch: staged file hash != expected_sha256
        """
        staged_path = self._staged_dir / f"{slot_name}.pkl"
        if not staged_path.exists():
            raise FileNotFoundError(f"Staged model not found: {staged_path}")

        actual_sha256 = _sha256_of(staged_path)
        if actual_sha256 != expected_sha256:
            raise ChecksumMismatch(
                f"Staged model checksum mismatch for {slot_name}: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )

        with self._lock:
            data = self._read_manifest()
            existing = data["models"].get(slot_name)
            active_path = Path(existing["model_path"]) if existing else self._dir / f"{slot_name}.pkl"

            # Atomic replace: new file lands at active_path
            os.replace(staged_path, active_path)

            version = new_version or _next_version(existing)
            slot = ModelSlot(
                slot_name=slot_name,
                model_path=str(active_path),
                model_version=version,
                sha256=expected_sha256,
                activated_at=datetime.now(timezone.utc).isoformat(),
            )
            data["models"][slot_name] = slot.to_dict()
            self._write_manifest(data)

        logger.info(
            "ActiveModelManager: promoted %s → %s (version=%s, sha256=%.12s…)",
            slot_name, active_path, version, expected_sha256,
        )

        self._fire_callbacks(slot_name, str(active_path))
        return slot

    # ------------------------------------------------------------------
    # Hot-reload callback registry
    # ------------------------------------------------------------------

    def add_reload_callback(self, slot_name: str, callback: Callable[[str], None]) -> None:
        """Register a callback invoked after a successful promotion of slot_name.

        callback(new_model_path: str) is called in the promoting thread.
        Engines use this to atomically swap their model reference.
        """
        with self._lock:
            self._reload_callbacks.setdefault(slot_name, []).append(callback)

    def _fire_callbacks(self, slot_name: str, new_path: str) -> None:
        callbacks = self._reload_callbacks.get(slot_name, [])[:]
        for cb in callbacks:
            try:
                cb(new_path)
            except Exception:
                logger.exception("ActiveModelManager: reload callback error for %s", slot_name)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _next_version(existing: dict[str, Any] | None) -> str:
    if not existing:
        return "v1"
    current = existing.get("model_version", "v0")
    try:
        n = int(current.lstrip("v")) + 1
        return f"v{n}"
    except ValueError:
        return f"{current}_next"
