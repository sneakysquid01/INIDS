"""ML model loading utilities with integrity verification.

PLAN.md Phase A Step 6 (A-06): All model loads must go through
load_model_with_verification() to prevent model substitution attacks.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

_VERIFY_MODE = None  # cached on first access


def _get_verify_mode() -> str:
    """Return the configured model verification mode.

    Modes:
        strict  — raise SecurityError on bad checksum (default)
        warn    — log CRITICAL and continue (rollback mode only)
        disabled — skip checksum check entirely
    """
    global _VERIFY_MODE
    if _VERIFY_MODE is None:
        _VERIFY_MODE = os.environ.get("INIDS_MODEL_VERIFY", "strict").strip().lower()
    return _VERIFY_MODE


def load_checksum_manifest(models_dir: str | Path) -> dict[str, str]:
    """Parse models/checksums.sha256 and return {filename: expected_hex_digest}."""
    manifest_path = Path(models_dir) / "checksums.sha256"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Model checksum manifest not found: {manifest_path}. "
            "Run scripts/generate_model_checksums.py to create it."
        )
    checksums: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            digest, name = parts
            checksums[name.strip()] = digest.strip().lower()
    return checksums


def update_checksum_manifest(models_dir: str | Path, filename: str, digest: str) -> None:
    """Add or replace the checksum entry for *filename* in checksums.sha256.

    Called by AnomalyEngine._persist_model() so freshly-saved models can be
    reloaded through load_model_with_verification() without a separate keygen step.
    """
    manifest_path = Path(models_dir) / "checksums.sha256"
    lines: list[str] = []
    if manifest_path.exists():
        lines = manifest_path.read_text(encoding="utf-8").splitlines()

    new_entry = f"{digest.lower()}  {filename}"
    updated = False
    for i, line in enumerate(lines):
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[1].strip() == filename:
            lines[i] = new_entry
            updated = True
            break
    if not updated:
        lines.append(new_entry)

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class SecurityError(RuntimeError):
    """Raised when a model fails integrity verification."""


def load_model_with_verification(path: str | Path, expected_sha256: str) -> Any:
    """Load a joblib model after verifying its SHA-256 checksum.

    PLAN.md Phase A Step 6 (A-06):
      - strict (default): raise SecurityError on mismatch
      - warn: log CRITICAL and continue (for rollback only — not for production use)
      - disabled: skip check (do not use in production)
    """
    path = Path(path)
    mode = _get_verify_mode()

    actual = _compute_sha256(path)
    if actual != expected_sha256.lower():
        msg = (
            f"Model integrity check FAILED for {path.name}: "
            f"expected={expected_sha256[:12]}... actual={actual[:12]}..."
        )
        if mode == "strict":
            logger.critical("model_integrity_check_failed path=%s mode=strict", path)
            raise SecurityError(msg)
        elif mode == "warn":
            # SHIM: Remove warn option in Phase F after full observation window.
            logger.critical(
                "WARN MODE ACTIVE — model integrity check failed but continuing. "
                "This mode is for rollback ONLY. path=%s", path
            )
        else:
            logger.debug("model_integrity_check_disabled path=%s", path)
    else:
        logger.debug("model_integrity_check_passed path=%s", path.name)

    return joblib.load(path)
