"""False-positive feedback manager.

Provides an API surface for analysts to mark alerts as false positives.
Tracks FP rates per rule/engine and can auto-suppress rules that exceed a
configurable threshold.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class FalsePositiveManager:
    """Collects FP feedback and derives auto-suppression recommendations.

    Parameters
    ----------
    suppress_threshold:
        Fraction of FP feedback (0.0–1.0) above which a rule/engine is
        recommended for suppression.
    min_samples:
        Minimum number of feedback entries before suppression kicks in.
    ops_store:
        Optional OpsStore for persisting suppression decisions. When provided,
        suppression state survives restarts.
    """

    def __init__(
        self,
        *,
        suppress_threshold: float = 0.7,
        min_samples: int = 10,
        ops_store=None,
    ) -> None:
        self.suppress_threshold = suppress_threshold
        self.min_samples = min_samples
        self._ops_store = ops_store

        # Keyed by (engine_id, rule_id or "model")
        self._total: dict[tuple[str, str], int] = defaultdict(int)
        self._fp_count: dict[tuple[str, str], int] = defaultdict(int)
        self._suppressed: set[tuple[str, str]] = set()
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_from_store(self) -> int:
        """Load persisted suppression state from ops_store. Returns count loaded."""
        if self._ops_store is None:
            return 0
        try:
            rows = self._ops_store.list_fp_suppressions()
        except Exception:
            logger.exception("Failed to load FP suppressions from store")
            return 0
        loaded = 0
        with self._lock:
            for row in rows:
                if row.get("suppressed"):
                    key = (str(row["engine_id"]), str(row["rule_id"]))
                    self._suppressed.add(key)
                    loaded += 1
        if loaded:
            logger.info("Loaded %d FP suppressions from persistent store", loaded)
        return loaded

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def report_fp(self, engine_id: str, rule_id: str = "model", *, alert_id: str = "") -> None:
        """Record that a particular alert was a false positive."""
        key = (engine_id, rule_id)
        with self._lock:
            self._total[key] += 1
            self._fp_count[key] += 1
            self._check_suppress(key)
        logger.info("FP reported: engine=%s rule=%s alert=%s", engine_id, rule_id, alert_id)

    def report_tp(self, engine_id: str, rule_id: str = "model") -> None:
        """Record that a particular alert was a true positive."""
        key = (engine_id, rule_id)
        with self._lock:
            self._total[key] += 1

    # ------------------------------------------------------------------
    # Suppression
    # ------------------------------------------------------------------

    def is_suppressed(self, engine_id: str, rule_id: str = "model") -> bool:
        with self._lock:
            return (engine_id, rule_id) in self._suppressed

    def suppress(self, engine_id: str, rule_id: str = "model") -> bool:
        """Explicitly suppress a rule/engine (analyst-driven). Returns True if newly suppressed."""
        key = (engine_id, rule_id)
        with self._lock:
            already = key in self._suppressed
            self._suppressed.add(key)
        if not already and self._ops_store is not None:
            try:
                self._ops_store.save_fp_suppression(engine_id, rule_id)
            except Exception:
                logger.exception("Failed to persist FP suppression engine=%s rule=%s", engine_id, rule_id)
        logger.info("Suppressed engine=%s rule=%s (explicit)", engine_id, rule_id)
        return not already

    def unsuppress(self, engine_id: str, rule_id: str = "model") -> bool:
        with self._lock:
            key = (engine_id, rule_id)
            if key in self._suppressed:
                self._suppressed.discard(key)
                removed = True
            else:
                removed = False
        if removed and self._ops_store is not None:
            try:
                self._ops_store.delete_fp_suppression(engine_id, rule_id)
            except Exception:
                logger.exception("Failed to delete FP suppression engine=%s rule=%s", engine_id, rule_id)
        return removed

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def fp_rate(self, engine_id: str, rule_id: str = "model") -> float:
        with self._lock:
            key = (engine_id, rule_id)
            total = self._total.get(key, 0)
            if total == 0:
                return 0.0
            return self._fp_count.get(key, 0) / total

    def stats(self) -> list[dict[str, Any]]:
        with self._lock:
            rows: list[dict[str, Any]] = []
            for key in sorted(self._total.keys()):
                total = self._total[key]
                fps = self._fp_count.get(key, 0)
                rows.append({
                    "engine_id": key[0],
                    "rule_id": key[1],
                    "total": total,
                    "false_positives": fps,
                    "fp_rate": round(fps / total, 4) if total else 0.0,
                    "suppressed": key in self._suppressed,
                })
            return rows

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_suppress(self, key: tuple[str, str]) -> None:
        """Auto-suppress if FP rate exceeds threshold (called under lock)."""
        total = self._total.get(key, 0)
        if total < self.min_samples:
            return
        rate = self._fp_count.get(key, 0) / total
        if rate >= self.suppress_threshold and key not in self._suppressed:
            self._suppressed.add(key)
            logger.warning(
                "Auto-suppressed engine=%s rule=%s (FP rate=%.2f, n=%d)",
                key[0],
                key[1],
                rate,
                total,
            )
            # Persist outside lock to avoid potential deadlock on store call.
            if self._ops_store is not None:
                try:
                    self._ops_store.save_fp_suppression(key[0], key[1])
                except Exception:
                    logger.exception("Failed to persist auto-suppression engine=%s rule=%s", key[0], key[1])
