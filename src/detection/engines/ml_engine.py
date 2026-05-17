"""ML Detection Engine — wraps the existing DetectionService as a pluggable engine."""
from __future__ import annotations

import logging
import threading
from typing import Any

import pandas as pd

from src.detection.engine_base import DetectionEngine, EngineResult
from src.schema import DEFAULT_FEATURE_ROW, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

# B-02: module-level fallback counter; incremented whenever inference fails.
# CPython GIL makes int += 1 effectively atomic for simple reads/writes.
_ml_unknown_verdict_total: int = 0
_ml_unknown_verdict_lock = threading.Lock()


def _increment_unknown_verdict_counter() -> None:
    global _ml_unknown_verdict_total
    with _ml_unknown_verdict_lock:
        _ml_unknown_verdict_total += 1


def get_unknown_verdict_total() -> int:
    """Return the cumulative count of inference fallbacks (for metrics endpoints)."""
    with _ml_unknown_verdict_lock:
        return _ml_unknown_verdict_total


class MLEngine(DetectionEngine):
    """Wraps a trained scikit-learn model as a DetectionEngine.

    This is a thin adapter around the existing model inference logic from
    ``DetectionService`` so that the ML model participates in the multi-engine
    pipeline on equal footing with signature / anomaly / threshold engines.
    """

    def __init__(self, model: Any, *, engine_id: str = "ml_primary", fp_manager: Any = None) -> None:
        self._model = model
        self._engine_id = engine_id
        self._fp_manager = fp_manager

    # ------------------------------------------------------------------
    # DetectionEngine interface
    # ------------------------------------------------------------------

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def engine_type(self) -> str:
        return "ml"

    def is_ready(self) -> bool:
        ready = self._model is not None and hasattr(self._model, "predict")
        logger.debug(f"MLEngine.is_ready() = {ready} (model={self._model is not None}, predict={hasattr(self._model, 'predict') if self._model else False})")
        return ready

    def evaluate(self, features: dict[str, Any]) -> EngineResult:
        # Check if source is a known false positive (suppressed)
        source_ip = features.get("source_ip")
        if self._fp_manager is not None and source_ip:
            try:
                if self._fp_manager.is_suppressed(source_ip):
                    logger.debug("MLEngine: source %s suppressed by FP manager", source_ip)
                    return EngineResult(
                        engine_id=self._engine_id,
                        engine_type=self.engine_type,
                        verdict="normal",
                        confidence=100.0,
                        severity="low",
                        attack_type="normal",
                        metadata={"suppressed_by_fp_manager": True},
                    )
            except Exception:
                logger.exception("FP manager suppression check failed for %s", source_ip)
        
        # Validate required columns
        logger.debug(f"MLEngine.evaluate() called with {len(features)} features")
        required_columns = set(FEATURE_COLUMNS)
        provided_columns = set(features.keys())
        missing_columns = required_columns - provided_columns
        
        if missing_columns:
            logger.warning(
                "Missing features for ML evaluation: %s (using defaults)",
                ", ".join(sorted(missing_columns)),
            )
            # If too many features missing, return low-confidence result
            if len(missing_columns) > 10:
                logger.debug(f"Too many missing features ({len(missing_columns)}), returning unknown verdict")
                return EngineResult(
                    engine_id=self._engine_id,
                    engine_type=self.engine_type,
                    verdict="unknown",
                    confidence=0.0,
                    severity="low",
                    attack_type="unknown",
                    metadata={
                        "error": f"too_many_missing_features ({len(missing_columns)})",
                        "missing": list(sorted(missing_columns)),
                    },
                )
        
        row = DEFAULT_FEATURE_ROW.copy()
        for key, value in features.items():
            if key in FEATURE_COLUMNS:
                try:
                    # Type-check numeric columns
                    if key in NUMERIC_FEATURES:
                        row[key] = float(value)
                    else:
                        row[key] = str(value)
                except (ValueError, TypeError):
                    logger.debug("Type conversion failed for %s=%s, using default", key, value)

        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        logger.debug("MLEngine DataFrame shape=%s columns=%d", df.shape, len(df.columns))

        try:
            pred = int(self._model.predict(df)[0])
        except Exception as exc:
            logger.error(
                "MLEngine[%s] predict() failed: %s — returning unknown verdict",
                self._engine_id, exc,
            )
            _increment_unknown_verdict_counter()
            return EngineResult(
                engine_id=self._engine_id,
                engine_type=self.engine_type,
                verdict="unknown",
                confidence=0.0,
                severity="low",
                attack_type="unknown",
                metadata={"error": str(exc), "fallback": True},
            )

        try:
            proba = self._model.predict_proba(df)[0]
        except Exception as exc:
            logger.error(
                "MLEngine[%s] predict_proba() failed: %s — returning unknown verdict",
                self._engine_id, exc,
            )
            _increment_unknown_verdict_counter()
            return EngineResult(
                engine_id=self._engine_id,
                engine_type=self.engine_type,
                verdict="unknown",
                confidence=0.0,
                severity="low",
                attack_type="unknown",
                metadata={"error": str(exc), "fallback": True},
            )

        confidence = round(float(max(proba)) * 100, 2)
        verdict = "attack" if pred == 1 else "normal"
        attack_type = features.get("attack_type", "unknown") if pred == 1 else "normal"
        severity = self._compute_severity(verdict, confidence)

        return EngineResult(
            engine_id=self._engine_id,
            engine_type=self.engine_type,
            verdict=verdict,
            confidence=confidence,
            severity=severity,
            attack_type=str(attack_type),
            metadata={"model_class": type(self._model).__name__},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_severity(verdict: str, confidence: float) -> str:
        if verdict == "attack" and confidence >= 90:
            return "critical"
        if verdict == "attack":
            return "high"
        if confidence < 60:
            return "medium"
        return "low"
