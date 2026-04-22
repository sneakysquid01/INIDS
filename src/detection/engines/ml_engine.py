"""ML Detection Engine — wraps the existing DetectionService as a pluggable engine."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.detection.engine_base import DetectionEngine, EngineResult
from src.schema import DEFAULT_FEATURE_ROW, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)


class MLEngine(DetectionEngine):
    """Wraps a trained scikit-learn model as a DetectionEngine.

    This is a thin adapter around the existing model inference logic from
    ``DetectionService`` so that the ML model participates in the multi-engine
    pipeline on equal footing with signature / anomaly / threshold engines.
    """

    def __init__(self, model: Any, *, engine_id: str = "ml_primary") -> None:
        self._model = model
        self._engine_id = engine_id

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
        logger.debug(f"Created DataFrame shape: {df.shape}, columns: {len(df.columns)}")
        
        try:
            pred = int(self._model.predict(df)[0])
            logger.debug(f"Model prediction: {pred}")
        except Exception as e:
            logger.exception(f"Model.predict() failed: {e}")
            raise
        
        try:
            proba = self._model.predict_proba(df)[0]
            logger.debug(f"Model predict_proba: {proba}")
        except Exception as e:
            logger.exception(f"Model.predict_proba() failed: {e}")
            raise
            
        confidence = round(float(max(proba)) * 100, 2)
        logger.debug(f"Confidence: {confidence}%")

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
