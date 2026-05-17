"""Anomaly-based Detection Engine — unsupervised outlier detection using IsolationForest."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.detection.engine_base import DetectionEngine, EngineResult
from src.schema import NUMERIC_FEATURES

logger = logging.getLogger(__name__)


class AnomalyEngine(DetectionEngine):
    """Uses scikit-learn IsolationForest for unsupervised anomaly detection.

    The engine can be trained on "normal" traffic and will flag outliers as
    suspicious.  It complements the supervised ML engine by catching novel
    attacks that the classifier has never seen.

    Parameters
    ----------
    buffer_size:
        Number of "normal" feature samples to collect before auto-fitting.
        Set to 0 to disable auto-fitting.
    model_path:
        Optional path to persist the fitted model as a joblib file.
    """

    def __init__(
        self,
        *,
        engine_id: str = "anomaly",
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
        buffer_size: int = 3000,
        model_path: str | Path | None = None,
        fp_manager: Any = None,
    ) -> None:
        self._engine_id = engine_id
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state
        # B-04: atomic model reference — replaced via _set_model(), read via _get_model().
        # CPython GIL makes single ref assignment atomic. Non-CPython: wrap in RLock.
        self._model_ref: Any = None
        self._model_lock = threading.Lock()
        self._fp_manager = fp_manager
        self._feature_names: list[str] = list(NUMERIC_FEATURES)
        self._buffer_size = int(buffer_size)
        self._model_path = Path(model_path) if model_path else None
        self._buffer: list[list[float]] = []
        self._buffer_lock = threading.Lock()
        self._fitted = False
        self._feature_count_warning_logged = False

        # Try to load a previously persisted model.
        if self._model_path and self._model_path.exists():
            try:
                from src.detection.ml_utils import (
                    load_checksum_manifest,
                    load_model_with_verification,
                    SecurityError,
                )
                manifest = load_checksum_manifest(self._model_path.parent)
                expected = manifest.get(self._model_path.name)
                if expected is None:
                    raise FileNotFoundError(
                        f"No checksum entry for {self._model_path.name} in manifest"
                    )
                self._set_model(load_model_with_verification(self._model_path, expected))
                self._fitted = True
                logger.info("AnomalyEngine loaded persisted model from %s", self._model_path)
            except SecurityError:
                raise
            except Exception:
                logger.warning("Failed to load anomaly model from %s; starting fresh", self._model_path, exc_info=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray | Any) -> None:
        """Train the IsolationForest on normal traffic data."""
        from sklearn.ensemble import IsolationForest  # lazy import to keep module light

        model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=self._random_state,
            n_jobs=-1,
        )
        model.fit(X)
        self._set_model(model)
        with self._buffer_lock:
            self._fitted = True
        logger.info("AnomalyEngine fitted on %d samples", X.shape[0])
        if self._model_path is not None:
            self._persist_model()

    def _persist_model(self) -> None:
        try:
            import joblib
            from src.detection.ml_utils import _compute_sha256, update_checksum_manifest
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._get_model(), self._model_path)
            digest = _compute_sha256(self._model_path)
            update_checksum_manifest(self._model_path.parent, self._model_path.name, digest)
            logger.info("AnomalyEngine model persisted to %s", self._model_path)
        except Exception:
            logger.warning("Failed to persist anomaly model to %s", self._model_path, exc_info=True)

    def add_sample(self, features: dict[str, Any]) -> bool:
        """Add a feature row to the rolling training buffer.

        When the buffer reaches ``buffer_size`` this method auto-fits and
        enables the engine.  Returns True if the engine was just fitted.
        """
        if self._buffer_size <= 0:
            return False
        vector = [float(features.get(f, 0.0)) for f in self._feature_names]
        with self._buffer_lock:
            self._buffer.append(vector)
            if len(self._buffer) < self._buffer_size:
                return False
            X = np.array(self._buffer)
            self._buffer = []  # reset after fitting

        logger.info("AnomalyEngine auto-fitting on %d buffered samples", len(X))
        self.fit(X)
        return True

    def buffer_status(self) -> dict[str, Any]:
        with self._buffer_lock:
            buffered = len(self._buffer)
        return {
            "fitted": self._fitted,
            "buffer_collected": buffered,
            "buffer_target": self._buffer_size,
            "buffer_pct": round(buffered / self._buffer_size * 100, 1) if self._buffer_size > 0 else 0,
        }

    def set_model(self, model: Any) -> None:
        """Inject a pre-trained IsolationForest/LOF model."""
        self._set_model(model)
        self._fitted = True

    # B-04: thread-safe model accessors
    def _set_model(self, model: Any) -> None:
        with self._model_lock:
            self._model_ref = model

    def _get_model(self) -> Any:
        return self._model_ref  # single read — GIL-safe in CPython

    def _vector_from_features(self, features: dict[str, Any], model: Any | None = None) -> np.ndarray:
        if model is None:
            model = self._get_model()
        values = [float(features.get(f, 0.0)) for f in self._feature_names]
        expected = getattr(model, "n_features_in_", len(values))
        if expected != len(values):
            if not self._feature_count_warning_logged:
                logger.warning(
                    "AnomalyEngine feature count mismatch: schema=%d model=%d; applying compatibility padding",
                    len(values),
                    expected,
                )
                self._feature_count_warning_logged = True
            if expected > len(values):
                values.extend([0.0] * (expected - len(values)))
            else:
                values = values[:expected]
        return np.array(values, dtype=float).reshape(1, -1)

    # ------------------------------------------------------------------
    # DetectionEngine interface
    # ------------------------------------------------------------------

    @property
    def engine_id(self) -> str:
        return self._engine_id

    @property
    def engine_type(self) -> str:
        return "anomaly"

    def is_ready(self) -> bool:
        model = self._get_model()
        return model is not None and hasattr(model, "predict")

    def evaluate(self, features: dict[str, Any]) -> EngineResult:
        source_ip = features.get("source_ip")

        # Check if source is a known false positive (suppressed)
        if self._fp_manager is not None and source_ip:
            try:
                if self._fp_manager.is_suppressed(source_ip):
                    logger.debug("AnomalyEngine: source %s suppressed by FP manager", source_ip)
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

        # B-02 / B-04: capture model snapshot; return unknown if not yet fitted.
        model = self._get_model()
        if model is None:
            from src.detection.engines.ml_engine import _increment_unknown_verdict_counter
            _increment_unknown_verdict_counter()
            return EngineResult(
                engine_id=self._engine_id,
                engine_type=self.engine_type,
                verdict="unknown",
                confidence=0.0,
                severity="low",
                attack_type="unknown",
                metadata={"error": "model_not_fitted", "fallback": True},
            )

        try:
            vector = self._vector_from_features(features, model)
            pred = int(model.predict(vector)[0])  # -1 = anomaly, 1 = normal
            score = float(model.decision_function(vector)[0])
        except Exception as exc:
            logger.error(
                "AnomalyEngine[%s] inference failed: %s — returning unknown verdict",
                self._engine_id, exc,
            )
            from src.detection.engines.ml_engine import _increment_unknown_verdict_counter
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

        # Convert IsolationForest output to our taxonomy.
        is_anomaly = pred == -1
        verdict = "suspicious" if is_anomaly else "normal"
        # Map decision_function score to a 0-100 confidence.
        # Scores near 0 or negative → high anomaly confidence.
        confidence = round(max(0.0, min(100.0, (1.0 - score) * 50 + 50)), 2) if is_anomaly else round(max(0.0, min(100.0, score * 50 + 50)), 2)
        severity = "medium" if is_anomaly else "low"

        return EngineResult(
            engine_id=self._engine_id,
            engine_type=self.engine_type,
            verdict=verdict,
            confidence=confidence,
            severity=severity,
            attack_type="anomaly" if is_anomaly else "normal",
            metadata={"raw_score": round(score, 6), "isolation_pred": pred},
        )
