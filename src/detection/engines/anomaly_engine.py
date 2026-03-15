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
    ) -> None:
        self._engine_id = engine_id
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._model: Any = None
        self._feature_names: list[str] = list(NUMERIC_FEATURES)
        self._buffer_size = int(buffer_size)
        self._model_path = Path(model_path) if model_path else None
        self._buffer: list[list[float]] = []
        self._buffer_lock = threading.Lock()
        self._fitted = False

        # Try to load a previously persisted model.
        if self._model_path and self._model_path.exists():
            try:
                import joblib
                self._model = joblib.load(self._model_path)
                self._fitted = True
                logger.info("AnomalyEngine loaded persisted model from %s", self._model_path)
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
        with self._buffer_lock:
            self._model = model
            self._fitted = True
        logger.info("AnomalyEngine fitted on %d samples", X.shape[0])
        if self._model_path is not None:
            self._persist_model()

    def _persist_model(self) -> None:
        try:
            import joblib
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, self._model_path)
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
        self._model = model
        self._fitted = True

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
        return self._model is not None and hasattr(self._model, "predict")

    def evaluate(self, features: dict[str, Any]) -> EngineResult:
        vector = np.array(
            [float(features.get(f, 0.0)) for f in self._feature_names]
        ).reshape(1, -1)

        pred = int(self._model.predict(vector)[0])  # -1 = anomaly, 1 = normal
        score = float(self._model.decision_function(vector)[0])

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
