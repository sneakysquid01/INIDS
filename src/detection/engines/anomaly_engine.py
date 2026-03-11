"""Anomaly-based Detection Engine — unsupervised outlier detection using IsolationForest."""
from __future__ import annotations

import logging
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
    """

    def __init__(
        self,
        *,
        engine_id: str = "anomaly",
        contamination: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
    ) -> None:
        self._engine_id = engine_id
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._model: Any = None
        self._feature_names: list[str] = list(NUMERIC_FEATURES)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray | Any) -> None:
        """Train the IsolationForest on normal traffic data."""
        from sklearn.ensemble import IsolationForest  # lazy import to keep module light

        self._model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=self._random_state,
            n_jobs=-1,
        )
        self._model.fit(X)
        logger.info("AnomalyEngine fitted on %d samples", X.shape[0])

    def set_model(self, model: Any) -> None:
        """Inject a pre-trained IsolationForest/LOF model."""
        self._model = model

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
