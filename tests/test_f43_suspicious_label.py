"""Step 43 regression: suspicious = attack with low confidence (not just any low confidence)."""
from __future__ import annotations

import numpy as np


class _AttackLowConf:
    """Model that always predicts Attack with confidence below threshold."""
    def predict(self, df):
        return np.ones(len(df), dtype=int)

    def predict_proba(self, df):
        return np.array([[0.55, 0.45]] * len(df))


class _NormalLowConf:
    """Model that always predicts Normal with confidence below threshold."""
    def predict(self, df):
        return np.zeros(len(df), dtype=int)

    def predict_proba(self, df):
        return np.array([[0.45, 0.55]] * len(df))


class _AttackHighConf:
    """Model that always predicts Attack with confidence above threshold."""
    def predict(self, df):
        return np.ones(len(df), dtype=int)

    def predict_proba(self, df):
        return np.array([[0.05, 0.95]] * len(df))


def _make_service(model):
    from src.detection_service import DetectionService
    return DetectionService(model=model)


def test_attack_low_confidence_is_suspicious():
    """Attack prediction below threshold → suspicious=True."""
    svc = _make_service(_AttackLowConf())
    result = svc.predict_from_features({"src_bytes": 10}, profile="balanced")
    assert result.prediction == "Attack"
    assert result.suspicious is True


def test_normal_low_confidence_is_not_suspicious():
    """Normal prediction below threshold → suspicious=False (label inversion fix)."""
    svc = _make_service(_NormalLowConf())
    result = svc.predict_from_features({"src_bytes": 10}, profile="balanced")
    assert result.prediction == "Normal"
    assert result.suspicious is False


def test_attack_high_confidence_is_not_suspicious():
    """Attack prediction above threshold → suspicious=False."""
    svc = _make_service(_AttackHighConf())
    result = svc.predict_from_features({"src_bytes": 10}, profile="balanced")
    assert result.prediction == "Attack"
    assert result.suspicious is False
