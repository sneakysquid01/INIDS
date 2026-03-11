"""Class-imbalance handling utilities.

Provides wrappers around SMOTE and random under/over-sampling for the
highly imbalanced NSL-KDD dataset. Falls back gracefully when
``imbalanced-learn`` is not installed.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_IMBLEARN_AVAILABLE = False
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    _IMBLEARN_AVAILABLE = True
except ImportError:
    pass


def has_imblearn() -> bool:
    return _IMBLEARN_AVAILABLE


def apply_smote(X: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to minority class."""
    if not _IMBLEARN_AVAILABLE:
        logger.warning("imbalanced-learn not installed; skipping SMOTE")
        return X, y
    sm = SMOTE(random_state=random_state, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X, y)
    logger.info("SMOTE: %d → %d samples", len(y), len(y_res))
    return X_res, y_res


def apply_smote_tomek(X: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE + Tomek links cleaning."""
    if not _IMBLEARN_AVAILABLE:
        logger.warning("imbalanced-learn not installed; skipping SMOTETomek")
        return X, y
    smt = SMOTETomek(random_state=random_state, n_jobs=-1)
    X_res, y_res = smt.fit_resample(X, y)
    logger.info("SMOTETomek: %d → %d samples", len(y), len(y_res))
    return X_res, y_res


def apply_random_oversample(X: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Simple random oversampling of minority class."""
    if not _IMBLEARN_AVAILABLE:
        logger.warning("imbalanced-learn not installed; skipping oversampling")
        return X, y
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)
    logger.info("RandomOverSampler: %d → %d samples", len(y), len(y_res))
    return X_res, y_res


def apply_random_undersample(X: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Random under-sampling of majority class."""
    if not _IMBLEARN_AVAILABLE:
        logger.warning("imbalanced-learn not installed; skipping undersampling")
        return X, y
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    logger.info("RandomUnderSampler: %d → %d samples", len(y), len(y_res))
    return X_res, y_res


def class_distribution(y: np.ndarray) -> dict[str, Any]:
    """Return class counts and ratios."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {
        "total": total,
        "classes": {
            str(cls): {"count": int(cnt), "ratio": round(cnt / total, 4)}
            for cls, cnt in zip(unique, counts)
        },
    }
