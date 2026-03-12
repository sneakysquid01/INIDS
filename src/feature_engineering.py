"""Feature engineering utilities for enriching raw NSL-KDD flows.

Adds derived / interaction features that improve model discriminative power
beyond the 41 raw NSL-KDD columns.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from src.schema import NUMERIC_FEATURES

logger = logging.getLogger(__name__)


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio-based interaction features."""
    df = df.copy()

    required = {
        "src_bytes",
        "dst_bytes",
        "serror_rate",
        "rerror_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "count",
    }
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.debug("Skipping ratio features; missing columns: %s", ",".join(sorted(missing)))
        return df

    # Byte ratio
    total_bytes = df["src_bytes"] + df["dst_bytes"]
    df["byte_ratio"] = df["src_bytes"] / total_bytes.replace(0, 1)

    # Error rate delta
    df["error_rate_delta"] = (df["serror_rate"] - df["rerror_rate"]).abs()

    # Service diversity per host
    safe_count = df["dst_host_count"].replace(0, 1)
    df["service_diversity"] = df["dst_host_srv_count"] / safe_count

    # Connection intensity
    df["conn_intensity"] = df["count"] * df["serror_rate"]

    return df


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform heavy-tailed numeric features."""
    df = df.copy()
    heavy_tail = ["src_bytes", "dst_bytes", "duration", "count", "srv_count"]
    for col in heavy_tail:
        if col in df.columns:
            df[f"log_{col}"] = df[col].apply(lambda x: math.log1p(max(0, float(x))))
    return df


def add_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple entropy proxies from rate columns."""
    df = df.copy()
    rate_cols = [c for c in df.columns if c.endswith("_rate")]
    if not rate_cols:
        return df

    # Pseudo-entropy across rate features per row.
    # Rates are clipped to [1e-10, 1.0] since they represent proportions.
    rates = df[rate_cols].clip(lower=1e-10, upper=1.0)
    log_rates = rates.apply(lambda col: col.apply(math.log2))
    df["rate_entropy"] = -(rates * log_rates).sum(axis=1)
    return df


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to a dataframe."""
    original_cols = len(df.columns)
    df = add_ratio_features(df)
    df = add_log_transforms(df)
    df = add_entropy_features(df)
    logger.info("Feature engineering: added %d derived columns", len(df.columns) - original_cols)
    return df


def enrich_single_row(features: dict[str, Any]) -> dict[str, Any]:
    """Apply feature engineering on a single feature dict (for inference)."""
    df = pd.DataFrame([features])
    enriched = enrich_features(df)
    return enriched.iloc[0].to_dict()
