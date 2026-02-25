from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from src.schema import COLUMNS, LABEL_COLUMNS, NUMERIC_FEATURES
except ImportError:
    from schema import COLUMNS, LABEL_COLUMNS, NUMERIC_FEATURES


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BASELINE_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
CURRENT_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    status: str


def _bucket_ratio(series: pd.Series, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(series, bins=bins)
    ratios = counts / max(len(series), 1)
    return np.clip(ratios, 1e-6, None)


def population_stability_index(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
    baseline = pd.to_numeric(baseline, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if baseline.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 3:
        return 0.0

    base_ratio = _bucket_ratio(baseline, edges)
    current_ratio = _bucket_ratio(current, edges)
    psi = np.sum((current_ratio - base_ratio) * np.log(current_ratio / base_ratio))
    return float(psi)


def classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "moderate_drift"
    return "high_drift"


def compute_drift_report(
    baseline_file: str = BASELINE_FILE,
    current_file: str = CURRENT_FILE,
    output_dir: str = RESULTS_DIR,
) -> dict:
    baseline_df = pd.read_csv(baseline_file, names=COLUMNS)
    current_df = pd.read_csv(current_file, names=COLUMNS)

    baseline_X = baseline_df.drop(columns=LABEL_COLUMNS)
    current_X = current_df.drop(columns=LABEL_COLUMNS)

    drifts: list[FeatureDrift] = []
    for feature in NUMERIC_FEATURES:
        psi = population_stability_index(baseline_X[feature], current_X[feature])
        drifts.append(FeatureDrift(feature=feature, psi=round(psi, 6), status=classify_psi(psi)))

    high = [d for d in drifts if d.status == "high_drift"]
    moderate = [d for d in drifts if d.status == "moderate_drift"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_file": baseline_file,
        "current_file": current_file,
        "summary": {
            "total_features": len(drifts),
            "stable": len([d for d in drifts if d.status == "stable"]),
            "moderate_drift": len(moderate),
            "high_drift": len(high),
        },
        "top_drift_features": [d.__dict__ for d in sorted(drifts, key=lambda x: x.psi, reverse=True)[:10]],
    }

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["output_file"] = out
    return report


if __name__ == "__main__":
    result = compute_drift_report()
    print(json.dumps(result["summary"], indent=2))
    print(f"Report saved: {result['output_file']}")
