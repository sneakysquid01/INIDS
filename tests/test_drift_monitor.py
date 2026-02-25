import pandas as pd

from src.drift_monitor import classify_psi, population_stability_index


def test_classify_psi_thresholds():
    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate_drift"
    assert classify_psi(0.25) == "high_drift"


def test_population_stability_index_non_negative():
    baseline = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    current = pd.Series([1, 2, 2, 4, 4, 6, 6, 8, 9, 10])
    psi = population_stability_index(baseline, current, bins=5)
    assert psi >= 0
