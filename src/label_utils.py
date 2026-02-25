import pandas as pd


def normalize_label(value) -> str:
    """
    Normalize NSL-KDD/KDD labels to a canonical lowercase form.
    Handles variants like 'normal.' and surrounding whitespace.
    """
    if pd.isna(value):
        return ""
    return str(value).strip().lower().rstrip(".")


def labels_to_binary_series(series: pd.Series) -> pd.Series:
    """Map normalized labels to binary target: normal=0, attack=1."""
    return series.apply(lambda x: 0 if normalize_label(x) == "normal" else 1)
