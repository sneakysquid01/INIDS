from src.schema import (
    COLUMNS,
    LABEL_COLUMNS,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DEFAULT_FEATURE_ROW,
)


def test_feature_and_label_partition():
    assert set(FEATURE_COLUMNS).isdisjoint(set(LABEL_COLUMNS))
    assert len(COLUMNS) == len(FEATURE_COLUMNS) + len(LABEL_COLUMNS)
    assert set(COLUMNS) == set(FEATURE_COLUMNS).union(set(LABEL_COLUMNS))


def test_numeric_and_categorical_partition():
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))
    assert set(FEATURE_COLUMNS) == set(NUMERIC_FEATURES).union(set(CATEGORICAL_FEATURES))


def test_default_row_matches_features():
    assert set(DEFAULT_FEATURE_ROW.keys()) == set(FEATURE_COLUMNS)
