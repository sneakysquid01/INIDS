import pytest
from src.preprocess_train import make_binary_label, make_attack_type_label
from src.label_utils import normalize_label
import pandas as pd

def test_binary_label():
    df = pd.DataFrame({"label": ["normal", "neptune", "smurf"]})
    df = make_binary_label(df)
    assert (df["binary_label"].tolist() == [0, 1, 1])

def test_attack_type_label():
    df = pd.DataFrame({"label": ["neptune", "satan", "spy", "rootkit", "unknown"]})
    df = make_attack_type_label(df)
    assert (df["attack_type"].tolist() == ["DoS", "Probe", "R2L", "U2R", "Other"])

def test_label_normalization_variants():
    assert normalize_label("normal.") == "normal"
    assert normalize_label(" Normal ") == "normal"

    df = pd.DataFrame({"label": ["normal.", "normal", " neptune. "]})
    df = make_binary_label(df)
    assert df["binary_label"].tolist() == [0, 0, 1]

