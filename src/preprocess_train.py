import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
try:
    from src.label_utils import normalize_label, labels_to_binary_series
except ImportError:
    from label_utils import normalize_label, labels_to_binary_series
try:
    from src.schema import COLUMNS, CATEGORICAL_FEATURES
except ImportError:
    from schema import COLUMNS, CATEGORICAL_FEATURES

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------- Paths and Constants ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")

# ---------- Functions ----------

def load_train_data() -> pd.DataFrame:
    """Load the NSL-KDD training dataset safely."""
    try:
        df = pd.read_csv(TRAIN_FILE, names=COLUMNS)
        logging.info(f"Loaded training data. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading {TRAIN_FILE}: {e}")
        raise

def make_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Adds binary column: normal=0, attack=1"""
    df["binary_label"] = labels_to_binary_series(df["label"])
    logging.info("Binary attack label added.")
    return df

def make_attack_type_label(df: pd.DataFrame) -> pd.DataFrame:
    """Adds multiclass attack type column for advanced classification."""
    attack_map = {
        'normal': 'Normal',
        # DoS
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
        # Probe
        'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
        # R2L
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
        'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
        # U2R
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R'
    }
    df["attack_type"] = df["label"].apply(normalize_label).map(attack_map).fillna("Other")
    logging.info("Multiclass attack type label added.")
    return df

def build_preprocess_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, ColumnTransformer]:
    """
    Builds a ColumnTransformer pipeline for categorical and numerical features.
    Args:
        df (pd.DataFrame): Input data.
    Returns:
        tuple: (Features X, Preprocessor pipeline)
    """
    # Remove label columns and keep all feature columns
    X = df.drop(columns=["label", "difficulty_level", "binary_label", "attack_type"])
    cat_cols = CATEGORICAL_FEATURES
    num_cols = [col for col in X.columns if col not in cat_cols]
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    logging.info(f"Preprocessing pipeline built. Num features: {len(num_cols)}, Cat features: {len(cat_cols)}")
    return X, preprocessor

def save_data(obj, filepath: str):
    """Utility: Save object using joblib with error handling."""
    try:
        joblib.dump(obj, filepath)
        logging.info(f"Saved {filepath}")
    except Exception as e:
        logging.error(f"Could not save {filepath}: {e}")

def main():
    # Make sure output directory exists
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create models directory: {e}")
        raise

    # Load and label data
    logging.info("Loading training data...")
    df = load_train_data()
    # Normalize raw labels once to keep train/eval semantics consistent.
    df["label"] = df["label"].apply(normalize_label)
    df = make_binary_label(df)
    df = make_attack_type_label(df)

    logging.info("Building preprocessing pipeline...")
    X, preprocessor = build_preprocess_pipeline(df)

    # Choose binary and multiclass label columns
    y_bin = df["binary_label"]
    y_multiclass = df["attack_type"]

    logging.info("Splitting train and validation for binary classification...")
    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    logging.info("Splitting train and validation for multiclass classification...")
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
        X, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass
    )

    # Save everything robustly
    save_data(
        {
            "X_train": X_train_b,
            "X_val": X_val_b,
            "y_train": y_train_b,
            "y_val": y_val_b,
        },
        os.path.join(MODELS_DIR, "train_splits.pkl"),
    )
    save_data(
        {
            "X_train": X_train_m,
            "X_val": X_val_m,
            "y_train": y_train_m,
            "y_val": y_val_m,
        },
        os.path.join(MODELS_DIR, "train_splits_multiclass.pkl"),
    )
    save_data(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

    logging.info(
        "✅ Preprocessing complete: models/train_splits.pkl (binary), models/train_splits_multiclass.pkl (multiclass), preprocessor.pkl"
    )

if __name__ == "__main__":
    main()
