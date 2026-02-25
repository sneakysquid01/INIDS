import os
import joblib
import pandas as pd
import logging
try:
    from src.label_utils import labels_to_binary_series
except ImportError:
    from label_utils import labels_to_binary_series
try:
    from src.schema import COLUMNS, LABEL_COLUMNS
except ImportError:
    from schema import COLUMNS, LABEL_COLUMNS

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "rf_nsl_kdd.pkl")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

def main() -> None:
    """Loads the model, tests the first 10 samples, prints results to terminal."""
    if not os.path.exists(MODEL_PATH):
        logging.error("❌ Train the model first: python src/train_model.py")
        return
    if not os.path.exists(TEST_FILE):
        logging.error("❌ Test data file not found: expected at '%s'", TEST_FILE)
        return

    logging.info("Loading trained model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    logging.info("Loading test data from %s", TEST_FILE)
    df_test = pd.read_csv(TEST_FILE, names=COLUMNS)
    X_test = df_test.drop(columns=LABEL_COLUMNS)
    y_test = labels_to_binary_series(df_test["label"])

    logging.info("Making predictions on first 10 rows...")
    preds = model.predict(X_test.head(10))
    for i, (ypred, ytrue) in enumerate(zip(preds, y_test.head(10))):
        pred_label = "Normal" if ypred == 0 else "Attack"
        true_label = "Normal" if ytrue == 0 else "Attack"
        logging.info(f"Row {i}: Predicted={pred_label} | True={true_label}")

if __name__ == "__main__":
    main()
