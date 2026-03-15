import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
try:
    from src.label_utils import labels_to_binary_series
except ImportError:
    from label_utils import labels_to_binary_series
try:
    from src.schema import COLUMNS, LABEL_COLUMNS
except ImportError:
    from schema import COLUMNS, LABEL_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "static")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
MODEL_FILE = os.path.join(MODELS_DIR, "gb_nsl_kdd.pkl")


def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    model = joblib.load(MODEL_FILE)
    df_test = pd.read_csv(TEST_FILE, names=COLUMNS)
    X_test = df_test.drop(columns=LABEL_COLUMNS)
    y_test = labels_to_binary_series(df_test["label"])
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix - Gradient Boosting Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'confusion_matrix.png'), dpi=150)
    print("Confusion matrix saved.")


if __name__ == "__main__":
    main()
