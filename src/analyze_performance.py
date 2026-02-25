import os
import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
try:
    from src.label_utils import labels_to_binary_series
except ImportError:
    from label_utils import labels_to_binary_series
try:
    from src.schema import COLUMNS, LABEL_COLUMNS
except ImportError:
    from schema import COLUMNS, LABEL_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "static")

TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")

def plot_roc_curves():
    """Generate ROC curves for all binary models."""
    df_test = pd.read_csv(TEST_FILE, names=COLUMNS)
    X_test = df_test.drop(columns=LABEL_COLUMNS)
    y_test = labels_to_binary_series(df_test["label"])
    
    plt.figure(figsize=(10, 8))
    
    models = ['rf_nsl_kdd', 'gb_nsl_kdd', 'dt_nsl_kdd', 'ab_nsl_kdd', 'mlp_nsl_kdd']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for model_name, color in zip(models, colors):
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'roc_curves.png'), dpi=150)
    plt.close()
    logging.info("ROC curves saved")

if __name__ == "__main__":
    plot_roc_curves()
