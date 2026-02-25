import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "static")

def train_and_save(
    model_name: str,
    clf,
    X_train,
    y_train,
    X_val,
    y_val,
    preprocessor
) -> None:
    """
    Trains a pipeline with the given classifier, evaluates it, saves the model, and plots feature importance if possible.
    """
    try:
        pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
        logging.info(f"Training {model_name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        logging.info(f"{model_name} Validation Accuracy: {acc:.4f}")
        logging.info(f"Confusion matrix:\n{confusion_matrix(y_val, y_pred)}")
        logging.info("Classification report:\n" + classification_report(y_val, y_pred))
        # Save model
        out_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        joblib.dump(pipe, out_path)
        logging.info(f"✅ Model saved: {out_path}")

        # Feature importance (only for RandomForest-like models)
        if hasattr(clf, "feature_importances_"):
            try:
                importances = pipe.named_steps['clf'].feature_importances_
                # For simplicity, use F1, F2, ...
                feature_names = [f"F{i+1}" for i in range(len(importances))]
                plt.figure(figsize=(10, 5))
                plt.bar(feature_names, importances)
                plt.title(f'Feature Importances ({model_name})')
                plt.xlabel('Feature')
                plt.ylabel('Importance')
                plt.xticks(rotation=90)
                chart_path = os.path.join(STATIC_DIR, f"feature_importance_{model_name}.png")
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
                logging.info(f"Feature importance plot saved: {chart_path}")
            except Exception as e:
                logging.warning(f"Could not plot feature importance for {model_name}: {e}")
    except Exception as e:
        logging.error(f"Training failed for {model_name}: {e}")

def main():
    # --- Binary classification (Normal vs Attack)
    splits_path = os.path.join(MODELS_DIR, "train_splits.pkl")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    if not os.path.exists(splits_path) or not os.path.exists(preprocessor_path):
        logging.error("❌ First run: python src/preprocess_train.py")
        return

    # --- Multiclass classification (attack types)
    splits_multi_path = os.path.join(MODELS_DIR, "train_splits_multiclass.pkl")
    has_multi = os.path.exists(splits_multi_path)

    logging.info("Loading preprocessed data and pipeline...")
    splits = joblib.load(splits_path)
    preprocessor = joblib.load(preprocessor_path)
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    y_train = splits["y_train"]
    y_val = splits["y_val"]

    # Train binary models
    train_and_save(
        "rf_nsl_kdd",
        RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        X_train, y_train, X_val, y_val, preprocessor
    )
    train_and_save(
        "gb_nsl_kdd",
        GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42),
        X_train, y_train, X_val, y_val, preprocessor
    )

    # Train multiclass (attack type) model
    if has_multi:
        splits_multi = joblib.load(splits_multi_path)
        X_train_m = splits_multi["X_train"]
        X_val_m = splits_multi["X_val"]
        y_train_m = splits_multi["y_train"]
        y_val_m = splits_multi["y_val"]
        train_and_save(
            "rf_nsl_kdd_multi",
            RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
            X_train_m, y_train_m, X_val_m, y_val_m, preprocessor
        )

    logging.info("All model runs finished.")

if __name__ == "__main__":
    main()
