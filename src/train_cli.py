import argparse
import json
import logging
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "static")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REGISTRY_FILE = os.path.join(RESULTS_DIR, "model_registry.json")


def _load_required_artifacts() -> tuple[dict, object]:
    splits_path = os.path.join(MODELS_DIR, "train_splits.pkl")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    if not os.path.exists(splits_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Missing preprocessing artifacts. Run: python src/preprocess_train.py")
    return joblib.load(splits_path), joblib.load(preprocessor_path)


def _available_model_builders() -> dict[str, callable]:
    return {
        "rf_nsl_kdd": lambda: RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        "gb_nsl_kdd": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42),
        "dt_nsl_kdd": lambda: DecisionTreeClassifier(max_depth=20, random_state=42),
        "ab_nsl_kdd": lambda: AdaBoostClassifier(n_estimators=50, random_state=42),
        "mlp_nsl_kdd": lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100, random_state=42),
    }


def train_one(model_name: str, clf, X_train, y_train, X_val, y_val, preprocessor) -> dict | None:
    try:
        pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
        logging.info("Training %s...", model_name)
        start_time = datetime.now()
        pipe.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        y_pred = pipe.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted")
        cm = confusion_matrix(y_val, y_pred)

        out_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        joblib.dump(pipe, out_path)

        logging.info("%s - Accuracy: %.4f, F1: %.4f", model_name, accuracy, f1)
        ModelRegistry(REGISTRY_FILE).register(
            name=model_name,
            model_path=out_path,
            accuracy=accuracy,
            f1_score=f1,
            training_time=training_time,
        )

        if hasattr(pipe.named_steps["clf"], "feature_importances_"):
            try:
                importances = pipe.named_steps["clf"].feature_importances_
                feature_names = [f"F{i + 1}" for i in range(len(importances))]
                plt.figure(figsize=(10, 5))
                plt.bar(feature_names, importances)
                plt.title(f"Feature Importances ({model_name})")
                plt.xlabel("Feature")
                plt.ylabel("Importance")
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(STATIC_DIR, f"feature_importance_{model_name}.png"))
                plt.close()
            except Exception as plot_error:
                logging.warning("Feature-importance plot skipped for %s: %s", model_name, plot_error)

        return {
            "name": model_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "training_time": training_time,
            "confusion_matrix": cm.tolist(),
            "model_path": out_path,
        }
    except Exception as exc:
        logging.error("Failed to train %s: %s", model_name, exc)
        return None


def _plot_model_comparison(results: list[dict]) -> None:
    if not results:
        return
    df = pd.DataFrame(results)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(df["name"], df["accuracy"], color="steelblue")
    plt.xlabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xlim(0, 1)

    plt.subplot(1, 2, 2)
    plt.barh(df["name"], df["f1_score"], color="coral")
    plt.xlabel("F1 Score")
    plt.title("Model F1 Score Comparison")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(df["name"], df["training_time"], color="green", alpha=0.7)
    plt.ylabel("Training Time (seconds)")
    plt.title("Model Training Time Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "training_time_comparison.png"), dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified INIDS model trainer")
    parser.add_argument(
        "--suite",
        choices=["baseline", "full"],
        default="baseline",
        help="baseline: rf+gb, full: rf+gb+dt+ab+mlp",
    )
    parser.add_argument(
        "--include-multiclass",
        action="store_true",
        help="Train rf_nsl_kdd_multi when multiclass splits are available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected models and prerequisite checks without training.",
    )
    return parser.parse_args()


def main() -> None:
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    args = parse_args()
    builders = _available_model_builders()

    selected_names = ["rf_nsl_kdd", "gb_nsl_kdd"]
    if args.suite == "full":
        selected_names.extend(["dt_nsl_kdd", "ab_nsl_kdd", "mlp_nsl_kdd"])

    if args.dry_run:
        logging.info("Dry run enabled.")
        logging.info("Suite: %s", args.suite)
        logging.info("Selected binary models: %s", ", ".join(selected_names))
        logging.info("Include multiclass: %s", args.include_multiclass)
        try:
            _load_required_artifacts()
            logging.info("Prerequisite artifacts: OK")
        except FileNotFoundError as exc:
            logging.error(str(exc))
        return

    splits_bin, preprocessor = _load_required_artifacts()
    X_train = splits_bin["X_train"]
    X_val = splits_bin["X_val"]
    y_train = splits_bin["y_train"]
    y_val = splits_bin["y_val"]

    results = []
    for name in selected_names:
        result = train_one(name, builders[name](), X_train, y_train, X_val, y_val, preprocessor)
        if result:
            results.append(result)

    if results:
        _plot_model_comparison(results)
        results_path = os.path.join(RESULTS_DIR, f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logging.info("Saved comparison metrics to %s", results_path)

    if args.include_multiclass:
        multi_splits_path = os.path.join(MODELS_DIR, "train_splits_multiclass.pkl")
        if not os.path.exists(multi_splits_path):
            logging.warning("Multiclass split file not found; skipping multiclass training.")
        else:
            splits_multi = joblib.load(multi_splits_path)
            train_one(
                "rf_nsl_kdd_multi",
                RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
                splits_multi["X_train"],
                splits_multi["y_train"],
                splits_multi["X_val"],
                splits_multi["y_val"],
                preprocessor,
            )


if __name__ == "__main__":
    main()
