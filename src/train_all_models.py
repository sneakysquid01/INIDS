import os
import joblib
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "static")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model(model_name: str, clf, X_train, y_train, X_val, y_val, preprocessor) -> dict:
    """Train a model and return performance metrics."""
    try:
        pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
        logging.info(f"Training {model_name}...")
        
        start_time = datetime.now()
        pipe.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        y_pred = pipe.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
        cm = confusion_matrix(y_val, y_pred)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        joblib.dump(pipe, model_path)
        
        logging.info(f"✅ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return {
            'name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'training_time': training_time,
            'confusion_matrix': cm.tolist(),
            'model_path': model_path
        }
    except Exception as e:
        logging.error(f"Failed to train {model_name}: {e}")
        return None

def plot_model_comparison(results: list):
    """Generate comparison charts for all models."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(df['name'], df['accuracy'], color='steelblue')
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xlim(0, 1)
    
    # F1 Score comparison
    plt.subplot(1, 2, 2)
    plt.barh(df['name'], df['f1_score'], color='coral')
    plt.xlabel('F1 Score')
    plt.title('Model F1 Score Comparison')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # Training time comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['name'], df['training_time'], color='green', alpha=0.7)
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Training Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'training_time_comparison.png'), dpi=150)
    plt.close()
    
    logging.info("Comparison charts saved")

def main():
    # Load binary classification data
    splits_bin = joblib.load(os.path.join(MODELS_DIR, "train_splits.pkl"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    
    X_train = splits_bin["X_train"]
    X_val = splits_bin["X_val"]
    y_train = splits_bin["y_train"]
    y_val = splits_bin["y_val"]
    
    # Define models to train
    models = {
        'rf_nsl_kdd': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        'gb_nsl_kdd': GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=42),
        'dt_nsl_kdd': DecisionTreeClassifier(max_depth=20, random_state=42),
        'ab_nsl_kdd': AdaBoostClassifier(n_estimators=50, random_state=42),
        'mlp_nsl_kdd': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100, random_state=42),
    }
    
    results = []
    for name, clf in models.items():
        result = train_model(name, clf, X_train, y_train, X_val, y_val, preprocessor)
        if result:
            results.append(result)
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, f'model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {results_file}")
    
    # Generate comparison plots
    plot_model_comparison(results)
    
    # Train multiclass models
    if os.path.exists(os.path.join(MODELS_DIR, "train_splits_multiclass.pkl")):
        logging.info("Training multiclass models...")
        splits_multi = joblib.load(os.path.join(MODELS_DIR, "train_splits_multiclass.pkl"))
        train_model('rf_nsl_kdd_multi', 
                   RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
                   splits_multi["X_train"], splits_multi["y_train"],
                   splits_multi["X_val"], splits_multi["y_val"], preprocessor)
    
    logging.info("✅ All model training complete")

if __name__ == "__main__":
    main()
