from flask import Flask, render_template, request
import joblib
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "web_app", "static")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_nsl_kdd.pkl")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src.schema import (
    COLUMNS,
    LABEL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    DEFAULT_FEATURE_ROW,
)

app = Flask(__name__)

model = None
all_models = {}

INPUT_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "count", 
    "srv_count", "serror_rate", "same_srv_rate",
]

MODEL_INPUT_COLUMNS = FEATURE_COLUMNS
NUMERIC_MODEL_COLUMNS = NUMERIC_FEATURES

def ensure_model_loaded() -> None:
    """Lazily load models if not available in memory."""
    global model
    if model is None:
        load_models()


def _normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().rstrip(".")

def load_models():
    """Load all available models into memory for live or selectable prediction."""
    global model, all_models
    if all_models:
        return
    model_files = ['rf_nsl_kdd.pkl', 'gb_nsl_kdd.pkl', 'dt_nsl_kdd.pkl', 
                   'ab_nsl_kdd.pkl', 'mlp_nsl_kdd.pkl', 'rf_nsl_kdd_multi.pkl']
    for model_file in model_files:
        path = os.path.join(MODELS_DIR, model_file)
        if os.path.exists(path):
            model_name = model_file.replace('.pkl', '')
            all_models[model_name] = joblib.load(path)
            logging.info(f"✅ Loaded {model_name}")
    # Default model: rf_nsl_kdd
    if 'rf_nsl_kdd' in all_models:
        model = all_models['rf_nsl_kdd']

@app.route("/")
def home():
    """Landing page with navigation."""
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Live prediction page with suspicious activity logic."""
    ensure_model_loaded()
    prediction = None
    error_message = None
    confidence = None
    is_suspicious = False

    if model is None:
        return render_template("predict.html", features=INPUT_FEATURES,
                               prediction=None, error="No trained model found. Please train and load a model first.",
                               confidence=None, is_suspicious=False)

    if request.method == "POST":
        try:
            values = []
            for feat in INPUT_FEATURES:
                v = request.form.get(feat, None)
                if v is None or not v.replace('.', '', 1).replace('-', '', 1).isdigit():
                    raise ValueError(f"Invalid input for {feat}")
                values.append(float(v))

            # Build the full test row with smart defaults for categorical and unused numerical
            row = DEFAULT_FEATURE_ROW.copy()
            row.update({
                "duration": values[0],
                "src_bytes": values[1],
                "dst_bytes": values[2],
                "count": values[3],
                "srv_count": values[4],
                "serror_rate": values[5],
                "same_srv_rate": values[6],
            })
            df = pd.DataFrame([row], columns=MODEL_INPUT_COLUMNS)

            pred = model.predict(df)[0]
            proba = model.predict_proba(df)[0]
            confidence = round(max(proba) * 100, 2)
            # Suspicious activity logic
            if confidence < 60:
                is_suspicious = True
                prediction = "SUSPICIOUS - Low Confidence"
            else:
                prediction = "Normal" if pred == 0 else "Attack"
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            error_message = f"Error: {e}"

    return render_template("predict.html", features=INPUT_FEATURES, 
                         prediction=prediction, error=error_message, 
                         confidence=confidence, is_suspicious=is_suspicious)

@app.route("/dashboard")
def dashboard():
    """Visual dashboard of system predictions and accuracy."""
    ensure_model_loaded()
    try:
        if model is None:
            raise FileNotFoundError("No trained model found. Please train and load a model first.")

        if not os.path.exists(TEST_FILE):
            raise FileNotFoundError("Test data file not found!")

        df_test = pd.read_csv(TEST_FILE, names=COLUMNS)
        X_test = df_test.drop(columns=LABEL_COLUMNS)
        y_test = df_test["label"].apply(lambda x: 0 if _normalize_label(x) == "normal" else 1)
        y_pred = model.predict(X_test)
        
        total = len(y_test)
        attacks = int(sum(y_pred))
        normal = total - attacks
        accuracy = round(float(sum(y_pred == y_test)) / total * 100, 2)

        # Pie chart of prediction outcome
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie([normal, attacks], labels=['Normal', 'Attack'], autopct='%1.1f%%', 
               colors=['#28a745', '#dc3545'], startangle=90)
        ax.set_title("Test Data Predictions Distribution")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        chart_data = base64.b64encode(buf.getvalue()).decode()
        buf.close()
        plt.close(fig)

        # Recent predictions table
        results = []
        for i in range(min(20, len(y_test))):
            results.append({
                "Index": i,
                "True": "Normal" if y_test.iloc[i] == 0 else "Attack",
                "Predicted": "Normal" if y_pred[i] == 0 else "Attack",
                "Match": "✓" if y_test.iloc[i] == y_pred[i] else "✗"
            })

        return render_template("dashboard.html", total=total, attacks=attacks, 
                             normal=normal, accuracy=accuracy, 
                             chart_data=chart_data, results=results)

    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        return render_template("error.html", error=str(e)), 500

@app.route("/models")
def models_page():
    """Comparison page of all trained models and their metrics."""
    try:
        results_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
        if not results_files:
            return render_template("models.html", models=[], has_data=False)
        
        latest_results = sorted(results_files)[-1]
        with open(os.path.join(RESULTS_DIR, latest_results), 'r') as f:
            model_results = json.load(f)
        # Show best model first in UI
        model_results = sorted(model_results, key=lambda x: x.get("accuracy", 0), reverse=True)
        return render_template("models.html", models=model_results, has_data=True)
    except Exception as e:
        logging.error(f"Models page error: {e}")
        return render_template("models.html", models=[], has_data=False)

@app.route("/batch", methods=["GET", "POST"])
def batch_predict():
    """Batch prediction from CSV upload."""
    ensure_model_loaded()
    if request.method == "POST":
        try:
            if model is None:
                return render_template("batch.html", error="No trained model found. Please train and load a model first.")

            file = request.files.get('file')
            if not file:
                return render_template("batch.html", error="No file uploaded")
            df = pd.read_csv(file)
            if df.empty:
                return render_template("batch.html", error="Uploaded CSV is empty.")

            max_rows = 50000
            if len(df) > max_rows:
                return render_template("batch.html", error=f"CSV too large ({len(df)} rows). Max allowed is {max_rows}.")

            # Accept optional ground-truth columns and enforce model schema
            df = df.drop(columns=LABEL_COLUMNS, errors="ignore")
            missing_cols = [col for col in MODEL_INPUT_COLUMNS if col not in df.columns]
            if missing_cols:
                return render_template("batch.html", error=f"Missing required columns: {', '.join(missing_cols)}")

            # Ignore extra columns that the model does not use
            df = df[MODEL_INPUT_COLUMNS].copy()

            for col in NUMERIC_MODEL_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[NUMERIC_MODEL_COLUMNS].isna().any().any():
                return render_template("batch.html", error="One or more numeric columns contain invalid values.")

            predictions = model.predict(df)
            results = [{"row": i, "prediction": "Attack" if p == 1 else "Normal"} 
                      for i, p in enumerate(predictions)]
            return render_template("batch.html", results=results[:50], 
                                 total=len(results), shown=min(50, len(results)))
        except Exception as e:
            logging.error(f"Batch prediction error: {e}")
            return render_template("batch.html", error=str(e))
    return render_template("batch.html")

@app.route("/about")
def about():
    """About page with project information."""
    return render_template("about.html")

@app.route("/realtime")
def realtime():
    """Real-time detection simulator page."""
    return render_template("realtime.html")

@app.route("/capture")
def capture():
    """Live traffic capture page."""
    return render_template("capture.html")

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(Exception)
def handle_error(e):
    logging.error(str(e))
    return render_template("error.html", error=str(e)), 500

if __name__ == "__main__":
    load_models()
    app.run(
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
        host='0.0.0.0',
        port=int(os.getenv("PORT", "5000")),
    )
