"""
JobGuard AI — Fake Job Description Detector
Flask backend serving XGBoost & CatBoost ensemble predictions.
"""

import os
import re
import io
import traceback

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Model & preprocessor loading
# ---------------------------------------------------------------------------
preprocessor = None
xgb_model = None
cb_model = None
models_loaded = False


def load_models():
    """Load pre-trained models and preprocessor from disk."""
    global preprocessor, xgb_model, cb_model, models_loaded

    # --- Preprocessor (TF-IDF + label encoders) ---
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    try:
        preprocessor = joblib.load(preprocessor_path)
        print(f"[OK] Preprocessor loaded from {preprocessor_path}")
    except FileNotFoundError:
        print(f"[!!] Preprocessor not found at {preprocessor_path}")
        print("    Run `python create_dummy_models.py` to generate test models.")
        return
    except Exception as e:
        print(f"[!!] Error loading preprocessor: {e}")
        return

    # --- XGBoost ---
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        print(f"[OK] XGBoost model loaded from {xgb_path}")
    except FileNotFoundError:
        print(f"[!!] XGBoost model not found at {xgb_path}")
        return
    except Exception as e:
        print(f"[!!] Error loading XGBoost model: {e}")
        return

    # --- CatBoost ---
    cb_path = os.path.join(MODEL_DIR, "catboost_model.cbm")
    try:
        cb_model = CatBoostClassifier()
        cb_model.load_model(cb_path)
        print(f"[OK] CatBoost model loaded from {cb_path}")
    except FileNotFoundError:
        print(f"[!!] CatBoost model not found at {cb_path}")
        return
    except Exception as e:
        print(f"[!!] Error loading CatBoost model: {e}")
        return

    models_loaded = True
    print("[OK] All models loaded successfully.")


# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_label_encode(encoder, value: str, default: int = 0) -> int:
    """Encode a categorical value, falling back to *default* if unseen."""
    try:
        return int(encoder.transform([value])[0])
    except (ValueError, KeyError):
        return default


# ---------------------------------------------------------------------------
# Red-flag detection (rule-based, model-independent)
# ---------------------------------------------------------------------------
def detect_red_flags(data: dict) -> list:
    """Return a list of human-readable red-flag strings."""
    flags = []
    title = (data.get("title") or "").lower()
    description = (data.get("description") or "").lower()
    company_profile = (data.get("company_profile") or "").strip()
    requirements = (data.get("requirements") or "").strip()
    benefits = (data.get("benefits") or "").strip()
    has_logo = data.get("has_company_logo", False)

    if not company_profile:
        flags.append("No company profile provided")
    if re.search(r"\bweekly\b", description) or re.search(r"\$\s?\d{4,}", description):
        flags.append("Salary mentioned as 'weekly' or suspiciously high")
    if "work from home" in description and "no experience" in description:
        flags.append("Description combines 'work from home' with 'no experience'")
    if not requirements:
        flags.append("Requirements field is empty")
    if any(w in title for w in ["urgent", "immediately", "asap"]):
        flags.append("Job title contains urgency keywords")
    if not has_logo:
        flags.append("No company logo")
    if not benefits:
        flags.append("Benefits section is empty")

    return flags


# ---------------------------------------------------------------------------
# Core prediction pipeline
# ---------------------------------------------------------------------------
def preprocess_single(data: dict) -> np.ndarray:
    """
    Mirror the training preprocessing:
      1. Combine text fields
      2. Clean text
      3. TF-IDF transform
      4. Append label-encoded meta-features
    Returns a 2-D numpy array (1, n_features).
    """
    combined_text = " ".join([
        data.get("title", ""),
        data.get("company_profile", ""),
        data.get("description", ""),
        data.get("requirements", ""),
        data.get("benefits", ""),
    ])
    combined_text = clean_text(combined_text)

    tfidf = preprocessor["tfidf"]
    X_tfidf = tfidf.transform([combined_text])

    emp_enc = safe_label_encode(
        preprocessor["employment_type_encoder"],
        data.get("employment_type", "Other"),
    )
    exp_enc = safe_label_encode(
        preprocessor["experience_encoder"],
        data.get("required_experience", "Not Applicable"),
    )
    edu_enc = safe_label_encode(
        preprocessor["education_encoder"],
        data.get("required_education", "Unspecified"),
    )
    has_logo = 1 if data.get("has_company_logo") else 0
    has_q = 1 if data.get("has_questions") else 0

    meta = np.array([[emp_enc, exp_enc, edu_enc, has_logo, has_q]])
    X_combined = sp.hstack([X_tfidf, sp.csr_matrix(meta)]).toarray()
    return X_combined


def predict_single(data: dict) -> dict:
    """Run the full prediction pipeline for one job posting."""
    X = preprocess_single(data)
    model_choice = data.get("model_choice", "ensemble").lower()

    # Probabilities (probability of class 1 = FAKE)
    xgb_prob = float(xgb_model.predict_proba(X)[0][1])
    cb_prob = float(cb_model.predict_proba(X)[0][1])

    if model_choice == "xgboost":
        fake_prob = xgb_prob
        model_used = "XGBoost"
    elif model_choice == "catboost":
        fake_prob = cb_prob
        model_used = "CatBoost"
    else:
        fake_prob = (xgb_prob + cb_prob) / 2.0
        model_used = "Ensemble (XGBoost + CatBoost)"

    prediction = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = round(fake_prob * 100 if prediction == "FAKE" else (1 - fake_prob) * 100, 2)

    if fake_prob >= 0.75:
        risk_level = "HIGH"
    elif fake_prob >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    red_flags = detect_red_flags(data)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "xgb_prob": round(xgb_prob, 4),
        "cb_prob": round(cb_prob, 4),
        "risk_level": risk_level,
        "red_flags": red_flags,
        "model_used": model_used,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": models_loaded})


@app.route("/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Check server logs."}), 503

    data = request.get_json(force=True)
    if not data.get("description", "").strip():
        return jsonify({"error": "Job description is required."}), 400

    try:
        result = predict_single(data)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/batch", methods=["POST"])
def batch():
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Check server logs."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    try:
        df = pd.read_csv(file.stream)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

    results = []
    for idx, row in df.iterrows():
        row_data = {
            "title": str(row.get("title", "")),
            "company_profile": str(row.get("company_profile", "")),
            "description": str(row.get("description", "")),
            "requirements": str(row.get("requirements", "")),
            "benefits": str(row.get("benefits", "")),
            "employment_type": str(row.get("employment_type", "Other")),
            "required_experience": str(row.get("required_experience", "Not Applicable")),
            "required_education": str(row.get("required_education", "Unspecified")),
            "has_company_logo": bool(row.get("has_company_logo", False)),
            "has_questions": bool(row.get("has_questions", False)),
            "model_choice": str(row.get("model_choice", "ensemble")),
        }
        try:
            result = predict_single(row_data)
            result["row_index"] = int(idx)
            results.append(result)
        except Exception as e:
            results.append({"row_index": int(idx), "error": str(e)})

    return jsonify(results)


@app.route("/sample-csv", methods=["GET"])
def sample_csv():
    """Return a sample CSV for batch upload."""
    sample = pd.DataFrame([
        {
            "title": "Senior Software Engineer",
            "company_profile": "We are a leading tech company specializing in AI solutions.",
            "description": "Looking for an experienced engineer to join our backend team.",
            "requirements": "5+ years Python, REST APIs, cloud experience.",
            "benefits": "Health insurance, 401k, remote work.",
            "employment_type": "Full-time",
            "required_experience": "Mid-Senior level",
            "required_education": "Bachelor's Degree",
            "has_company_logo": True,
            "has_questions": True,
        },
        {
            "title": "URGENT - Work From Home Data Entry",
            "company_profile": "",
            "description": "Earn $3000 weekly work from home no experience needed start immediately.",
            "requirements": "",
            "benefits": "",
            "employment_type": "Other",
            "required_experience": "Not Applicable",
            "required_education": "Unspecified",
            "has_company_logo": False,
            "has_questions": False,
        },
    ])
    buf = io.BytesIO()
    sample.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="sample_jobs.csv")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
load_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
