"""
Utility script to create dummy pre-trained models for testing.
Run this once before starting the app if you don't have real trained models.

Usage:
    python create_dummy_models.py
"""

import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from catboost import CatBoostClassifier

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Create a TF-IDF Vectorizer trained on sample job descriptions ---
sample_texts = [
    "software engineer needed for full time position in new york experience with python required",
    "data scientist machine learning deep learning tensorflow pytorch nlp",
    "urgent hiring work from home no experience needed earn money fast weekly pay",
    "marketing manager digital marketing seo sem social media advertising",
    "customer service representative call center support help desk",
    "administrative assistant office management scheduling data entry",
    "sales representative b2b saas enterprise software account executive",
    "graphic designer adobe photoshop illustrator figma ui ux design",
    "accountant finance bookkeeping tax preparation audit cpa",
    "project manager agile scrum kanban jira product development",
    "earn $5000 weekly from home no experience required start immediately",
    "hiring now work at home easy money guaranteed income no skills needed",
    "senior developer java spring boot microservices cloud aws azure",
    "registered nurse healthcare hospital patient care clinical nursing",
    "teacher education curriculum instruction classroom management",
    "mechanical engineer cad solidworks manufacturing production design",
    "human resources recruiter talent acquisition onboarding training",
    "business analyst requirements gathering stakeholder management",
    "web developer html css javascript react angular node backend frontend",
    "operations manager logistics supply chain warehouse inventory",
]

sample_labels = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

print("[1/4] Fitting TF-IDF Vectorizer...")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(sample_texts)

# --- 2. Create label encoders for categorical features ---
employment_type_encoder = LabelEncoder()
employment_type_encoder.fit(["Full-time", "Part-time", "Contract", "Temporary", "Other"])

experience_encoder = LabelEncoder()
experience_encoder.fit([
    "Not Applicable", "Internship", "Entry level",
    "Associate", "Mid-Senior level", "Director", "Executive"
])

education_encoder = LabelEncoder()
education_encoder.fit([
    "Unspecified", "High School", "Some College",
    "Bachelor's Degree", "Master's Degree", "PhD"
])

# Save preprocessor bundle
preprocessor = {
    "tfidf": tfidf,
    "employment_type_encoder": employment_type_encoder,
    "experience_encoder": experience_encoder,
    "education_encoder": education_encoder,
}

preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
joblib.dump(preprocessor, preprocessor_path)
print(f"   Saved preprocessor to {preprocessor_path}")

# --- 3. Build feature matrix with meta-features ---
import scipy.sparse as sp

n_samples = len(sample_texts)
meta_features = np.zeros((n_samples, 5))
for i in range(n_samples):
    meta_features[i, 0] = np.random.randint(0, 5)   # employment_type
    meta_features[i, 1] = np.random.randint(0, 7)   # experience
    meta_features[i, 2] = np.random.randint(0, 6)   # education
    meta_features[i, 3] = np.random.randint(0, 2)   # has_company_logo
    meta_features[i, 4] = np.random.randint(0, 2)   # has_questions

X_combined = sp.hstack([X_tfidf, sp.csr_matrix(meta_features)]).tocsr()
y = np.array(sample_labels)

# --- 4. Train and save XGBoost model ---
print("[2/4] Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
xgb_model.fit(X_combined.toarray(), y)
xgb_path = os.path.join(MODEL_DIR, "xgboost_model.json")
xgb_model.save_model(xgb_path)
print(f"   Saved XGBoost model to {xgb_path}")

# --- 5. Train and save CatBoost model ---
print("[3/4] Training CatBoost model...")
cb_model = CatBoostClassifier(
    iterations=50,
    depth=3,
    learning_rate=0.1,
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
)
cb_model.fit(X_combined.toarray(), y)
cb_path = os.path.join(MODEL_DIR, "catboost_model.cbm")
cb_model.save_model(cb_path)
print(f"   Saved CatBoost model to {cb_path}")

print("[4/4] Done! All models and preprocessor saved to ./model/")
print(f"   TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
print(f"   Total feature count: {X_combined.shape[1]}")
print(f"   Training samples: {n_samples}")
