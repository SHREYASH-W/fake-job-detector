# JobGuard AI — Fake Job Description Detector

JobGuard AI is a complete web application powered by Machine Learning that detects fraudulent job postings. It loads pre-trained XGBoost and CatBoost models and allows users to either paste a single job description or upload a batch CSV to get real-time fraud predictions, confidence scores, risk levels, and identified red flags.

## Features

- **Single Job Analysis**: Paste job details (title, description, requirements, etc.) to get an instant verdict on whether the job is real or fake.
- **Batch CSV Analysis**: Upload a CSV file containing multiple job postings to get predictions for all of them at once.
- **Ensemble ML Models**: Choose between XGBoost, CatBoost, or an Ensemble of both for the prediction.
- **Rule-Based Red Flags**: Automatically detects common red flags such as missing company profiles, suspiciously high weekly salaries, "work from home + no experience" combinations, and more.
- **Cybersecurity Theme**: A premium, dark-themed UI with glassmorphism panels, glowing elements, and dynamic animations for an investigative feel.

## Project Structure

```
fake_job_detector/
├── app.py                      # Main Flask application and API routes
├── train_models.py             # Script used to generate/train the models
├── requirements.txt            # Python dependencies
├── model/                      # Pre-trained models and preprocessors
│   ├── xgboost_model.json
│   ├── catboost_model.cbm
│   └── preprocessor.pkl
├── static/
│   ├── css/style.css           # Custom styling and animations
│   └── js/main.js              # Client-side interactivity and API calls
└── templates/
    └── index.html              # Main frontend HTML template
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SHREYASH-W/fake-job-detector.git
   cd fake-job-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Open in Browser:**
   Navigate to `http://localhost:5000` in your web browser.

## Tech Stack

- **Backend**: Python, Flask, Pandas, NumPy
- **Machine Learning**: XGBoost, CatBoost, Scikit-Learn
- **Frontend**: HTML5, Vanilla CSS3 (Custom Variables, Animations), Vanilla JavaScript
- **Fonts & Icons**: Google Fonts (JetBrains Mono, DM Sans), Font Awesome

## How It Works

1. **Paste Job**: Provide the job listing details into the form fields.
2. **AI Analysis**: The XGBoost & CatBoost ensemble scans text patterns (via TF-IDF) and metadata for fraud signals.
3. **Get Result**: Receive a clear verdict, confidence score, risk level, and a detailed red flag breakdown.

## Author

Developed by Shreyash.
