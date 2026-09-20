# Halo CME Detection – Coronal Mass Ejection Classifier

A web application that classifies **Halo Coronal Mass Ejections (CMEs)** from real solar wind
plasma data (Aditya-L1 SWIS Level-2). A physics-informed ensemble model runs behind a **FastAPI**
JSON API, and a dependency-free **HTML / CSS / JavaScript** frontend presents the verdict,
ensemble votes, derived features and parameter time series.

- Deployed link: https://halo-cme-detection.onrender.com

---

## Features

- Accepts 5-minute cadence SWIS CSV data (finer data is resampled, sparser data is rejected)
- Predicts CME occurrence from four physics-informed features:
  - Alpha–proton density ratio
  - Proton speed variability (15-minute rolling std)
  - Alpha / speed-variability index
  - Alpha–temperature ratio
- Trained on labelled CACTUS Halo CME events
- Soft-voting ensemble (Random Forest + XGBoost + Logistic Regression)
- FastAPI backend with a JSON API (`/api/predict`, `/api/health`)
- Claymorphic, animated frontend: drag-and-drop upload, probability gauge, per-estimator votes,
  feature cards, interactive plasma time-series charts and a data preview
- Bundled sample window for a one-click demo
- Deployable on Render or locally; inference runs entirely on the backend

---

## Project structure

```
.
├── main.py                  # FastAPI app: API routes + static frontend
├── app/
│   ├── model/cme_model.joblib
│   └── Utils/features.py    # feature engineering (unchanged model pipeline)
├── static/
│   ├── index.html           # single-page frontend
│   ├── css/style.css        # design system (claymorphism)
│   ├── js/app.js            # upload flow, charts, animations
│   ├── samples/             # sample SWIS window for the demo button
│   └── favicon.svg
├── streamlit_app.py         # legacy Streamlit UI (kept for reference)
├── requirements.txt
├── render.yaml / start.sh   # Render deployment
└── debug_input.csv          # test dataset
```

---

## Example usage

1. Prepare a CSV file with **2–3 days of 5-min averaged data**.
2. Required headers:
   - `timestamp`
   - `proton_density`
   - `proton_speed`
   - `proton_temperature`
   - `alpha_density`
3. Upload on the web UI (or click *Load a sample SWIS window*) and run detection.

### API

```
GET  /api/health            -> model name, estimators, threshold, feature list
POST /api/predict  (multipart "file") -> prediction, probability, threshold, features,
                                          per-estimator votes, dataset summary,
                                          preview rows, downsampled series
```

---

## CME classification model overview

This document provides a technical overview of the machine learning model used in the **CME Classifier Web App** for detecting Coronal Mass Ejections from solar wind data.

---

### 📌 Objective

To classify whether a given 3-day window of Aditya-L1 SWIS Level-2 data contains a **Halo CME event** (label = 1) or not (label = 0) using physics-informed features and ensemble learning.

---

### 🧮 Model Architecture

We use a **VotingClassifier** that ensembles the predictions of the following base models:

- `RandomForestClassifier`
  - class_weight: `"balanced"`
  - random_state: 42
- `XGBClassifier`
  - scale_pos_weight: ratio of class imbalance
  - eval_metric: `"logloss"`
  - random_state: 42
- `LogisticRegression`
  - penalty: `"l2"`
  - solver: `"liblinear"`
  - class_weight: `"balanced"`

#### Ensemble Strategy

- **Voting Type:** Soft (averages class probabilities)
- **Rationale:** Reduces false negatives while balancing overfitting and generalization

---

### 📈 Model Performance

- **Accuracy:** 85% on held-out test set
- **Precision:** High
- **False Negatives:** 0 (No CME missed)
- **ROC AUC:** ~0.91

---

### ⚙️ Features Used (Derived from Raw Data)

Each feature is computed over a **3-day window (T-1 to T+2)** around an event timestamp:

| Feature Name         | Formula                                                            | Description                                      |
|----------------------|---------------------------------------------------------------------|--------------------------------------------------|
| Alpha/Proton Ratio   | `alpha_density / proton_density`                                   | Indicates ion composition changes during CME     |
| Alpha / Vp Std       | `std(alpha_density / proton_speed)`                                | Measures variability in heavy-ion speed          |
| Alpha/TP Ratio       | `alpha_density / (proton_temperature * proton_density)`            | Combines temperature and composition signal      |
| Vp Std 15min         | `std(proton_speed)` on 15-min scale within 3-day window            | Measures solar wind variability                  |

---

### 🧪 Data Summary

- **Labeled Events:**
  - 13 Halo CME events (from CACTUS LASCO catalog)
  - 30 Non-CME events (randomly sampled)
- **Dataset Source:** Aditya-L1 SWIS Level-2 (5-minute downsampled)
- **File used:** `swis_downsampled_5min.csv`

---

### 🗃️ Training Pipeline

1. **Preprocessing**
   - Load and clean solar wind CSV data
   - Extract 3-day windows around each CME/Non-CME timestamp

2. **Feature Engineering**
   - Compute 4 derived features for each window
   - Normalize where necessary

3. **Model Training**
   - Train individual classifiers
   - Fit VotingClassifier on training data
   - Evaluate on test set (30% split)

4. **Model Export**
   - Saved as `model/classifier.pkl` using `joblib.dump()`

---

### 🧠 Why Physics-Informed Features?

Using domain-specific ratios (like Alpha/Proton) helps generalize across time periods and missions, and avoids black-box dependence. The model is **interpretable**, making it useful for both scientific and operational use.

---

### 🔍 Future Improvements

- Add liveness checks for real-time onboard systems
- Train on more labeled events (using CACTUS & SEEDS)
- Experiment with LSTM for temporal embedding
- Add feature importance plots in UI

---

## Run locally

> Requires Python 3.10+

```bash
git clone https://github.com/yourusername/cme-classifier.git
cd cme-classifier

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000.

---

## Authors

**Arnav Joshi · Pulkit Garg**  
- B.Tech CSE @ Thapar Institute of Engineering & Technology  
- Passionate about space-AI and physics-informed ML

