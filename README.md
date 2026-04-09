# 🌞 CME Classifier –  Coronal Mass Ejection Detection

- A web-based ML tool for classifying Coronal Mass Ejections (CMEs) using **real solar wind data** from Aditya-L1 SWIS Level-2 dataset. Powered by a physics-informed ML model deployed using **FastAPI**.
- Deployed link: [https://heliopredict.streamlit.app/](https://heliopredict.streamlit.app/)

---

## 🚀 Features

- 🌞 Accepts real-time 5-minute SWIS CSV data
- 📊 Predicts CME occurrence using:
  - Alpha-Proton density ratios
  - Proton temperature to speed ratio
  - Velocity variability
- 🧠 Trained on labeled CACTUS-Halo CME events
- 💡 FastAPI-based backend with clean HTML frontend
- ✅ No internet model inference — runs fully on backend
- ⚙️ Deployable via Render or locally

---

## 🧪 Example Usage

1. Prepare a CSV file with **2–3 days of 5-min averaged data**.
2. Required headers:
   - `proton_density`
   - `proton_speed`
   - `proton_temperature`
   - `alpha_density`
3. Upload on the web UI → get prediction and confidence score.

---

## 🧠 CME Classification Model Overview

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

## 🖥️ Run Locally

> Requires Python 3.10+

```bash
git clone https://github.com/yourusername/cme-classifier.git
cd cme-classifier

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload

---

## 👨‍💻 Author

**Arnav Joshi**  
- B.Tech CSE @ Thapar University  
- Passionate about space-AI and physics-informed ML

