<div align="center">

# Halo CME Detection

**Physics-informed machine learning for detecting Halo Coronal Mass Ejections from in-situ solar wind plasma data**

[![Python](https://img.shields.io/badge/Python-3.10%2B-1c1a16?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-1c1a16?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-1c1a16?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-1c1a16?style=flat-square)](https://xgboost.readthedocs.io/)
[![Deploy](https://img.shields.io/badge/Deployed%20on-Render-1c1a16?style=flat-square&logo=render&logoColor=white)](https://halo-cme-detection.onrender.com)

[Live demo](https://halo-cme-detection.onrender.com) · [Methodology](#methodology) · [API](#api) · [Run locally](#run-locally)

</div>

<br>

![Landing page](public/landing.png)

## Overview

Halo CMEs are the most geo-effective solar events: they drive geomagnetic storms, disrupt satellite operations and degrade GNSS accuracy. Conventional detection relies on coronagraph imagery (LASCO / CACTUS), which requires ground processing and introduces delay.

This project takes a different route. It scores **five-minute plasma measurements from the Aditya-L1 Solar Wind Ion Spectrometer (SWIS)** directly, using four interpretable, physics-derived features and a soft-voting ensemble. No imaging is required, inference is lightweight enough for on-board execution, and every prediction is traceable to a handful of plasma ratios.

The system is delivered as a FastAPI service with a purpose-built web interface: upload a SWIS window, receive a calibrated probability, the per-estimator vote, the derived features and an interactive view of the underlying plasma parameters.

## Highlights

- **100 % recall on Halo CME windows** on the held-out test split, with zero false negatives
- **Four physics-informed features** instead of black-box embeddings, keeping the model interpretable for scientific use
- **Soft-voting ensemble** of Random Forest, XGBoost and Logistic Regression
- **Production-style web service**: FastAPI JSON API, dependency-free frontend, drag-and-drop upload, bundled sample dataset
- **Robust input handling**: unsorted rows, UTF-8 BOM, sub-5-minute cadence (resampled) and sparse data (rejected with a clear message)

![Detection results](public/detection.png)

## Methodology

### Data

| | |
|---|---|
| **Source** | Aditya-L1 SWIS Level-2 plasma moments, downsampled to 5-minute cadence |
| **Parameters** | Proton density N<sub>p</sub>, proton speed V<sub>p</sub>, proton temperature T<sub>p</sub>, alpha density N<sub>α</sub> |
| **Ground truth** | Halo CME events from the CACTUS LASCO catalogue |
| **Windows** | T − 1 d to T + 2 d around each event timestamp; 13 CME and 30 non-CME windows |

### Features

Each feature is computed row-wise and averaged over the window.

| Feature | Definition | Physical rationale |
|---|---|---|
| Alpha–proton ratio | N<sub>α</sub> / N<sub>p</sub> | CME ejecta carries enhanced helium abundance relative to ambient wind |
| Speed variability | σ(V<sub>p</sub>), centred 15-min rolling window | Shock-driven turbulence produces short-scale speed fluctuations |
| Alpha over V<sub>p</sub> std | (N<sub>α</sub> / N<sub>p</sub>) / σ(V<sub>p</sub>) | Isolates alpha-rich, low-turbulence ejecta cores behind the sheath |
| Alpha–temperature ratio | N<sub>α</sub> / T<sub>p</sub> | Magnetic clouds are anomalously cool for their speed |

### Model

A scikit-learn `VotingClassifier` (soft voting) over three complementary learners:

| Estimator | Key configuration | Contribution |
|---|---|---|
| `RandomForestClassifier` | `class_weight="balanced"` | Robust to outliers, non-linear interactions |
| `XGBClassifier` | `scale_pos_weight` = class ratio, `eval_metric="logloss"` | Bias reduction on the minority class |
| `LogisticRegression` | `penalty="l2"`, `solver="liblinear"`, balanced | Calibrated probabilities, linear baseline |

The decision threshold is set to **0.45** to prioritise recall — a missed CME is far costlier than a false alarm.

### Performance (30 % held-out split)

| Accuracy | Precision (CME) | Recall (CME) | F1 (CME) | ROC AUC | False negatives |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 85 % | 67 % | 100 % | 80 % | 0.91 | 0 |

## Architecture

```
Browser ──► static/index.html + app.js ──► POST /api/predict ──► main.py (FastAPI)
                                                                     │
                                                    app/Utils/features.py  (feature engineering)
                                                                     │
                                                    app/model/cme_model.joblib  (VotingClassifier)
```

| Layer | Technology |
|---|---|
| Model | scikit-learn, XGBoost, pandas, NumPy |
| API | FastAPI, Uvicorn |
| Frontend | HTML, CSS, vanilla JavaScript — no build step, no chart library |
| Deployment | Render (`render.yaml`, `start.sh`) |

## API

```
GET  /api/health
     → model name, estimator list, threshold, feature names

POST /api/predict        multipart/form-data, field "file" (.csv)
     → prediction, probability, threshold, derived features,
       per-estimator probabilities, dataset summary, preview rows,
       downsampled time series for plotting
```

**Input CSV columns:** `timestamp`, `proton_density`, `proton_speed`, `proton_temperature`, `alpha_density` — two to three days of data at ≤ 5-minute cadence. Validation errors are returned as `400` with a human-readable `detail`.

## Run locally

Requires Python 3.10+.

```bash
git clone https://github.com/Arnav020/Halo-CME-Detection.git
cd Halo-CME-Detection

python -m venv venv
venv\Scripts\activate          # macOS / Linux: source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 and either upload a CSV or click **Load a sample SWIS window**.

## Project structure

```
.
├── main.py                       FastAPI application: API routes + static hosting
├── app/
│   ├── model/cme_model.joblib    Trained VotingClassifier
│   └── Utils/features.py         Feature engineering pipeline
├── static/
│   ├── index.html                Single-page frontend
│   ├── css/style.css             Design system
│   ├── js/app.js                 Upload flow, charts, animations
│   └── samples/                  Sample SWIS window for the demo button
├── public/                       Screenshots
├── requirements.txt
├── render.yaml · start.sh        Render deployment
└── debug_input.csv               Test dataset
```

## Roadmap

- Shock and sheath region segmentation
- Multi-year SWIS dataset expansion (CACTUS and SEEDS catalogues)
- Temporal embeddings via recurrent models
- ONNX export for on-board inference

## Authors

**Arnav Joshi** · **Pulkit Garg**
B.Tech Computer Science & Engineering, Thapar Institute of Engineering & Technology
