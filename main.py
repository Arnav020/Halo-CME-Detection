"""
Halo CME Detection — FastAPI backend.

Serves the static frontend from /static and exposes a small JSON API:
    GET  /api/health   -> model + threshold info
    POST /api/predict  -> upload a SWIS CSV, get prediction + analysis payload
"""

import warnings
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.Utils.features import extract_features_from_window

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "app" / "model" / "cme_model.joblib"
THRESHOLD = 0.45
REQUIRED_COLS = ["timestamp", "proton_density", "proton_speed", "proton_temperature", "alpha_density"]
MAX_SERIES_POINTS = 480  # downsample uploaded data for the client-side charts

app = FastAPI(title="Halo CME Detection", version="2.0.0")
model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(v):
    """Convert numpy scalars / NaN to JSON-safe Python values."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def _series_payload(df: pd.DataFrame) -> dict:
    """Sorted, downsampled copy of the raw parameters for plotting."""
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp")

    if len(work) > MAX_SERIES_POINTS:
        idx = np.linspace(0, len(work) - 1, MAX_SERIES_POINTS).round().astype(int)
        work = work.iloc[idx]

    return {
        "timestamp": [t.isoformat() for t in work["timestamp"]],
        **{
            col: [_clean(v) for v in pd.to_numeric(work[col], errors="coerce")]
            for col in REQUIRED_COLS[1:]
        },
    }


def _dataset_summary(df: pd.DataFrame) -> dict:
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna().sort_values()
    span_hours = None
    cadence = None
    if len(ts) > 1:
        span_hours = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600)
        diffs = ts.diff().dropna().dt.total_seconds()
        if not diffs.empty:
            cadence = float(diffs.mode()[0])

    stats = {}
    for col in REQUIRED_COLS[1:]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        stats[col] = {
            "mean": _clean(s.mean()),
            "min": _clean(s.min()),
            "max": _clean(s.max()),
            "std": _clean(s.std()),
        }

    return {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "start": ts.iloc[0].isoformat() if len(ts) else None,
        "end": ts.iloc[-1].isoformat() if len(ts) else None,
        "span_hours": span_hours,
        "cadence_seconds": cadence,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": type(model).__name__,
        "estimators": [name for name, _ in getattr(model, "estimators", [])],
        "threshold": THRESHOLD,
        "features": list(getattr(model, "feature_names_in_", [])),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    raw = await file.read()
    try:
        df = pd.read_csv(StringIO(raw.decode("utf-8-sig")))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file contains no rows.")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail="Missing required columns: " + ", ".join(missing),
        )

    try:
        features_df = extract_features_from_window(df.copy())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if features_df.isnull().values.any():
        raise HTTPException(
            status_code=400,
            detail="Not enough valid data to compute features (at least ~15 minutes of 5-minute samples needed).",
        )

    prob = float(model.predict_proba(features_df)[0][1])
    prediction = "CME" if prob >= THRESHOLD else "Non-CME"

    # Per-estimator probabilities give a nice "how the ensemble voted" view.
    votes = []
    for name, est in getattr(model, "named_estimators_", {}).items():
        try:
            votes.append({"name": name, "probability": float(est.predict_proba(features_df)[0][1])})
        except Exception:  # noqa: BLE001
            continue

    preview = df.head(8).copy()
    preview_rows = [
        {str(k): _clean(v) for k, v in row.items()} for row in preview.to_dict(orient="records")
    ]

    return {
        "prediction": prediction,
        "probability": prob,
        "threshold": THRESHOLD,
        "features": {k: _clean(v) for k, v in features_df.iloc[0].to_dict().items()},
        "votes": votes,
        "dataset": _dataset_summary(df),
        "preview": preview_rows,
        "series": _series_payload(df),
        "filename": file.filename,
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
