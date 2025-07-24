from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from io import StringIO
from app.Utils.features import extract_features_from_window

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="app/templates")

model = joblib.load("app/model/cme_model.joblib")
THRESHOLD = 0.45

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Save file for debugging
        df.to_csv("debug_input.csv", index=False)

        # Validate required columns
        required_cols = {"proton_density", "proton_speed", "proton_temperature", "alpha_density"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError("Missing required columns: " + ", ".join(required_cols - set(df.columns)))

        # Check time resolution
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")
            time_deltas = df["timestamp"].diff().dropna().dt.total_seconds()
            if not (time_deltas.between(240, 360).mean() > 0.75):
                raise ValueError("Time resolution not close to 5 minutes. Please average your data.")

        # Feature extraction
        features_df = extract_features_from_window(df)
        if features_df.isnull().values.any():
            raise ValueError("Feature extraction failed. Ensure enough valid data is present (~15 min).")

        # Prediction
        prob = model.predict_proba(features_df)[0][1]
        prediction = int(prob >= THRESHOLD)
        result = "CME" if prediction else "Non-CME"

        # Pass preview rows (first 5 rows)
        preview_rows = df.head(5)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "submitted": True,
            "result": result,
            "confidence": round(prob * 100, 2),
            "preview_rows": preview_rows.to_html(classes="preview-table", index=False, border=0, escape=False)
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "submitted": True,
            "error": str(e)
        })
