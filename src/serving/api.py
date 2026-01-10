import os
import shutil
import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

from src.dsa.lru_cache import LRUCache
from src.serving.shadow_predictor import ShadowModel
from src.risk_engine.flood_risk_classifier import FloodRiskClassifier
from src.feedback.collector import save_feedback
from src.models.retrain_pipeline import retrain_and_evaluate
from src.monitoring.drift_detector import DriftDetector
from src.features.feature_builder import FeatureBuilder

# =================================================
# PATHS
# =================================================
CHAMPION_PATH = "data/models/champion.pkl"        # RF
SHADOW_LSTM_PATH = "data/models/lstm_shadow.pkl"  # LSTM / Transformer
STATS_PATH = "data/stats/training_stats.pkl"

# =================================================
# APP
# =================================================
app = FastAPI(title="Flood Risk Prediction Platform – Phase 6A")

app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
templates = Jinja2Templates(directory="src/web/templates")

# =================================================
# GLOBALS
# =================================================
champion_model = None
FEATURE_COLUMNS = None
shadow_model = None
drift_detector = None

feature_builder = FeatureBuilder()
risk_engine = FloodRiskClassifier()
cache = LRUCache(100_000)

REQ = Counter("requests_total", "Total requests")
LAT = Histogram("latency_seconds", "Request latency")

# =================================================
# STARTUP
# =================================================
@app.on_event("startup")
def load_models():
    global champion_model, FEATURE_COLUMNS, shadow_model, drift_detector

    if not os.path.exists(CHAMPION_PATH):
        raise RuntimeError("Champion model missing. Run `make all`.")

    bundle = joblib.load(CHAMPION_PATH)
    champion_model = bundle["model"]
    FEATURE_COLUMNS = bundle["features"]

    # ---- Shadow LSTM (Phase 6A) ----
    if os.path.exists(SHADOW_LSTM_PATH):
        shadow_model = ShadowModel(SHADOW_LSTM_PATH)
        print("LSTM/Transformer shadow model loaded")
    else:
        shadow_model = None
        print("No LSTM shadow model found (optional)")

    # ---- Drift Detector ----
    if os.path.exists(STATS_PATH):
        drift_detector = DriftDetector(STATS_PATH)

    print("Phase 6A models loaded successfully")

# =================================================
# SCHEMAS
# =================================================
class PredictRequest(BaseModel):
    date: str
    temperature_c: float
    humidity_percent: float
    wind_speed_ms: float
    sea_level_pressure_hpa: float


class FeedbackSchema(BaseModel):
    actual_rainfall: float

# =================================================
# ROUTES
# =================================================
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# =================================================
# PREDICTION (Champion RF + Shadow LSTM)
# =================================================
@app.post("/predict")
@LAT.time()
async def predict(req: PredictRequest):
    REQ.inc()

    cache_key = tuple(req.dict().values())
    cached = cache.get(cache_key)
    if cached:
        return cached

    # -----------------------------
    # Feature Engineering (Infer)
    # -----------------------------
    raw_df = pd.DataFrame([req.dict()])
    feat_df = feature_builder.build(raw_df, mode="infer")

    missing = set(FEATURE_COLUMNS) - set(feat_df.columns)
    if missing:
        raise ValueError(f"Missing inference features: {missing}")

    X = feat_df[FEATURE_COLUMNS]

    # -----------------------------
    # Champion RF Prediction
    # -----------------------------
    rf_preds = np.array([t.predict(X)[0] for t in champion_model.estimators_])
    mean_pred = float(rf_preds.mean())
    uncertainty = float(rf_preds.std())

    # -----------------------------
    # Shadow LSTM / Transformer
    # -----------------------------
    if shadow_model is not None:
        shadow_model.predict(X, silent=True)

    # -----------------------------
    # Feedback
    # -----------------------------
    save_feedback(
        features=req.dict(),
        prediction=mean_pred
    )

    response = {
        "prediction_mm": mean_pred,
        "uncertainty_mm": uncertainty,
        "risk_level": risk_engine.classify(mean_pred),
        "model": "random_forest_champion"
    }

    cache.put(cache_key, response)
    return response

# =================================================
# DRIFT DETECTION (PHASE 5)
# =================================================
@app.post("/drift")
def detect_drift(req: PredictRequest):
    if drift_detector is None:
        return {"status": "stats_not_available"}

    raw_df = pd.DataFrame([req.dict()])
    feat_df = feature_builder.build(raw_df, mode="infer")

    drift_report = drift_detector.detect(feat_df[FEATURE_COLUMNS])

    return {
        "status": "ok",
        "drift_report": drift_report
    }

# =================================================
# FEEDBACK
# =================================================
@app.post("/feedback")
def submit_feedback(feedback: FeedbackSchema):
    save_feedback(
        features={},
        prediction=0.0,
        actual=feedback.actual_rainfall
    )
    return {"status": "feedback saved"}

# =================================================
# RETRAINING
# =================================================
@app.post("/retrain")
def retrain():
    return retrain_and_evaluate()

# =================================================
# METRICS
# =================================================
@app.get("/metrics")
def metrics():
    return generate_latest()
