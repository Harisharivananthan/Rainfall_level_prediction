import joblib
import numpy as np
import pandas as pd

from src.features.feature_builder import FeatureBuilder
from src.risk_engine.flood_risk_classifier import FloodRiskClassifier

MODEL_PATH = "data/models/champion.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURE_COLUMNS = bundle["features"]

feature_builder = FeatureBuilder()
risk_engine = FloodRiskClassifier()

def predict_from_event(event: dict) -> dict:
    df = pd.DataFrame([event])
    feat_df = feature_builder.build(df, mode="infer")

    X = feat_df[FEATURE_COLUMNS]

    preds = np.array([t.predict(X)[0] for t in model.estimators_])
    mean = float(preds.mean())
    std = float(preds.std())

    return {
        "prediction_mm": mean,
        "uncertainty_mm": std,
        "risk_level": risk_engine.classify(mean),
        "raw_event": event
    }
