import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from src.models.trainer import ModelTrainer

# -------------------------------------------------
# Paths
# -------------------------------------------------
MODEL_DIR = "data/models"
STATS_DIR = "data/stats"

CHAMPION_PATH = f"{MODEL_DIR}/champion.pkl"
CANDIDATE_PATH = f"{MODEL_DIR}/candidate.pkl"
FEEDBACK_PATH = "data/feedback/rainfall_feedback.csv"
STATS_PATH = f"{STATS_DIR}/training_stats.pkl"


def retrain_and_evaluate():
    """
    Phase 3:
    - Retrain candidate model from feedback
    - Compare against champion
    - Promote if better

    Phase 4:
    - Persist training feature distributions
      for drift detection
    """

    # -------------------------------------------------
    # Preconditions
    # -------------------------------------------------
    if not os.path.exists(FEEDBACK_PATH):
        return {"status": "no_feedback"}

    if not os.path.exists(CHAMPION_PATH):
        return {"status": "no_champion"}

    df = pd.read_csv(FEEDBACK_PATH)

    if "actual_rainfall" not in df.columns:
        return {"status": "missing_labels"}

    df = df.dropna(subset=["actual_rainfall"])

    if len(df) < 100:
        return {"status": "insufficient_data"}

    # Training contract
    df = df.rename(columns={"actual_rainfall": "rainfall"})

    # -------------------------------------------------
    # Train candidate (DO NOT TOUCH CHAMPION)
    # -------------------------------------------------
    trainer = ModelTrainer()
    train_metrics = trainer.train(
        df,
        output_path=CANDIDATE_PATH
    )

    # -------------------------------------------------
    # Load models
    # -------------------------------------------------
    champion = joblib.load(CHAMPION_PATH)
    candidate = joblib.load(CANDIDATE_PATH)

    X = df[champion["features"]]
    y = df["rainfall"]

    champ_preds = champion["model"].predict(X)
    cand_preds = candidate["model"].predict(X)

    champ_rmse = np.sqrt(mean_squared_error(y, champ_preds))
    cand_rmse = np.sqrt(mean_squared_error(y, cand_preds))

    # -------------------------------------------------
    # Promotion decision
    # -------------------------------------------------
    if cand_rmse < champ_rmse:
        # Atomic promotion
        joblib.dump(candidate, CHAMPION_PATH)
        decision = "promoted"

        # -------------------------------
        # Phase 4: Save training stats
        # -------------------------------
        os.makedirs(STATS_DIR, exist_ok=True)

        training_stats = {
            feature: X[feature].values
            for feature in champion["features"]
        }

        joblib.dump(training_stats, STATS_PATH)

    else:
        decision = "rejected"

    return {
        "status": decision,
        "champion_rmse": float(champ_rmse),
        "candidate_rmse": float(cand_rmse),
        "train_rmse": float(train_metrics["rmse"])
    }
