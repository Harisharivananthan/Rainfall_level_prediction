import os
import shutil
import joblib
from datetime import datetime

from src.serving.model_governor import ModelGovernor
from src.models.retrain_pipeline import retrain_and_evaluate

MODEL_DIR = "data/models"
CHAMPION = f"{MODEL_DIR}/champion.pkl"
CANDIDATE = f"{MODEL_DIR}/candidate.pkl"
METRICS_PATH = "data/stats/model_metrics.pkl"


def main():
    governor = ModelGovernor()
    decision = governor.should_retrain()

    print("Governance decision:", decision)

    if decision["action"] != "RETRAIN":
        print("Retraining skipped:", decision["reason"])
        return

    # -----------------------------
    # Run retraining
    # -----------------------------
    print("Starting retraining job...")
    result = retrain_and_evaluate(save_as=CANDIDATE)

    if not os.path.exists(CANDIDATE):
        raise RuntimeError("Candidate model not produced")

    # -----------------------------
    # Compare models
    # -----------------------------
    champion = joblib.load(CHAMPION)
    candidate = joblib.load(CANDIDATE)

    old_rmse = champion.get("rmse", float("inf"))
    new_rmse = result["rmse"]

    if new_rmse >= old_rmse:
        print("Candidate rejected (worse performance)")
        os.remove(CANDIDATE)
        return

    # -----------------------------
    # Atomic promotion
    # -----------------------------
    backup = f"{MODEL_DIR}/champion_backup_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pkl"
    shutil.copy(CHAMPION, backup)
    shutil.move(CANDIDATE, CHAMPION)

    # -----------------------------
    # Persist metrics
    # -----------------------------
    metrics = {
        "event": "retrain",
        "rmse": new_rmse,
        "timestamp": datetime.utcnow()
    }

    if os.path.exists(METRICS_PATH):
        history = joblib.load(METRICS_PATH)
    else:
        history = []

    history.append(metrics)
    joblib.dump(history, METRICS_PATH)

    print("New model promoted successfully")


if __name__ == "__main__":
    main()
