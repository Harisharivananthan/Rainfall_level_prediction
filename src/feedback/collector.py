import os
import pandas as pd
from datetime import datetime

FEEDBACK_PATH = "data/feedback/rainfall_feedback.csv"


def save_feedback(
    features: dict,
    prediction: float,
    actual: float | None = None
):
    os.makedirs("data/feedback", exist_ok=True)

    row = {
        **features,
        "prediction": prediction,
        "actual_rainfall": actual,
        "timestamp": datetime.utcnow().isoformat()
    }

    if os.path.exists(FEEDBACK_PATH):
        df = pd.read_csv(FEEDBACK_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(FEEDBACK_PATH, index=False)
