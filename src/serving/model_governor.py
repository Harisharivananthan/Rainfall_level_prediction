import os
import joblib
from datetime import datetime, timedelta

DRIFT_PATH = "data/stats/drift_history.pkl"
METRICS_PATH = "data/stats/model_metrics.pkl"
GOVERNANCE_LOG = "data/stats/governance_log.csv"

# Governance thresholds (industry-grade defaults)
MIN_DAYS_BETWEEN_RETRAINS = 7
MAX_ACCEPTABLE_RMSE_INCREASE = 0.15
DRIFT_SEVERITY_TRIGGER = {"medium", "high"}


class ModelGovernor:
    def __init__(self):
        self._load_state()

    def _load_state(self):
        self.drift_history = joblib.load(DRIFT_PATH) if os.path.exists(DRIFT_PATH) else []
        self.metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else []

    def _last_retrain_time(self):
        retrains = [m for m in self.metrics if m.get("event") == "retrain"]
        if not retrains:
            return None
        return max(r["timestamp"] for r in retrains)

    def _recent_drift(self):
        if not self.drift_history:
            return None
        return self.drift_history[-1]

    def should_retrain(self) -> dict:
        """
        Core governance decision engine.
        Returns explicit decision + reason.
        """

        now = datetime.utcnow()

        # -----------------------------
        # Rule 1: Drift must exist
        # -----------------------------
        drift = self._recent_drift()
        if drift is None:
            return self._decision("WAIT", "No drift data available")

        if drift["severity"] not in DRIFT_SEVERITY_TRIGGER:
            return self._decision(
                "WAIT",
                f"Drift severity '{drift['severity']}' below retrain threshold"
            )

        # -----------------------------
        # Rule 2: Cooldown enforcement
        # -----------------------------
        last_retrain = self._last_retrain_time()
        if last_retrain:
            delta = now - last_retrain
            if delta < timedelta(days=MIN_DAYS_BETWEEN_RETRAINS):
                return self._decision(
                    "BLOCK",
                    f"Cooldown active ({delta.days} days since last retrain)"
                )

        # -----------------------------
        # Rule 3: Performance regression check
        # -----------------------------
        rmses = [m["rmse"] for m in self.metrics if "rmse" in m]
        if len(rmses) >= 2:
            prev, latest = rmses[-2], rmses[-1]
            increase = (latest - prev) / prev
            if increase > MAX_ACCEPTABLE_RMSE_INCREASE:
                return self._decision(
                    "BLOCK",
                    f"RMSE regression too high ({increase:.2%})"
                )

        # -----------------------------
        # Approved
        # -----------------------------
        return self._decision(
            "RETRAIN",
            f"Drift severity '{drift['severity']}' with governance approval"
        )

    def _decision(self, action: str, reason: str) -> dict:
        decision = {
            "action": action,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._log(decision)
        return decision

    def _log(self, decision: dict):
        import pandas as pd

        os.makedirs(os.path.dirname(GOVERNANCE_LOG), exist_ok=True)

        df = pd.DataFrame([decision])
        if os.path.exists(GOVERNANCE_LOG):
            old = pd.read_csv(GOVERNANCE_LOG)
            df = pd.concat([old, df], ignore_index=True)

        df.to_csv(GOVERNANCE_LOG, index=False)
