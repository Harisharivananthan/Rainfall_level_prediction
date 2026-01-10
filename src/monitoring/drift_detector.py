import numpy as np
import joblib
import os

DRIFT_THRESHOLD = 0.2  # industry standard


class DriftDetector:
    def __init__(self, stats_path: str):
        if not os.path.exists(stats_path):
            raise RuntimeError("Training stats not found")

        self.train_stats = joblib.load(stats_path)

    def psi(self, expected, actual, bins=10):
        expected = np.asarray(expected)
        actual = np.asarray(actual)

        breakpoints = np.linspace(0, 100, bins + 1)
        expected_perc = np.percentile(expected, breakpoints)
        actual_perc = np.percentile(actual, breakpoints)

        psi_value = 0.0
        for i in range(len(expected_perc) - 1):
            e = np.mean(
                (expected >= expected_perc[i]) &
                (expected < expected_perc[i + 1])
            )
            a = np.mean(
                (actual >= actual_perc[i]) &
                (actual < actual_perc[i + 1])
            )

            e = max(e, 1e-6)
            a = max(a, 1e-6)

            psi_value += (a - e) * np.log(a / e)

        return psi_value

    def detect(self, live_df):
        """
        live_df: DataFrame with same feature columns as training
        """
        report = {}

        for feature in live_df.columns:
            if feature not in self.train_stats:
                continue

            psi_score = self.psi(
                self.train_stats[feature],
                live_df[feature]
            )

            if psi_score < 0.1:
                severity = "low"
            elif psi_score < DRIFT_THRESHOLD:
                severity = "medium"
            else:
                severity = "high"

            report[feature] = {
                "psi": float(psi_score),
                "drift": psi_score > DRIFT_THRESHOLD,
                "severity": severity
            }

        return report
