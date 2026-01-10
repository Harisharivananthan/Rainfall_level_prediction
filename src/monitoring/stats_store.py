import numpy as np

class StatsStore:
    def __init__(self):
        self.predictions = []

    def add_prediction(self, value: float):
        self.predictions.append(value)

        if len(self.predictions) > 2000:
            self.predictions = self.predictions[-2000:]

    def get_baseline_and_recent(self):
        if len(self.predictions) < 1000:
            return None, None

        baseline = np.array(self.predictions[:500])
        recent = np.array(self.predictions[-500:])
        return baseline, recent
