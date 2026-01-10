class FloodRiskClassifier:
    def classify(self, rainfall: float) -> str:
        if rainfall < 50:
            return "LOW"
        if rainfall < 100:
            return "MEDIUM"
        return "HIGH"
