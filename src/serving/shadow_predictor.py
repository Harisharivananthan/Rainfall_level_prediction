import joblib

class ShadowModel:
    def __init__(self, path):
        bundle = joblib.load(path)
        self.model = bundle["model"]

    def predict(self, df):
        return self.model.predict(df)[0]
