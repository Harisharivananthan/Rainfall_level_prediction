import joblib
from sklearn.metrics import mean_squared_error

class ChampionEvaluator:
    def compare(self, X, y):
        champ = joblib.load("data/models/champion.pkl")["model"]
        cand = joblib.load("data/models/candidate.pkl")["model"]

        return (
            mean_squared_error(y, cand.predict(X), squared=False),
            mean_squared_error(y, champ.predict(X), squared=False),
        )
