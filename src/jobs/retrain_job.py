import pandas as pd
from src.models.retrainer import AutoRetrainer
from src.models.champion_evaluator import ChampionEvaluator
from src.models.promoter import ChampionPromoter

def run():
    AutoRetrainer().run()

    df = pd.read_csv("data/feedback/rainfall_feedback.csv")
    X = df.drop(columns=["rainfall"])
    y = df["rainfall"]

    cand_rmse, champ_rmse = ChampionEvaluator().compare(X, y)

    if cand_rmse < champ_rmse:
        ChampionPromoter().promote()
        print("✅ Champion updated")
    else:
        print("❌ Candidate rejected")
