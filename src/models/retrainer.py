import pandas as pd
from src.models.model_trainer import ModelTrainer

class AutoRetrainer:
    def run(self):
        df = pd.read_csv("data/feedback/rainfall_feedback.csv")
        trainer = ModelTrainer()
        return trainer.train(df)
