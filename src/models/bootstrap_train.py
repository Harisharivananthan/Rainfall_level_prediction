import os
import pandas as pd

from src.features.feature_builder import FeatureBuilder
from src.models.trainer import ModelTrainer

DATA_PATH = "data/raw/real_chennai_1984_2024.csv"


def main():
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Training data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    fb = FeatureBuilder()
    df = fb.build(df, mode="train")  # 🔴 CRITICAL

    trainer = ModelTrainer()
    metrics = trainer.train(df)

    print("Bootstrap training successful")
    print(metrics)


if __name__ == "__main__":
    main()
