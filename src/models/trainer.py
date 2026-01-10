import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class ModelTrainer:
    def train(self, df):
        # -------------------------------------------------
        # ONLINE FEATURE CONTRACT (API-SAFE)
        # -------------------------------------------------
        UI_FEATURES = [
            "temperature",
            "humidity",
            "wind_speed",
            "pressure",

            "month",
            "day_of_year",
            "month_sin",
            "month_cos",
            "doy_sin",
            "doy_cos",

            "humidity_temp_ratio"
        ]

        # -------------------------------------------------
        # Split features / target
        # -------------------------------------------------
        X = df[UI_FEATURES]
        y = df["rainfall"]

        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # -------------------------------------------------
        # Model (Champion)
        # -------------------------------------------------
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
            n_jobs=-1,
            random_state=42
        )

        model.fit(X_train, y_train)

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # -------------------------------------------------
        # Persist model + feature schema
        # -------------------------------------------------
        bundle = {
            "model": model,
            "features": UI_FEATURES
        }

        joblib.dump(bundle, "data/models/champion.pkl")

        return {
            "rmse": float(rmse),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
