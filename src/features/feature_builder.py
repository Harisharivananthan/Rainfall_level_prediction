import numpy as np
import pandas as pd


class FeatureBuilder:
    def build(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        mode = "train"     → production training (NO rainfall lags)
        mode = "infer"     → API prediction (NO rainfall)
        mode = "research"  → offline experiments (rainfall lags allowed)
        """

        df = df.copy()

        # -------------------------------------------------
        # Date (MANDATORY)
        # -------------------------------------------------
        if "date" not in df.columns:
            raise ValueError("Missing required column: date")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        # -------------------------------------------------
        # Weather normalization
        # -------------------------------------------------
        df["temperature"] = df["temperature_c"]
        df["humidity"] = df["humidity_percent"]
        df["wind_speed"] = df["wind_speed_ms"]
        df["pressure"] = df["sea_level_pressure_hpa"]

        # -------------------------------------------------
        # Time features
        # -------------------------------------------------
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # -------------------------------------------------
        # Interaction feature
        # -------------------------------------------------
        df["humidity_temp_ratio"] = df["humidity"] / (df["temperature"] + 1e-3)

        # -------------------------------------------------
        # TRAIN MODE (TARGET ONLY, NO LAGS)
        # -------------------------------------------------
        if mode == "train":
            if "rainfall_mm" not in df.columns:
                raise ValueError("rainfall_mm required for training")

            df["rainfall"] = df["rainfall_mm"]
            return df.reset_index(drop=True)

        # -------------------------------------------------
        # RESEARCH MODE (LAGS + TARGET)
        # -------------------------------------------------
        if mode == "research":
            if "rainfall_mm" not in df.columns:
                raise ValueError("rainfall_mm required for research")

            df["rainfall"] = df["rainfall_mm"]

            for lag in [1, 3, 7, 14, 30]:
                df[f"rainfall_lag_{lag}"] = df["rainfall"].shift(lag)

            for win in [7, 14, 30]:
                df[f"rainfall_roll_mean_{win}"] = (
                    df["rainfall"].shift(1).rolling(win).mean()
                )
                df[f"rainfall_roll_std_{win}"] = (
                    df["rainfall"].shift(1).rolling(win).std()
                )

            df["heavy_rain_flag"] = (df["rainfall"] > 100).astype(int)
            return df.dropna().reset_index(drop=True)

        # -------------------------------------------------
        # INFERENCE MODE
        # -------------------------------------------------
        return df.reset_index(drop=True)
