class DataValidator:
    REQUIRED_COLUMNS = {
        "date",
        "rainfall_mm",      # canonical rainfall column in this dataset
    }

    OPTIONAL_COLUMNS = {
        "temperature_c",
        "humidity_percent",
        "wind_speed_ms",
        "surface_pressure_pa",
    }

    def validate(self, df):
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Schema mismatch. Missing columns: {missing}")

        # Keep only known columns (defensive)
        allowed = self.REQUIRED_COLUMNS | self.OPTIONAL_COLUMNS
        return df[[c for c in df.columns if c in allowed]]
