import pandas as pd

class WeatherIngestor:
    def __init__(self, path: str, chunk_size: int = 1_000_000):
        self.path = path
        self.chunk_size = chunk_size

    def stream(self):
        for chunk in pd.read_csv(
            self.path,
            parse_dates=["date"],
            chunksize=self.chunk_size
        ):
            chunk.sort_values("date", inplace=True)
            yield chunk
