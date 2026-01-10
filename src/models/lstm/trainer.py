import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

from src.models.lstm.dataset import SequenceDataset
from src.models.lstm.model import build_lstm

class LSTMTrainer:
    def train(self, X, y, features, seq_len=30):
        ds = SequenceDataset(X, y, seq_len)
        X_seq, y_seq = ds.build()

        split = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        model = build_lstm(X_train.shape[1:])
        model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=20,
            batch_size=32,
            verbose=1
        )

        preds = model.predict(X_test).ravel()
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        model.save("data/models/lstm_candidate")
        joblib.dump(features, "data/models/lstm_features.pkl")

        return {
            "rmse": float(rmse),
            "model": "LSTM"
        }
