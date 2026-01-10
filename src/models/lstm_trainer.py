import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class LSTMTrainer:
    def __init__(self, lookback=30, epochs=25, batch_size=64):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()

    def _make_seq(self, X, y):
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i-self.lookback:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def train(self, df):
        cols = [
            "rainfall","temperature","humidity",
            "wind_speed","pressure",
            "month_sin","month_cos","doy_sin","doy_cos"
        ]

        data = self.scaler.fit_transform(df[cols])
        X, y = data[:,1:], data[:,0]

        Xs, ys = self._make_seq(X, y)
        split = int(len(Xs) * 0.8)

        model = Sequential([
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")
        model.fit(
            Xs[:split], ys[:split],
            validation_split=0.1,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )

        model.save("data/models/lstm_model")
        joblib.dump(self.scaler, "data/models/lstm_scaler.pkl")
