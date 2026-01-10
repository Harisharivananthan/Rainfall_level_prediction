import joblib
import numpy as np
import tensorflow as tf

class LSTMPredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            "data/models/lstm_candidate"
        )
        self.features = joblib.load(
            "data/models/lstm_features.pkl"
        )
        self.seq_len = 30
        self.buffer = []

    def predict(self, x):
        self.buffer.append(x)
        if len(self.buffer) < self.seq_len:
            return None

        seq = np.array(self.buffer[-self.seq_len:])[None, ...]
        return float(self.model.predict(seq)[0][0])
