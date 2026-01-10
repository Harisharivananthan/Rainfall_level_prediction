import numpy as np

class SequenceDataset:
    def __init__(self, X, y, seq_len=30):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def build(self):
        Xs, ys = [], []
        for i in range(len(self.X) - self.seq_len):
            Xs.append(self.X[i:i+self.seq_len])
            ys.append(self.y[i+self.seq_len])
        return np.array(Xs), np.array(ys)
