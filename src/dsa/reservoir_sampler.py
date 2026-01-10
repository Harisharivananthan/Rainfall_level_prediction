import random

class ReservoirSampler:
    def __init__(self, k: int):
        self.k = k
        self.sample = []
        self.n = 0

    def process(self, value):
        self.n += 1
        if len(self.sample) < self.k:
            self.sample.append(value)
        else:
            j = random.randint(0, self.n - 1)
            if j < self.k:
                self.sample[j] = value
