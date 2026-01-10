class RingBuffer:
    def __init__(self, size=10000):
        self.buffer = [0.0] * size
        self.index = 0

    def add(self, value):
        self.buffer[self.index % len(self.buffer)] = value
        self.index += 1
