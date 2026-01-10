from collections import deque

class SlidingWindow:
    def __init__(self, size: int):
        self.size = size
        self.window = deque()
        self.total = 0.0

    def add(self, value: float):
        self.window.append(value)
        self.total += value
        if len(self.window) > self.size:
            self.total -= self.window.popleft()

    def mean(self):
        return self.total / len(self.window)
