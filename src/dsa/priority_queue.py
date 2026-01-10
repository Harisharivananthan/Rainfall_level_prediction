import heapq

class AlertQueue:
    def __init__(self):
        self.heap = []

    def push(self, severity: int, message: str):
        heapq.heappush(self.heap, (-severity, message))

    def pop(self):
        return heapq.heappop(self.heap)
