# flop_counter.py

class FlopCounter:
    def __init__(self):
        self.count = 0

    def add(self, n):
        self.count += int(n)

    def reset(self):
        self.count = 0

    def snapshot(self):
        return int(self.count)

# Single global instance used across all modules
GLOBAL_FLOPS = FlopCounter()
