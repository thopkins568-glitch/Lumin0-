# flop_counter.py
from dataclasses import dataclass

@dataclass
class FlopCounter:
    """Simple global FLOP counter (counts arithmetic operations only)."""
    count: int = 0

    def add(self, n: int):
        self.count += int(n)

    def reset(self):
        self.count = 0

    def snapshot(self) -> int:
        return int(self.count)

# single global instance used by all modules
GLOBAL_FLOPS = FlopCounter()
