# flop_counter.py
from dataclasses import dataclass
from typing import Callable
import functools

@dataclass
class FlopCounter:
    count: int = 0

    def add(self, n: int):
        self.count += int(n)

    def reset(self):
        self.count = 0

    def snapshot(self) -> int:
        return int(self.count)

# a small decorator to annotate functions where we want to increment cost
def counted(cost: int):
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # note: this expects a global FLOPS variable imported by callers
            from flop_counter import GLOBAL_FLOPS  # local import to avoid circular on module load
            GLOBAL_FLOPS.add(cost)
            return fn(*args, **kwargs)
        return wrapped
    return deco

# single global instance for convenience
GLOBAL_FLOPS = FlopCounter()
