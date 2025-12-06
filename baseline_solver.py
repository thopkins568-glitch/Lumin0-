# baseline_solver.py
import numpy as np
from typing import Callable, Tuple
from flop_counter import GLOBAL_FLOPS as FLOPS

def baseline_step(pop: np.ndarray, func: Callable[[np.ndarray], float],
                  step_size: float = 0.1) -> np.ndarray:
    """One iteration of baseline random local search with greedy accept."""
    new_pop = pop.copy()
    n, d = pop.shape

    for i in range(n):
        # random step
        FLOPS.add(d * 3)
        delta = np.random.normal(loc=0.0, scale=step_size, size=(d,))
        candidate = pop[i] + delta

        # evaluate
        cur_val = func(pop[i])
        cand_val = func(candidate)
        FLOPS.add(20)

        if cand_val < cur_val:
            new_pop[i] = candidate

    return new_pop


def baseline_run(func: Callable[[np.ndarray], float],
                 dim: int = 8,
                 pop_size: int = 32,
                 steps: int = 100,
                 step_size: float = 0.1):
    """
    Run baseline solver; returns:
      - final population
      - path of mean objective values
      - total FLOPs used
      - best final value
    """

    # reset FLOP counter (important)
    FLOPS.reset()

    pop = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))
    path = []

    for _ in range(steps):
        pop = baseline_step(pop, func, step_size=step_size)
        path.append(float(func(pop.mean(axis=0))))

    best_val = float(func(pop.mean(axis=0)))
    total_flops = FLOPS.snapshot()

    return pop, path, best_val, total_flops
