"""
tension_solver.py
Core tension-based optimizer for Lumen0 benchmark.
"""

import numpy as np
from typing import Callable, Tuple
from flop_counter import GLOBAL_FLOPS as FLOPS


def tension_step(
    pop: np.ndarray,
    func: Callable[[np.ndarray], float],
    tension: float = 0.1,
    step_size: float = 0.1
) -> np.ndarray:
    """
    One iteration of the tension optimizer:
    - baseline random step
    - coupling toward population mean
    """

    n, d = pop.shape
    new_pop = pop.copy()

    # compute center
    center = np.mean(pop, axis=0)
    FLOPS.add(n * d * 2)   # heuristic FLOPs for mean calculation

    for i in range(n):
        # baseline random step
        FLOPS.add(d * 3)
        delta = np.random.normal(0.0, step_size, size=d)

        # tension coupling
        FLOPS.add(d * 2)
        coupling = tension * (center - pop[i])

        # candidate point
        candidate = pop[i] + delta + coupling

        # evaluate old + new
        cur_val = func(pop[i])
        cand_val = func(candidate)
        FLOPS.add(20)

        if cand_val < cur_val:
            new_pop[i] = candidate

    return new_pop


def tension_run(
    func: Callable[[np.ndarray], float],
    dim: int = 8,
    pop_size: int = 32,
    steps: int = 100,
    tension: float = 0.1,
    step_size: float = 0.1
) -> Tuple[np.ndarray, list, float, int]:
    """
    Runs the tension solver and returns:
      - final population
      - history of mean-objective scores
      - best final value (min over pop)
      - total FLOPs used
    """

    FLOPS.reset()

    pop = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))
    path = []

    for _ in range(steps):
        pop = tension_step(pop, func, tension=tension, step_size=step_size)
        path.append(float(func(pop.mean(axis=0))))

    # compute best value properly
    best_val = min(float(func(ind)) for ind in pop)

    total_flops = FLOPS.snapshot()

    return pop, path, best_val, total_flops
