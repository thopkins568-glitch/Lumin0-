# tension_solver.py
import numpy as np
from typing import Callable, Tuple
from flop_counter import GLOBAL_FLOPS as FLOPS

def tension_step(pop: np.ndarray, func: Callable[[np.ndarray], float],
                 tension: float = 0.1, step_size: float = 0.1) -> np.ndarray:
    """One iteration of tension-augmented optimizer: baseline + coupling toward mean."""
    new_pop = pop.copy()
    n, d = pop.shape
    center = np.mean(pop, axis=0)
    FLOPS.add(n * d * 2)  # mean computation heuristic
    for i in range(n):
        FLOPS.add(d * 3)
        delta = np.random.normal(loc=0.0, scale=step_size, size=(d,))
        coupling = tension * (center - pop[i])
        FLOPS.add(d * 2)
        candidate = pop[i] + delta + coupling
        cur_val = func(pop[i])
        cand_val = func(candidate)
        FLOPS.add(20)
        if cand_val < cur_val:
            new_pop[i] = candidate
    return new_pop

def tension_run(func: Callable[[np.ndarray], float],
                dim: int = 8,
                pop_size: int = 32,
                steps: int = 100,
                tension: float = 0.1,
                step_size: float = 0.1) -> Tuple[np.ndarray, list]:
    """Run tension solver; returns final population and path (mean-objective per step)."""
    pop = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))
    path = []
    for _ in range(steps):
        pop = tension_step(pop, func, tension=tension, step_size=step_size)
        path.append(float(func(pop.mean(axis=0))))
    return pop, path
