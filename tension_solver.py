# tension_solver.py
"""
Tension-augmented solver with convergence and FLOP reporting.
"""
import numpy as np
from typing import Callable, Tuple
from flop_counter import GLOBAL_FLOPS as FLOPS

def tension_step(pop: np.ndarray, func: Callable[[np.ndarray], float],
                 tension: float = 0.1, step_size: float = 0.1) -> np.ndarray:
    new_pop = pop.copy()
    n, d = pop.shape
    center = np.mean(pop, axis=0)
    FLOPS.add(n * d * 2)
    for i in range(n):
        FLOPS.add(d * 3)
        delta = np.random.normal(0.0, step_size, size=(d,))
        FLOPS.add(d * 2)
        coupling = tension * (center - pop[i])
        cur_val = func(pop[i])
        candidate = pop[i] + delta + coupling
        cand_val = func(candidate)
        FLOPS.add(5)
        if cand_val < cur_val:
            new_pop[i] = candidate
    return new_pop

def tension_run(func: Callable[[np.ndarray], float],
                dim: int = 8,
                pop_size: int = 32,
                steps: int = 100,
                tension: float = 0.1,
                step_size: float = 0.1,
                tol: float = 0.0,
                patience: int = 5,
                seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    FLOPS.reset()
    pop = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))
    history = []
    best = float("inf")
    stagnant = 0
    iters = 0
    for _ in range(steps):
        pop = tension_step(pop, func, tension=tension, step_size=step_size)
        iters += 1
        mean_val = float(func(pop.mean(axis=0)))
        history.append(mean_val)
        if mean_val < best - tol:
            best = mean_val
            stagnant = 0
        else:
            stagnant += 1
        if tol is not None and stagnant >= max(1, patience):
            break
    total_flops = FLOPS.snapshot()
    best_final = float(min(func(ind) for ind in pop))
    converged = (stagnant >= max(1, patience))
    return pop, history, best_final, int(total_flops), iters, converged
