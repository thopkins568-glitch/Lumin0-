"""
lumin0_core.py

Clean, minimal full replacement.
Uses the shared GLOBAL_FLOPS from flop_counter.py.
Provides:
 - ExperimentResult class
 - evaluate_population()
 - run_baseline()
 - run_tension()
 - run_experiment()
"""

import time
import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS


# -------------------------------------------------
# BENCHMARK FUNCTIONS
# -------------------------------------------------
def rastrigin(x: np.ndarray) -> float:
    n = x.size
    FLOPS.add(n * 6)
    A = 10.0
    return A * n + np.sum(x * x - A * np.cos(2.0 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    n = x.size
    FLOPS.add(n * 10)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


FUNCTIONS = {
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
}


# -------------------------------------------------
# EXPERIMENT RESULT
# -------------------------------------------------
class ExperimentResult:
    def __init__(
        self,
        problem,
        dim,
        pop_size,
        steps,
        tension_strength,
        baseline_best,
        baseline_flops,
        tension_best,
        tension_flops,
        wall_time_s,
    ):
        self.problem = problem
        self.dim = dim
        self.pop_size = pop_size
        self.steps = steps
        self.tension_strength = tension_strength
        self.baseline_best = baseline_best
        self.baseline_flops = baseline_flops
        self.tension_best = tension_best
        self.tension_flops = tension_flops
        self.wall_time_s = wall_time_s

    def to_dict(self):
        return {
            "problem": self.problem,
            "dim": self.dim,
            "population": self.pop_size,
            "steps": self.steps,
            "tension_strength": self.tension_strength,
            "baseline": {
                "best_value": self.baseline_best,
                "flops": self.baseline_flops,
            },
            "tension": {
                "best_value": self.tension_best,
                "flops": self.tension_flops,
            },
            "flop_savings": self.baseline_flops - self.tension_flops,
            "wall_time_s": self.wall_time_s,
        }


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def evaluate_population(pop, func):
    """Return mean fitness of population."""
    vals = np.array([func(ind) for ind in pop])
    return float(np.mean(vals))


# -------------------------------------------------
# BASELINE SOLVER
# -------------------------------------------------
def run_baseline(pop, func, steps):
    """Simple random local search with greedy accept."""
    for _ in range(steps):
        for i in range(pop.shape[0]):
            delta = np.random.normal(0, 0.15, size=pop.shape[1])
            cand = pop[i] + delta
            if func(cand) < func(pop[i]):
                pop[i] = cand
    return evaluate_population(pop, func)


# -------------------------------------------------
# TENSION SOLVER
# -------------------------------------------------
def run_tension(pop, func, steps, tension_strength):
    """Local search with center-pull tension field."""
    for _ in range(steps):
        center = pop.mean(axis=0)
        FLOPS.add(pop.size)  # tension overhead cost

        for i in range(pop.shape[0]):
            delta = np.random.normal(0, 0.15, size=pop.shape[1])
            cand = pop[i] + delta + tension_strength * (center - pop[i])
            if func(cand) < func(pop[i]):
                pop[i] = cand

    return evaluate_population(pop, func)


# -------------------------------------------------
# MAIN EXPERIMENT API
# -------------------------------------------------
def run_experiment(
    func_name="rastrigin",
    dim=8,
    pop_size=32,
    steps=100,
    tension_strength=0.1,
    seed=None,
):
    """Runs baseline + tension solver and returns ExperimentResult."""

    if seed is not None:
        np.random.seed(seed)

    if func_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {func_name}")

    func = FUNCTIONS[func_name]
    pop0 = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))

    start = time.time()

    # baseline
    pop_b = pop0.copy()
    FLOPS.reset()
    best_base = run_baseline(pop_b, func, steps)
    base_flops = FLOPS.snapshot()

    # tension
    pop_t = pop0.copy()
    FLOPS.reset()
    best_t = run_tension(pop_t, func, steps, tension_strength)
    tension_flops = FLOPS.snapshot()

    wall = time.time() - start

    return ExperimentResult(
        func_name,
        dim,
        pop_size,
        steps,
        tension_strength,
        best_base,
        base_flops,
        best_t,
        tension_flops,
        wall,
 )
