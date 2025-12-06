"""
lumin0_core.py

Core Lumin0 benchmark engine.
Provides:
 - ExperimentResult class
 - FLOPCounter for consistent FLOP measurement
 - evaluate_population()
 - run_baseline()
 - run_tension()
 - run_experiment()

This is a minimal, clean implementation designed to match the README.
"""

import time
import numpy as np


# -------------------------------------------------
# FLOP COUNTER
# -------------------------------------------------
class FLOPCounter:
    def __init__(self):
        self.count = 0

    def add(self, n: int):
        self.count += int(n)

    def reset(self):
        self.count = 0

    def value(self):
        return int(self.count)


FLOPS = FLOPCounter()


# -------------------------------------------------
# Standard benchmark functions
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
# DATA CLASS
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
            "flop_savings": self.baseline_flops - self.tension_flops
            if (self.baseline_flops is not None and self.tension_flops is not None)
            else None,
            "wall_time_s": self.wall_time_s,
        }


# -------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------

def evaluate_population(pop, func):
    """Evaluate average fitness of the population (returns float)."""
    vals = np.array([func(ind) for ind in pop])
    return float(np.mean(vals))


def run_baseline(pop, func, steps):
    """Simple hill-climb baseline."""
    for _ in range(steps):
        for i in range(pop.shape[0]):
            delta = np.random.normal(0, 0.15, size=pop.shape[1])
            cand = pop[i] + delta
            if func(cand) < func(pop[i]):
                pop[i] = cand
    return evaluate_population(pop, func)


def run_tension(pop, func, steps, tension_strength):
    """Same as baseline, but pulls toward center each step."""
    for _ in range(steps):
        center = pop.mean(axis=0)
        FLOPS.add(pop.size)  # tension pull cost
        for i in range(pop.shape[0]):
            delta = np.random.normal(0, 0.15, size=pop.shape[1])
            cand = pop[i] + delta + tension_strength * (center - pop[i])
            if func(cand) < func(pop[i]):
                pop[i] = cand
    return evaluate_population(pop, func)


# -------------------------------------------------
# MAIN API
# -------------------------------------------------

def run_experiment(
    func_name="rastrigin",
    dim=8,
    pop_size=32,
    steps=100,
    tension_strength=0.1,
    seed=None,
    save_json=False,
):
    """Runs baseline and tension solver, returns ExperimentResult."""

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
    base_flops = FLOPS.value()

    # tension
    pop_t = pop0.copy()
    FLOPS.reset()
    best_t = run_tension(pop_t, func, steps, tension_strength)
    tension_flops = FLOPS.value()

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
